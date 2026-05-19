# Juggle Classifier — Design

**Date:** 2026-05-19
**Status:** Approved, awaiting implementation plan
**Predecessor:** Logan1904/JuggleNet (kept upstream for ball/pose detection; this project replaces the counter)

## Goal

Replace JuggleNet's heuristic juggle counter with a learned, evaluable, generalizable counter that hits **500 ± 5 juggles** on the ground-truth `Juggles_New.mp4` benchmark while reporting per-event precision/recall on held-out validation video(s).

### Success criteria

- Total count on `Juggles_New.mp4`: **500 ± 5**
- Validation F1 on held-out 2nd video (with ±5-frame matching tolerance): **≥ 0.92**
- Per-foot precision and recall: **≥ 0.90 each**
- Pipeline reproducible: anyone clones the repo, follows the README, can train and reproduce results
- No per-video hand-tuning required — the same trained model and thresholds work on new videos

## Motivation

The current heuristic counter peaks at ~472 juggles on `Juggles_New.mp4` (vs. ground truth ~500). The remaining ~28 missed events cluster in two patterns:

1. **Ball-out-of-frame** moments where ball-Y peak detection has no signal
2. **Back-to-camera** moments where MediaPipe pose tracking flickers but feet are still partially visible

Heuristic fusion of more signals (foot-Y peaks, inter-foot spread) closes some of the gap but plateaus before the target, with each new signal source needing its own hand-tuned prominence threshold and validation window — fragile and unlikely to generalize.

A learned classifier trained on per-event labels can combine all signals (ball, pose, derivatives, distances) without manual rule-writing, and produces calibrated confidence scores that enable a real evaluation framework.

## Architecture

The existing JuggleNet detection pipeline (YOLO ball detector → MediaPipe pose → Kalman smoother) is **kept as-is**. It produces the features. The change is only at the counting layer: replace the heuristic peak-detection counter with a learned per-frame classifier.

```
┌──────────────────────────┐
│ Existing JuggleNet       │
│ detection pipeline       │
│ (YOLO + MediaPipe        │
│ + Kalman)                │
└────────────┬─────────────┘
             │
             ▼
   ┌──────────────────┐    ┌──────────────────────┐
   │ Feature extractor│    │ Labeling CLI         │
   │ → features.csv   │    │ (two-pass tap-mark)  │
   └────────┬─────────┘    │ → labels.csv         │
            │              └──────────┬───────────┘
            │                         │
            ▼                         ▼
        ┌──────────────────────────────┐
        │ Trainer (LightGBM, windowed   │
        │ ±15-frame features, 3-class)  │
        │ → juggle_classifier.pkl       │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐    ┌────────────┐
        │ Predictor (NMS over per-frame │ ─▶ │ Evaluator  │
        │ probs → events.csv)           │    │ (P/R/F1)   │
        └──────────────────────────────┘    └────────────┘
```

All components communicate via CSV files. This lets us run the slow feature extractor *once per video* (~3 min on Colab CUDA, ~8 min local MPS), then iterate on training and evaluation in *seconds*.

### Data interfaces

| File | Path | Schema | Notes |
|---|---|---|---|
| Labels | `labels/<video>.csv` | `frame,foot` | One row per ground-truth juggle. `foot` ∈ {`Left_Foot`, `Right_Foot`} |
| Features | `features/<video>.csv` | `frame,<~30 feature columns>` | One row per video frame |
| Predicted events | `events/<video>.csv` | `frame,foot,confidence` | One row per predicted juggle, post-NMS |

## Components

### 1. Labeling CLI — `tools/label.py`

**Purpose:** Capture ground-truth juggle events from a user watching a video.

**Design:** Two-pass tap-to-mark. Pass 1: user presses space for every left-foot juggle. Pass 2: same video again, space for every right-foot juggle. Reduces cognitive load vs. one-pass left/right discrimination at speed.

**Controls:**
- `space` — mark juggle at current frame
- `p` — pause/resume
- `↑`/`↓` — increase/decrease playback speed (default 0.5×)
- `←`/`→` — seek ±2s
- `u` — undo last mark
- `q` — quit and save

**Output:** appends to `labels/<video>.csv` with `frame,foot`. The ±5-frame matching tolerance during evaluation absorbs human reaction-time error.

**Local-only.** Needs a display and keyboard input — not for Colab.

### 2. Feature extractor — `tools/extract_features.py`

**Purpose:** Run the existing JuggleNet detection pipeline on a video, write one CSV row per frame with ~30 hand-crafted features.

**Features:**

| Group | Columns |
|---|---|
| Ball | `ball_x, ball_y, ball_w, ball_h, ball_conf, ball_visible` |
| L foot | `lfoot_x, lfoot_y, lfoot_dy, lfoot_visible` |
| R foot | `rfoot_x, rfoot_y, rfoot_dy, rfoot_visible` |
| L knee | `lknee_x, lknee_y, lknee_dy` |
| R knee | `rknee_x, rknee_y, rknee_dy` |
| Hip reference | `hip_x, hip_y` |
| Derived distances | `ball_lfoot_dist, ball_rfoot_dist, lfoot_rfoot_dy, lfoot_rfoot_dist` |
| Ball motion | `ball_dy, ball_dy2` (velocity, acceleration in Y) |

Conventions:
- All XY values normalized to [0, 1] by frame dimensions
- `_dy` = first derivative over a centered 3-frame window
- `_visible` = 1 if the underlying detector returned a real detection that frame, 0 if the value is Kalman-extrapolated. This is **signal**, not metadata — the classifier should learn that low-visibility frames are still useful (back-to-camera juggle recovery)
- Missing values represented as NaN; classifier handles natively (LightGBM supports NaN)

### 3. Trainer — `tools/train.py`

**Purpose:** Fit a LightGBM multiclass classifier mapping per-frame windowed features to `{none, Left_Foot, Right_Foot}`.

**Procedure:**
1. Load `features/<video>.csv` + `labels/<video>.csv`
2. Build per-sample feature vector: a window of ±15 frames around each target frame, flattened to ~30 × 31 = 930 features
3. Build labels: 3-class, default `none`, set to the foot at frames marked in labels CSV
4. Class weights computed inversely to frequency (`none` is ~10× more common than `Left_Foot` or `Right_Foot`)
5. Train `LGBMClassifier` with `n_estimators=500`, `num_leaves=31`, `max_depth=-1`, `learning_rate=0.05`, fixed seed (42)
6. Save model to `models/juggle_classifier.pkl`

**Why LightGBM over 1D CNN:** 500 positive examples is too small for a deep model to outperform a boosted tree. Trees handle NaN natively, give us feature importances for diagnosis, train in seconds, and require no GPU. If LightGBM plateaus below target on validation, a temporal CNN is the obvious next step using the same evaluation harness.

### 4. Predictor — `tools/predict.py`

**Purpose:** Run trained classifier over a video's features, emit final juggle events.

**Procedure:**
1. Load model + `features/<video>.csv`
2. For each frame, predict `P(none), P(Left_Foot), P(Right_Foot)`
3. **Non-max suppression:** in a 5-frame sliding window, keep only local maxima of `max(P_Left_Foot, P_Right_Foot)` above a configurable threshold (default 0.5)
4. Assign foot = `argmax(P_Left_Foot, P_Right_Foot)` at each kept frame
5. Write `events/<video>.csv` with `frame,foot,confidence`

The NMS threshold and window are exposed as CLI args so we can sweep them in the evaluation notebook.

### 5. Evaluator — `tools/evaluate.py`

**Purpose:** Compute per-event metrics between predicted events and ground truth.

**Matching:** Bipartite, greedy, with ±5-frame tolerance. A predicted event at frame F is a **true positive** iff there's a ground-truth event of the same foot within ±5 frames AND no other prediction has already claimed that ground-truth event.

**Reports:**
- Precision, recall, F1 — per foot and combined
- Total-count error (predicted total − ground-truth total)
- Confusion: Left-as-Right and Right-as-Left misattribution counts
- Lists of false positives and misses by frame number (for manual inspection in the analyzed video)

### 6. `main.py` integration

Add `--events <events.csv>` flag that, when present, replaces the heuristic counter with an `EventListCounter` driven by a pre-computed events CSV. The canonical classifier flow is therefore two commands:

```
extract_features.py  →  predict.py  →  events/<video>.csv  →  main.py --events ...
```

This decouples model inference from rendering. Inference (slow, CSV-in / CSV-out) runs in Colab; rendering (also slow but interactive) runs locally. Existing flags (`--save-plot`, `--save`, `--feet-only`, `--headless`) keep working — they're independent of the counting layer.

> *Earlier design note (revised during implementation planning):* an in-`main.py` `--classifier <model.pkl>` flag was considered. Streaming inference would have required a rolling-buffer design that re-implements feature derivative computation and edge handling already covered correctly in the offline path. Decoupling via a pre-computed events CSV is the simpler choice.

## Repo structure

```
juggle-classifier/
├── README.md                          # setup for both local + Colab
├── requirements.txt                   # local Mac (MPS) deps
├── requirements-colab.txt             # Colab-specific (CUDA torch)
├── .gitignore                         # .venv, *.mp4, save/, __pycache__
├── docs/superpowers/specs/            # this design doc + future ones
├── notebooks/
│   ├── 01_extract_features.ipynb     # Colab — runs feature extractor
│   ├── 02_train.ipynb                # LightGBM training + sanity eval
│   └── 03_predict_evaluate.ipynb     # batch predict + metrics
├── tools/                             # CLI scripts
│   ├── label.py                       # local-only
│   ├── extract_features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── utils/                             # existing JuggleNet modules
│   ├── vision_estimate.py
│   ├── KalmanFilter.py
│   ├── juggle_counter.py             # kept as heuristic baseline
│   ├── update_predict.py
│   ├── draw_POI.py
│   └── plot_graph.py
├── main.py                            # entry point with --classifier flag
├── models/
│   ├── finetuned.pt                   # YOLO weights (6 MB, in git)
│   └── juggle_classifier.pkl          # trained classifier
├── labels/                            # committed — small text
│   ├── Juggles_New.csv
│   └── <2nd_video>.csv
├── features/                          # committed — ~1.5 MB per video
│   └── Juggles_New.csv
└── events/                            # outputs, gitignored
    └── .gitkeep
```

### Where artifacts live

| Artifact | Git | Drive | Local Desktop |
|---|---|---|---|
| Source code, notebooks | ✓ | (via repo clone) | (via repo clone) |
| YOLO weights | ✓ | | |
| Source `.mp4` videos | ✗ | `/MyDrive/JuggleNet/videos/` | `~/Desktop/*.mp4` |
| Label CSVs | ✓ | (via repo) | (via repo) |
| Feature CSVs | ✓ | (via repo) | (via repo) |
| Trained classifier `.pkl` | ✓ | (via repo) | (via repo) |
| Annotated output `.mp4` | ✗ | `/MyDrive/JuggleNet/output/` | `~/Desktop/` |

Videos stay out of git (size). Everything else is reproducible from labels + features + code.

## Workflows

### Local Mac (interactive)

- **Label collection:** `python tools/label.py ~/Desktop/Juggles_New.mp4` → writes `labels/Juggles_New.csv`. Commit to git.
- **Quick visual sanity:** run prediction + render annotated video for spot-checking
- **Development:** edit code, run small tests

### Colab (CUDA, batch)

- **`01_extract_features.ipynb`** — mounts Drive, clones repo, runs feature extractor on all videos in `/MyDrive/JuggleNet/videos/`, commits feature CSVs back to repo. ~3 min per video on T4.
- **`02_train.ipynb`** — loads features + labels from repo, trains LightGBM, saves model, commits. Seconds.
- **`03_predict_evaluate.ipynb`** — loads model, runs predictor on all features, evaluates against labels, reports metrics. Seconds. This is the iteration loop for tuning the classifier.

### Sequence to first trained model

1. Create repo, push current state (one-time)
2. **Local:** label `Juggles_New.mp4` (Pass 1 = left, Pass 2 = right). Commit `labels/Juggles_New.csv`.
3. **Local:** same for 2nd video (`Clark_Juggles1.mov` or `0511.mp4`). Commit.
4. **Colab:** extract features for both videos. Commit features.
5. **Colab:** train classifier on `Juggles_New` features + labels. Save model.
6. **Colab:** evaluate on 2nd video. Iterate on hyperparameters / features in `03_predict_evaluate.ipynb` until validation F1 stops improving.
7. **Local:** run final classifier on `Juggles_New` (sanity), check total count vs 500 ± 5.

### Colab specifics

- Drive mount at notebook start: `from google.colab import drive; drive.mount('/content/drive')`
- Repo cloned fresh each session: `git clone https://github.com/<user>/juggle-classifier.git`
- For `git push` of generated artifacts (features, model), use a GitHub PAT stored in Colab Secrets (or pull manually)
- Pro session budget ~24h; far more than needed (extraction ~3 min/video, training seconds)

## Testing strategy

| Component | Test |
|---|---|
| Evaluator | Unit tests with synthetic predictions + labels covering: exact match, ±5-frame match, foot mismatch, missing GT, extra prediction, double-claim of one GT event |
| Feature extractor | Integration: run on bundled `source_data/Vid1.mp4`, assert CSV exists, row count = frame count, all expected columns present |
| Trainer | Smoke test: tiny synthetic features + labels, model trains without error, can predict |
| Predictor | Smoke test: feed extractor output through, assert events CSV is non-empty |
| End-to-end | One script that runs all 5 stages on `Vid1.mp4` |

Skipping deep unit tests on labeling (interactive input) and on YOLO/MediaPipe internals (already exercised by upstream). Model accuracy isn't tested by assertion — it's tested by cross-validation and held-out evaluation in `03_predict_evaluate.ipynb`.

## Error handling

Fail loud at boundaries, lenient inside:

- Missing video or labels file → clear error message, exit 1
- Missing detection at a frame (no ball, no foot) → fill NaN, set `*_visible=0`. This is signal, not error
- Bad row in CSV → drop with warning, continue
- Frame mismatch between labels and features (e.g. labeling tool used different fps reading) → fail loudly with diagnostic
- Empty labels → trainer refuses with `"need ≥ N positive examples per class"` message
- Two labels at same frame → keep one, warn
- Labels with frame number out of video range → drop with warning

## Reproducibility

- Random seed pinned in trainer (`seed=42`)
- `requirements.txt` with exact versions — `mediapipe==0.10.21` is the known-good pin (newer versions drop `mp.solutions`)
- Feature extractor is deterministic given the same video + same device
- Note in README: CUDA Colab and MPS local will produce *very slightly* different YOLO confidence scores — within noise floor, won't affect classifier outputs meaningfully
- All artifacts (labels, features, model) are checked into git; full re-run is `git clone && python tools/predict.py && python tools/evaluate.py`

## Open questions deferred to implementation

- **Drive folder layout convention**: stick with `/MyDrive/JuggleNet/{videos,output}/` or use a project-specific name? Decision in README.
- **Second labeled video**: `Clark_Juggles1.mov` vs `0511.mp4`. Either works; user picks during implementation based on which has clearer juggle action.
- **NMS threshold default**: 0.5 is a starting guess. The eval notebook will sweep and pick the value that maximizes F1 on validation.

## Out of scope (explicitly)

- Multi-ball juggling
- Multi-player video
- Real-time / live inference (the system runs on recorded video)
- Skill-level analysis (height of kicks, control quality)
- Web UI / mobile app
- Auto-labeling via existing model (user explicitly chose manual labeling for higher quality)
