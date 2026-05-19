# juggle-classifier

A learned juggle counter built on top of [Logan1904/JuggleNet](https://github.com/Logan1904/JuggleNet).

JuggleNet's YOLO ball detector + MediaPipe pose pipeline produces per-frame features.
This project replaces its heuristic peak-detection counter with a LightGBM classifier
trained on per-event labels.

## Layout

```
juggle-classifier/
├── tools/                  CLI scripts
│   ├── label.py            two-pass tap-to-mark labeler (local only)
│   ├── extract_features.py YOLO+MediaPipe+Kalman → per-frame feature CSV
│   ├── train.py            LightGBM trainer
│   ├── predict.py          model + NMS → events CSV
│   └── evaluate.py         per-event precision/recall vs ground truth
├── notebooks/              Colab notebooks for the cloud workflow
├── utils/                  upstream JuggleNet detection modules (kept as-is)
├── main.py                 entry point with --events flag
├── models/                 finetuned YOLO weights + trained classifier
├── labels/                 ground-truth labels (frame, foot) — committed
├── features/               per-frame feature CSVs — committed
└── docs/superpowers/       design spec + implementation plan
```

## Local setup (Mac, Apple Silicon)

```bash
git clone https://github.com/PatrickJReed/juggle-classifier.git
cd juggle-classifier
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`mediapipe==0.10.21` is the pinned version (newer versions removed `mp.solutions`).

## Workflows

### Local (interactive labeling, visual sanity check)

```bash
# Label a video (two-pass: Left, then Right)
.venv/bin/python -m tools.label ~/Desktop/Juggles_New.mp4

# Predict events with the classifier (offline)
.venv/bin/python -m tools.extract_features --video ~/Desktop/Juggles_New.mp4 \
    --output features/Juggles_New.csv
.venv/bin/python -m tools.predict --features features/Juggles_New.csv \
    --model models/juggle_classifier.pkl --output events/Juggles_New.csv

# Render annotated video using those events
.venv/bin/python main.py --video ~/Desktop/Juggles_New.mp4 \
    --events events/Juggles_New.csv --save ~/Desktop --feet-only
```

### Colab (feature extraction, training, evaluation)

Open these notebooks in Colab (with a CUDA runtime):

1. `notebooks/01_extract_features.ipynb` — runs the feature extractor on every video in `/MyDrive/JuggleNet/videos/`
2. `notebooks/02_train.ipynb` — trains on a labeled video's features + labels
3. `notebooks/03_predict_evaluate.ipynb` — predicts on any feature CSV, evaluates against labels, sweeps threshold

Each notebook:
- Mounts Google Drive
- Clones this repo to `/content/juggle-classifier`
- Installs `requirements-colab.txt`
- (Optional) Pushes generated artifacts back to GitHub via a PAT stored in Colab Secrets

## Sequence to first trained model

1. **Local:** label `Juggles_New.mp4` (~500 events) and a second held-out video. Commit `labels/*.csv`.
2. **Colab (01):** extract features for both videos. Commit `features/*.csv`.
3. **Colab (02):** train on `Juggles_New`. Commit `models/juggle_classifier.pkl`.
4. **Colab (03):** evaluate on the held-out video. Sweep threshold/window to maximize F1.
5. **Local:** run the final model end-to-end on `Juggles_New`, confirm total count is 500 ± 5.

## Data interfaces

All artifacts are CSV.

| File | Schema | Notes |
|------|--------|-------|
| `labels/<video>.csv` | `frame,foot` | One row per ground-truth juggle. `foot` ∈ {`Left_Foot`, `Right_Foot`} |
| `features/<video>.csv` | `frame, <~28 feature columns>` | One row per video frame |
| `events/<video>.csv` | `frame,foot,confidence` | One row per predicted juggle |

## Running the tests

```bash
.venv/bin/python -m pytest -v
```

The end-to-end smoke test runs feature extraction on `source_data/Vid1.mp4` — takes ~30-60s.

## Attribution

Detection pipeline (YOLO ball detector + MediaPipe pose + Kalman) is from [Logan1904/JuggleNet](https://github.com/Logan1904/JuggleNet). The classifier counter and supporting infrastructure are added on top.
