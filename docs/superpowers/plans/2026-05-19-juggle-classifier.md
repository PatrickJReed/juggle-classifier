# Juggle Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace JuggleNet's heuristic juggle counter with a learned per-frame classifier, hitting 500 ± 5 juggles on the `Juggles_New.mp4` benchmark with measurable precision/recall on held-out validation video.

**Architecture:** Keep JuggleNet's detection pipeline (YOLO + MediaPipe + Kalman) as the feature source. Replace the counter layer with LightGBM trained on per-event labels, using CSV files as interfaces between extract / label / train / predict / evaluate stages. Local Mac for labeling, Colab for training.

**Tech Stack:** Python 3.12, OpenCV, MediaPipe 0.10.21, Ultralytics YOLOv8, scipy, scikit-learn, LightGBM, pandas, Jupyter (Colab).

**Source spec:** `docs/superpowers/specs/2026-05-19-juggle-classifier-design.md`

**Working directory:** `/Users/patrickreed/Sandbox/Juggles/juggle-classifier/` (already exists; spec is committed there).

**Existing code source:** `/Users/patrickreed/Sandbox/Juggles/JuggleNet/` — copy from here.

---

## Task 1: Repo bootstrap (scaffolding + .gitignore + requirements)

**Files:**
- Create: `juggle-classifier/.gitignore`
- Create: `juggle-classifier/requirements.txt`
- Create: `juggle-classifier/requirements-colab.txt`
- Create: `juggle-classifier/README.md` (skeleton)
- Create: `juggle-classifier/labels/.gitkeep`
- Create: `juggle-classifier/features/.gitkeep`
- Create: `juggle-classifier/events/.gitkeep`
- Create: `juggle-classifier/tests/__init__.py`
- Create: `juggle-classifier/tools/__init__.py`

- [ ] **Step 1.1: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
.venv/
venv/
env/
*.egg-info/

# Videos and large media (kept on Desktop / Drive, not in git)
*.mp4
*.mov
*.avi
*.mkv

# Generated outputs
events/*.csv
save/
runs/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
.DS_Store

# Pyenv
.python-version
```

- [ ] **Step 1.2: Create `requirements.txt`** (local Mac, Python 3.12)

```
numpy<2
ultralytics>=8.4
mediapipe==0.10.21
opencv-python>=4.10
scipy>=1.13
matplotlib>=3.10
pandas>=2.2
scikit-learn>=1.5
lightgbm>=4.5
torch>=2.7
torchvision>=0.22
pytest>=8.3
```

- [ ] **Step 1.3: Create `requirements-colab.txt`** (Colab provides torch+CUDA; we install on top)

```
# requirements-colab.txt — installed in addition to Colab's pre-baked torch.
# Colab already has torch with CUDA; do not reinstall torch here.
numpy<2
ultralytics>=8.4
mediapipe==0.10.21
opencv-python>=4.10
scipy>=1.13
matplotlib>=3.10
pandas>=2.2
scikit-learn>=1.5
lightgbm>=4.5
pytest>=8.3
```

- [ ] **Step 1.4: Create `README.md` skeleton** (we flesh out full content in Task 11)

```markdown
# juggle-classifier

A learned juggle counter built on top of [Logan1904/JuggleNet](https://github.com/Logan1904/JuggleNet).

JuggleNet's per-frame YOLO ball detector + MediaPipe pose pipeline produces the features.
This project replaces the heuristic juggle counter with a LightGBM classifier trained on
per-event labels.

**Status:** Implementation in progress. See `docs/superpowers/specs/` for design and
`docs/superpowers/plans/` for the implementation plan.
```

- [ ] **Step 1.5: Create empty marker files**

```bash
cd /Users/patrickreed/Sandbox/Juggles/juggle-classifier
touch labels/.gitkeep features/.gitkeep events/.gitkeep
mkdir -p tools tests
touch tools/__init__.py tests/__init__.py
```

- [ ] **Step 1.6: Initialize git and make first commit**

```bash
cd /Users/patrickreed/Sandbox/Juggles/juggle-classifier
git init
git add .gitignore requirements.txt requirements-colab.txt README.md
git add labels/.gitkeep features/.gitkeep events/.gitkeep
git add tools/__init__.py tests/__init__.py
git add docs/superpowers/specs/2026-05-19-juggle-classifier-design.md
git add docs/superpowers/plans/2026-05-19-juggle-classifier.md
git commit -m "chore: initial repo scaffolding"
```

Expected: clean commit, no errors.

---

## Task 2: Copy existing JuggleNet detection code

**Files:**
- Copy from `/Users/patrickreed/Sandbox/Juggles/JuggleNet/utils/` → `juggle-classifier/utils/`
- Copy from `/Users/patrickreed/Sandbox/Juggles/JuggleNet/main.py` → `juggle-classifier/main.py`
- Copy from `/Users/patrickreed/Sandbox/Juggles/JuggleNet/models/finetuned.pt` → `juggle-classifier/models/finetuned.pt`
- Copy from `/Users/patrickreed/Sandbox/Juggles/JuggleNet/source_data/Vid1.mp4` → `juggle-classifier/source_data/Vid1.mp4` (used as a small test fixture)

- [ ] **Step 2.1: Copy the directories and files**

```bash
cd /Users/patrickreed/Sandbox/Juggles/juggle-classifier
cp -R /Users/patrickreed/Sandbox/Juggles/JuggleNet/utils ./utils
cp /Users/patrickreed/Sandbox/Juggles/JuggleNet/main.py ./main.py
mkdir -p models source_data
cp /Users/patrickreed/Sandbox/Juggles/JuggleNet/models/finetuned.pt ./models/finetuned.pt
cp /Users/patrickreed/Sandbox/Juggles/JuggleNet/source_data/Vid1.mp4 ./source_data/Vid1.mp4
```

- [ ] **Step 2.2: Verify file inventory**

Run: `ls -lh utils/ models/ main.py`

Expected: see `vision_estimate.py`, `KalmanFilter.py`, `juggle_counter.py`, `update_predict.py`, `draw_POI.py`, `plot_graph.py` in `utils/`; `finetuned.pt` (~6 MB) in `models/`; `main.py` at repo root.

- [ ] **Step 2.3: Force-add Vid1.mp4 (overrides .gitignore for this fixture)**

Vid1.mp4 is 8.5 MB and we want it in git as a test fixture so anyone can run the smoke test. `.gitignore` excludes all `*.mp4`; override with `-f`.

```bash
git add -f source_data/Vid1.mp4
git add models/finetuned.pt
git add utils/*.py main.py
git commit -m "chore: import existing JuggleNet detection pipeline"
```

Expected: commit succeeds. ~6 MB YOLO weights + 8.5 MB test video committed.

---

## Task 3: Set up local venv and verify pipeline still works

**Files:**
- Create: `juggle-classifier/.venv/` (local only, gitignored)

- [ ] **Step 3.1: Create virtualenv with Python 3.12**

```bash
cd /Users/patrickreed/Sandbox/Juggles/juggle-classifier
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
```

- [ ] **Step 3.2: Install requirements**

```bash
.venv/bin/pip install -r requirements.txt
```

Expected: completes without error. mediapipe 0.10.21 is the version that retains `mp.solutions` (newer versions removed it).

- [ ] **Step 3.3: Smoke-test pipeline on bundled Vid1.mp4**

```bash
mkdir -p save
.venv/bin/python -u main.py --video source_data/Vid1.mp4 --save save --headless --feet-only 2>&1 | grep -vE 'inference_feedback_manager|gl_context|XNNPACK|landmark_projection|absl::InitializeLog|NMS time limit' | tail -15
```

Expected output ends with:
```
Running on pre-recorded video: source_data/Vid1.mp4
Video FPS: 30.0
Video stream ended or camera disconnected.
```
And `save/Vid1_Analysed.mp4` exists.

If it fails: most likely cause is mediapipe version. Verify `.venv/bin/pip show mediapipe | grep Version` returns `0.10.21`.

- [ ] **Step 3.4: Commit (no code changed, but mark milestone)**

```bash
# Nothing new to commit unless the smoke surfaced a bug.
# Skip the commit if there are no changes.
git status
```

If files were modified to fix something, commit:

```bash
git commit -am "fix: <describe the fix>"
```

---

## Task 4: Evaluator (TDD)

The evaluator is the most testable component and benefits most from TDD because the matching logic has multiple subtle cases.

**Files:**
- Create: `juggle-classifier/tools/evaluate.py`
- Create: `juggle-classifier/tests/test_evaluate.py`

- [ ] **Step 4.1: Write the failing test file**

Create `tests/test_evaluate.py`:

```python
"""Tests for the per-event evaluator.

The evaluator does greedy bipartite matching between predicted and ground-truth
juggle events with a ±tolerance frame window and same-foot constraint.
"""
import pandas as pd
import pytest

from tools.evaluate import match_events, metrics


def _df(rows, cols=('frame', 'foot')):
    return pd.DataFrame(rows, columns=cols)


def test_exact_match_same_foot():
    pred = _df([(10, 'Left_Foot')])
    gt = _df([(10, 'Left_Foot')])
    tp_p, tp_g, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == [0]
    assert tp_g == [0]
    assert fp_p == []
    assert fn_g == []


def test_within_tolerance_matches():
    pred = _df([(15, 'Left_Foot')])
    gt = _df([(10, 'Left_Foot')])
    tp_p, _, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == [0]
    assert fp_p == []
    assert fn_g == []


def test_outside_tolerance_does_not_match():
    pred = _df([(16, 'Left_Foot')])
    gt = _df([(10, 'Left_Foot')])
    tp_p, _, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == []
    assert fp_p == [0]
    assert fn_g == [0]


def test_foot_mismatch_is_not_a_match():
    pred = _df([(10, 'Left_Foot')])
    gt = _df([(10, 'Right_Foot')])
    tp_p, _, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == []
    assert fp_p == [0]
    assert fn_g == [0]


def test_extra_prediction_within_window_is_false_positive():
    """Two predictions within tolerance of one GT — first matches, second is FP."""
    pred = _df([(10, 'Left_Foot'), (11, 'Left_Foot')])
    gt = _df([(10, 'Left_Foot')])
    tp_p, _, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == [0]
    assert fp_p == [1]
    assert fn_g == []


def test_missing_gt_is_false_negative():
    pred = _df([])
    gt = _df([(10, 'Left_Foot')])
    tp_p, _, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == []
    assert fp_p == []
    assert fn_g == [0]


def test_multiple_gt_one_prediction():
    """Two GT, one pred — pred matches the first GT in frame order, second GT is a miss."""
    pred = _df([(10, 'Left_Foot')])
    gt = _df([(10, 'Left_Foot'), (15, 'Left_Foot')])
    tp_p, tp_g, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    assert tp_p == [0]
    assert tp_g == [0]
    assert fn_g == [1]


def test_one_to_one_greedy_assignment():
    """Each GT can only be matched once."""
    pred = _df([(10, 'Left_Foot'), (12, 'Left_Foot')])
    gt = _df([(10, 'Left_Foot'), (12, 'Left_Foot')])
    tp_p, tp_g, fp_p, fn_g = match_events(pred, gt, tolerance=5)
    # Each prediction should claim its closest still-unclaimed GT in frame order.
    assert sorted(tp_p) == [0, 1]
    assert sorted(tp_g) == [0, 1]
    assert fp_p == []
    assert fn_g == []


def test_metrics_basic():
    p, r, f1 = metrics(tp=8, fp=2, fn=3)
    assert abs(p - 8/10) < 1e-9
    assert abs(r - 8/11) < 1e-9
    assert abs(f1 - 2 * (8/10) * (8/11) / ((8/10) + (8/11))) < 1e-9


def test_metrics_zero_division_safe():
    # No predictions, no ground truth → all zero (and no exception)
    p, r, f1 = metrics(tp=0, fp=0, fn=0)
    assert p == 0.0 and r == 0.0 and f1 == 0.0
```

- [ ] **Step 4.2: Run tests, verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: ImportError or ModuleNotFoundError on `from tools.evaluate import ...` (file doesn't exist yet).

- [ ] **Step 4.3: Implement `tools/evaluate.py`**

```python
"""Per-event juggle counter evaluator.

Reads a predictions CSV (frame, foot, confidence) and a ground-truth labels
CSV (frame, foot). Matches predictions to ground truth using a greedy bipartite
algorithm with a ±tolerance frame window AND same-foot constraint. Reports
precision, recall, F1 (combined and per foot), total count error, and lists
of false positives and misses for manual inspection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def match_events(pred_df: pd.DataFrame, gt_df: pd.DataFrame, tolerance: int = 5):
    """
    Greedy bipartite matching of predictions to ground-truth events.

    A predicted event matches a ground-truth event when:
      - same foot, AND
      - |pred.frame - gt.frame| <= tolerance, AND
      - the ground-truth event has not been claimed by a previous prediction

    Iteration is in pred frame-order; for each prediction the first unclaimed
    GT (in GT frame-order) within tolerance and matching foot is taken.

    Returns:
      tp_p: list[int] - indices into pred_df of true positives
      tp_g: list[int] - indices into gt_df paired with tp_p
      fp_p: list[int] - indices into pred_df of false positives
      fn_g: list[int] - indices into gt_df of misses
    """
    pred = pred_df.sort_values('frame').reset_index()  # preserves original index in 'index'
    gt = gt_df.sort_values('frame').reset_index()

    gt_claimed = np.zeros(len(gt), dtype=bool)
    tp_p: list[int] = []
    tp_g: list[int] = []
    fp_p: list[int] = []

    for _, prow in pred.iterrows():
        pframe = int(prow['frame'])
        pfoot = prow['foot']
        match_gt_pos: int | None = None
        for gi, grow in gt.iterrows():
            if gt_claimed[gi]:
                continue
            if grow['foot'] != pfoot:
                continue
            if abs(int(grow['frame']) - pframe) <= tolerance:
                match_gt_pos = gi
                break
        if match_gt_pos is not None:
            gt_claimed[match_gt_pos] = True
            tp_p.append(int(prow['index']))
            tp_g.append(int(gt.iloc[match_gt_pos]['index']))
        else:
            fp_p.append(int(prow['index']))

    fn_g = [int(gt.iloc[i]['index']) for i in range(len(gt)) if not gt_claimed[i]]
    return tp_p, tp_g, fp_p, fn_g


def metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return precision, recall, F1. All zero when no predictions and no GT."""
    if tp + fp == 0 and tp + fn == 0:
        return 0.0, 0.0, 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _print_report(pred: pd.DataFrame, gt: pd.DataFrame, tolerance: int) -> None:
    tp_p, _, fp_p, fn_g = match_events(pred, gt, tolerance)
    p, r, f1 = metrics(len(tp_p), len(fp_p), len(fn_g))

    print(f"Total predictions: {len(pred)}")
    print(f"Total ground truth: {len(gt)}")
    print(f"Count error: {len(pred) - len(gt):+d}")
    print()
    print(f"Combined (tol=±{tolerance}): P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    for foot in ['Left_Foot', 'Right_Foot']:
        pred_f = pred[pred['foot'] == foot].reset_index(drop=True)
        gt_f = gt[gt['foot'] == foot].reset_index(drop=True)
        tp_pf, _, fp_pf, fn_gf = match_events(pred_f, gt_f, tolerance)
        pf, rf, f1f = metrics(len(tp_pf), len(fp_pf), len(fn_gf))
        print(f"{foot:11s}: P={pf:.3f}  R={rf:.3f}  F1={f1f:.3f}  "
              f"(GT={len(gt_f)}, Pred={len(pred_f)})")

    fps = pred.iloc[fp_p].sort_values('frame')
    misses = gt.iloc[fn_g].sort_values('frame')
    print(f"\nFalse positives ({len(fps)}):")
    if len(fps) == 0:
        print("  (none)")
    else:
        cols = [c for c in ('frame', 'foot', 'confidence') if c in fps.columns]
        print(fps[cols].to_string(index=False))
    print(f"\nMisses ({len(misses)}):")
    if len(misses) == 0:
        print("  (none)")
    else:
        print(misses[['frame', 'foot']].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted juggle events against ground truth.")
    parser.add_argument('--pred', type=str, required=True, help='Predictions CSV (frame,foot,confidence)')
    parser.add_argument('--gt', type=str, required=True, help='Ground-truth labels CSV (frame,foot)')
    parser.add_argument('--tolerance', type=int, default=5, help='Frame-distance tolerance for a match (default 5)')
    args = parser.parse_args()

    pred = pd.read_csv(args.pred)
    gt = pd.read_csv(args.gt)
    _print_report(pred, gt, args.tolerance)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4.4: Run tests, verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: 9 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add tools/evaluate.py tests/test_evaluate.py
git commit -m "feat: per-event evaluator with greedy bipartite matching"
```

---

## Task 5: Feature extractor

**Files:**
- Create: `juggle-classifier/tools/extract_features.py`
- Create: `juggle-classifier/tests/test_extract_features.py`

- [ ] **Step 5.1: Write the integration test**

Create `tests/test_extract_features.py`:

```python
"""Integration test for the feature extractor.

Runs the extractor on the bundled Vid1.mp4 test fixture and asserts the
output CSV has the expected schema and a sensible row count.
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
VID = REPO / "source_data" / "Vid1.mp4"


@pytest.fixture(scope="module")
def features_csv(tmp_path_factory):
    if not VID.exists():
        pytest.skip(f"test fixture missing: {VID}")
    out = tmp_path_factory.mktemp("features") / "vid1.csv"
    subprocess.run(
        [sys.executable, "-m", "tools.extract_features",
         "--video", str(VID), "--output", str(out)],
        check=True, cwd=REPO,
    )
    return out


def test_csv_exists_and_nonempty(features_csv):
    df = pd.read_csv(features_csv)
    assert len(df) > 0, "feature CSV is empty"


def test_csv_has_expected_columns(features_csv):
    df = pd.read_csv(features_csv)
    expected = {
        'frame',
        'ball_x', 'ball_y', 'ball_w', 'ball_h', 'ball_visible',
        'lfoot_x', 'lfoot_y', 'lfoot_visible',
        'rfoot_x', 'rfoot_y', 'rfoot_visible',
        'lknee_x', 'lknee_y',
        'rknee_x', 'rknee_y',
        'head_x', 'head_y',
        'ball_dy', 'ball_dy2',
        'lfoot_dy', 'rfoot_dy',
        'lknee_dy', 'rknee_dy',
        'lfoot_rfoot_dy', 'lfoot_rfoot_dist',
        'ball_lfoot_dist', 'ball_rfoot_dist',
    }
    missing = expected - set(df.columns)
    assert not missing, f"missing columns: {missing}"


def test_frame_column_is_dense_index(features_csv):
    df = pd.read_csv(features_csv)
    assert df['frame'].tolist() == list(range(len(df))), \
        "frame column should be a dense 0..N-1 index"
```

- [ ] **Step 5.2: Implement `tools/extract_features.py`**

```python
"""Per-frame feature extractor for juggle classification.

Runs the existing JuggleNet detection pipeline (YOLO + MediaPipe + Kalman) on
a video and writes one CSV row per frame with ~28 hand-crafted features.

For each tracked body part / ball we record:
  - the Kalman-smoothed XY position (used as the position feature)
  - a *_visible flag: 1 if the underlying detector returned a real measurement
    on that frame, 0 if the value is Kalman-extrapolated from prior frames.

This split is important: low-visibility frames are signal, not noise — they're
how the classifier learns to attribute juggles when one signal source briefly
drops out (back-to-camera, ball-out-of-frame).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Make sibling 'utils' package importable when run as a script.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.vision_estimate import get_POI  # noqa: E402
from utils.update_predict import update_measurements, predict_KF  # noqa: E402

PARTS = ['Ball', 'Head', 'Left_Knee', 'Right_Knee', 'Left_Foot', 'Right_Foot']


def _initial_history():
    return {part: np.empty(shape=(0, 4)) for part in PARTS}


def extract(video_path: Path, ball_conf: float, track: bool) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    measurements = _initial_history()
    predictions = _initial_history()
    rows: list[dict] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pois = get_POI(frame, ball_conf=ball_conf, track=track)
        # Raw measurements drive _visible flags
        raw_visible = {
            'Ball':       not (pois['Ball'][0] is None or pois['Ball'][0] != pois['Ball'][0]),
            'Left_Foot':  not (pois['Left_Foot'][0] is None or pois['Left_Foot'][0] != pois['Left_Foot'][0]),
            'Right_Foot': not (pois['Right_Foot'][0] is None or pois['Right_Foot'][0] != pois['Right_Foot'][0]),
        }
        measurements = update_measurements(measurements, pois)
        predictions = predict_KF(measurements, predictions)

        # Kalman-smoothed XY for each part (last row of history)
        def _xy(part):
            arr = predictions[part]
            if len(arr) == 0:
                return (np.nan, np.nan)
            return (float(arr[-1, 0]), float(arr[-1, 1]))

        ball_x, ball_y = _xy('Ball')
        ball_w = float(predictions['Ball'][-1, 2]) if len(predictions['Ball']) else np.nan
        ball_h = float(predictions['Ball'][-1, 3]) if len(predictions['Ball']) else np.nan
        lfoot_x, lfoot_y = _xy('Left_Foot')
        rfoot_x, rfoot_y = _xy('Right_Foot')
        lknee_x, lknee_y = _xy('Left_Knee')
        rknee_x, rknee_y = _xy('Right_Knee')
        head_x, head_y = _xy('Head')

        rows.append({
            'frame': frame_idx,
            'ball_x': ball_x, 'ball_y': ball_y, 'ball_w': ball_w, 'ball_h': ball_h,
            'ball_visible': int(raw_visible['Ball']),
            'lfoot_x': lfoot_x, 'lfoot_y': lfoot_y,
            'lfoot_visible': int(raw_visible['Left_Foot']),
            'rfoot_x': rfoot_x, 'rfoot_y': rfoot_y,
            'rfoot_visible': int(raw_visible['Right_Foot']),
            'lknee_x': lknee_x, 'lknee_y': lknee_y,
            'rknee_x': rknee_x, 'rknee_y': rknee_y,
            'head_x': head_x, 'head_y': head_y,
        })
        frame_idx += 1

    cap.release()
    df = pd.DataFrame(rows)

    # Derived features. .diff() introduces NaN at row 0 — LightGBM handles NaN.
    df['ball_dy'] = df['ball_y'].diff()
    df['ball_dy2'] = df['ball_dy'].diff()
    df['lfoot_dy'] = df['lfoot_y'].diff()
    df['rfoot_dy'] = df['rfoot_y'].diff()
    df['lknee_dy'] = df['lknee_y'].diff()
    df['rknee_dy'] = df['rknee_y'].diff()
    df['lfoot_rfoot_dy'] = df['lfoot_y'] - df['rfoot_y']
    df['lfoot_rfoot_dist'] = np.sqrt(
        (df['lfoot_x'] - df['rfoot_x'])**2 + (df['lfoot_y'] - df['rfoot_y'])**2
    )
    df['ball_lfoot_dist'] = np.sqrt(
        (df['ball_x'] - df['lfoot_x'])**2 + (df['ball_y'] - df['lfoot_y'])**2
    )
    df['ball_rfoot_dist'] = np.sqrt(
        (df['ball_x'] - df['rfoot_x'])**2 + (df['ball_y'] - df['rfoot_y'])**2
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame features from a video.")
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--ball-conf', type=float, default=0.10,
                        help='YOLO confidence threshold for ball (default 0.10)')
    parser.add_argument('--no-track', action='store_true',
                        help='Disable ByteTrack temporal association (default: enabled)')
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.output)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    print(f"Extracting features from {video} ...")
    df = extract(video, ball_conf=args.ball_conf, track=not args.no_track)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} frames -> {out}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 5.3: Run the extractor manually on Vid1.mp4 once to warm Ultralytics + verify it runs**

```bash
.venv/bin/python -m tools.extract_features --video source_data/Vid1.mp4 --output features/Vid1.csv 2>&1 | grep -vE 'inference_feedback_manager|gl_context|XNNPACK|landmark_projection|absl::InitializeLog|NMS time limit' | tail -10
```

Expected: prints `Wrote 937 frames -> features/Vid1.csv` (or similar count).

- [ ] **Step 5.4: Run the test suite**

```bash
.venv/bin/python -m pytest tests/test_extract_features.py -v -s
```

Expected: 3 tests pass. (`-s` so you can see the extractor's print output during the run, which is reassuring on a long-ish test.)

- [ ] **Step 5.5: Commit**

```bash
git add tools/extract_features.py tests/test_extract_features.py features/Vid1.csv
git commit -m "feat: per-frame feature extractor"
```

(Including the generated `features/Vid1.csv` is intentional — it makes the trainer's smoke test runnable without re-extracting.)

---

## Task 6: Labeling CLI

The labeling tool plays a video at user-controlled speed and captures per-frame keypresses. Two-pass design: pass 1 marks left-foot juggles, pass 2 marks right-foot juggles. Output is `labels/<video>.csv` with `frame,foot` rows.

**Files:**
- Create: `juggle-classifier/tools/label.py`

(No automated test — interactive input is hard to test cleanly. Manual smoke test in Step 6.3.)

- [ ] **Step 6.1: Implement `tools/label.py`**

```python
"""Two-pass tap-to-mark juggle labeling tool.

Pass 1: video plays, user presses SPACE for every Left_Foot juggle.
Pass 2: same video, SPACE for every Right_Foot juggle.

Output appends to labels/<video_stem>.csv with columns (frame, foot).
If the labels file already exists for the current foot, those rows are
preserved unchanged when the other-foot pass writes its own.

Controls during playback:
  SPACE - mark juggle at current frame
  p     - pause / resume
  u     - undo last mark in current pass
  j     - slow down (multiply speed by 0.75)
  k     - speed up (multiply speed by 1/0.75)
  h     - seek -2s
  l     - seek +2s
  q     - quit and save
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

SPACE, P, U, J, K, H, L, Q = (ord(c) for c in " pujkhlq")


def _draw_overlay(frame, *, foot, frame_idx, total, paused, speed, marks_count, last_mark):
    """Draw labeling HUD on the frame."""
    h, w = frame.shape[:2]
    top = h - 110
    cv2.rectangle(frame, (0, top), (w, h), (0, 0, 0), -1)
    color_foot = (0, 200, 255) if foot == 'Left_Foot' else (255, 200, 0)
    lines = [
        f"Pass: {foot}   Marks: {marks_count}   Last: {last_mark if last_mark is not None else '-'}",
        f"Frame: {frame_idx}/{total - 1}   Speed: {speed:.2f}x   {'PAUSED' if paused else 'PLAYING'}",
        "SPACE=mark  p=pause  u=undo  j/k=slower/faster  h/l=seek-2s/+2s  q=quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, top + 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_foot if i == 0 else (255, 255, 255), 1, cv2.LINE_AA)


def _load_existing(out_csv: Path) -> list[tuple[int, str]]:
    if not out_csv.exists():
        return []
    rows: list[tuple[int, str]] = []
    with out_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((int(row['frame']), row['foot']))
            except (KeyError, ValueError):
                continue
    return rows


def _save(out_csv: Path, rows: list[tuple[int, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'foot'])
        writer.writerows(rows_sorted)


def label_pass(video: Path, out_csv: Path, foot: str, initial_speed: float = 0.5) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    existing = _load_existing(out_csv)
    other_foot_rows = [r for r in existing if r[1] != foot]
    this_foot_rows: list[tuple[int, str]] = [r for r in existing if r[1] == foot]
    initial_count = len(this_foot_rows)

    speed = initial_speed
    paused = False
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    win = f"Label {foot}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_render = time.time()

    while ok:
        target_dt = 1.0 / max(fps * speed, 1e-3)
        now = time.time()
        if not paused and (now - last_render) >= target_dt:
            frame_idx += 1
            if frame_idx >= total:
                break
            ok, frame = cap.read()
            if not ok:
                break
            last_render = now

        display = frame.copy()
        _draw_overlay(display, foot=foot, frame_idx=frame_idx, total=total,
                      paused=paused, speed=speed,
                      marks_count=len(this_foot_rows),
                      last_mark=this_foot_rows[-1][0] if this_foot_rows else None)
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 0xFF:
            continue
        if key == Q:
            break
        if key == SPACE:
            this_foot_rows.append((frame_idx, foot))
        elif key == P:
            paused = not paused
        elif key == U and this_foot_rows:
            this_foot_rows.pop()
        elif key == J:
            speed = max(0.1, speed * 0.75)
        elif key == K:
            speed = min(4.0, speed / 0.75)
        elif key == H:
            frame_idx = max(0, frame_idx - int(2 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        elif key == L:
            frame_idx = min(total - 1, frame_idx + int(2 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    combined = other_foot_rows + this_foot_rows
    _save(out_csv, combined)
    print(f"\n{foot} pass: {len(this_foot_rows) - initial_count:+d} marks "
          f"(total this foot: {len(this_foot_rows)})  ->  {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-pass juggle labeler.")
    parser.add_argument('video', type=str)
    parser.add_argument('--foot', choices=['Left_Foot', 'Right_Foot', 'both'], default='both',
                        help="Which pass to run. 'both' runs Left_Foot then Right_Foot. Default: both.")
    parser.add_argument('--out', type=str, default=None,
                        help="Output CSV path. Default: labels/<video_stem>.csv")
    parser.add_argument('--speed', type=float, default=0.5, help="Initial playback speed multiplier.")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    out_csv = Path(args.out) if args.out else (Path('labels') / f"{video.stem}.csv")

    if args.foot in ('Left_Foot', 'both'):
        print(f"\n=== Pass 1: Left_Foot ===  press SPACE every left-foot juggle, q to finish")
        label_pass(video, out_csv, foot='Left_Foot', initial_speed=args.speed)
    if args.foot in ('Right_Foot', 'both'):
        print(f"\n=== Pass 2: Right_Foot ===  press SPACE every right-foot juggle, q to finish")
        label_pass(video, out_csv, foot='Right_Foot', initial_speed=args.speed)


if __name__ == '__main__':
    main()
```

- [ ] **Step 6.2: Verify it imports cleanly** (without running the GUI)

```bash
.venv/bin/python -c "from tools import label; print('import ok')"
```

Expected: prints `import ok`.

- [ ] **Step 6.3: Manual smoke test** (only when local with a display)

```bash
.venv/bin/python -m tools.label source_data/Vid1.mp4 --foot Left_Foot --speed 0.5
```

A window appears playing Vid1.mp4 at half speed. Tap space a few times, press q. Verify `labels/Vid1.csv` is created with rows like `frame,foot` / `123,Left_Foot`.

If the window doesn't open (headless environment), this test is skipped. Document in the commit message.

- [ ] **Step 6.4: Commit**

```bash
git add tools/label.py
git commit -m "feat: two-pass tap-to-mark labeling CLI"
```

---

## Task 7: Trainer

**Files:**
- Create: `juggle-classifier/tools/train.py`
- Create: `juggle-classifier/tests/test_train.py`

- [ ] **Step 7.1: Write the failing test**

Create `tests/test_train.py`:

```python
"""Smoke tests for the trainer module — build_features_matrix and build_labels."""
import numpy as np
import pandas as pd
import pytest

from tools.train import build_features_matrix, build_labels, WINDOW_RADIUS


def _toy_features(n_frames=50, n_feats=3):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n_frames, n_feats)),
                      columns=[f"f{i}" for i in range(n_feats)])
    df.insert(0, 'frame', range(n_frames))
    return df


def test_windowed_matrix_shape():
    df = _toy_features(n_frames=50, n_feats=3)
    X = build_features_matrix(df, window_radius=2)  # window=5
    assert X.shape == (50, 3 * 5)


def test_center_frame_in_middle_slot():
    """The center slot of the window holds the current frame's features."""
    df = _toy_features(n_frames=10, n_feats=2)
    X = build_features_matrix(df, window_radius=1)  # window=3, slots [-1,0,+1]
    # For frame 5, center slot occupies feature columns [2:4]
    feature_arr = df[['f0', 'f1']].to_numpy()
    center_cols = slice(2, 4)
    np.testing.assert_array_equal(X[5, center_cols], feature_arr[5])


def test_window_edges_get_nan():
    """At frame 0 with window_radius=2, the past-2 and past-1 slots have no data => NaN."""
    df = _toy_features(n_frames=5, n_feats=1)
    X = build_features_matrix(df, window_radius=2)
    # Slot 0 corresponds to offset=-2 => columns [0:1]
    # Slot 1 corresponds to offset=-1 => columns [1:2]
    assert np.isnan(X[0, 0])  # past-2
    assert np.isnan(X[0, 1])  # past-1


def test_build_labels_basic():
    features = _toy_features(n_frames=20, n_feats=2)
    labels = pd.DataFrame({
        'frame': [3, 8, 15],
        'foot': ['Left_Foot', 'Right_Foot', 'Left_Foot'],
    })
    y = build_labels(features, labels)
    assert y.shape == (20,)
    assert y[3] == 1  # Left
    assert y[8] == 2  # Right
    assert y[15] == 1  # Left
    # All other frames are 0 (none)
    other = np.delete(y, [3, 8, 15])
    assert np.all(other == 0)


def test_build_labels_ignores_out_of_range():
    features = _toy_features(n_frames=10, n_feats=2)
    labels = pd.DataFrame({
        'frame': [5, 999],  # second is out of range
        'foot': ['Left_Foot', 'Right_Foot'],
    })
    y = build_labels(features, labels)
    assert y[5] == 1
    # No exception; out-of-range row silently dropped
    assert y.shape == (10,)
```

- [ ] **Step 7.2: Run tests, verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`
Expected: ImportError on `tools.train`.

- [ ] **Step 7.3: Implement `tools/train.py`**

```python
"""Train a LightGBM 3-class classifier on per-frame windowed features.

Reads features/<video>.csv + labels/<video>.csv, builds a per-frame feature
matrix where each row contains the features for a ±WINDOW_RADIUS window around
that frame (flattened), and fits a multiclass LightGBM classifier.

Class 0 = no juggle, 1 = Left_Foot, 2 = Right_Foot.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

WINDOW_RADIUS = 15  # ±15 frames => 31-frame window
SEED = 42

FOOT_TO_CLASS = {'Left_Foot': 1, 'Right_Foot': 2}
CLASS_TO_FOOT = {1: 'Left_Foot', 2: 'Right_Foot'}


def build_features_matrix(features_df: pd.DataFrame, window_radius: int = WINDOW_RADIUS) -> np.ndarray:
    """
    Build a 2D matrix of shape (n_frames, n_features * window_size) where
    window_size = 2*window_radius + 1.

    Row i contains the feature vector for frame i, formed by stacking the
    per-frame features from offsets -window_radius .. +window_radius. Out-of-range
    rows (e.g. for the first/last few frames) are filled with NaN.
    """
    feature_cols = [c for c in features_df.columns if c != 'frame']
    n_frames = len(features_df)
    n_features = len(feature_cols)
    window_size = 2 * window_radius + 1

    arr = features_df[feature_cols].to_numpy()
    X = np.full((n_frames, n_features * window_size), np.nan, dtype=float)

    for offset in range(-window_radius, window_radius + 1):
        slot = offset + window_radius
        src_lo = max(0, -offset)
        src_hi = min(n_frames, n_frames - offset)
        dst_lo = max(0, offset)
        dst_hi = min(n_frames, n_frames + offset)
        cols_lo = slot * n_features
        cols_hi = (slot + 1) * n_features
        X[dst_lo:dst_hi, cols_lo:cols_hi] = arr[src_lo:src_hi]
    return X


def build_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> np.ndarray:
    """Return a length-n_frames array of class labels: 0=none, 1=Left, 2=Right."""
    n_frames = len(features_df)
    y = np.zeros(n_frames, dtype=int)
    for _, row in labels_df.iterrows():
        try:
            frame = int(row['frame'])
        except (KeyError, ValueError):
            continue
        foot = row.get('foot', None)
        if foot not in FOOT_TO_CLASS:
            continue
        if 0 <= frame < n_frames:
            y[frame] = FOOT_TO_CLASS[foot]
    return y


def train(features_df: pd.DataFrame, labels_df: pd.DataFrame,
          window_radius: int = WINDOW_RADIUS) -> lgb.LGBMClassifier:
    X = build_features_matrix(features_df, window_radius)
    y = build_labels(features_df, labels_df)

    counts = np.bincount(y, minlength=3)
    print(f"Class counts: none={counts[0]}, Left={counts[1]}, Right={counts[2]}")
    if counts[1] < 5 or counts[2] < 5:
        raise SystemExit(
            f"need >= 5 positive examples per class; got Left={counts[1]}, Right={counts[2]}"
        )

    class_weight = {i: float(counts.max()) / max(int(c), 1) for i, c in enumerate(counts)}
    print(f"Class weights: {class_weight}")

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=500,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        random_state=SEED,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)

    pred = model.predict(X)
    train_acc = float((pred == y).mean())
    recall = [(pred[y == c] == c).mean() if (y == c).sum() else 0.0 for c in range(3)]
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Training recall by class: none={recall[0]:.3f}, Left={recall[1]:.3f}, Right={recall[2]:.3f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train juggle classifier.")
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="Output .pkl path")
    parser.add_argument('--window-radius', type=int, default=WINDOW_RADIUS)
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    labels_df = pd.read_csv(args.labels)
    model = train(features_df, labels_df, window_radius=args.window_radius)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump({'model': model, 'window_radius': args.window_radius}, f)
    print(f"Saved model to {out}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 7.4: Run tests, verify they pass**

```bash
.venv/bin/python -m pytest tests/test_train.py -v
```

Expected: 5 tests pass.

- [ ] **Step 7.5: Commit**

```bash
git add tools/train.py tests/test_train.py
git commit -m "feat: LightGBM juggle classifier trainer"
```

---

## Task 8: Predictor

**Files:**
- Create: `juggle-classifier/tools/predict.py`
- Create: `juggle-classifier/tests/test_predict.py`

- [ ] **Step 8.1: Write the failing test**

Create `tests/test_predict.py`:

```python
"""Smoke tests for the predictor — NMS and event extraction."""
import numpy as np
import pandas as pd

from tools.predict import nms


def test_nms_keeps_highest_in_window():
    """Two events 3 frames apart, window=5 — only the higher score is kept."""
    scores = np.array([0.0, 0.6, 0.0, 0.0, 0.8, 0.0, 0.0])
    keep = nms(scores, window=5, threshold=0.5)
    # Frame 4 has higher score; frame 1 falls inside its ±5 suppression window
    assert list(keep) == [4]


def test_nms_keeps_separate_events():
    """Two events well outside the suppression window — both kept."""
    scores = np.array([0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0])
    keep = nms(scores, window=2, threshold=0.5)
    assert list(keep) == [1, 8]


def test_nms_applies_threshold():
    """Score below threshold is dropped even with no other events nearby."""
    scores = np.array([0.0, 0.4, 0.0])
    keep = nms(scores, window=2, threshold=0.5)
    assert list(keep) == []


def test_nms_empty_input_returns_empty():
    keep = nms(np.array([]), window=5, threshold=0.5)
    assert list(keep) == []
```

- [ ] **Step 8.2: Run tests, verify they fail**

```bash
.venv/bin/python -m pytest tests/test_predict.py -v
```

Expected: ImportError.

- [ ] **Step 8.3: Implement `tools/predict.py`**

```python
"""Run the trained classifier over a feature CSV and emit predicted juggle events.

Procedure:
  1. Load model bundle from .pkl (model + window_radius)
  2. Build per-frame windowed feature matrix from features CSV
  3. Run model.predict_proba -> (n_frames, 3) array of [P(none), P(L), P(R)]
  4. Compute per-frame max foot probability and foot class (1=L, 2=R)
  5. NMS: in a 2*window+1 sliding window, keep only local maxima above threshold
  6. Write events CSV (frame, foot, confidence)
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tools.train import build_features_matrix, CLASS_TO_FOOT


def nms(scores: np.ndarray, window: int = 5, threshold: float = 0.5) -> np.ndarray:
    """Greedy non-max suppression by frame distance.

    Returns: indices into `scores` of kept events (sorted ascending).
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=int)
    above = scores >= threshold
    cand = np.where(above)[0]
    if cand.size == 0:
        return np.array([], dtype=int)
    order = cand[np.argsort(-scores[cand], kind='stable')]
    suppressed = np.zeros_like(scores, dtype=bool)
    kept: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        kept.append(int(i))
        lo = max(0, int(i) - window)
        hi = min(scores.size - 1, int(i) + window)
        suppressed[lo:hi + 1] = True
    return np.array(sorted(kept), dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict juggle events from per-frame features.")
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, help="Path to .pkl saved by train.py")
    parser.add_argument('--output', type=str, required=True, help="Output events CSV path")
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--nms-window', type=int, default=5)
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    with open(args.model, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    window_radius = bundle.get('window_radius', 15)

    X = build_features_matrix(features_df, window_radius=window_radius)
    proba = model.predict_proba(X)  # (n_frames, 3)

    foot_proba = proba[:, 1:].max(axis=1)
    foot_class = proba[:, 1:].argmax(axis=1) + 1  # 1=Left, 2=Right

    keep = nms(foot_proba, window=args.nms_window, threshold=args.threshold)
    events = pd.DataFrame({
        'frame': features_df['frame'].iloc[keep].to_numpy(),
        'foot': [CLASS_TO_FOOT[c] for c in foot_class[keep]],
        'confidence': foot_proba[keep],
    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    print(f"Wrote {len(events)} events to {out}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 8.4: Run tests, verify they pass**

```bash
.venv/bin/python -m pytest tests/test_predict.py -v
```

Expected: 4 tests pass.

- [ ] **Step 8.5: Commit**

```bash
git add tools/predict.py tests/test_predict.py
git commit -m "feat: classifier predictor with non-max suppression"
```

---

## Task 9: main.py — `--events` flag (pre-computed events visualization)

The existing `main.py` runs the detection pipeline + heuristic counter. For classifier-based runs the canonical flow is:

```
extract_features.py  →  predict.py  →  events/<video>.csv
                                              │
                                              ▼
                            main.py --events events/<video>.csv --video <video>.mp4
                            (renders annotated video using pre-computed events)
```

Why not live model inference inside `main.py`? Streaming inference would need a rolling-buffer design that handles edge cases at start/end of video and reproduces the offline derived-feature computation (1st and 2nd derivatives, distances). It's a non-trivial chunk of code that the offline `predict.py` already handles correctly. Decoupling keeps each tool simple.

**Files:**
- Modify: `juggle-classifier/main.py`

- [ ] **Step 9.1: Add the `--events` CLI flag**

In `main.py`'s `parse_args()`, just before `return parser.parse_args()`, add:

```python
    parser.add_argument('--events', type=str, default=None,
                        help='Path to a pre-computed events CSV (frame,foot,confidence). '
                             'When set, overrides the heuristic counter — the rendered count '
                             'reflects these events as the video plays through their frames.')
```

- [ ] **Step 9.2: Add an `EventListCounter` class**

Near the top of `main.py` (after the existing imports, before `def parse_args`), add:

```python
import csv as _csv
from collections import defaultdict


class EventListCounter:
    """
    Counter driven by a pre-computed events CSV. On each update() call we advance
    through the events sorted by frame and increment the per-foot count whenever
    the playback frame index has reached an event's frame.
    """

    def __init__(self, events_csv: str):
        rows: list[tuple[int, str]] = []
        with open(events_csv) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    rows.append((int(row['frame']), row['foot']))
                except (KeyError, ValueError):
                    continue
        self._events = sorted(rows, key=lambda r: r[0])
        self._next = 0
        self._counts: dict[str, int] = defaultdict(int)

    def update(self, frame_idx: int) -> dict:
        while self._next < len(self._events) and self._events[self._next][0] <= frame_idx:
            _, foot = self._events[self._next]
            self._counts[foot] += 1
            self._next += 1
        return dict(self._counts)
```

- [ ] **Step 9.3: Branch the counter setup on `args.events`**

Find the existing `juggle_counter = JuggleCounter(...)` call in `main.py` and replace the block with:

```python
    if args.events:
        juggle_counter = EventListCounter(args.events)
    else:
        juggle_counter = JuggleCounter(
            prominence=0.02,
            min_gap_frames=5,
            max_distance=0.3,
            min_history_length=10,
            candidate_parts=["Left_Foot", "Right_Foot"] if args.feet_only else None,
            mode=('spread' if args.foot_spread else ('foot' if args.foot_peaks else 'ball')),
            foot_prominence=args.foot_prominence,
            foot_validation_window=args.foot_window,
            spread_prominence=args.spread_prominence,
        )
```

- [ ] **Step 9.4: Add a frame counter and branch the per-frame update call**

Just before the main `while cap.isOpened()` loop, add:

```python
    frame_index = 0
```

Inside the loop, locate the existing `count = juggle_counter.update(predictions)` and replace it with:

```python
        # count juggle
        if isinstance(juggle_counter, EventListCounter):
            count = juggle_counter.update(frame_index)
        else:
            count = juggle_counter.update(predictions)
```

At the bottom of the loop body (just before the `# show image` comment), add:

```python
        frame_index += 1
```

- [ ] **Step 9.5: Syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parse ok')"
```

Expected: prints `parse ok`.

- [ ] **Step 9.6: Test heuristic path still works (no `--events`)**

```bash
.venv/bin/python -u main.py --video source_data/Vid1.mp4 --save save --headless --feet-only 2>&1 | grep -vE 'inference_feedback_manager|gl_context|XNNPACK|landmark_projection|absl::InitializeLog|NMS time limit' | tail -5
```

Expected: ends with `Video stream ended or camera disconnected.`. No regressions vs Task 3.

- [ ] **Step 9.7: Smoke-test the `--events` path with a hand-made events CSV**

```bash
cat > /tmp/fake_events.csv <<EOF
frame,foot,confidence
100,Left_Foot,0.9
200,Right_Foot,0.85
EOF

.venv/bin/python -u main.py --video source_data/Vid1.mp4 --events /tmp/fake_events.csv --save save --headless 2>&1 | grep -vE 'inference_feedback_manager|gl_context|XNNPACK|landmark_projection|absl::InitializeLog|NMS time limit' | tail -5
```

Expected: runs through without error. The annotated `save/Vid1_Analysed.mp4` shows the count climbing from 0 → 1 around frame 100 → 2 around frame 200.

- [ ] **Step 9.8: Commit**

```bash
git add main.py
git commit -m "feat: --events flag renders video using pre-computed events CSV"
```

---

## Task 10: End-to-end smoke test

This ties everything together: extract → (synthetic labels) → train → predict → evaluate. The labels are hand-rolled (we won't use the labeling tool — we just want to exercise the rest of the pipeline).

**Files:**
- Create: `juggle-classifier/tests/test_smoke_e2e.py`

- [ ] **Step 10.1: Write the smoke test**

Create `tests/test_smoke_e2e.py`:

```python
"""End-to-end smoke test: extract -> train -> predict -> evaluate on Vid1.mp4.

We synthesize labels by sampling 30 frames at intervals and assigning feet
alternately. This isn't accurate — the goal is only to assert that all five
stages run without error and produce well-shaped outputs.
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
VID = REPO / "source_data" / "Vid1.mp4"


def _run(*args, cwd=REPO):
    subprocess.run(
        [sys.executable, *args],
        check=True, cwd=cwd,
    )


@pytest.mark.skipif(not VID.exists(), reason="Vid1.mp4 fixture missing")
def test_full_pipeline(tmp_path):
    features = tmp_path / "vid1.csv"
    labels = tmp_path / "labels.csv"
    model = tmp_path / "model.pkl"
    events = tmp_path / "events.csv"

    # 1. Extract features (uses real video, ~20s wall clock)
    _run("-m", "tools.extract_features", "--video", str(VID), "--output", str(features))

    feats = pd.read_csv(features)
    assert len(feats) > 100, f"unexpectedly few frames: {len(feats)}"

    # 2. Synthesize labels: every 30th frame, alternating feet.
    rows = []
    for i, fr in enumerate(range(30, len(feats), 30)):
        rows.append({'frame': fr, 'foot': 'Left_Foot' if i % 2 == 0 else 'Right_Foot'})
    pd.DataFrame(rows).to_csv(labels, index=False)

    # 3. Train
    _run("-m", "tools.train", "--features", str(features),
         "--labels", str(labels), "--output", str(model))
    assert model.exists()

    # 4. Predict
    _run("-m", "tools.predict", "--features", str(features),
         "--model", str(model), "--output", str(events),
         "--threshold", "0.3")  # lower threshold so we get *some* output
    assert events.exists()

    # 5. Evaluate (no assertion on metrics — model is junk; just check it runs)
    _run("-m", "tools.evaluate", "--pred", str(events),
         "--gt", str(labels), "--tolerance", "5")
```

- [ ] **Step 10.2: Run the smoke test**

```bash
.venv/bin/python -m pytest tests/test_smoke_e2e.py -v -s
```

Expected: passes. Takes ~30-60s (feature extraction is the dominant cost).

- [ ] **Step 10.3: Commit**

```bash
git add tests/test_smoke_e2e.py
git commit -m "test: end-to-end smoke test (extract -> train -> predict -> evaluate)"
```

---

## Task 11: Colab notebooks

Three notebooks. All share boilerplate for mounting Drive, cloning the repo, and installing requirements. We write them by saving JSON to .ipynb files (notebooks are JSON).

**Files:**
- Create: `juggle-classifier/notebooks/01_extract_features.ipynb`
- Create: `juggle-classifier/notebooks/02_train.ipynb`
- Create: `juggle-classifier/notebooks/03_predict_evaluate.ipynb`

- [ ] **Step 11.1: Create `notebooks/` directory**

```bash
mkdir -p notebooks
```

- [ ] **Step 11.2: Write `notebooks/01_extract_features.ipynb`**

Save the following JSON as `notebooks/01_extract_features.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 01 — Extract features\n",
    "\n",
    "Runs the feature extractor on every video in `/MyDrive/JuggleNet/videos/` and commits the resulting CSV(s) to `features/` in the repo.\n",
    "\n",
    "**Inputs:** videos in `/MyDrive/JuggleNet/videos/*.mp4`\n",
    "**Outputs:** `features/<stem>.csv` in the repo\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import os, subprocess, getpass\n",
    "REPO_URL = 'https://github.com/<your-user>/juggle-classifier.git'  # EDIT\n",
    "REPO_DIR = '/content/juggle-classifier'\n",
    "if not os.path.exists(REPO_DIR):\n",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)\n",
    "os.chdir(REPO_DIR)\n",
    "!git pull\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "!pip install -q -r requirements-colab.txt\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "import glob, os, subprocess, sys\n",
    "VIDEOS_DIR = '/content/drive/MyDrive/JuggleNet/videos'\n",
    "vids = sorted(glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')))\n",
    "print(f'Found {len(vids)} videos.')\n",
    "for v in vids:\n",
    "    stem = os.path.splitext(os.path.basename(v))[0]\n",
    "    out = f'features/{stem}.csv'\n",
    "    if os.path.exists(out):\n",
    "        print(f'skip (exists): {out}')\n",
    "        continue\n",
    "    print(f'extracting {v} -> {out}')\n",
    "    subprocess.run([sys.executable, '-m', 'tools.extract_features',\n",
    "                    '--video', v, '--output', out], check=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Commit and push generated features. Requires a GitHub PAT in Colab Secrets.\n",
    "from google.colab import userdata\n",
    "user = 'YOUR_GITHUB_USERNAME'  # EDIT\n",
    "token = userdata.get('GITHUB_PAT')\n",
    "!git config user.email 'colab@example.com'\n",
    "!git config user.name 'Colab'\n",
    "!git add features/*.csv\n",
    "!git -c credential.helper= -c credential.helper=\"!f() { echo username={user}; echo password={token}; }; f\" commit -m 'data: extract features' || echo 'nothing to commit'\n",
    "!git push https://{user}:{token}@github.com/{user}/juggle-classifier.git\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 11.3: Write `notebooks/02_train.ipynb`**

Save:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 02 — Train\n",
    "\n",
    "Trains a LightGBM classifier on a single video's features + labels. Saves the model to `models/juggle_classifier.pkl`.\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "import os, subprocess\n",
    "REPO_URL = 'https://github.com/<your-user>/juggle-classifier.git'  # EDIT\n",
    "REPO_DIR = '/content/juggle-classifier'\n",
    "if not os.path.exists(REPO_DIR):\n",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)\n",
    "os.chdir(REPO_DIR)\n",
    "!git pull\n",
    "!pip install -q -r requirements-colab.txt\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "TRAIN_VIDEO = 'Juggles_New'  # EDIT — stem matching features/<stem>.csv and labels/<stem>.csv\n",
    "OUT = 'models/juggle_classifier.pkl'\n",
    "!python -m tools.train --features features/{TRAIN_VIDEO}.csv --labels labels/{TRAIN_VIDEO}.csv --output {OUT}\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Push model + labels back to GitHub\n",
    "from google.colab import userdata\n",
    "user = 'YOUR_GITHUB_USERNAME'\n",
    "token = userdata.get('GITHUB_PAT')\n",
    "!git config user.email 'colab@example.com'\n",
    "!git config user.name 'Colab'\n",
    "!git add models/juggle_classifier.pkl\n",
    "!git commit -m 'model: retrain juggle classifier' || echo 'nothing to commit'\n",
    "!git push https://{user}:{token}@github.com/{user}/juggle-classifier.git\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 11.4: Write `notebooks/03_predict_evaluate.ipynb`**

Save:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 03 — Predict and evaluate\n",
    "\n",
    "Runs the trained classifier on a video's features, then evaluates against labels (if present).\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "import os, subprocess\n",
    "REPO_URL = 'https://github.com/<your-user>/juggle-classifier.git'  # EDIT\n",
    "REPO_DIR = '/content/juggle-classifier'\n",
    "if not os.path.exists(REPO_DIR):\n",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)\n",
    "os.chdir(REPO_DIR)\n",
    "!git pull\n",
    "!pip install -q -r requirements-colab.txt\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "VIDEO = 'Juggles_New'  # EDIT — predict + evaluate against this video\n",
    "MODEL = 'models/juggle_classifier.pkl'\n",
    "THRESHOLD = 0.5\n",
    "NMS_WINDOW = 5\n",
    "!python -m tools.predict --features features/{VIDEO}.csv --model {MODEL} \\\n",
    "    --output events/{VIDEO}.csv --threshold {THRESHOLD} --nms-window {NMS_WINDOW}\n",
    "import os\n",
    "if os.path.exists(f'labels/{VIDEO}.csv'):\n",
    "    !python -m tools.evaluate --pred events/{VIDEO}.csv --gt labels/{VIDEO}.csv --tolerance 5\n",
    "else:\n",
    "    print(f'No labels for {VIDEO}; printing total only.')\n",
    "    import pandas as pd\n",
    "    print(pd.read_csv(f'events/{VIDEO}.csv').groupby('foot').size())\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Optional: sweep threshold/window to find best F1 on validation\n",
    "import subprocess, sys\n",
    "VAL_VIDEO = 'Clark_Juggles1'  # EDIT — held-out validation video stem\n",
    "for thr in [0.3, 0.4, 0.5, 0.6]:\n",
    "    for w in [3, 5, 7]:\n",
    "        print(f'\\n--- threshold={thr} nms_window={w} ---')\n",
    "        subprocess.run([sys.executable, '-m', 'tools.predict',\n",
    "                        '--features', f'features/{VAL_VIDEO}.csv',\n",
    "                        '--model', 'models/juggle_classifier.pkl',\n",
    "                        '--output', f'events/{VAL_VIDEO}.csv',\n",
    "                        '--threshold', str(thr), '--nms-window', str(w)], check=True)\n",
    "        subprocess.run([sys.executable, '-m', 'tools.evaluate',\n",
    "                        '--pred', f'events/{VAL_VIDEO}.csv',\n",
    "                        '--gt', f'labels/{VAL_VIDEO}.csv',\n",
    "                        '--tolerance', '5'], check=True)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 11.5: Verify all three notebooks parse as valid JSON**

```bash
.venv/bin/python -c "
import json, sys
for p in ['notebooks/01_extract_features.ipynb',
          'notebooks/02_train.ipynb',
          'notebooks/03_predict_evaluate.ipynb']:
    with open(p) as f: json.load(f)
    print(f'{p}: ok')
"
```

Expected: three `ok` lines.

- [ ] **Step 11.6: Commit**

```bash
git add notebooks/*.ipynb
git commit -m "feat: colab notebooks (extract, train, predict+evaluate)"
```

---

## Task 12: README + GitHub repo creation + first push

**Files:**
- Modify: `juggle-classifier/README.md`

- [ ] **Step 12.1: Replace README skeleton with full content**

Overwrite `README.md`:

````markdown
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
├── main.py                 entry point with --classifier flag
├── models/                 finetuned YOLO weights + trained classifier
├── labels/                 ground-truth labels (frame, foot) — committed
├── features/               per-frame feature CSVs — committed
└── docs/superpowers/       design spec + implementation plan
```

## Local setup (Mac, Apple Silicon)

```bash
git clone https://github.com/<your-user>/juggle-classifier.git
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
````

- [ ] **Step 12.2: Commit README**

```bash
git add README.md
git commit -m "docs: full README with local + Colab workflows"
```

- [ ] **Step 12.3: Create public GitHub repo**

```bash
gh repo create juggle-classifier --public --description "Learned juggle counter built on JuggleNet (YOLO + MediaPipe + LightGBM)" --source=. --remote=origin
```

Expected: prints the new repo URL, sets `origin` remote.

- [ ] **Step 12.4: Push everything**

```bash
git branch -M main
git push -u origin main
```

Expected: push succeeds; repo is live at `https://github.com/<your-user>/juggle-classifier`.

- [ ] **Step 12.5: Verify the repo on GitHub**

```bash
gh repo view --web
```

Expected: opens the repo in browser. Confirm README renders and `docs/superpowers/specs/2026-05-19-juggle-classifier-design.md` is visible.

---

## Done condition

After Task 12, the repo is set up, the pipeline is implemented + tested, and the user can begin the data work:

1. Label `Juggles_New.mp4` locally → `labels/Juggles_New.csv` (commit)
2. Label a second video → `labels/<second>.csv` (commit)
3. Upload both `.mp4`s to `/MyDrive/JuggleNet/videos/`
4. Open `notebooks/01_extract_features.ipynb` in Colab → extract both videos' features (commit)
5. Open `notebooks/02_train.ipynb` in Colab → train on `Juggles_New` (commit model)
6. Open `notebooks/03_predict_evaluate.ipynb` in Colab → evaluate on the second video, sweep hyperparameters
7. Locally, run final model on `Juggles_New.mp4`, confirm count is 500 ± 5

This plan ends when the repo is on GitHub and ready for the data-collection + training cycle to begin.
