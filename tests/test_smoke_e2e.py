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
