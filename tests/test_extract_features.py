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
