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
