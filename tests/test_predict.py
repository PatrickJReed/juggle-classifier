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
