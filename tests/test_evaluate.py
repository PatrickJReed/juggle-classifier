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
