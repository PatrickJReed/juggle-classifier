"""Per-event juggle counter evaluator.

Reads a predictions CSV (frame, foot, confidence) and a ground-truth labels
CSV (frame, foot). Matches predictions to ground truth using a greedy bipartite
algorithm with a ±tolerance frame window AND same-foot constraint. Reports
precision, recall, F1 (combined and per foot), total count error, and lists
of false positives and misses for manual inspection.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def match_events(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, tolerance: int = 5
) -> tuple[list[int], list[int], list[int], list[int]]:
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

    # Per-foot breakdown only when the labels actually distinguish feet.
    foot_values = set(pred.get('foot', pd.Series([], dtype=object)).unique()) | set(gt.get('foot', pd.Series([], dtype=object)).unique())
    per_foot_classes = [f for f in ('Left_Foot', 'Right_Foot') if f in foot_values]
    for foot in per_foot_classes:
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
