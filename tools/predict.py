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

from tools.train import build_features_matrix, CLASS_TO_FOOT, CLASS_TO_FOOT_BINARY


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
    mode = bundle.get('mode', 'feet')  # 'feet' for legacy models without explicit mode

    X = build_features_matrix(features_df, window_radius=window_radius)
    proba = model.predict_proba(X)  # (n_frames, num_class)

    if mode == 'binary':
        # Class 0 = none, Class 1 = Juggle
        foot_proba = proba[:, 1]
        foot_class = np.ones(len(foot_proba), dtype=int)
        class_to_foot = CLASS_TO_FOOT_BINARY
    else:
        # Class 0 = none, 1 = Left_Foot, 2 = Right_Foot
        foot_proba = proba[:, 1:].max(axis=1)
        foot_class = proba[:, 1:].argmax(axis=1) + 1
        class_to_foot = CLASS_TO_FOOT

    keep = nms(foot_proba, window=args.nms_window, threshold=args.threshold)
    events = pd.DataFrame({
        'frame': features_df['frame'].iloc[keep].to_numpy(),
        'foot': [class_to_foot[c] for c in foot_class[keep]],
        'confidence': foot_proba[keep],
    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    print(f"Wrote {len(events)} events to {out}")


if __name__ == '__main__':
    main()
