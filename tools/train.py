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

# Binary "Juggle" mode (no L/R distinction)
FOOT_TO_CLASS_BINARY = {'Juggle': 1}
CLASS_TO_FOOT_BINARY = {1: 'Juggle'}


def detect_mode(labels_df: pd.DataFrame) -> str:
    """Return 'binary' if labels only contain 'Juggle', else 'feet'."""
    foot_values = set(labels_df['foot'].dropna().unique())
    if foot_values == {'Juggle'}:
        return 'binary'
    return 'feet'


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
        # slot index: offset=-window_radius => slot 0 (past), offset=0 => center
        slot = offset + window_radius
        # For destination row i, we want arr[i + offset].
        # Valid when 0 <= i+offset < n_frames, i.e. -offset <= i < n_frames-offset.
        # dst range: max(0, -offset) .. min(n_frames, n_frames - offset)
        # src range: dst + offset => max(0, offset) .. min(n_frames, n_frames + offset)
        dst_lo = max(0, -offset)
        dst_hi = min(n_frames, n_frames - offset)
        src_lo = dst_lo + offset
        src_hi = dst_hi + offset
        cols_lo = slot * n_features
        cols_hi = (slot + 1) * n_features
        if dst_lo < dst_hi:
            X[dst_lo:dst_hi, cols_lo:cols_hi] = arr[src_lo:src_hi]
    return X


def build_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame,
                 mode: str = 'feet') -> np.ndarray:
    """Return a length-n_frames array of class labels.

    mode='feet':   0=none, 1=Left_Foot, 2=Right_Foot
    mode='binary': 0=none, 1=Juggle
    """
    mapping = FOOT_TO_CLASS_BINARY if mode == 'binary' else FOOT_TO_CLASS
    n_frames = len(features_df)
    y = np.zeros(n_frames, dtype=int)
    for _, row in labels_df.iterrows():
        try:
            frame = int(row['frame'])
        except (KeyError, ValueError):
            continue
        foot = row.get('foot', None)
        if foot not in mapping:
            continue
        if 0 <= frame < n_frames:
            y[frame] = mapping[foot]
    return y


def train(features_df: pd.DataFrame, labels_df: pd.DataFrame,
          window_radius: int = WINDOW_RADIUS,
          mode: str | None = None):
    """Train a LightGBM classifier. Auto-detects mode from labels if not specified."""
    if mode is None:
        mode = detect_mode(labels_df)
    num_class = 2 if mode == 'binary' else 3

    X = build_features_matrix(features_df, window_radius)
    y = build_labels(features_df, labels_df, mode=mode)

    counts = np.bincount(y, minlength=num_class)
    if mode == 'binary':
        print(f"Mode: binary (Juggle)  |  counts: none={counts[0]}, Juggle={counts[1]}")
        if counts[1] < 5:
            raise ValueError(f"need >= 5 Juggle examples; got {counts[1]}")
    else:
        print(f"Mode: feet  |  counts: none={counts[0]}, Left={counts[1]}, Right={counts[2]}")
        if counts[1] < 5 or counts[2] < 5:
            raise ValueError(
                f"need >= 5 positive examples per foot; got Left={counts[1]}, Right={counts[2]}"
            )

    class_weight = {i: float(counts.max()) / max(int(c), 1) for i, c in enumerate(counts)}
    print(f"Class weights: {class_weight}")

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_class,
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
    recall = [(pred[y == c] == c).mean() if (y == c).sum() else 0.0 for c in range(num_class)]
    print(f"Training accuracy: {train_acc:.4f}")
    if mode == 'binary':
        print(f"Training recall: none={recall[0]:.3f}, Juggle={recall[1]:.3f}")
    else:
        print(f"Training recall: none={recall[0]:.3f}, Left={recall[1]:.3f}, Right={recall[2]:.3f}")
    return model, mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Train juggle classifier.")
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="Output .pkl path")
    parser.add_argument('--window-radius', type=int, default=WINDOW_RADIUS)
    parser.add_argument('--mode', choices=['feet', 'binary', 'auto'], default='auto',
                        help="Label mode. 'auto' detects from labels CSV.")
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    labels_df = pd.read_csv(args.labels)
    mode_arg = None if args.mode == 'auto' else args.mode
    model, mode = train(features_df, labels_df, window_radius=args.window_radius, mode=mode_arg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump({'model': model, 'window_radius': args.window_radius, 'mode': mode}, f)
    print(f"Saved model ({mode} mode) to {out}")


if __name__ == '__main__':
    main()
