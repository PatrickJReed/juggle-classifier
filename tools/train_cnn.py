"""Train a 1D-CNN juggle classifier on per-frame windowed features.

Mirrors `tools/train.py` (LightGBM) but uses a PyTorch CNN. Auto-detects
binary (Juggle) vs feet (Left/Right) mode from labels.

Saves a pickle bundle of {'model': CNNJuggleClassifier, 'window_radius', 'mode',
'arch'} compatible with `tools/predict.py`.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tools.cnn import CNNJuggleClassifier, JuggleCNN, best_device
from tools.train import (
    SEED, WINDOW_RADIUS,
    build_features_matrix, build_labels, detect_mode,
)


def _split_indices(n_frames: int, val_frac: float, gap: int = 50, rng_seed: int = SEED):
    """Temporal split: first (1-val_frac)*n frames train, last val_frac val.

    A small gap is left out between the train tail and val head so windowed
    features (±window_radius) don't leak training data into the val set.
    """
    n_val = max(1, int(n_frames * val_frac))
    val_lo = n_frames - n_val
    val_hi = n_frames
    train_hi = max(0, val_lo - gap)
    train_idx = np.arange(0, train_hi)
    val_idx = np.arange(val_lo, val_hi)
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(train_idx)
    return train_idx, val_idx


def train_cnn(features_df: pd.DataFrame, labels_df: pd.DataFrame, *,
              window_radius: int = WINDOW_RADIUS,
              mode: str | None = None,
              epochs: int = 30,
              batch_size: int = 256,
              lr: float = 1e-3,
              val_frac: float = 0.15,
              weight_decay: float = 1e-4):
    if mode is None:
        mode = detect_mode(labels_df)
    num_class = 2 if mode == 'binary' else 3
    device = best_device()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_flat = build_features_matrix(features_df, window_radius)
    y = build_labels(features_df, labels_df, mode=mode)

    n_frames = X_flat.shape[0]
    feature_cols = [c for c in features_df.columns if c != 'frame']
    n_features = len(feature_cols)
    window_size = 2 * window_radius + 1
    assert X_flat.shape[1] == n_features * window_size

    counts = np.bincount(y, minlength=num_class)
    if mode == 'binary':
        print(f"Mode: binary (Juggle)  |  counts: none={counts[0]}, Juggle={counts[1]}")
    else:
        print(f"Mode: feet  |  counts: none={counts[0]}, Left={counts[1]}, Right={counts[2]}")
    print(f"Device: {device}, frames: {n_frames}, features: {n_features}, window: {window_size}")

    # Reshape to (N, n_features, window_size)
    X_filled = np.nan_to_num(X_flat, nan=0.0).astype(np.float32, copy=False)
    X_3d = X_filled.reshape(n_frames, window_size, n_features).transpose(0, 2, 1).copy()

    # Temporal split
    train_idx, val_idx = _split_indices(n_frames, val_frac)
    X_train = torch.from_numpy(X_3d[train_idx])
    y_train = torch.from_numpy(y[train_idx]).long()
    X_val = torch.from_numpy(X_3d[val_idx])
    y_val = torch.from_numpy(y[val_idx]).long()
    print(f"Split: train={len(X_train)} (val_frame_range={val_idx[0]}..{val_idx[-1]}), val={len(X_val)}")

    # Class weights for CE loss (inverse frequency)
    cw = torch.tensor([float(counts.max()) / max(int(c), 1) for c in counts], dtype=torch.float32, device=device)
    print(f"Class weights: {cw.tolist()}")

    model = JuggleCNN(n_features=n_features, window_size=window_size, num_class=num_class).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNN params: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_val_loss = float('inf')
    best_state = None
    train_loader_idx = torch.arange(len(X_train))

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        ep_loss = 0.0; n_seen = 0
        for i in range(0, len(X_train), batch_size):
            b = perm[i:i + batch_size]
            xb = X_train[b].to(device); yb = y_train[b].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0); n_seen += xb.size(0)
        sched.step()
        train_loss = ep_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0; n_val = 0
            preds_val = []
            for i in range(0, len(X_val), batch_size):
                xb = X_val[i:i + batch_size].to(device)
                yb = y_val[i:i + batch_size].to(device)
                out = model(xb)
                val_loss_sum += loss_fn(out, yb).item() * xb.size(0); n_val += xb.size(0)
                preds_val.append(out.argmax(dim=1).cpu().numpy())
            val_loss = val_loss_sum / max(n_val, 1)
            val_pred = np.concatenate(preds_val) if preds_val else np.array([])
        val_acc = float((val_pred == y_val.numpy()).mean()) if len(val_pred) else 0.0
        val_recall_pos = float((val_pred[y_val.numpy() > 0] > 0).mean()) if (y_val.numpy() > 0).any() else 0.0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep+1:02d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}  val_pos_recall={val_recall_pos:.3f}{'  *' if improved else ''}")

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    arch = dict(n_features=n_features, window_size=window_size, num_class=num_class)
    wrapper = CNNJuggleClassifier(state_dict=model.state_dict(), arch=arch, device=str(device))
    return wrapper, mode, arch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN juggle classifier.")
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help="Output .pkl path")
    parser.add_argument('--window-radius', type=int, default=WINDOW_RADIUS)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--mode', choices=['feet', 'binary', 'auto'], default='auto')
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    labels_df = pd.read_csv(args.labels)
    mode_arg = None if args.mode == 'auto' else args.mode
    wrapper, mode, arch = train_cnn(
        features_df, labels_df,
        window_radius=args.window_radius,
        mode=mode_arg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump({
            'model': wrapper,
            'window_radius': args.window_radius,
            'mode': mode,
            'arch': arch,
        }, f)
    print(f"Saved CNN model ({mode} mode) to {out}")


if __name__ == '__main__':
    main()
