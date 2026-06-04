"""Head-to-head: LightGBM vs CNN on All_Training_Juggles with a temporal holdout."""
import sys, pickle
import pandas as pd, numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn

# Set early before any tensor work to avoid M1 thread thrashing.
torch.set_num_threads(8)

sys.path.insert(0, '.')
from tools.train import (build_features_matrix, build_labels, detect_mode,
                          WINDOW_RADIUS, SEED)
from tools.predict import nms
from tools.evaluate import match_events, metrics
from tools.cnn import JuggleCNN, CNNJuggleClassifier, best_device

VAL_FRAC = 0.15
GAP = 50
NMS_WIN = 5
EPOCHS = 30
BS = 256
import os
# 'auto' = use best_device() (CUDA on Colab, MPS on Mac). On Mac MPS we hit a
# conv1d hang, so default to CPU there. Override with env DEVICE=cuda|mps|cpu.
_DEVICE_ENV = os.environ.get('DEVICE', 'auto').lower()

print("Loading features + labels...", flush=True)
features_df = pd.read_csv('features/All_Training_Juggles.csv')
labels_df = pd.read_csv('labels/All_Training_Juggles.csv')
mode = detect_mode(labels_df)
num_class = 2 if mode == 'binary' else 3
print(f"mode={mode}, num_class={num_class}, frames={len(features_df)}, events={len(labels_df)}", flush=True)

X_flat = build_features_matrix(features_df, WINDOW_RADIUS)
y = build_labels(features_df, labels_df, mode=mode)

n_frames = len(features_df)
n_val = int(n_frames * VAL_FRAC)
val_lo = n_frames - n_val
train_hi = max(0, val_lo - GAP)
train_idx = np.arange(0, train_hi)
val_idx = np.arange(val_lo, n_frames)
print(f"Train: 0..{train_hi-1} ({len(train_idx)} frames)  Val: {val_lo}..{n_frames-1} ({len(val_idx)} frames)", flush=True)
print(f"Train juggles: {int((y[train_idx]>0).sum())}  Val juggles: {int((y[val_idx]>0).sum())}", flush=True)

X_train, y_train = X_flat[train_idx], y[train_idx]
X_val_lgbm = X_flat[val_idx]
labels_val = labels_df[(labels_df['frame'] >= val_lo) & (labels_df['frame'] < n_frames)].copy().reset_index(drop=True)

# ====== LightGBM ======
print("\n=== LightGBM training ===", flush=True)
counts = np.bincount(y_train, minlength=num_class)
cw = {i: float(counts.max())/max(int(c),1) for i, c in enumerate(counts)}
print(f"  class counts (train): {counts.tolist()}, weights: {cw}", flush=True)
import time
t0 = time.time()
lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=num_class,
                          n_estimators=500, num_leaves=31, learning_rate=0.05,
                          random_state=SEED, class_weight=cw, n_jobs=-1, verbose=-1)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lgbm.fit(X_train, y_train)
print(f"  trained in {time.time()-t0:.1f}s", flush=True)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    proba_lgbm = lgbm.predict_proba(X_val_lgbm)
foot_proba_lgbm = proba_lgbm[:, 1]

# ====== CNN ======
print("\n=== CNN training ===", flush=True)
feature_cols = [c for c in features_df.columns if c != 'frame']
n_features = len(feature_cols)
window_size = 2 * WINDOW_RADIUS + 1
if _DEVICE_ENV == 'auto':
    auto = best_device()
    # Avoid the MPS conv1d hang we observed on M-series Macs.
    device = torch.device('cpu') if str(auto) == 'mps' else auto
elif _DEVICE_ENV in ('cuda', 'mps', 'cpu'):
    device = torch.device(_DEVICE_ENV)
else:
    device = best_device()
print(f"  device={device} (env DEVICE={_DEVICE_ENV}), n_features={n_features}, window={window_size}", flush=True)

print(f"  building tensors...", flush=True)
t0 = time.time()
X_filled = np.nan_to_num(X_flat, nan=0.0).astype(np.float32, copy=False)
X_3d = X_filled.reshape(n_frames, window_size, n_features).transpose(0, 2, 1).copy()
print(f"  X_3d built in {time.time()-t0:.1f}s, shape={X_3d.shape}, dtype={X_3d.dtype}", flush=True)

t0 = time.time()
# Pin tensors to target device once (small enough to fit; avoids per-batch copies that stall MPS).
X_train_t = torch.from_numpy(X_3d[train_idx]).contiguous().to(device)
y_train_t = torch.from_numpy(y[train_idx]).long().to(device)
X_val_t = torch.from_numpy(X_3d[val_idx]).contiguous().to(device)
y_val_t = torch.from_numpy(y[val_idx]).long().to(device)
print(f"  tensors on {X_train_t.device} in {time.time()-t0:.1f}s, train={tuple(X_train_t.shape)}, val={tuple(X_val_t.shape)}, threads={torch.get_num_threads()}", flush=True)

torch.manual_seed(SEED); np.random.seed(SEED)
cnn = JuggleCNN(n_features=n_features, window_size=window_size, num_class=num_class).to(device)
print(f"  params: {sum(p.numel() for p in cnn.parameters()):,}", flush=True)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
cw_t = torch.tensor([float(counts.max())/max(int(c),1) for c in counts], dtype=torch.float32, device=device)
loss_fn = nn.CrossEntropyLoss(weight=cw_t)

best_val_loss = float('inf'); best_state = None
t0 = time.time()
for ep in range(EPOCHS):
    cnn.train()
    perm = torch.randperm(len(X_train_t), device=device)
    ep_loss = 0.0; n_seen = 0
    t_ep = time.time()
    for bi, i in enumerate(range(0, len(X_train_t), BS)):
        b = perm[i:i+BS]
        xb = X_train_t[b]; yb = y_train_t[b]
        opt.zero_grad()
        out = cnn(xb)
        l = loss_fn(out, yb)
        l.backward(); opt.step()
        ep_loss += l.item()*xb.size(0); n_seen += xb.size(0)
        # Heartbeat on first epoch so a hang is obvious.
        if ep == 0 and bi in (0, 5, 25, 60):
            print(f"    ep0 batch {bi+1} done at {time.time()-t_ep:.1f}s", flush=True)
    sched.step()
    train_loss = ep_loss/max(n_seen,1)

    cnn.eval()
    with torch.no_grad():
        vl = 0.0; nv = 0
        for i in range(0, len(X_val_t), BS):
            xb = X_val_t[i:i+BS]; yb = y_val_t[i:i+BS]
            out = cnn(xb)
            vl += loss_fn(out, yb).item()*xb.size(0); nv += xb.size(0)
        val_loss = vl/max(nv,1)
    star = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in cnn.state_dict().items()}
        star = ' *'
    print(f"  ep {ep+1:02d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}{star}  ({time.time()-t0:.0f}s)", flush=True)
print(f"  best val_loss: {best_val_loss:.4f}", flush=True)
cnn.load_state_dict(best_state); cnn.eval()

with torch.no_grad():
    probs = []
    for i in range(0, len(X_val_t), BS):
        xb = X_val_t[i:i+BS]
        out = cnn(xb)
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
proba_cnn = np.concatenate(probs)
foot_proba_cnn = proba_cnn[:, 1]

# ====== Threshold sweep + evaluation ======
val_frames = features_df['frame'].iloc[val_idx].to_numpy()

print(f"\n=== Validation (last 15%, NMS window={NMS_WIN}) ===", flush=True)
hdr = f"{'Model':<10} {'thr':>5} {'pred':>5} {'GT':>4} {'P':>6} {'R':>6} {'F1':>6}"
print(hdr); print('-'*len(hdr), flush=True)

def evaluate_block(name, foot_proba):
    best = None
    for thr in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]:
        keep = nms(foot_proba, window=NMS_WIN, threshold=thr)
        events = pd.DataFrame({
            'frame': val_frames[keep],
            'foot': ['Juggle']*len(keep),
            'confidence': foot_proba[keep],
        })
        tp, _, fp, fn = match_events(events, labels_val, tolerance=5)
        p, r, f1 = metrics(len(tp), len(fp), len(fn))
        mark = ''
        if best is None or f1 > best[3]:
            best = (thr, p, r, f1); mark = ' *'
        print(f"{name:<10} {thr:>5.2f} {len(events):>5d} {len(labels_val):>4d} {p:>6.3f} {r:>6.3f} {f1:>6.3f}{mark}", flush=True)
    return best

best_lgbm = evaluate_block('LightGBM', foot_proba_lgbm)
best_cnn = evaluate_block('CNN', foot_proba_cnn)

print(f"\nLightGBM best: thr={best_lgbm[0]} P={best_lgbm[1]:.3f} R={best_lgbm[2]:.3f} F1={best_lgbm[3]:.3f}", flush=True)
print(f"CNN      best: thr={best_cnn[0]} P={best_cnn[1]:.3f} R={best_cnn[2]:.3f} F1={best_cnn[3]:.3f}", flush=True)
if best_cnn[3] > best_lgbm[3]:
    print(f"\nCNN wins by F1 +{best_cnn[3]-best_lgbm[3]:.3f}", flush=True)
else:
    print(f"\nLightGBM wins by F1 +{best_lgbm[3]-best_cnn[3]:.3f}", flush=True)
