"""Small 1D-CNN for juggle classification on windowed per-frame features.

Input layout per sample: (n_features, window_size). Features are channels,
the temporal window is the conv axis. ~80k parameters, trains in minutes.

The `CNNJuggleClassifier` wraps a trained model in a sklearn-like
`predict_proba` API so that `tools/predict.py` can consume CNN bundles and
LightGBM bundles interchangeably.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class JuggleCNN(nn.Module):
    """Three-block 1D CNN. Input (B, n_features, window_size) -> (B, num_class)."""

    def __init__(self, n_features: int, window_size: int, num_class: int,
                 dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.num_class = num_class

        self.block1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class CNNJuggleClassifier:
    """sklearn-like wrapper exposing `predict_proba(X)`.

    X is the SAME flattened windowed feature matrix produced by
    `tools.train.build_features_matrix` — shape (n_frames, n_features * window_size).
    Internally we reshape to (n_frames, n_features, window_size) for the CNN.
    NaNs in the input are filled with 0 (model uses *_visible flags to know
    when a detection was absent).
    """

    def __init__(self, state_dict: dict, arch: dict, device: str | None = None):
        self.arch = arch
        self.device = torch.device(device) if device else best_device()
        self.model = JuggleCNN(
            n_features=arch['n_features'],
            window_size=arch['window_size'],
            num_class=arch['num_class'],
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @property
    def n_features_(self) -> int:
        return int(self.arch['n_features'])

    @property
    def window_size_(self) -> int:
        return int(self.arch['window_size'])

    @property
    def num_class_(self) -> int:
        return int(self.arch['num_class'])

    def _reshape(self, X_flat: np.ndarray) -> torch.Tensor:
        X = np.nan_to_num(X_flat, nan=0.0).astype(np.float32, copy=False)
        n_frames = X.shape[0]
        # X_flat is laid out (slot0_feat0..slot0_featN, slot1_feat0..slot1_featN, ...)
        # i.e. shape (n_frames, window_size * n_features).
        X = X.reshape(n_frames, self.window_size_, self.n_features_)
        # Swap to (n_frames, n_features, window_size) for Conv1d.
        X = np.transpose(X, (0, 2, 1))
        return torch.from_numpy(X.copy())

    def predict_proba(self, X_flat: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        x = self._reshape(X_flat)
        n = x.shape[0]
        out = np.zeros((n, self.num_class_), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = x[i:i + batch_size].to(self.device)
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                out[i:i + batch_size] = probs
        return out
