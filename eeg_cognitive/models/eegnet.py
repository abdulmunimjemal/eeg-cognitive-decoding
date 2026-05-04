"""
EEGNet (Lawhern et al. 2018) — a compact convolutional network for EEG.

Architecture:
    Input  : (1, n_channels, n_samples)
    Block 1: temporal Conv2d (1, kernel_length=64) → BN
             depthwise Conv2d across channels → BN → ELU → AvgPool(1, 4)
             → Dropout
    Block 2: separable Conv2d (1, 16) → BN → ELU → AvgPool(1, 8) → Dropout
    Head   : Linear → softmax

The whole thing has ~2k parameters for typical EEG configs, so it trains
in seconds on CPU and remains competitive with much larger models.

Implemented in PyTorch with the standard hyper-parameters from the paper.
The trainer is a small scikit-learn-compatible wrapper so it slots into the
same CV harness as FBCSP.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        n_samples: int,
        sfreq: float = 250.0,
        F1: int = 8,             # number of temporal filters
        D: int = 2,              # depth multiplier (spatial filters per temporal)
        F2: int | None = None,   # F1 * D
        kernel_length: int | None = None,
        dropout: float = 0.25,
    ):
        super().__init__()
        F2 = F2 or F1 * D
        # Lawhern et al. set kernel_length ≈ sfreq/2 (so each temporal filter
        # spans ~500 ms — half the lowest interesting EEG period).
        if kernel_length is None:
            kernel_length = max(16, int(sfreq // 2))

        # Block 1: temporal conv + depthwise conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise conv across channels (groups=F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv (depthwise + pointwise)
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                                   groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute classifier input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self._forward_features(dummy)
            n_feat = feat.numel()
        self.classifier = nn.Linear(n_feat, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        return self.classifier(x)


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper. fit() trains EEGNet with early stopping on
    a held-out 20% validation slice."""

    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        patience: int = 30,
        val_frac: float = 0.2,
        device: str = "cpu",
        verbose: bool = False,
        seed: int = 0,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_frac = val_frac
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        cls_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([cls_to_idx[c] for c in y], dtype=np.int64)

        n_trials, n_ch, n_samples = X.shape
        self.n_channels_, self.n_samples_ = n_ch, n_samples

        # Train / val split
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_trials)
        n_val = int(n_trials * self.val_frac)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr, ytr = X[tr_idx], y_idx[tr_idx]
        Xv, yv = X[val_idx], y_idx[val_idx]

        # Build model — pass sfreq + optional kernel override if the caller
        # stashed them on us (used by run_one.py to budget training time).
        sfreq = getattr(self, "sfreq", 250.0)
        kernel_length = getattr(self, "eegnet_kernel_length", None)
        self.model_ = EEGNet(n_classes=n_classes, n_channels=n_ch,
                             n_samples=n_samples, sfreq=sfreq,
                             kernel_length=kernel_length).to(self.device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        # Tensors. Add the channel dim expected by Conv2d: (N, 1, C, T)
        def to_loader(X_, y_, shuffle):
            Xt = torch.from_numpy(X_[:, None, :, :].astype(np.float32))
            yt = torch.from_numpy(y_)
            return DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size,
                              shuffle=shuffle)

        tr_loader = to_loader(Xtr, ytr, shuffle=True)
        # For validation we just batch through once
        Xv_t = torch.from_numpy(Xv[:, None, :, :].astype(np.float32)).to(self.device)
        yv_t = torch.from_numpy(yv).to(self.device)

        best_val = float("inf")
        best_state = None
        wait = 0
        self.history_ = {"train_loss": [], "val_loss": [], "val_acc": []}

        for ep in range(self.epochs):
            self.model_.train()
            tr_losses = []
            for xb, yb in tr_loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                out = self.model_(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                tr_losses.append(loss.item())

            self.model_.eval()
            with torch.no_grad():
                out = self.model_(Xv_t)
                vl = loss_fn(out, yv_t).item()
                va = (out.argmax(dim=1) == yv_t).float().mean().item()

            self.history_["train_loss"].append(float(np.mean(tr_losses)))
            self.history_["val_loss"].append(vl)
            self.history_["val_acc"].append(va)

            if vl < best_val - 1e-4:
                best_val = vl
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print(f"[EEGNet] early stop at epoch {ep+1}")
                    break

            if self.verbose and (ep + 1) % 20 == 0:
                print(f"[EEGNet] ep {ep+1:3d}  train {np.mean(tr_losses):.3f}"
                      f"  val_loss {vl:.3f}  val_acc {va:.3f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        Xt = torch.from_numpy(X[:, None, :, :].astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model_(Xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[probs.argmax(axis=1)]


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eeg_cognitive.data import make_motor_imagery_synthetic
    from eeg_cognitive.preprocess import preprocess_epochs

    ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=20, seed=0)
    X = preprocess_epochs(ds.X, ds.sfreq)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ds.y))
    split = int(len(ds.y) * 0.8)
    tr, te = perm[:split], perm[split:]
    clf = EEGNetClassifier(epochs=80, batch_size=32, verbose=True, seed=0)
    clf.fit(X[tr], ds.y[tr])
    yhat = clf.predict(X[te])
    acc = (yhat == ds.y[te]).mean()
    print(f"EEGNet smoke-test accuracy on motor imagery: {acc:.3f}")
    print(f"Total params: {sum(p.numel() for p in clf.model_.parameters())}")
