"""
EEG preprocessing.

A small, dependency-light preprocessing module that takes a stack of epochs
and returns clean, model-ready arrays. Steps:

    1. Bandpass (default 4-40 Hz, Butterworth, zero-phase via filtfilt)
    2. Common-average reference
    3. Per-channel z-score using TRAIN-FOLD statistics only (caller passes
       the fitted scaler back in for the validation fold)

We deliberately keep ICA / artifact rejection out of the default path: it is
slow, requires hand-tuning, and CSP is robust to typical EEG artifacts.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from dataclasses import dataclass


def bandpass(
    X: np.ndarray, sfreq: float, l_freq: float = 4.0, h_freq: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass. X shape: (..., n_samples)."""
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt along the last axis; broadcast across leading dims
    return filtfilt(b, a, X, axis=-1).astype(np.float32)


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """Subtract the across-channel mean from each channel.
    X shape: (n_trials, n_channels, n_samples)."""
    return (X - X.mean(axis=1, keepdims=True)).astype(np.float32)


@dataclass
class ChannelStandardizer:
    """Per-channel z-score. Fit on train epochs, transform train + val."""
    mean_: np.ndarray = None
    std_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "ChannelStandardizer":
        # Pool over trials and time → one mean/std per channel
        self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = X.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def preprocess_epochs(
    X: np.ndarray, sfreq: float,
    l_freq: float = 4.0, h_freq: float = 40.0,
    do_car: bool = True,
) -> np.ndarray:
    """Full epoch-level preprocessing (filter + CAR). Standardization is done
    later, per CV fold, by ChannelStandardizer."""
    X = bandpass(X, sfreq, l_freq=l_freq, h_freq=h_freq)
    if do_car:
        X = common_average_reference(X)
    return X


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 22, 500)).astype(np.float32)
    Xp = preprocess_epochs(X, sfreq=250.0)
    print("Preprocessed:", Xp.shape, "dtype", Xp.dtype)
    sc = ChannelStandardizer().fit(Xp)
    Xz = sc.transform(Xp)
    print("Standardized: mean ≈", Xz.mean(), "std ≈", Xz.std())
