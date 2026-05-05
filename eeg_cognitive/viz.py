"""
Visualizations for the EEG project.

Each function takes a results JSON (or numpy arrays) and writes a PNG to
results/figures/. Designed to be called from the demo notebook OR from
generate_figures.py for batch output.
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Note: scripts that need Agg should call matplotlib.use("Agg") themselves
# before importing this module — we don'''t set it here so notebooks can
# render plots inline.

from .data import CHANNEL_POS_22, CHANNELS_22


# ---------------- Common style ----------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], title: str,
                          out_path: str) -> None:
    """Pretty confusion-matrix heatmap with row-normalized percentages."""
    cm = np.asarray(cm)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(0.7 * len(classes) + 3.0, 0.7 * len(classes) + 2.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", color=text_color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_topomap(weights: np.ndarray, ch_names: list[str], title: str,
                 out_path: str, cmap: str = "RdBu_r", vmax: float | None = None) -> None:
    """Simple topomap on a 2D head outline. weights is shape (n_channels,)."""
    if vmax is None:
        vmax = float(np.max(np.abs(weights)) + 1e-8)
    pos = np.array([CHANNEL_POS_22[c] for c in ch_names])

    # Grid for interpolation
    xi = np.linspace(-1.0, 1.0, 200)
    yi = np.linspace(-1.0, 1.0, 200)
    XI, YI = np.meshgrid(xi, yi)
    # Inverse-distance-weighted interpolation
    Z = np.zeros_like(XI)
    eps = 1e-6
    for (x, y), w in zip(pos, weights):
        d = np.sqrt((XI - x) ** 2 + (YI - y) ** 2) + eps
        Z += w / d ** 2
    norm = np.zeros_like(XI)
    for (x, y) in pos:
        d = np.sqrt((XI - x) ** 2 + (YI - y) ** 2) + eps
        norm += 1.0 / d ** 2
    Z = Z / norm

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    # Mask outside the head circle
    mask = (XI ** 2 + YI ** 2) > 1.05 ** 2
    Z = np.where(mask, np.nan, Z)
    im = ax.contourf(XI, YI, Z, levels=20, cmap=cmap, vmin=-vmax, vmax=vmax)
    # Head outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", lw=1.5)
    # Nose
    ax.plot([-0.10, 0, 0.10], [1.0, 1.15, 1.0], color="k", lw=1.5)
    # Channels
    ax.scatter(pos[:, 0], pos[:, 1], c="black", s=18, zorder=5)
    for (x, y), name in zip(pos, ch_names):
        ax.text(x, y - 0.07, name, ha="center", va="top", fontsize=7)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_fold_accuracy(results: dict, out_path: str) -> None:
    """Bar chart: each (dataset, model) pair with mean ± std and per-fold dots."""
    rows = []
    for key, d in results.items():
        rows.append((key, d["per_fold_accuracy"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(rows))
    means = [np.mean(r[1]) for r in rows]
    stds = [np.std(r[1]) for r in rows]
    bars = ax.bar(xs, means, yerr=stds, capsize=4,
                  color=["#3367d6", "#e8662e", "#3367d6", "#e8662e"],
                  alpha=0.85, edgecolor="black", linewidth=0.8)
    for x, (_, accs) in zip(xs, rows):
        ax.scatter([x] * len(accs), accs, color="black", s=20, zorder=5,
                   edgecolor="white", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k, _ in rows], fontsize=9)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="chance (4-class)")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
               label="chance (2-class)")
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-fold accuracy by dataset and model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_eegnet_temporal_filters(model, sfreq: float, out_path: str) -> None:
    """First-layer temporal filter responses, transformed to the spectral
    domain so we can see which frequencies each filter is selective for."""
    import torch
    W = model.conv1.weight.detach().cpu().numpy().squeeze()  # (F1, kernel_length)
    n_filters, kl = W.shape
    fig, axes = plt.subplots(2, n_filters, figsize=(2.0 * n_filters, 4.5))
    for i in range(n_filters):
        # Time-domain
        axes[0, i].plot(np.arange(kl) / sfreq * 1000, W[i], color="C0", lw=1)
        axes[0, i].set_title(f"filter {i}")
        if i == 0:
            axes[0, i].set_ylabel("amplitude")
        axes[0, i].set_xlabel("ms")
        # Frequency-domain magnitude response
        F = np.abs(np.fft.rfft(W[i]))
        freqs = np.fft.rfftfreq(kl, d=1.0 / sfreq)
        axes[1, i].plot(freqs, F, color="C3", lw=1)
        axes[1, i].set_xlim(0, 40)
        if i == 0:
            axes[1, i].set_ylabel("|FFT|")
        axes[1, i].set_xlabel("Hz")
    fig.suptitle("EEGNet first-layer temporal filters: time and spectral views",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_erd_time_frequency(X: np.ndarray, y: np.ndarray, ch_names: list[str],
                             sfreq: float, ch_of_interest: str, classes: list[str],
                             out_path: str) -> None:
    """Mean log-power spectrogram per class at one channel — visualizes the
    class-specific oscillatory differences (ERD/ERS pattern)."""
    from scipy.signal import spectrogram
    ch_idx = ch_names.index(ch_of_interest)
    n_classes = len(classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(3.2 * n_classes, 3.5),
                             sharey=True)
    if n_classes == 1:
        axes = [axes]
    vmin, vmax = None, None
    spec_per_cls = []
    for c in range(n_classes):
        Xc = X[y == c, ch_idx, :]
        f, t, S = spectrogram(Xc, fs=sfreq, nperseg=int(sfreq // 2),
                              noverlap=int(sfreq // 4), axis=-1)
        S_mean = np.log(np.mean(S, axis=0) + 1e-12)
        spec_per_cls.append((f, t, S_mean))
        if vmin is None:
            vmin = S_mean.min(); vmax = S_mean.max()
        else:
            vmin = min(vmin, S_mean.min()); vmax = max(vmax, S_mean.max())

    for c in range(n_classes):
        f, t, S_mean = spec_per_cls[c]
        ax = axes[c]
        im = ax.pcolormesh(t, f, S_mean, shading="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 40)
        ax.set_title(f"class: {classes[c]}")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel(f"frequency at {ch_of_interest} (Hz)")
    fig.suptitle(f"Per-class mean log-power spectrogram at {ch_of_interest}",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, label="log power")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Quick smoke
    cm = np.array([[80, 0, 0, 0], [0, 78, 2, 0], [2, 1, 76, 1], [1, 0, 1, 78]])
    plot_confusion_matrix(cm, ["L", "R", "F", "T"], "smoke",
                          "/tmp/smoke_cm.png")
    print("OK")
