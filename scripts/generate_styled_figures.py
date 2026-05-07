"""
Regenerate the figures used in the slide deck with the 'Cortex Editorial'
palette so they integrate visually with the slide design.

Outputs are written alongside the original figures with a `_styled` suffix
so the originals (used in the README/report) remain unchanged.
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from eeg_cognitive.data import CHANNELS_22, CHANNEL_POS_22

# --- Cortex Editorial palette ---
BONE      = "#F2EFE7"
CARD      = "#FAF7EE"
INK       = "#101319"
CHARCOAL  = "#3A3F4A"
STONE     = "#82806E"
HAIRLINE  = "#D8D2C4"
SIGNAL    = "#1F4F47"   # deep teal (primary accent)
EMBER     = "#C25F3C"   # warm rust (secondary accent)

# Use a serif display + clean sans throughout
DISPLAY_FONT = "Georgia"
BODY_FONT    = "Helvetica"

plt.rcParams.update({
    "figure.facecolor": BONE,
    "axes.facecolor":   BONE,
    "savefig.facecolor": BONE,
    "axes.edgecolor":   HAIRLINE,
    "axes.labelcolor":  CHARCOAL,
    "xtick.color":      CHARCOAL,
    "ytick.color":      CHARCOAL,
    "text.color":       INK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom": True,
    "axes.linewidth":    0.7,
    "axes.titleweight":  "bold",
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "savefig.dpi": 220,
    "figure.dpi": 220,
})


def load_results():
    out = {}
    for ds in ["motor_imagery", "mental_arithmetic"]:
        for m in ["fbcsp", "eegnet"]:
            path = os.path.abspath(os.path.join(ROOT, "results",
                                                f"{ds}_{m}.json"))
            with open(path) as f:
                out[(ds, m)] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# 1. Per-fold accuracy bar chart — editorial restyle
# ---------------------------------------------------------------------------
def make_accuracy_chart(results, out_path):
    rows = [
        ("motor_imagery",     "fbcsp",  "Motor imagery\nFBCSP+LDA",      SIGNAL),
        ("motor_imagery",     "eegnet", "Motor imagery\nEEGNet",         EMBER),
        ("mental_arithmetic", "fbcsp",  "Mental arithmetic\nFBCSP+LDA",  SIGNAL),
        ("mental_arithmetic", "eegnet", "Mental arithmetic\nEEGNet",     EMBER),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BONE)
    ax.set_facecolor(BONE)

    xs = np.arange(len(rows))
    means = [np.mean(results[(ds, m)]["per_fold_accuracy"]) for ds, m, _, _ in rows]
    stds  = [np.std(results[(ds, m)]["per_fold_accuracy"])  for ds, m, _, _ in rows]
    accs_per = [results[(ds, m)]["per_fold_accuracy"] for ds, m, _, _ in rows]
    colors = [c for _, _, _, c in rows]

    # bars
    bars = ax.bar(xs, means, width=0.55, color=colors, edgecolor="none", zorder=2)
    # error bars
    ax.errorbar(xs, means, yerr=stds, fmt="none",
                ecolor=CHARCOAL, elinewidth=1.0, capsize=5, capthick=1.0, zorder=3)

    # per-fold dots
    for x, accs in zip(xs, accs_per):
        jitter = (np.random.default_rng(int(x * 13 + 7)).uniform(-0.08, 0.08, size=len(accs)))
        ax.scatter(x + jitter, accs, s=22, color=BONE, edgecolor=INK,
                   linewidth=1.0, zorder=4)

    # mean labels above bars
    for x, m, sd in zip(xs, means, stds):
        ax.text(x, m + sd + 0.03, f"{m*100:.1f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=INK)

    # chance lines, very subtle (labels on the LEFT so they don't collide
    # with downstream layout that places content on the right of the chart)
    ax.axhline(0.25, color=STONE, linestyle=(0, (1, 4)), linewidth=0.7)
    ax.axhline(0.50, color=STONE, linestyle=(0, (1, 4)), linewidth=0.7)
    ax.text(-0.45, 0.255, "chance · 4-class",
            color=STONE, fontsize=9, ha="left", va="bottom", style="italic")
    ax.text(-0.45, 0.505, "chance · 2-class",
            color=STONE, fontsize=9, ha="left", va="bottom", style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([r[2] for r in rows], color=CHARCOAL, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=STONE, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=10)

    # remove all spines except a hairline at the bottom
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.spines["bottom"].set_linewidth(0.7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=BONE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Confusion matrices — editorial restyle
# ---------------------------------------------------------------------------
def make_confusion(cm, classes, title, out_path, accent=SIGNAL):
    cm = np.asarray(cm)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    n = len(classes)

    fig, ax = plt.subplots(figsize=(0.95 * n + 1.6, 0.95 * n + 1.6))
    fig.patch.set_facecolor(BONE)
    ax.set_facecolor(BONE)

    # build a custom monochromatic colormap from BONE → accent
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ed", [BONE, accent], N=256
    )
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    # numeric labels
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            color = BONE if v > 0.55 else INK
            ax.text(j, i, f"{cm[i,j]}",
                    ha="center", va="center", color=color,
                    fontsize=14, fontweight="bold")
            ax.text(j, i + 0.32, f"{v*100:.0f}%",
                    ha="center", va="center", color=color, fontsize=8.5,
                    style="italic", alpha=0.85)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=20, ha="right",
                       color=CHARCOAL, fontsize=10)
    ax.set_yticklabels(classes, color=CHARCOAL, fontsize=10)

    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title(title, color=INK, fontsize=12, fontweight="bold",
                 loc="left", pad=14)

    # hairline frame
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=BONE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. CSP topomap — editorial restyle (transparent bg, palette accents)
# ---------------------------------------------------------------------------
def make_topomap(weights, ch_names, title, subtitle, out_path):
    weights = np.asarray(weights)
    vmax = float(np.max(np.abs(weights)) + 1e-8)
    pos = np.array([CHANNEL_POS_22[c] for c in ch_names])

    xi = np.linspace(-1.0, 1.0, 240)
    yi = np.linspace(-1.0, 1.0, 240)
    XI, YI = np.meshgrid(xi, yi)
    Z = np.zeros_like(XI); norm = np.zeros_like(XI)
    eps = 1e-6
    for (x, y), w in zip(pos, weights):
        d = np.sqrt((XI - x) ** 2 + (YI - y) ** 2) + eps
        Z += w / d ** 2
        norm += 1.0 / d ** 2
    Z = Z / norm

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ed_div", [SIGNAL, "#9EBFB1", BONE, "#E0AE99", EMBER], N=256
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(BONE)
    ax.set_facecolor(BONE)
    mask = (XI ** 2 + YI ** 2) > 1.05 ** 2
    Z = np.where(mask, np.nan, Z)
    ax.contourf(XI, YI, Z, levels=24, cmap=cmap, vmin=-vmax, vmax=vmax)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=INK, lw=1.5)
    ax.plot([-0.10, 0, 0.10], [1.0, 1.15, 1.0], color=INK, lw=1.5)
    ax.scatter(pos[:, 0], pos[:, 1], c=INK, s=14, zorder=5)
    for (x, y), name in zip(pos, ch_names):
        ax.text(x, y - 0.10, name, ha="center", va="top",
                fontsize=7, color=CHARCOAL)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.set_axis_off()

    ax.text(-1.18, 1.18, title.upper(),
            color=SIGNAL, fontsize=10, fontweight="bold",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.1", fc="none", ec="none"),
            )
    ax.text(-1.18, 1.06, subtitle,
            color=CHARCOAL, fontsize=10, ha="left", va="top", style="italic")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=BONE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. EEGNet temporal filters — editorial restyle
# ---------------------------------------------------------------------------
def make_eegnet_kernels(model, sfreq, out_path):
    import torch
    W = model.conv1.weight.detach().cpu().numpy().squeeze()
    n_filters, kl = W.shape
    fig, axes = plt.subplots(2, n_filters, figsize=(1.7 * n_filters, 4.0))
    fig.patch.set_facecolor(BONE)
    for i in range(n_filters):
        # time
        ax = axes[0, i]
        ax.set_facecolor(BONE)
        ax.plot(np.arange(kl) / sfreq * 1000, W[i],
                color=SIGNAL, lw=1.2)
        ax.set_title(f"k{i}", color=CHARCOAL, fontsize=10, loc="left")
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(HAIRLINE)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(length=0, colors=STONE, labelsize=8)
        if i == 0:
            ax.set_ylabel("amp.", color=STONE, fontsize=9)
        # freq
        ax = axes[1, i]
        ax.set_facecolor(BONE)
        F = np.abs(np.fft.rfft(W[i]))
        freqs = np.fft.rfftfreq(kl, d=1.0 / sfreq)
        ax.fill_between(freqs, F, color=EMBER, alpha=0.35, linewidth=0)
        ax.plot(freqs, F, color=EMBER, lw=1.2)
        ax.set_xlim(0, 40)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(HAIRLINE)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(length=0, colors=STONE, labelsize=8)
        if i == 0:
            ax.set_ylabel("|FFT|", color=STONE, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=BONE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    FIG = os.path.abspath(os.path.join(ROOT, "results", "figures"))
    os.makedirs(FIG, exist_ok=True)

    results = load_results()

    # 1. accuracy chart
    np.random.seed(0)
    make_accuracy_chart(results, os.path.join(FIG, "styled_accuracy_bar.png"))
    print("✔ styled accuracy bar")

    # 2. confusion matrices (4)
    for (ds, m), d in results.items():
        accent = SIGNAL if m == "fbcsp" else EMBER
        title = f"{'Motor imagery' if ds == 'motor_imagery' else 'Mental arithmetic'}  ·  {'FBCSP+LDA' if m == 'fbcsp' else 'EEGNet'}"
        make_confusion(np.array(d["confusion"]), d["classes"], title,
                       os.path.join(FIG, f"styled_confusion_{ds}_{m}.png"),
                       accent=accent)
    print("✔ styled confusion matrices")

    # 3. fresh CSP topomaps (one per class) using existing-style data
    print("training fresh FBCSP for topomaps...")
    from eeg_cognitive.data import make_motor_imagery_synthetic
    from eeg_cognitive.preprocess import preprocess_epochs, ChannelStandardizer
    from eeg_cognitive.models import FBCSPClassifier
    mi = make_motor_imagery_synthetic(n_subjects=4, trials_per_class=20,
                                      snr_db=-12, seed=0)
    Xmi = preprocess_epochs(mi.X, mi.sfreq)
    sc = ChannelStandardizer().fit(Xmi)
    Xz = sc.transform(Xmi)
    fb = FBCSPClassifier(sfreq=mi.sfreq, n_components=4, n_features=20).fit(Xz, mi.y)
    for cls_idx, cls_name in enumerate(mi.class_names):
        w = fb.get_csp_filter(band_idx=1, class_idx=cls_idx, comp_idx=0)
        make_topomap(w, CHANNELS_22,
                     title=f"CSP top filter · {cls_name}",
                     subtitle="mu band 8–12 Hz, one-vs-rest",
                     out_path=os.path.join(FIG, f"styled_topomap_{cls_name}.png"))
    print("✔ styled topomaps")

    # 4. fresh EEGNet for temporal filter view
    print("training fresh EEGNet for filter visualization...")
    from eeg_cognitive.models import EEGNetClassifier
    en = EEGNetClassifier(epochs=30, batch_size=64, lr=1.5e-3, weight_decay=1e-4,
                          patience=12, val_frac=0.2, seed=0)
    en.sfreq = mi.sfreq
    en.eegnet_kernel_length = max(16, int(mi.sfreq // 4))
    en.fit(Xz, mi.y)
    make_eegnet_kernels(en.model_, mi.sfreq,
                        os.path.join(FIG, "styled_eegnet_kernels.png"))
    print("✔ styled EEGNet kernels")

    print("\nAll styled figures written.")


if __name__ == "__main__":
    main()
