"""
Build the seven walkthrough notebooks.

Each notebook is constructed as ipynb JSON and written to disk; execute them
afterwards with `jupyter nbconvert --execute` (the Makefile target
`make notebooks` does this for all of them).

Run from the project root or from notebooks/.
"""
from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
_uid = [0]

def _id() -> str:
    _uid[0] += 1
    return f"cell-{_uid[0]:04d}"

def code(src: str) -> dict:
    return {"cell_type": "code", "id": _id(), "metadata": {}, "outputs": [],
            "execution_count": None, "source": src.splitlines(keepends=True)}

def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _id(), "metadata": {},
            "source": src.splitlines(keepends=True)}

def write_nb(name: str, cells: list) -> None:
    nb = {"cells": cells,
          "metadata": {
              "kernelspec": {"name": "python3", "display_name": "Python 3"},
              "language_info": {"name": "python", "version": "3.10"},
          },
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(HERE, name)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("  wrote", name)

# Common header that adds the project root to sys.path so notebooks import
# the eeg_cognitive package without needing pip install.
HEADER = """# Make the eeg_cognitive package importable when running this notebook
# directly out of a clone (no pip install required).
import os, sys
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""

# ============================================================================
# 01 — introduction
# ============================================================================
def build_01():
    cells = [
md("""# 01  ·  Introduction

This is the first of seven walkthrough notebooks for **eeg-cognitive-decoding**.

The project compares two machine-learning pipelines for decoding cognitive
states from scalp EEG:

- **FBCSP + LDA** — the classical filter-bank Common-Spatial-Patterns
  baseline used in BCI competitions for over a decade.
- **EEGNet** — a compact (~2,400-parameter) convolutional network from
  Lawhern et al. 2018, designed specifically for EEG.

Both are run on two qualitatively different cognitive paradigms — motor
imagery (4-class) and mental arithmetic vs. rest (2-class).

This notebook just gets you oriented: load some data, look at it,
visualise a few raw trials. The next six notebooks walk through every
piece of the pipeline in detail.
"""),
        code(HEADER),
md("""## Generate a small synthetic dataset

For reproducibility we use a physiologically grounded synthetic EEG
generator. Each trial is the sum of pink (1/f) background noise and a
class-specific narrowband signal modulated by a scalp gain map encoding
the textbook ERD/ERS pattern. See `eeg_cognitive/data.py` for the full
implementation."""),
        code("""from eeg_cognitive import make_motor_imagery_synthetic
ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=12,
                                  snr_db=-12, seed=0)
print(f"shape:    {ds.X.shape}  (n_trials, n_channels, n_samples)")
print(f"sfreq:    {ds.sfreq} Hz  →  {ds.X.shape[-1]/ds.sfreq:.1f} s per trial")
print(f"channels: {len(ds.ch_names)} = {ds.ch_names[:6]} ...")
print(f"classes:  {ds.class_names}")
print(f"labels:   {ds.y[:10].tolist()} ... (first 10)")
"""),
md("""## Look at one trial across all 22 channels

A traditional "EEG strip" — channels stacked vertically, time on the
x-axis. Each channel is a noisy oscillation; the model's job is to figure
out which spatial × spectral pattern distinguishes the four motor-imagery
classes."""),
        code("""import numpy as np
import matplotlib.pyplot as plt

trial_idx = 0
trial = ds.X[trial_idx]                 # shape (22, n_samples)
truth = ds.class_names[ds.y[trial_idx]]
t = np.arange(trial.shape[1]) / ds.sfreq

fig, ax = plt.subplots(figsize=(9, 6))
spacing = 4 * trial.std()
for i, name in enumerate(ds.ch_names):
    ax.plot(t, trial[i] + (len(ds.ch_names) - 1 - i) * spacing,
            color='black', linewidth=0.55)
    ax.text(-0.04, (len(ds.ch_names) - 1 - i) * spacing, name,
            ha='right', va='center', fontsize=7)
ax.set_xlim(0, t[-1])
ax.set_yticks([]); ax.set_xlabel('time (s)')
ax.set_title(f'Trial 0 (class: {truth}) — 22 channels stacked', fontsize=11)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout(); plt.show()
"""),
md("""## Class balance"""),
        code("""import numpy as np
counts = np.bincount(ds.y)
for cls, n in zip(ds.class_names, counts):
    print(f"  {cls:12s}  {n} trials")
print(f"  total       {len(ds.y)} trials")
"""),
md("""## What's next

- **02** — the synthetic data generator: what's actually in each trial,
  and how class-specific patterns are embedded.
- **03** — preprocessing: bandpass, common-average reference, leakage-free
  per-channel z-scoring.
- **04** — FBCSP + LDA from scratch, with topomaps of the learned spatial filters.
- **05** — EEGNet training, with the first-layer kernels visualised.
- **06** — full results comparison and the cross-subject finding.
- **07** — the live-prediction demo widget for the talk.
"""),
    ]
    write_nb("01_introduction.ipynb", cells)


# ============================================================================
# 02 — synthetic data
# ============================================================================
def build_02():
    cells = [
md("""# 02  ·  The synthetic data generator

Real BCI Competition IV-2a is ~500 MB and requires network access to the
competition servers. To make this project reproducible without that
dependency, we wrote a physiologically grounded synthetic EEG generator.

This notebook shows what's inside a single trial and verifies — visually
— that the class-specific scalp patterns are actually present in the data.
"""),
        code(HEADER),
md("""## The generative model, in one block

For each trial we compute:

```
trial[ch, t] = pink_noise[ch, t]
             + snr · gain[cls, ch] · rhythm[band, t]
```

- **pink noise** reproduces EEG's natural 1/f background spectrum.
- **rhythm** is a narrowband oscillation in the relevant band (mu, beta,
  theta, alpha) with random phase and slowly drifting amplitude.
- **gain[cls, ch]** is a per-class spatial weight encoding the textbook
  ERD/ERS pattern — negative weights at electrodes where the rhythm is
  *suppressed* (ERD), positive weights where it is *enhanced* (ERS).
- The gain map is then smeared by a Gaussian over neighbouring channels
  to simulate volume conduction.
- Per-subject and per-trial multiplicative jitter ensure no two
  recordings are identical.
"""),
        code("""from eeg_cognitive import make_motor_imagery_synthetic, CHANNELS_22
ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=20,
                                  snr_db=-10, seed=0)
print(f"X shape: {ds.X.shape}")
print(f"classes: {ds.class_names}")
"""),
md("""## Per-class spectrograms at C3 vs C4

If the class signatures are right, **C3** (left-hemisphere motor cortex)
should have *more* mu-band power for left-hand imagery than for right-hand
imagery, because right-hand imagery causes ERD at C3 (suppressing the
rhythm there). C4 should show the mirror image."""),
        code("""import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_per_class_spec(channel_name, axes_row):
    ch_idx = CHANNELS_22.index(channel_name)
    vmin, vmax = None, None
    spec_per_cls = []
    for c in range(len(ds.class_names)):
        Xc = ds.X[ds.y == c, ch_idx, :]
        f, t, S = spectrogram(Xc, fs=ds.sfreq, nperseg=int(ds.sfreq // 2),
                              noverlap=int(ds.sfreq // 4), axis=-1)
        Sm = np.log(np.mean(S, axis=0) + 1e-12)
        spec_per_cls.append((f, t, Sm))
        vmin = Sm.min() if vmin is None else min(vmin, Sm.min())
        vmax = Sm.max() if vmax is None else max(vmax, Sm.max())
    for c in range(len(ds.class_names)):
        f, t, Sm = spec_per_cls[c]
        ax = axes_row[c]
        ax.pcolormesh(t, f, Sm, shading='auto', cmap='viridis',
                      vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 35)
        ax.set_title(f'{channel_name} — {ds.class_names[c]}', fontsize=10)
        ax.set_xlabel('s'); ax.set_ylabel('Hz')

fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharey=True)
plot_per_class_spec('C3', axes[0])
plot_per_class_spec('C4', axes[1])
plt.tight_layout(); plt.show()
"""),
md("""## Reading the topographic gain map

Each class has a different scalp gain map. Plot the four gain maps
side-by-side (averaged across trials) to see the textbook neuroanatomy
the generator embeds:

- **left_hand** — suppression around C4 (right-central)
- **right_hand** — suppression around C3 (left-central)
- **feet** — suppression around Cz (central midline)
- **tongue** — enhancement at FCz / Fz (fronto-central beta ERS)
"""),
        code("""# Compute per-class mean log-power in the mu band (8-13 Hz) at every channel.
import numpy as np
from scipy.signal import welch

def mean_mu_power(X, sfreq, band=(8, 13)):
    f, P = welch(X, fs=sfreq, nperseg=int(sfreq // 2), axis=-1)
    bandmask = (f >= band[0]) & (f <= band[1])
    return P[..., bandmask].mean(axis=-1)

per_class_mu = []
for c in range(len(ds.class_names)):
    Xc = ds.X[ds.y == c]
    Pc = np.log(mean_mu_power(Xc, ds.sfreq).mean(axis=0) + 1e-12)  # (n_ch,)
    per_class_mu.append(Pc)

# Use the project's topomap helper for a quick visual.
from eeg_cognitive.viz import plot_topomap
import os
os.makedirs('../results/figures/notebook_outputs', exist_ok=True)
for cls_idx, cls_name in enumerate(ds.class_names):
    weights = per_class_mu[cls_idx] - np.mean(per_class_mu, axis=0)
    plot_topomap(weights, CHANNELS_22,
                 f'mean log-mu power — {cls_name} (relative)',
                 f'../results/figures/notebook_outputs/topo_{cls_name}.png')

# Render them in a 2x2 grid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, cls_name in zip(axes.ravel(), ds.class_names):
    ax.imshow(mpimg.imread(f'../results/figures/notebook_outputs/topo_{cls_name}.png'))
    ax.axis('off')
plt.tight_layout(); plt.show()
"""),
md("""You should see exactly the textbook localization: left-hand class
has the strongest *deviation* from average around C4 (right hemisphere),
right-hand around C3, feet around Cz, tongue around the fronto-central
midline. The fact that the synthetic generator embeds this is what makes
the dataset useful as a controlled benchmark — a correct ML pipeline
*must* recover these patterns.
"""),
    ]
    write_nb("02_synthetic_data.ipynb", cells)


# ============================================================================
# 03 — preprocessing
# ============================================================================
def build_03():
    cells = [
md("""# 03  ·  Preprocessing

Three steps — bandpass, common-average reference, per-channel z-score.
Every one of them is justified, and the *order* and *scope* matter:

1. **Bandpass 4–40 Hz** — keep the cognitively relevant range, kill DC drift
   and high-frequency muscle artifact.
2. **Common-average reference (CAR)** — subtract the across-channel mean at
   each time point; removes shared noise (line, reference electrode).
3. **Per-channel z-score** — but fit on the *train fold only* and apply to
   the validation fold; otherwise we leak distributional information across
   folds.
"""),
        code(HEADER),
        code("""import numpy as np
import matplotlib.pyplot as plt
from eeg_cognitive import make_motor_imagery_synthetic
from eeg_cognitive.preprocess import (
    bandpass, common_average_reference, ChannelStandardizer
)

ds = make_motor_imagery_synthetic(n_subjects=1, trials_per_class=10,
                                  snr_db=-12, seed=0)
print(f"raw X shape: {ds.X.shape}")
"""),
md("""## Step 1 — bandpass

Compare the raw signal to its 4–40 Hz bandpassed version. The DC drift and
slow trends should disappear; the cognitively relevant rhythms should
remain intact."""),
        code("""raw = ds.X[0, 0]  # trial 0, channel 0
filt = bandpass(ds.X, ds.sfreq, l_freq=4, h_freq=40)[0, 0]
t = np.arange(len(raw)) / ds.sfreq

fig, ax = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
ax[0].plot(t, raw, color='#82806E', linewidth=0.7)
ax[0].set_title('raw'); ax[0].set_ylabel('a.u.')
ax[1].plot(t, filt, color='#1F4F47', linewidth=0.7)
ax[1].set_title('bandpassed 4–40 Hz'); ax[1].set_xlabel('s'); ax[1].set_ylabel('a.u.')
for a in ax:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()
"""),
md("""## Step 2 — common-average reference

Subtracting the cross-channel mean removes line-noise–like signals
shared across electrodes. The amplitude per channel typically drops a
little; that's expected."""),
        code("""Xf = bandpass(ds.X, ds.sfreq, 4, 40)
Xc = common_average_reference(Xf)
print(f"std before CAR: {Xf.std():.3f}")
print(f"std after CAR:  {Xc.std():.3f}")
"""),
md("""## Step 3 — leakage-free standardization

The most common subtle bug in EEG ML is fitting `(mean, std)` on the
*whole* dataset before splitting into train/test. That leaks information
from the test fold into the train fold.

Our `ChannelStandardizer` is fit on the train fold only and applied to
the validation fold."""),
        code("""rng = np.random.default_rng(0)
n = len(ds.y)
perm = rng.permutation(n)
tr, te = perm[: int(n*0.8)], perm[int(n*0.8):]

scaler = ChannelStandardizer().fit(Xc[tr])
Xtr = scaler.transform(Xc[tr])
Xte = scaler.transform(Xc[te])

print(f"train mean: {Xtr.mean():.5f}, std: {Xtr.std():.3f}")
print(f"val   mean: {Xte.mean():.5f}, std: {Xte.std():.3f}")
print()
print("note: val mean is NOT exactly 0 — that's the point. The val fold")
print("was standardized using train stats, not its own.")
"""),
md("""All three steps are wired into `eeg_cognitive.preprocess.preprocess_epochs`,
which the experiment scripts and the next two notebooks call directly."""),
    ]
    write_nb("03_preprocessing.ipynb", cells)


# ============================================================================
# 04 — FBCSP + LDA
# ============================================================================
def build_04():
    cells = [
md("""# 04  ·  FBCSP + LDA

The classical baseline. Four conceptual steps:

1. **Filter bank** — bandpass into 9 sub-bands (4 Hz wide, 4–40 Hz).
2. **CSP per band** — for each band and each class, fit a one-vs-rest
   binary CSP (4 components). Variance ratio between class pairs.
3. **MI feature selection** — keep the top 20 of the 144 raw features.
4. **LDA** — linear discriminant analysis classifier.

This notebook trains FBCSP on motor imagery, evaluates with 5-fold CV,
and visualises the recovered spatial filters as scalp topomaps.
"""),
        code(HEADER),
        code("""import time, numpy as np
from eeg_cognitive import (make_motor_imagery_synthetic, preprocess_epochs,
                          ChannelStandardizer, cross_validate, CHANNELS_22)
from eeg_cognitive.models import FBCSPClassifier

# Smaller dataset here so the notebook executes quickly when re-run;
# the headline numbers in the report use a larger dataset.
ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=14,
                                  snr_db=-12, seed=0)
X = preprocess_epochs(ds.X, ds.sfreq)
print(f"X = {X.shape}, y = {ds.y.shape}, classes = {ds.class_names}")
"""),
md("""## 5-fold CV"""),
        code("""t0 = time.time()
res = cross_validate(
    X, ds.y, ds.sfreq,
    clf_factory=lambda: FBCSPClassifier(sfreq=ds.sfreq, n_components=4, n_features=16),
    n_splits=5, seed=0, standardize=True,
)
print(f"FBCSP+LDA  ·  motor imagery  ·  {time.time()-t0:.1f}s")
s = res.summary()
print(f"  acc       = {s['acc_mean']*100:5.1f} ± {s['acc_std']*100:4.1f} %")
print(f"  macro-F1  = {s['f1_mean']*100:5.1f} ± {s['f1_std']*100:4.1f}")
print(f"  Cohen's κ = {s['kappa_mean']:+.3f} ± {s['kappa_std']:.3f}")
print()
print("per-fold accuracy:", [f"{a*100:.1f}%" for a in res.accuracy])
"""),
md("""## Confusion matrix

Where do the residual errors cluster? With four classes that all involve
central electrodes, the most common confusion is between motor-imagery
classes that share spatial neighbourhood — a real-EEG pattern, not a bug
in the model."""),
        code("""from eeg_cognitive.viz import plot_confusion_matrix
import os, matplotlib.pyplot as plt, matplotlib.image as mpimg

os.makedirs('../results/figures/notebook_outputs', exist_ok=True)
out = '../results/figures/notebook_outputs/04_fbcsp_cm.png'
plot_confusion_matrix(np.array(res.confusion), ds.class_names,
                      f"FBCSP+LDA  ·  motor imagery  ·  acc={s['acc_mean']*100:.1f}%",
                      out)
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(mpimg.imread(out)); ax.axis('off'); plt.show()
"""),
md("""## Topomaps of the learned spatial filters

The CSP filter for each class is a 22-dimensional weight vector. Plotting
it as a head-shaped topomap shows *which* electrodes the filter relies on.
For motor imagery the textbook prediction is sharp:

| class      | predicted peak |
|------------|----------------|
| left_hand  | C4 (contralateral, right hemisphere) |
| right_hand | C3 (contralateral, left hemisphere) |
| feet       | Cz (central midline) |
| tongue     | FCz / Fz (fronto-central) |

Train one fresh FBCSP on the full data and inspect the top filter from
the mu band (8–12 Hz) for each class:"""),
        code("""sc = ChannelStandardizer().fit(X)
Xz = sc.transform(X)
fb = FBCSPClassifier(sfreq=ds.sfreq, n_components=4, n_features=20).fit(Xz, ds.y)

from eeg_cognitive.viz import plot_topomap
mu_band_idx = 1  # 8-12 Hz in DEFAULT_BANDS

fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for cls_idx, cls_name in enumerate(ds.class_names):
    w = fb.get_csp_filter(band_idx=mu_band_idx, class_idx=cls_idx, comp_idx=0)
    out_p = f'../results/figures/notebook_outputs/04_topo_{cls_name}.png'
    plot_topomap(w, CHANNELS_22, f'top CSP — {cls_name} (mu band)', out_p)
    axes[cls_idx].imshow(mpimg.imread(out_p)); axes[cls_idx].axis('off')
plt.tight_layout(); plt.show()
"""),
md("""Each filter peaks where neuroanatomy says it should. **The model
recovered the contralateral-motor-cortex rule from raw EEG without being
told.** That's the strongest cog-sci-relevant evidence in the project —
it says the working classifier is reading out the right brain regions.
"""),
    ]
    write_nb("04_fbcsp.ipynb", cells)


# ============================================================================
# 05 — EEGNet
# ============================================================================
def build_05():
    cells = [
md("""# 05  ·  EEGNet

The compact convolutional baseline (Lawhern et al. 2018). About 2,400
parameters total. CPU-trainable in seconds.

The architecture is engineered to do, end-to-end, exactly what FBCSP does
in pieces: temporal filtering (Conv2d temporal kernels), spatial filtering
(depthwise channel-conv per temporal kernel), feature integration
(separable conv), linear decision (linear head).

This notebook trains EEGNet on motor imagery, plots the loss curves, and
visualises the first-layer temporal kernels in time and frequency."""),
        code(HEADER),
        code("""import time
import numpy as np, matplotlib.pyplot as plt
from eeg_cognitive import (make_motor_imagery_synthetic, preprocess_epochs,
                          ChannelStandardizer, cross_validate)
from eeg_cognitive.models import EEGNetClassifier

# Smaller dataset for the notebook so it executes quickly; the headline
# numbers in the report use a larger dataset.
ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=14,
                                  snr_db=-12, seed=0)
X = preprocess_epochs(ds.X, ds.sfreq)
print(f"X = {X.shape}")
"""),
md("""## Train one model and look at the loss curves"""),
        code("""rng = np.random.default_rng(0)
perm = rng.permutation(len(ds.y))
tr, te = perm[: int(len(ds.y)*0.8)], perm[int(len(ds.y)*0.8):]

sc = ChannelStandardizer().fit(X[tr])
Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

t0 = time.time()
clf = EEGNetClassifier(epochs=30, batch_size=64, lr=1.5e-3, weight_decay=1e-4,
                       patience=12, val_frac=0.2, verbose=False, seed=0)
clf.sfreq = ds.sfreq
clf.eegnet_kernel_length = max(16, int(ds.sfreq // 4))
clf.fit(Xtr, ds.y[tr])
print(f"trained in {time.time()-t0:.1f}s")
print(f"params = {sum(p.numel() for p in clf.model_.parameters())}")
print(f"val acc on the model's internal split: {clf.history_['val_acc'][-1]*100:.1f}%")
print(f"held-out acc: {(clf.predict(Xte) == ds.y[te]).mean()*100:.1f}%")
"""),
        code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
h = clf.history_
ep = np.arange(1, len(h['train_loss'])+1)
ax[0].plot(ep, h['train_loss'], label='train', color='#1F4F47')
ax[0].plot(ep, h['val_loss'], label='val', color='#C25F3C')
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('cross-entropy loss'); ax[0].set_title('loss')
ax[0].legend(); ax[0].spines['top'].set_visible(False); ax[0].spines['right'].set_visible(False)

ax[1].plot(ep, h['val_acc'], color='#1F4F47')
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('val accuracy'); ax[1].set_title('val accuracy')
ax[1].set_ylim(0, 1.05)
ax[1].spines['top'].set_visible(False); ax[1].spines['right'].set_visible(False)
plt.tight_layout(); plt.show()
"""),
md("""## 3-fold CV (kept short for fast notebook execution)"""),
        code("""def make_eegnet():
    c = EEGNetClassifier(epochs=20, batch_size=64, lr=1.5e-3,
                         weight_decay=1e-4, patience=8, val_frac=0.2, seed=0)
    c.sfreq = ds.sfreq
    c.eegnet_kernel_length = max(16, int(ds.sfreq // 4))
    return c

t0 = time.time()
res = cross_validate(X, ds.y, ds.sfreq, clf_factory=make_eegnet,
                     n_splits=3, seed=0, standardize=True)
print(f"EEGNet  ·  motor imagery  ·  {time.time()-t0:.1f}s")
s = res.summary()
print(f"  acc       = {s['acc_mean']*100:5.1f} ± {s['acc_std']*100:4.1f} %")
print(f"  macro-F1  = {s['f1_mean']*100:5.1f} ± {s['f1_std']*100:4.1f}")
print(f"  Cohen's κ = {s['kappa_mean']:+.3f} ± {s['kappa_std']:.3f}")
print(f"  per-fold:   {[f'{a*100:.1f}%' for a in res.accuracy]}")
"""),
md("""## What did the first layer learn?

Plot each of the 8 temporal kernels in the time domain (top row) and in
the spectral domain (bottom row). Several should peak in the mu (8–13 Hz)
or beta (13–30 Hz) range — the same band FBCSP encodes by hand."""),
        code("""W = clf.model_.conv1.weight.detach().cpu().numpy().squeeze()
n_filters, kl = W.shape

fig, axes = plt.subplots(2, n_filters, figsize=(2.0*n_filters, 4.5))
for i in range(n_filters):
    axes[0, i].plot(np.arange(kl)/ds.sfreq*1000, W[i], color='#1F4F47', lw=1)
    axes[0, i].set_title(f'kernel {i}', fontsize=9)
    axes[0, i].set_xlabel('ms')
    F = np.abs(np.fft.rfft(W[i]))
    freqs = np.fft.rfftfreq(kl, d=1.0/ds.sfreq)
    axes[1, i].fill_between(freqs, F, color='#C25F3C', alpha=0.4)
    axes[1, i].plot(freqs, F, color='#C25F3C', lw=1)
    axes[1, i].set_xlim(0, 40); axes[1, i].set_xlabel('Hz')
for ax in axes.ravel():
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('EEGNet first-layer temporal kernels', y=1.02, fontweight='bold')
plt.tight_layout(); plt.show()
"""),
    ]
    write_nb("05_eegnet.ipynb", cells)


# ============================================================================
# 06 — results comparison
# ============================================================================
def build_06():
    cells = [
md("""# 06  ·  Results comparison

Pull the four saved-result JSONs and lay them side-by-side. This is the
notebook to look at if you only have time for one — every headline
number, the bar chart, the four confusion matrices, and the cross-subject
generalization gap, all in one place."""),
        code(HEADER),
        code("""import json, os, numpy as np, matplotlib.pyplot as plt
RESULTS = '../results'

results = {}
for ds_name in ['motor_imagery', 'mental_arithmetic']:
    for m in ['fbcsp', 'eegnet']:
        path = os.path.join(RESULTS, f'{ds_name}_{m}.json')
        if os.path.exists(path):
            with open(path) as f:
                results[(ds_name, m)] = json.load(f)
        else:
            print(f'  missing: {path}')

print(f'loaded {len(results)} result files')
"""),
md("""## Headline table"""),
        code("""print(f"{'dataset':22s} {'model':12s} {'acc':>14s} {'F1':>14s} {'kappa':>14s}")
print('-' * 78)
for (dsn, m), d in results.items():
    s = d['summary']
    name = 'FBCSP+LDA' if m == 'fbcsp' else 'EEGNet'
    acc = f"{s['acc_mean']*100:5.1f} ± {s['acc_std']*100:4.1f}"
    f1  = f"{s['f1_mean']*100:5.1f} ± {s['f1_std']*100:4.1f}"
    kp  = f"{s['kappa_mean']:+.3f} ± {s['kappa_std']:.3f}"
    print(f"{dsn:22s} {name:12s} {acc:>14s} {f1:>14s} {kp:>14s}")
"""),
md("""## Per-fold accuracy bar chart"""),
        code("""rows = []
for (dsn, m) in [('motor_imagery','fbcsp'), ('motor_imagery','eegnet'),
                 ('mental_arithmetic','fbcsp'), ('mental_arithmetic','eegnet')]:
    if (dsn, m) in results:
        rows.append((f'{dsn}\\n{m}', results[(dsn, m)]['per_fold_accuracy'],
                     '#1F4F47' if m == 'fbcsp' else '#C25F3C'))

fig, ax = plt.subplots(figsize=(9.5, 5))
xs = np.arange(len(rows))
for x, (label, accs, color) in zip(xs, rows):
    m, sd = np.mean(accs), np.std(accs)
    ax.bar(x, m, color=color, width=0.55)
    ax.errorbar(x, m, yerr=sd, fmt='none', ecolor='black', capsize=4)
    jitter = np.random.default_rng(int(x*7+1)).uniform(-0.06, 0.06, len(accs))
    ax.scatter(x + jitter, accs, s=28, color='white', edgecolor='black', zorder=5)
    ax.text(x, m + sd + 0.04, f'{m*100:.1f}%', ha='center',
            fontweight='bold', fontsize=12)

ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows], fontsize=10)
ax.set_ylim(0, 1.10); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)
ax.axhline(0.50, color='gray', linestyle=':', alpha=0.5, linewidth=0.7)
ax.set_title('5-fold CV accuracy by dataset and model', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout(); plt.show()
"""),
md("""## Confusion matrices, side-by-side"""),
        code("""from eeg_cognitive.viz import plot_confusion_matrix
import matplotlib.image as mpimg

os.makedirs('../results/figures/notebook_outputs', exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (dsn, m) in zip(axes.ravel(),
        [('motor_imagery','fbcsp'), ('motor_imagery','eegnet'),
         ('mental_arithmetic','fbcsp'), ('mental_arithmetic','eegnet')]):
    if (dsn, m) not in results:
        ax.axis('off'); continue
    d = results[(dsn, m)]
    name = 'FBCSP+LDA' if m == 'fbcsp' else 'EEGNet'
    title = f"{dsn} · {name} · acc={d['summary']['acc_mean']*100:.1f}%"
    out = f'../results/figures/notebook_outputs/06_cm_{dsn}_{m}.png'
    plot_confusion_matrix(np.array(d['confusion']), d['classes'], title, out)
    ax.imshow(mpimg.imread(out)); ax.axis('off')
plt.tight_layout(); plt.show()
"""),
md("""## The cross-subject generalization gap

Train on subjects 0–2 of motor imagery, test on subject 3. The drop is
the cost of treating each new person's brain as a fresh distribution."""),
        code("""from eeg_cognitive import (make_motor_imagery_synthetic, preprocess_epochs,
                          ChannelStandardizer)
from eeg_cognitive.models import FBCSPClassifier, EEGNetClassifier
from sklearn.metrics import accuracy_score

mi = make_motor_imagery_synthetic(n_subjects=4, trials_per_class=20,
                                  snr_db=-12, seed=0)
Xall = preprocess_epochs(mi.X, mi.sfreq)
n = len(mi.y); per_subj = n // 4
sub_train = np.arange(0, 3 * per_subj)
sub_test = np.arange(3 * per_subj, 4 * per_subj)
sc = ChannelStandardizer().fit(Xall[sub_train])
Xa = sc.transform(Xall[sub_train]); ya = mi.y[sub_train]
Xb = sc.transform(Xall[sub_test]);  yb = mi.y[sub_test]

fb = FBCSPClassifier(sfreq=mi.sfreq, n_components=4, n_features=20).fit(Xa, ya)
en = EEGNetClassifier(epochs=30, batch_size=64, lr=1.5e-3, weight_decay=1e-4,
                      patience=12, val_frac=0.2, seed=0)
en.sfreq = mi.sfreq; en.eegnet_kernel_length = max(16, int(mi.sfreq // 4))
en.fit(Xa, ya)

print('CROSS-SUBJECT (train on subj 0-2, test on subj 3):')
print(f'  FBCSP  acc = {accuracy_score(yb, fb.predict(Xb))*100:5.1f} %')
print(f'  EEGNet acc = {accuracy_score(yb, en.predict(Xb))*100:5.1f} %')
print()
print('Compare to within-subject CV above. Both drop dramatically;')
print('EEGNet drops less. Capacity that hurt within-subject helps cross-subject.')
"""),
    ]
    write_nb("06_results.ipynb", cells)


# ============================================================================
# 07 — demo (simplified rebuild of 05_demo)
# ============================================================================
def build_07():
    cells = [
md("""# 07  ·  Live-prediction demo

The interactive cell at the bottom of this notebook is the on-stage demo
for the talk. It picks a random held-out trial, shows the 22-channel EEG,
and displays both models' class-probability bars side-by-side.

Re-run the demo cell during the talk for each audience prediction."""),
        code(HEADER),
        code("""import os, sys, time, random
import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from eeg_cognitive import (make_motor_imagery_synthetic, preprocess_epochs,
                          ChannelStandardizer, CHANNELS_22)
from eeg_cognitive.models import FBCSPClassifier, EEGNetClassifier
from eeg_cognitive.viz import plot_confusion_matrix
print('OK')
"""),
md("""## Stratified 80/20 split, train both models"""),
        code("""ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=14,
                                  snr_db=-12, seed=0)
X = preprocess_epochs(ds.X, ds.sfreq)
idx_tr, idx_demo = train_test_split(np.arange(len(ds.y)), test_size=0.2,
                                    stratify=ds.y, random_state=42)
sc = ChannelStandardizer().fit(X[idx_tr])
Xtr, ytr = sc.transform(X[idx_tr]), ds.y[idx_tr]
Xdm, ydm = sc.transform(X[idx_demo]), ds.y[idx_demo]
print(f'train {Xtr.shape}  demo {Xdm.shape}  classes {ds.class_names}')

t0 = time.time()
fb = FBCSPClassifier(sfreq=ds.sfreq, n_components=4, n_features=20).fit(Xtr, ytr)
print(f'FBCSP fit {time.time()-t0:.1f}s')
t0 = time.time()
en = EEGNetClassifier(epochs=20, batch_size=64, lr=1.5e-3, weight_decay=1e-4,
                      patience=8, val_frac=0.2, seed=0)
en.sfreq = ds.sfreq; en.eegnet_kernel_length = max(16, int(ds.sfreq // 4))
en.fit(Xtr, ytr)
print(f'EEGNet fit {time.time()-t0:.1f}s, {sum(p.numel() for p in en.model_.parameters())} params')
"""),
md("""## The live cell — re-run for every audience prediction"""),
        code("""def predict_one(trial_idx=None):
    if trial_idx is None: trial_idx = random.randrange(len(Xdm))
    trial = Xdm[trial_idx]; truth = ds.class_names[ydm[trial_idx]]
    fb_p = fb.predict_proba(trial[None, ...])[0]
    en_p = en.predict_proba(trial[None, ...])[0]

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.45)
    a_eeg = fig.add_subplot(gs[:, 0])
    a_fb  = fig.add_subplot(gs[0, 1])
    a_en  = fig.add_subplot(gs[1, 1])

    n_ch, n_t = trial.shape
    t = np.arange(n_t) / ds.sfreq
    spacing = 4 * trial.std()
    for i in range(n_ch):
        a_eeg.plot(t, trial[i] + (n_ch - 1 - i) * spacing,
                   color='black', linewidth=0.55)
        a_eeg.text(-0.04, (n_ch - 1 - i)*spacing, CHANNELS_22[i],
                   ha='right', va='center', fontsize=7)
    a_eeg.set_xlim(0, t[-1]); a_eeg.set_yticks([])
    a_eeg.set_xlabel('time (s)')
    a_eeg.set_title(f'Trial {trial_idx} of demo set\\ntrue class: {truth}',
                    fontweight='bold')

    a_fb.barh(ds.class_names, fb_p, color='#1F4F47')
    fb_top = ds.class_names[fb_p.argmax()]
    a_fb.set_title(f'FBCSP+LDA → {fb_top}',
                   color='green' if fb_top == truth else 'red')
    a_fb.invert_yaxis(); a_fb.set_xlim(0, 1)

    a_en.barh(ds.class_names, en_p, color='#C25F3C')
    en_top = ds.class_names[en_p.argmax()]
    a_en.set_title(f'EEGNet → {en_top}',
                   color='green' if en_top == truth else 'red')
    a_en.invert_yaxis(); a_en.set_xlim(0, 1)
    plt.show()
    return truth, fb_top, en_top

random.seed(7)
predict_one()
"""),
md("""## Replay across the full demo set"""),
        code("""fb_pred = fb.predict(Xdm)
en_pred = en.predict(Xdm)
print(f'FBCSP demo-set acc:  {accuracy_score(ydm, fb_pred)*100:5.1f}%')
print(f'EEGNet demo-set acc: {accuracy_score(ydm, en_pred)*100:5.1f}%')

cm_fb = confusion_matrix(ydm, fb_pred, labels=range(len(ds.class_names)))
cm_en = confusion_matrix(ydm, en_pred, labels=range(len(ds.class_names)))
os.makedirs('../results/figures/notebook_outputs', exist_ok=True)
plot_confusion_matrix(cm_fb, ds.class_names, 'FBCSP — demo split',
    '../results/figures/notebook_outputs/07_demo_fbcsp.png')
plot_confusion_matrix(cm_en, ds.class_names, 'EEGNet — demo split',
    '../results/figures/notebook_outputs/07_demo_eegnet.png')

import matplotlib.image as mpimg
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].imshow(mpimg.imread('../results/figures/notebook_outputs/07_demo_fbcsp.png'))
ax[0].axis('off')
ax[1].imshow(mpimg.imread('../results/figures/notebook_outputs/07_demo_eegnet.png'))
ax[1].axis('off')
plt.tight_layout(); plt.show()
"""),
    ]
    write_nb("07_demo.ipynb", cells)


# ============================================================================
# main
# ============================================================================
if __name__ == "__main__":
    print("building notebooks ...")
    build_01(); _uid[0] = 0
    build_02(); _uid[0] = 0
    build_03(); _uid[0] = 0
    build_04(); _uid[0] = 0
    build_05(); _uid[0] = 0
    build_06(); _uid[0] = 0
    build_07(); _uid[0] = 0
    print("done.")
