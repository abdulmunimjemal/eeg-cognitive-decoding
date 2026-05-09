# eeg-cognitive-decoding

> Comparing classical **FBCSP + LDA** against deep **EEGNet** for decoding
> cognitive states from scalp EEG, across two qualitatively different
> paradigms — **motor imagery** (4-class) and **mental arithmetic vs. rest**
> (2-class).

A small, modular Python pipeline + reproducible notebooks + a 12-slide
editorial deck + a 24-page teaching brief PDF. CPU-only, runs in seconds,
no proprietary datasets required.

---

## Headline results

| Dataset | Model | Accuracy | macro-F1 | Cohen's κ |
|---|---|---|---|---|
| Motor imagery (4-class)      | FBCSP + LDA | **99.4 ± 0.8 %** | 99.4 ± 0.8 | 0.99 ± 0.01 |
| Motor imagery (4-class)      | EEGNet      | 87.5 ± 13.0 %    | 87.6 ± 13.0 | 0.83 ± 0.17 |
| Mental arithmetic (2-class)  | FBCSP + LDA | **98.1 ± 1.2 %** | 98.1 ± 1.2 | 0.96 ± 0.02 |
| Mental arithmetic (2-class)  | EEGNet      | 79.7 ± 23.6 %    | 74.4 ± 30.1 | 0.59 ± 0.47 |

5-fold stratified cross-validation, leakage-free per-channel z-scoring.
The classical baseline outperforms the deep model on both tasks at this
data scale — consistent with the published BCI literature on small motor-
imagery datasets. EEGNet's higher across-fold variance is itself a
finding (see `REPORT.md` §4 and the brief).

---

## Quickstart

```bash
git clone https://github.com/abdulmunimjemal/eeg-cognitive-decoding
cd eeg-cognitive-decoding

# Install the package + dependencies
pip install -e .[dev]

# Reproduce all results, figures, and notebook outputs
make all

# Or run individual stages
make experiments   # run all experiments → results/<dataset>_<model>.json
make figures       # render figures → results/figures/
make notebooks     # execute notebooks end-to-end
make clean         # remove generated outputs
```

CPU is sufficient throughout — EEGNet has only ~2 400 parameters.

---

## What's inside

```
eeg-cognitive-decoding/
├── eeg_cognitive/               # the actual library
│   ├── data.py                  # synthetic + real-data loaders
│   ├── preprocess.py            # filter, CAR, leakage-free z-score
│   ├── evaluate.py              # K-fold CV harness with Cohen's κ
│   ├── viz.py                   # topomaps, confusion matrices
│   └── models/
│       ├── fbcsp.py             # filter-bank CSP + LDA
│       └── eegnet.py            # EEGNet (Lawhern 2018) in PyTorch
├── scripts/                     # CLI entry points
│   ├── run_experiment.py        # run a single (dataset, model) pair
│   ├── run_all_experiments.py   # run all four
│   ├── generate_figures.py      # all matplotlib figures
│   └── generate_styled_figures.py
├── notebooks/                   # 7 executed walkthrough notebooks
│   ├── 01_introduction.ipynb
│   ├── 02_synthetic_data.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_fbcsp.ipynb
│   ├── 05_eegnet.ipynb
│   ├── 06_results.ipynb
│   └── 07_demo.ipynb            # the live-prediction widget
├── slides/                      # 12-slide editorial deck (.pptx + .pdf)
├── brief/                       # 24-page teaching brief PDF
├── results/                     # JSONs, CSV, all 16 figures
├── REPORT.md                    # the long-form write-up
├── pyproject.toml
└── Makefile                     # one-command reproducibility
```

---

## The two paradigms

**Motor imagery** is the mental rehearsal of a movement without producing it.
The classic scalp signature is **mu (8–13 Hz) and beta (13–30 Hz) ERD over
contralateral central electrodes** — left-hand imagery suppresses C4
(right hemisphere), right-hand imagery suppresses C3, foot imagery suppresses
Cz, tongue imagery shows fronto-central beta ERS. This is the canonical
brain-computer-interface paradigm.

**Mental arithmetic** (silent serial subtraction) recruits a fronto-parietal
control network rather than the motor system. Signature: **frontal-midline
theta (4–7 Hz) increase at Fz/FCz** plus **parietal alpha suppression**.

We use a single ML pipeline on both — same preprocessing, same evaluation
harness, same two model families — to study how well the machinery
transfers across qualitatively different cognitive states.

## The two models

**FBCSP + LDA** (Ang et al. 2008). Filter-bank Common Spatial Patterns:
bandpass into 9 sub-bands, fit one-vs-rest CSPs per band (with shrinkage
regularization for numerical stability), take log-variance of the projected
signals, mutual-information feature selection, LDA. The classical BCI
baseline. Fast, interpretable, and competitive with deep models on small
datasets.

**EEGNet** (Lawhern et al. 2018). Compact convolutional network for EEG —
~2 400 parameters total, trains on CPU in seconds. First layer learns
temporal filters (analogue of a frequency bank), second layer is a depthwise
spatial convolution per temporal filter (analogue of CSP), separable conv
integrates them, linear head reads out. Implemented from scratch in PyTorch
with an sklearn-compatible wrapper.

## On the synthetic data

The default loader generates physiologically grounded synthetic EEG —
1/f pink noise plus class-specific narrowband oscillations modulated by
scalp gain maps that encode the textbook ERD/ERS pattern, smeared by a
Gaussian volume-conduction model. Class patterns are **known**, so a
correctly-functioning pipeline must recover them — which we verify by
inspecting CSP topomaps (left-hand imagery → C4 peak ✓, right-hand → C3 ✓,
feet → Cz ✓, tongue → FCz ✓).

A real-data loader (`load_physionet_eegmmi()`) is included for grading-time
or research-time reproduction on the actual PhysioNet EEG Motor Movement/
Imagery DB. The pipeline is identical for real data.

---

## Reading guide

If you want to learn the project, read in this order:

1. `notebooks/01_introduction.ipynb` — get oriented in 5 minutes
2. `brief/EEG_Project_Brief.pdf` — 24-page editorial walkthrough of the field
   and the project, end-to-end
3. `notebooks/03–05` — preprocessing, FBCSP, EEGNet walkthroughs with
   embedded figures and metrics
4. `notebooks/06_results.ipynb` — the comparison and the cross-subject finding
5. `slides/presentation.pdf` — the 12-slide talk version
6. `REPORT.md` — the dense write-up

---

## License

MIT.
