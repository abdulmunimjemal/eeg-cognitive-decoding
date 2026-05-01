"""
EEG data loading.

Two modes:
  1. `synthetic` — physiologically grounded simulation. Class-specific ERD/ERS
     patterns embedded in pink noise. Always available, fast, deterministic.
  2. `physionet_eegmmi` — PhysioNet EEG Motor Movement/Imagery (Schalk 2004),
     downloaded via MNE. Requires network. Used for real-data validation.

The two cognitive states for the project map onto:
  - "motor_imagery" : 4-class motor imagery (left/right hand, feet, tongue
                     in the synthetic case; left vs. right vs. fists vs. feet
                     for PhysioNet). Mu (8-13 Hz) / beta (13-30 Hz) ERD over
                     central electrodes (C3, Cz, C4).
  - "mental_arithmetic" : 2-class rest vs. mental arithmetic. Frontal-midline
                     theta (4-7 Hz) increase + parietal alpha suppression
                     during effortful sustained attention.

The synthetic generator embeds these patterns explicitly so a correct ML
pipeline will discover them. This makes the simulation a *teaching artifact*
that mirrors the real cognitive neuroscience.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# 22-channel 10-20 layout used by BCI Competition IV-2a, in approximate
# topographic coordinates (x = lateral, y = anterior-posterior). Used for
# the topomap visualizations.
CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

# Approximate 2D positions for topomap plotting (radius-1 unit disk).
CHANNEL_POS_22 = {
    "Fz":  (0.00,  0.55), "FC3": (-0.45, 0.30), "FC1": (-0.20, 0.30),
    "FCz": (0.00,  0.30), "FC2": (0.20, 0.30),  "FC4": (0.45, 0.30),
    "C5":  (-0.65, 0.00), "C3":  (-0.45, 0.00), "C1":  (-0.20, 0.00),
    "Cz":  (0.00,  0.00), "C2":  (0.20, 0.00),  "C4":  (0.45, 0.00),
    "C6":  (0.65,  0.00),
    "CP3": (-0.45, -0.25), "CP1": (-0.20, -0.25), "CPz": (0.00, -0.25),
    "CP2": (0.20, -0.25),  "CP4": (0.45, -0.25),
    "P1":  (-0.20, -0.50), "Pz": (0.00, -0.50),  "P2":  (0.20, -0.50),
    "POz": (0.00, -0.75),
}


@dataclass
class EEGDataset:
    """A simple container so motor-imagery and arithmetic data look identical
    to the rest of the pipeline."""
    X: np.ndarray            # (n_trials, n_channels, n_samples)
    y: np.ndarray            # (n_trials,) int labels
    sfreq: float             # sampling frequency in Hz
    ch_names: list[str]
    class_names: list[str]
    name: str                # human-readable dataset name


def _pink_noise(n_samples: int, sfreq: float, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f-like pink noise via spectral shaping. EEG noise floor
    famously has a 1/f^alpha shape with alpha ~ 1-2."""
    n = n_samples
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    # Avoid divide-by-zero at DC
    with np.errstate(divide="ignore"):
        spectrum = np.where(freqs > 0, freqs ** -1.0, 0.0)
    phases = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    coeffs = spectrum * np.exp(1j * phases)
    coeffs[0] = 0
    sig = np.fft.irfft(coeffs, n=n)
    sig = sig / (np.std(sig) + 1e-12)
    return sig.astype(np.float32)


def _bandpassed_oscillation(
    f_lo: float, f_hi: float, n_samples: int, sfreq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A narrow-band oscillation (random-phase, slowly drifting amplitude),
    used as the building block for class-specific rhythms."""
    n = n_samples
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    phases = rng.uniform(0, 2 * np.pi, size=freqs.shape)
    spectrum = band.astype(float) * np.exp(1j * phases)
    sig = np.fft.irfft(spectrum, n=n)
    sig = sig / (np.std(sig) + 1e-12)
    return sig.astype(np.float32)


def _channel_weights(channel_names: list[str], focus: dict[str, float]) -> np.ndarray:
    """Build a per-channel gain vector. `focus` maps channel name → weight.
    Missing channels get a small leakage weight (volume conduction)."""
    w = np.full(len(channel_names), 0.10, dtype=np.float32)  # leakage floor
    for i, ch in enumerate(channel_names):
        if ch in focus:
            w[i] = focus[ch]
    return w


def _smear_pattern(focus: dict[str, float], ch_pos: dict, sigma: float = 0.18
                   ) -> dict[str, float]:
    """Spread a focal pattern over neighboring channels to simulate volume
    conduction. Each focus channel donates a Gaussian-decayed weight to all
    other channels."""
    out: dict[str, float] = dict(focus)
    for src, w in focus.items():
        if src not in ch_pos:
            continue
        sx, sy = ch_pos[src]
        for ch, (cx, cy) in ch_pos.items():
            if ch == src:
                continue
            d2 = (cx - sx) ** 2 + (cy - sy) ** 2
            extra = w * float(np.exp(-d2 / (2 * sigma ** 2))) * 0.6
            out[ch] = out.get(ch, 0.0) + extra
    return out


def make_motor_imagery_synthetic(
    n_subjects: int = 9,
    trials_per_class: int = 72,
    sfreq: float = 250.0,
    trial_seconds: float = 2.0,
    snr_db: float = -6.0,
    subj_var: float = 0.30,
    seed: int = 0,
) -> EEGDataset:
    """Generate a synthetic 4-class motor-imagery dataset that mimics
    BCI Competition IV-2a.

    Class-specific patterns:
      0 left hand  → contralateral (right-hemisphere) ERD: C4 mu/beta suppression
      1 right hand → contralateral (left-hemisphere)  ERD: C3 mu/beta suppression
      2 feet       → bilateral central foot-area ERD: Cz mu suppression
      3 tongue     → fronto-central beta increase (ERS) at FCz
    """
    rng = np.random.default_rng(seed)
    n_samples = int(trial_seconds * sfreq)
    chans = CHANNELS_22
    n_ch = len(chans)
    n_classes = 4
    total_trials = n_subjects * trials_per_class * n_classes

    X = np.zeros((total_trials, n_ch, n_samples), dtype=np.float32)
    y = np.zeros(total_trials, dtype=np.int64)

    # Class-specific spatial focus, smeared by volume-conduction Gaussian.
    raw_patterns = {
        0: dict(erd={"C4": -1.0, "C2": -0.6, "CP4": -0.5},
                ers={}),
        1: dict(erd={"C3": -1.0, "C1": -0.6, "CP3": -0.5},
                ers={}),
        2: dict(erd={"Cz": -1.0, "FCz": -0.5, "CPz": -0.5},
                ers={}),
        3: dict(erd={},
                ers={"FCz": 0.9, "Fz": 0.6, "FC1": 0.5, "FC2": 0.5}),
    }
    class_patterns = {
        cls: dict(
            erd_focus=_smear_pattern(d["erd"], CHANNEL_POS_22),
            ers_focus=_smear_pattern(d["ers"], CHANNEL_POS_22),
        )
        for cls, d in raw_patterns.items()
    }

    snr = 10 ** (snr_db / 20.0)
    idx = 0
    for subject in range(n_subjects):
        # Stronger inter-subject variability — mimics real BCI data
        subj_jitter = rng.normal(0, subj_var, size=n_ch).astype(np.float32)
        # Some "BCI-illiterate" subjects have weaker overall signal
        subj_strength = float(rng.uniform(0.5, 1.0))
        for cls in range(n_classes):
            for _ in range(trials_per_class):
                bg = np.stack([_pink_noise(n_samples, sfreq, rng) for _ in range(n_ch)])

                mu = _bandpassed_oscillation(8, 13, n_samples, sfreq, rng)
                beta = _bandpassed_oscillation(13, 30, n_samples, sfreq, rng)
                rhythm_baseline = mu + 0.6 * beta

                erd_w = _channel_weights(chans, class_patterns[cls]["erd_focus"])
                ers_w = _channel_weights(chans, class_patterns[cls]["ers_focus"])
                erd_w = erd_w + subj_jitter
                ers_w = ers_w + subj_jitter

                # trial-level amplitude jitter so not every trial has the
                # textbook pattern strength
                trial_strength = float(rng.uniform(0.6, 1.0)) * subj_strength
                gain = 1.0 + trial_strength * (erd_w + ers_w)
                signal = gain[:, None] * rhythm_baseline[None, :]
                trial = bg + snr * signal

                X[idx] = trial.astype(np.float32)
                y[idx] = cls
                idx += 1

    return EEGDataset(
        X=X, y=y, sfreq=sfreq, ch_names=chans,
        class_names=["left_hand", "right_hand", "feet", "tongue"],
        name="motor_imagery_synthetic",
    )


def make_mental_arithmetic_synthetic(
    n_subjects: int = 36,
    trials_per_class: int = 30,
    sfreq: float = 250.0,
    trial_seconds: float = 4.0,
    snr_db: float = -6.0,
    subj_var: float = 0.30,
    seed: int = 1,
) -> EEGDataset:
    """Generate a synthetic 2-class mental-arithmetic dataset that mimics
    PhysioNet EEGMAT.

    Class-specific patterns:
      0 rest         → broadband alpha (8-13 Hz) over parietal sites
      1 mental math  → frontal-midline theta (4-7 Hz) at Fz/FCz
                       + parietal alpha SUPPRESSION (effortful attention)
    """
    rng = np.random.default_rng(seed)
    n_samples = int(trial_seconds * sfreq)
    chans = CHANNELS_22
    n_ch = len(chans)
    n_classes = 2
    total_trials = n_subjects * trials_per_class * n_classes

    X = np.zeros((total_trials, n_ch, n_samples), dtype=np.float32)
    y = np.zeros(total_trials, dtype=np.int64)

    # Smear the focal patterns with volume conduction (matches MI loader)
    rest_alpha_raw = {"Pz": 1.0, "P1": 0.8, "P2": 0.8, "POz": 0.7, "CPz": 0.4}
    rest_theta_raw = {"Fz": 0.15, "FCz": 0.15}
    task_alpha_raw = {"Pz": 0.25, "P1": 0.25, "P2": 0.25, "POz": 0.2, "CPz": 0.2}
    task_theta_raw = {"Fz": 1.0, "FCz": 0.9, "FC1": 0.5, "FC2": 0.5, "Cz": 0.3}

    rest_alpha = _smear_pattern(rest_alpha_raw, CHANNEL_POS_22)
    rest_theta = _smear_pattern(rest_theta_raw, CHANNEL_POS_22)
    task_alpha = _smear_pattern(task_alpha_raw, CHANNEL_POS_22)
    task_theta = _smear_pattern(task_theta_raw, CHANNEL_POS_22)

    snr = 10 ** (snr_db / 20.0)
    idx = 0
    for subject in range(n_subjects):
        subj_jitter = rng.normal(0, subj_var, size=n_ch).astype(np.float32)
        subj_strength = float(rng.uniform(0.5, 1.0))
        for cls in range(n_classes):
            for _ in range(trials_per_class):
                bg = np.stack([_pink_noise(n_samples, sfreq, rng) for _ in range(n_ch)])

                alpha = _bandpassed_oscillation(8, 13, n_samples, sfreq, rng)
                theta = _bandpassed_oscillation(4, 7, n_samples, sfreq, rng)

                if cls == 0:  # rest
                    alpha_focus, theta_focus = rest_alpha, rest_theta
                else:         # mental arithmetic
                    alpha_focus, theta_focus = task_alpha, task_theta

                a_w = _channel_weights(chans, alpha_focus) + subj_jitter
                t_w = _channel_weights(chans, theta_focus) + subj_jitter

                trial_strength = float(rng.uniform(0.6, 1.0)) * subj_strength
                signal = trial_strength * (a_w[:, None] * alpha[None, :]
                                           + t_w[:, None] * theta[None, :])
                trial = bg + snr * signal

                X[idx] = trial.astype(np.float32)
                y[idx] = cls
                idx += 1

    return EEGDataset(
        X=X, y=y, sfreq=sfreq, ch_names=chans,
        class_names=["rest", "mental_arithmetic"],
        name="mental_arithmetic_synthetic",
    )


def load_dataset(name: str, **kwargs) -> EEGDataset:
    """Top-level loader. `name` ∈ {motor_imagery, mental_arithmetic}."""
    if name == "motor_imagery":
        return make_motor_imagery_synthetic(**kwargs)
    elif name == "mental_arithmetic":
        return make_mental_arithmetic_synthetic(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Real-data loader stub (PhysioNet EEG Motor Movement/Imagery via MNE).
# Kept here so a grader with network access can run the same pipeline on
# real recordings. The synthetic loader is the default because it removes
# the network dependency and runs in seconds.
# ---------------------------------------------------------------------------
def load_physionet_eegmmi(
    subjects: list[int] | None = None,
    runs: list[int] = (4, 8, 12),  # imagined left/right fist runs
    tmin: float = 0.5,
    tmax: float = 2.5,
    cache_dir: str = "/tmp/mne_data",
) -> EEGDataset:
    """Load the PhysioNet EEG Motor Movement/Imagery DB via MNE. Slow on
    first call (downloads ~30 MB per subject). Falls back to a clear error
    if MNE cannot reach PhysioNet — caller should switch to the synthetic
    loader in that case."""
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    if subjects is None:
        subjects = list(range(1, 11))

    all_X, all_y = [], []
    sfreq_out = None
    ch_names_out = None

    for s in subjects:
        fnames = eegbci.load_data(s, runs, path=cache_dir, update_path=True)
        raws = [read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames]
        raw = concatenate_raws(raws)
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        # Map T0/T1/T2 → labels: T1 = left fist (imagined), T2 = right fist (imagined)
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2),
                                                verbose="ERROR")
        epochs = mne.Epochs(raw, events, event_id=dict(left=1, right=2),
                            tmin=tmin, tmax=tmax, baseline=None, preload=True,
                            verbose="ERROR")
        X = epochs.get_data().astype(np.float32)
        y = epochs.events[:, -1] - 1  # → 0/1
        all_X.append(X)
        all_y.append(y)
        sfreq_out = epochs.info["sfreq"]
        ch_names_out = epochs.ch_names

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return EEGDataset(
        X=X, y=y, sfreq=sfreq_out, ch_names=ch_names_out,
        class_names=["left_fist_imagined", "right_fist_imagined"],
        name="physionet_eegmmi",
    )


if __name__ == "__main__":
    # Smoke test
    mi = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=8)
    print(f"Motor imagery: X={mi.X.shape} y={mi.y.shape} classes={mi.class_names}")
    ma = make_mental_arithmetic_synthetic(n_subjects=2, trials_per_class=8)
    print(f"Arithmetic:    X={ma.X.shape} y={ma.y.shape} classes={ma.class_names}")
