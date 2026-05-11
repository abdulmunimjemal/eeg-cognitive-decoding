"""
Run a single (dataset, model) experiment and save results JSON.

Usage:
    python scripts/run_experiment.py motor_imagery fbcsp
    python scripts/run_experiment.py motor_imagery eegnet
    python scripts/run_experiment.py mental_arithmetic fbcsp
    python scripts/run_experiment.py mental_arithmetic eegnet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eeg_cognitive.data import make_motor_imagery_synthetic, make_mental_arithmetic_synthetic
from eeg_cognitive.preprocess import preprocess_epochs
from eeg_cognitive.evaluate import cross_validate
from eeg_cognitive.models import FBCSPClassifier
from eeg_cognitive.models import EEGNetClassifier


RESULTS_DIR = os.path.abspath(os.path.join(ROOT, "results"))
DEFAULT_SNR_DB = -12.0


def _make_eegnet(sfreq: float) -> EEGNetClassifier:
    clf = EEGNetClassifier(
        epochs=30, batch_size=64, lr=1.5e-3, weight_decay=1e-4,
        patience=12, val_frac=0.2, seed=0,
    )
    # Use a smaller kernel (256 ms instead of 500 ms) to keep training time
    # in the time budget; still wide enough to capture mu/beta rhythms.
    clf.sfreq = sfreq
    clf.eegnet_kernel_length = max(16, int(sfreq // 4))
    return clf


def make_dataset(name: str, snr_db: float):
    if name == "motor_imagery":
        return make_motor_imagery_synthetic(
            n_subjects=4, trials_per_class=20, snr_db=snr_db, seed=0,
        )
    if name == "mental_arithmetic":
        return make_mental_arithmetic_synthetic(
            n_subjects=8, trials_per_class=20, trial_seconds=2.0,
            snr_db=snr_db, seed=1,
        )
    raise ValueError(name)


def model_factory(name: str, sfreq: float):
    if name == "fbcsp":
        return lambda: FBCSPClassifier(sfreq=sfreq, n_components=4, n_features=20)
    if name == "eegnet":
        return lambda: _make_eegnet(sfreq)
    raise ValueError(name)


def cv_to_dict(res, classes):
    return {
        "summary": res.summary(),
        "per_fold_accuracy": res.accuracy,
        "per_fold_macro_f1": res.macro_f1,
        "per_fold_kappa": res.kappa,
        "confusion": res.confusion.tolist() if res.confusion is not None else None,
        "classes": list(map(str, classes)),
        "y_true": res.y_true_all.tolist() if res.y_true_all is not None else None,
        "y_pred": res.y_pred_all.tolist() if res.y_pred_all is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run one synthetic EEG decoding experiment and save its JSON summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("dataset", choices=["motor_imagery", "mental_arithmetic"])
    ap.add_argument("model", choices=["fbcsp", "eegnet"])
    ap.add_argument(
        "--snr-db",
        type=float,
        default=DEFAULT_SNR_DB,
        help="SNR (dB) used when generating the synthetic dataset.",
    )
    ap.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory where the experiment JSON will be written.",
    )
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    print(f"--- {args.dataset} × {args.model} (SNR={args.snr_db} dB) ---")
    ds = make_dataset(args.dataset, args.snr_db)
    X = preprocess_epochs(ds.X, ds.sfreq)
    print(f"  X={X.shape}  classes={ds.class_names}")

    t0 = time.time()
    res = cross_validate(
        X, ds.y, ds.sfreq,
        clf_factory=model_factory(args.model, ds.sfreq),
        n_splits=5, seed=0, standardize=True,
    )
    print(f"  summary: {res.summary()}  ({time.time()-t0:.1f}s)")
    print(f"  confusion:\n{np.array(res.confusion)}")

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{args.dataset}_{args.model}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cv_to_dict(res, ds.class_names), f, indent=2)
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
