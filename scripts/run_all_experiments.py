"""
Main experiment runner.

Runs FBCSP+LDA and EEGNet on both cognitive states (motor imagery, mental
arithmetic) with stratified 5-fold CV. Dumps results to results/results.csv
and per-experiment JSON files for downstream plotting.

Usage:
    python src/run_experiments.py [--quick]

`--quick` shrinks the data and EEGNet epochs for fast iteration; the default
runs the full settings reported in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import csv
from dataclasses import asdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eeg_cognitive.data import make_motor_imagery_synthetic, make_mental_arithmetic_synthetic
from eeg_cognitive.preprocess import preprocess_epochs
from eeg_cognitive.evaluate import cross_validate
from eeg_cognitive.models import FBCSPClassifier
from eeg_cognitive.models import EEGNetClassifier


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _cv_to_dict(res, classes):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller data + fewer EEGNet epochs (smoke run)")
    parser.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--snr-db", type=float, default=-14.0,
                        help="SNR (dB) for synthetic data (default: -14)")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if args.quick:
        n_sub_mi, tpc_mi = 3, 24
        n_sub_ma, tpc_ma = 8, 20
        eeg_epochs = 60
    else:
        n_sub_mi, tpc_mi = 9, 48
        n_sub_ma, tpc_ma = 24, 24
        eeg_epochs = 120

    print(f"=== EEG project full run (SNR={args.snr_db} dB) ===")
    print(f"Motor imagery:    {n_sub_mi} subjects × {tpc_mi} trials/class × 4 classes")
    print(f"Mental arithmetic:{n_sub_ma} subjects × {tpc_ma} trials/class × 2 classes")

    # --------- Load both cognitive states ---------
    t0 = time.time()
    mi = make_motor_imagery_synthetic(n_subjects=n_sub_mi, trials_per_class=tpc_mi,
                                      snr_db=args.snr_db, seed=0)
    ma = make_mental_arithmetic_synthetic(n_subjects=n_sub_ma, trials_per_class=tpc_ma,
                                          snr_db=args.snr_db, seed=1)
    print(f"  generated MI {mi.X.shape}, MA {ma.X.shape}  ({time.time()-t0:.1f}s)")

    Xmi = preprocess_epochs(mi.X, mi.sfreq)
    Xma = preprocess_epochs(ma.X, ma.sfreq)
    print(f"  preprocessed.")

    rows = []  # for the headline CSV

    for ds_name, ds, X in [
        ("motor_imagery", mi, Xmi),
        ("mental_arithmetic", ma, Xma),
    ]:
        print(f"\n--- {ds_name} ---")

        # ------ FBCSP ------
        print("  FBCSP+LDA: running 5-fold CV...")
        t0 = time.time()
        res_fb = cross_validate(
            X, ds.y, ds.sfreq,
            clf_factory=lambda: FBCSPClassifier(
                sfreq=ds.sfreq, n_components=4, n_features=20),
            n_splits=5, seed=0, standardize=True,
        )
        print(f"    summary: {res_fb.summary()}  ({time.time()-t0:.1f}s)")
        _save_json(os.path.join(results_dir, f"{ds_name}_fbcsp.json"),
                   _cv_to_dict(res_fb, ds.class_names))
        rows.append(dict(dataset=ds_name, model="FBCSP+LDA",
                         **{k: v for k, v in res_fb.summary().items() if k != "n_folds"}))

        # ------ EEGNet ------
        print("  EEGNet: running 5-fold CV...")
        t0 = time.time()
        res_en = cross_validate(
            X, ds.y, ds.sfreq,
            clf_factory=lambda: EEGNetClassifier(
                epochs=eeg_epochs, batch_size=64, lr=1e-3,
                weight_decay=5e-4, patience=20, val_frac=0.2, seed=0),
            n_splits=5, seed=0, standardize=True,
        )
        print(f"    summary: {res_en.summary()}  ({time.time()-t0:.1f}s)")
        _save_json(os.path.join(results_dir, f"{ds_name}_eegnet.json"),
                   _cv_to_dict(res_en, ds.class_names))
        rows.append(dict(dataset=ds_name, model="EEGNet",
                         **{k: v for k, v in res_en.summary().items() if k != "n_folds"}))

    # --------- Headline CSV ---------
    csv_path = os.path.join(results_dir, "results.csv")
    fieldnames = ["dataset", "model", "acc_mean", "acc_std",
                  "f1_mean", "f1_std", "kappa_mean", "kappa_std", "mean_fit_sec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote headline results to {csv_path}")

    # Pretty print final summary
    print("\n========== HEADLINE RESULTS ==========")
    print(f"{'dataset':22s} {'model':12s} {'acc':>14s} {'F1':>14s} {'κ':>14s}")
    for r in rows:
        acc = f"{r['acc_mean']*100:5.1f} ± {r['acc_std']*100:4.1f}"
        f1  = f"{r['f1_mean']*100:5.1f} ± {r['f1_std']*100:4.1f}"
        kp  = f"{r['kappa_mean']:+.3f} ± {r['kappa_std']:.3f}"
        print(f"{r['dataset']:22s} {r['model']:12s} {acc:>14s} {f1:>14s} {kp:>14s}")


if __name__ == "__main__":
    main()
