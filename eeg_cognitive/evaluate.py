"""
Cross-validated evaluation harness.

Runs stratified K-fold CV with a given classifier factory on a given EEG
dataset. Returns per-fold accuracy, macro-F1, Cohen's κ, and the pooled
confusion matrix.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from typing import Callable, Any

from .preprocess import preprocess_epochs, ChannelStandardizer


@dataclass
class CVResult:
    accuracy: list[float] = field(default_factory=list)
    macro_f1: list[float] = field(default_factory=list)
    kappa: list[float] = field(default_factory=list)
    confusion: np.ndarray | None = None      # pooled across folds
    y_true_all: np.ndarray | None = None     # for downstream confusion matrix etc.
    y_pred_all: np.ndarray | None = None
    proba_all: np.ndarray | None = None
    fit_seconds: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        return dict(
            acc_mean=float(np.mean(self.accuracy)),
            acc_std=float(np.std(self.accuracy)),
            f1_mean=float(np.mean(self.macro_f1)),
            f1_std=float(np.std(self.macro_f1)),
            kappa_mean=float(np.mean(self.kappa)),
            kappa_std=float(np.std(self.kappa)),
            n_folds=len(self.accuracy),
            mean_fit_sec=float(np.mean(self.fit_seconds)),
        )


def cross_validate(
    X: np.ndarray, y: np.ndarray, sfreq: float,
    clf_factory: Callable[[], Any],
    n_splits: int = 5,
    seed: int = 0,
    standardize: bool = True,
) -> CVResult:
    """Run K-fold stratified CV.

    `clf_factory` is a zero-arg callable returning a fresh, unfitted classifier.
    `standardize=True` z-scores per-channel using train-fold statistics only,
    which is the correct way to avoid leakage in EEG pipelines.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    res = CVResult()
    y_true_all, y_pred_all, proba_all = [], [], []

    classes_seen = np.unique(y)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        if standardize:
            sc = ChannelStandardizer().fit(Xtr)
            Xtr = sc.transform(Xtr)
            Xte = sc.transform(Xte)

        t0 = time.time()
        clf = clf_factory()
        clf.fit(Xtr, ytr)
        fit_sec = time.time() - t0

        ypred = clf.predict(Xte)
        try:
            yprob = clf.predict_proba(Xte)
        except Exception:
            yprob = None

        res.accuracy.append(float(accuracy_score(yte, ypred)))
        res.macro_f1.append(float(f1_score(yte, ypred, average="macro")))
        res.kappa.append(float(cohen_kappa_score(yte, ypred)))
        res.fit_seconds.append(fit_sec)
        y_true_all.append(yte); y_pred_all.append(ypred)
        if yprob is not None:
            proba_all.append(yprob)

    res.y_true_all = np.concatenate(y_true_all)
    res.y_pred_all = np.concatenate(y_pred_all)
    res.confusion = confusion_matrix(res.y_true_all, res.y_pred_all,
                                     labels=classes_seen)
    if proba_all:
        res.proba_all = np.concatenate(proba_all, axis=0)
    return res


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eeg_cognitive.data import make_mental_arithmetic_synthetic
    from eeg_cognitive.models import FBCSPClassifier

    ds = make_mental_arithmetic_synthetic(n_subjects=4, trials_per_class=20)
    X = preprocess_epochs(ds.X, ds.sfreq)
    res = cross_validate(X, ds.y, ds.sfreq,
                         clf_factory=lambda: FBCSPClassifier(sfreq=ds.sfreq))
    print("CV summary:", res.summary())
    print("Confusion:\n", res.confusion)
