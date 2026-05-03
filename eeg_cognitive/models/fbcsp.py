"""
Filter-Bank Common Spatial Patterns (FBCSP) + LDA.

Reference: Ang et al. 2008, "Filter Bank Common Spatial Pattern (FBCSP) in
Brain-Computer Interface". This is the canonical *classical* baseline for
motor-imagery BCI and remains competitive with deep models on small datasets.

Pipeline:
    1. For each of K frequency sub-bands, bandpass-filter the epochs.
    2. Per band, fit Common Spatial Patterns: solve the generalized eigenvalue
       problem on per-class covariance matrices.
    3. Project epochs onto the top + bottom k CSP components per band, take
       log-variance → (n_trials, K * 2k) feature vector.
    4. Mutual-information feature selection → top M features.
    5. LDA (or linear SVM) classifier.

We implement multi-class CSP via the one-vs-rest scheme, which gives n_classes
sets of CSP filters per band. Plain CSP is the binary case.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Default 9-band filter bank (4 Hz wide, 4-40 Hz)
DEFAULT_BANDS: list[tuple[float, float]] = [
    (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
    (24, 28), (28, 32), (32, 36), (36, 40),
]


def _bandpass_bank(X: np.ndarray, sfreq: float, bands: list[tuple[float, float]],
                   order: int = 4) -> np.ndarray:
    """Apply a bank of bandpass filters to X (n_trials, n_ch, n_samples).
    Returns shape (n_bands, n_trials, n_ch, n_samples)."""
    nyq = 0.5 * sfreq
    out = np.empty((len(bands),) + X.shape, dtype=np.float32)
    for i, (lo, hi) in enumerate(bands):
        b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
        out[i] = filtfilt(b, a, X, axis=-1).astype(np.float32)
    return out


class _CSP:
    """Binary CSP. Fits W : (n_components, n_channels) such that projected
    log-variance maximally discriminates the two classes."""

    def __init__(self, n_components: int = 4, reg: float = 1e-3):
        self.n_components = n_components
        self.reg = reg  # Ledoit-Wolf-style shrinkage toward scaled identity
        self.W: np.ndarray | None = None  # (n_components, n_channels)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CSP":
        classes = np.unique(y)
        assert len(classes) == 2, "binary CSP requires exactly 2 classes"
        n_ch = X.shape[1]
        eye = np.eye(n_ch, dtype=np.float64)

        def cov(trials: np.ndarray) -> np.ndarray:
            covs = []
            for t in trials:
                c = t @ t.T
                c = c / (np.trace(c) + 1e-8)
                covs.append(c)
            C = np.mean(covs, axis=0).astype(np.float64)
            # Shrinkage regularization → guarantees positive-definiteness
            tr = np.trace(C) / n_ch
            C = (1 - self.reg) * C + self.reg * tr * eye
            return C

        c1 = cov(X[y == classes[0]])
        c2 = cov(X[y == classes[1]])
        # Generalized eigenvalue problem: c1 v = lambda (c1 + c2) v
        # eigh returns eigenvalues in ascending order
        evals, evecs = eigh(c1, c1 + c2)
        # Reorder so largest eigenvalues come first
        order = np.argsort(np.abs(evals - 0.5))[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        # Take the top n_components/2 from each end (largest and smallest evals,
        # i.e., maximally discriminative for class 0 vs. class 1)
        n = self.n_components
        idx = np.r_[np.arange(n // 2), np.arange(evecs.shape[1] - n // 2,
                                                  evecs.shape[1])]
        self.W = evecs[:, idx].T.astype(np.float32)  # (n_components, n_channels)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Project, take log-variance per component
        proj = self.W @ X  # (n_trials, n_components, n_samples) via broadcasting
        # We need to apply per-trial: einsum is cleanest
        proj = np.einsum("cd,nds->ncs", self.W, X)
        var = proj.var(axis=-1)
        var = var / (var.sum(axis=1, keepdims=True) + 1e-8)
        return np.log(var + 1e-8).astype(np.float32)


class FBCSPClassifier(BaseEstimator, ClassifierMixin):
    """Filter-bank CSP + MI selection + LDA.

    Multi-class via one-vs-rest CSP. This means for K classes and B bands, we
    fit K * B CSPs, each with `n_components` filters, giving K * B * n_components
    raw features per trial. Mutual-information selects the top `n_features`.
    """

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        n_features: int = 20,
        sfreq: float = 250.0,
        classifier: str = "lda",  # 'lda' or 'svm'
    ):
        self.bands = bands if bands is not None else DEFAULT_BANDS
        self.n_components = n_components
        self.n_features = n_features
        self.sfreq = sfreq
        self.classifier = classifier

    def _build_features(self, X: np.ndarray, fit: bool, y: np.ndarray | None = None
                        ) -> np.ndarray:
        Xb = _bandpass_bank(X, self.sfreq, self.bands)  # (B, N, C, T)
        feats_per_band = []
        if fit:
            self.csps_: list[list[_CSP]] = []
        for bi in range(len(self.bands)):
            Xfb = Xb[bi]
            if fit:
                csps_for_band = []
                for cls in self.classes_:
                    y_bin = (y == cls).astype(int)
                    csp = _CSP(n_components=self.n_components).fit(Xfb, y_bin)
                    csps_for_band.append(csp)
                self.csps_.append(csps_for_band)
            feats = []
            for csp in self.csps_[bi]:
                feats.append(csp.transform(Xfb))
            feats_per_band.append(np.concatenate(feats, axis=1))  # (N, K*ncomp)
        F = np.concatenate(feats_per_band, axis=1)  # (N, B*K*ncomp)
        return F

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPClassifier":
        self.classes_ = np.unique(y)
        F = self._build_features(X, fit=True, y=y)

        # Cap n_features at what's actually available
        k = min(self.n_features, F.shape[1])
        self.selector_ = SelectKBest(mutual_info_classif, k=k).fit(F, y)
        Fs = self.selector_.transform(F)

        if self.classifier == "lda":
            clf = LinearDiscriminantAnalysis()
        elif self.classifier == "svm":
            from sklearn.svm import LinearSVC
            clf = Pipeline([("scaler", StandardScaler()),
                            ("svm", LinearSVC(C=1.0, max_iter=5000))])
        else:
            raise ValueError(self.classifier)
        self.classifier_ = clf.fit(Fs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self._build_features(X, fit=False)
        Fs = self.selector_.transform(F)
        return self.classifier_.predict(Fs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = self._build_features(X, fit=False)
        Fs = self.selector_.transform(F)
        if hasattr(self.classifier_, "predict_proba"):
            return self.classifier_.predict_proba(Fs)
        # Fallback for SVM: decision_function → softmax
        df = self.classifier_.decision_function(Fs)
        if df.ndim == 1:
            df = np.stack([-df, df], axis=1)
        e = np.exp(df - df.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # Convenience: expose a CSP filter for visualization (band, class, comp)
    def get_csp_filter(self, band_idx: int, class_idx: int, comp_idx: int
                       ) -> np.ndarray:
        return self.csps_[band_idx][class_idx].W[comp_idx]


if __name__ == "__main__":
    # Smoke test on tiny synthetic data
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eeg_cognitive.data import make_motor_imagery_synthetic
    from eeg_cognitive.preprocess import preprocess_epochs

    ds = make_motor_imagery_synthetic(n_subjects=2, trials_per_class=20, seed=0)
    X = preprocess_epochs(ds.X, ds.sfreq)
    n = len(ds.y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    split = int(n * 0.8)
    tr, te = perm[:split], perm[split:]
    clf = FBCSPClassifier(sfreq=ds.sfreq, n_components=4, n_features=16)
    clf.fit(X[tr], ds.y[tr])
    yhat = clf.predict(X[te])
    acc = (yhat == ds.y[te]).mean()
    print(f"FBCSP smoke-test accuracy on motor imagery: {acc:.3f}")
