"""
Models — classical and deep classifiers for EEG, with sklearn-compatible
interfaces so they slot into the same cross-validation harness.
"""
from .fbcsp import FBCSPClassifier, DEFAULT_BANDS
from .eegnet import EEGNet, EEGNetClassifier

__all__ = ["FBCSPClassifier", "DEFAULT_BANDS", "EEGNet", "EEGNetClassifier"]
