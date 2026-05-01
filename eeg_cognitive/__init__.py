"""
eeg_cognitive — a small, modular pipeline for decoding cognitive states
from scalp EEG.

Public API
----------
>>> from eeg_cognitive import (
...     load_dataset,
...     preprocess_epochs, ChannelStandardizer,
...     cross_validate,
... )
>>> from eeg_cognitive.models import FBCSPClassifier, EEGNetClassifier

The package has two concrete models (filter-bank CSP + LDA classical
baseline; compact EEGNet in PyTorch) wired into a shared cross-validation
harness, plus a synthetic-EEG generator with physiologically grounded
class-specific patterns.
"""

from .data import (
    EEGDataset,
    CHANNELS_22, CHANNEL_POS_22,
    load_dataset,
    make_motor_imagery_synthetic,
    make_mental_arithmetic_synthetic,
    load_physionet_eegmmi,
)
from .preprocess import (
    bandpass, common_average_reference, ChannelStandardizer, preprocess_epochs,
)
from .evaluate import cross_validate, CVResult

__version__ = "0.1.0"
__all__ = [
    "EEGDataset", "CHANNELS_22", "CHANNEL_POS_22",
    "load_dataset",
    "make_motor_imagery_synthetic",
    "make_mental_arithmetic_synthetic",
    "load_physionet_eegmmi",
    "bandpass", "common_average_reference",
    "ChannelStandardizer", "preprocess_epochs",
    "cross_validate", "CVResult",
]
