"""
Generate every figure for the report and slides from the saved results JSONs
plus a freshly trained FBCSP/EEGNet for visualization purposes.

Outputs to results/figures/:
    confusion_<dataset>_<model>.png
    accuracy_bar.png
    csp_topomap_motor_imagery_<class>.png
    eegnet_temporal_filters_motor_imagery.png
    erd_spectrogram_motor_imagery_C3_C4.png
    erd_spectrogram_mental_arithmetic_Fz_Pz.png
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eeg_cognitive.data import (
    make_motor_imagery_synthetic, make_mental_arithmetic_synthetic, CHANNELS_22,
)
from eeg_cognitive.preprocess import preprocess_epochs, ChannelStandardizer
from eeg_cognitive.models import FBCSPClassifier
from eeg_cognitive.models import EEGNetClassifier
from eeg_cognitive.viz import (
    plot_confusion_matrix, plot_topomap, plot_per_fold_accuracy,
    plot_eegnet_temporal_filters, plot_erd_time_frequency,
)


RESULTS = os.path.abspath(os.path.join(ROOT, "results"))
FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)


def _load_results():
    out = {}
    for ds in ["motor_imagery", "mental_arithmetic"]:
        for m in ["fbcsp", "eegnet"]:
            with open(os.path.join(RESULTS, f"{ds}_{m}.json")) as f:
                out[(ds, m)] = json.load(f)
    return out


def main():
    results = _load_results()

    # ------ 1. Per-(dataset, model) confusion matrices ------
    for (ds, m), d in results.items():
        cm = np.array(d["confusion"])
        title = f"{ds}  ·  {'FBCSP+LDA' if m=='fbcsp' else 'EEGNet'}\n" \
                f"acc={d['summary']['acc_mean']*100:.1f}±{d['summary']['acc_std']*100:.1f}%, " \
                f"κ={d['summary']['kappa_mean']:.2f}"
        plot_confusion_matrix(cm, d["classes"], title,
                              os.path.join(FIG, f"confusion_{ds}_{m}.png"))
    print("✔ confusion matrices")

    # ------ 2. Per-fold accuracy bar chart ------
    plot_per_fold_accuracy(results, os.path.join(FIG, "accuracy_bar.png"))
    print("✔ accuracy bar chart")

    # ------ 3. CSP topomaps for motor imagery ------
    print("training a fresh FBCSP on full motor-imagery data for topomaps...")
    mi = make_motor_imagery_synthetic(n_subjects=4, trials_per_class=20,
                                      snr_db=-12, seed=0)
    Xmi = preprocess_epochs(mi.X, mi.sfreq)
    sc = ChannelStandardizer().fit(Xmi)
    Xmi_z = sc.transform(Xmi)
    fb = FBCSPClassifier(sfreq=mi.sfreq, n_components=4, n_features=20).fit(
        Xmi_z, mi.y)
    # For each class, plot the top CSP filter from the mu band (band index 1: 8-12 Hz)
    mu_band_idx = 1  # 8-12 Hz
    for cls_idx, cls_name in enumerate(mi.class_names):
        w = fb.get_csp_filter(band_idx=mu_band_idx, class_idx=cls_idx, comp_idx=0)
        plot_topomap(
            weights=w, ch_names=CHANNELS_22,
            title=f"CSP top filter — {cls_name}\n(mu band 8-12 Hz, one-vs-rest)",
            out_path=os.path.join(FIG, f"csp_topomap_motor_imagery_{cls_name}.png"),
        )
    print("✔ CSP topomaps")

    # ------ 4. EEGNet first-layer temporal filters ------
    print("training a fresh EEGNet for filter visualization...")
    en = EEGNetClassifier(epochs=30, batch_size=64, lr=1.5e-3, weight_decay=1e-4,
                          patience=12, val_frac=0.2, seed=0)
    en.sfreq = mi.sfreq
    en.eegnet_kernel_length = max(16, int(mi.sfreq // 4))
    en.fit(Xmi_z, mi.y)
    plot_eegnet_temporal_filters(en.model_, mi.sfreq,
                                 os.path.join(FIG, "eegnet_temporal_filters_motor_imagery.png"))
    print("✔ EEGNet temporal filters")

    # ------ 5. Per-class spectrograms (the "raw cog-sci" view) ------
    plot_erd_time_frequency(
        Xmi_z, mi.y, CHANNELS_22, mi.sfreq, "C3", mi.class_names,
        os.path.join(FIG, "erd_spectrogram_motor_imagery_C3.png"))
    plot_erd_time_frequency(
        Xmi_z, mi.y, CHANNELS_22, mi.sfreq, "C4", mi.class_names,
        os.path.join(FIG, "erd_spectrogram_motor_imagery_C4.png"))
    print("✔ MI spectrograms")

    print("loading mental arithmetic data for spectrograms...")
    ma = make_mental_arithmetic_synthetic(n_subjects=8, trials_per_class=20,
                                          trial_seconds=2.0, snr_db=-12, seed=1)
    Xma = preprocess_epochs(ma.X, ma.sfreq)
    plot_erd_time_frequency(
        Xma, ma.y, CHANNELS_22, ma.sfreq, "Fz", ma.class_names,
        os.path.join(FIG, "erd_spectrogram_mental_arithmetic_Fz.png"))
    plot_erd_time_frequency(
        Xma, ma.y, CHANNELS_22, ma.sfreq, "Pz", ma.class_names,
        os.path.join(FIG, "erd_spectrogram_mental_arithmetic_Pz.png"))
    print("✔ MA spectrograms")

    print(f"\nAll figures written to {FIG}")
    for f in sorted(os.listdir(FIG)):
        sz = os.path.getsize(os.path.join(FIG, f))
        print(f"  {f}  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
