"""
Generate clean visualisation artefacts into artifacts/plots/.
Each output file contains exactly ONE self-contained plot.

Produces (10 images total):
  1.  Class distribution bar chart — all splits
  2.  Feature Inter-Correlation Matrix — train set
  3.  Confusion matrix — validation set, optimised threshold
  4.  Confusion matrix — test set, default threshold 0.50
  5.  Confusion matrix — test set, optimised threshold (Youden's J)
  6.  Prediction score distribution — test set
  7.  Feature distribution: MFCC          (Normal vs Depressed, per coefficient)
  8.  Feature distribution: Delta-MFCC    (Normal vs Depressed, per coefficient)
  9.  Feature distribution: Chroma        (Normal vs Depressed, per bin)
  10. Feature distribution: Spectral Contrast (Normal vs Depressed, per band)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PLOTS_DIR    = os.path.join(PROJECT_ROOT, "artifacts", "plots")
DATA_V2      = os.path.join(PROJECT_ROOT, "data", "processed", "combined_v2")
MODEL_V2_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models", "multi_feature_v2")
os.makedirs(PLOTS_DIR, exist_ok=True)

NORMAL_CLR    = "#4C9BE8"
DEPRESSED_CLR = "#E84C4C"

sns.set_theme(style="whitegrid", font_scale=1.1)

FEATURE_SLICES = {
    "MFCC"             : (0,  13),
    "Delta-MFCC"       : (13, 26),
    "Chroma"           : (26, 38),
    "Spectral Contrast": (38, 45),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_split(name: str):
    path = os.path.join(DATA_V2, f"{name}_features.npz")
    d    = np.load(path)
    return d["X"], d["y"]   # X: (N, 46, 313)  y: (N,)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Class distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_class_distribution(datasets: dict):
    print("  [1] Class distribution...")

    labels_map  = {"train": "Train", "val": "Validation", "test": "Test", "ravdess": "RAVDESS"}
    split_names = list(labels_map.values())
    dep_counts  = [int(datasets[k][1].sum())         for k in labels_map]
    norm_counts = [int((datasets[k][1] == 0).sum())  for k in labels_map]

    x     = np.arange(len(split_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_n = ax.bar(x - width / 2, norm_counts, width, label="Normal",    color=NORMAL_CLR,    alpha=0.9)
    bars_d = ax.bar(x + width / 2, dep_counts,  width, label="Depressed", color=DEPRESSED_CLR, alpha=0.9)

    for bar in list(bars_n) + list(bars_d):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(split_names, fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_title("Class Distribution Across Data Splits", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(norm_counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "1_class_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Feature correlation matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_correlation_matrix(X):
    print("  [2] Feature correlation matrix...")

    all_slices = {
        "MFCC"             : (0,  13),
        "Delta-MFCC"       : (13, 26),
        "Chroma"           : (26, 38),
        "Spectral Contrast": (38, 45),
        "ZCR"              : (45, 46),
    }

    labels = []
    cols   = []
    for fname, (s, e) in all_slices.items():
        chunk = X[:, s:e, :].mean(axis=2)   # (N, bins)
        short = {"Spectral Contrast": "SC", "Delta-MFCC": "dMFCC"}.get(fname, fname)
        for b in range(e - s):
            labels.append(f"{short}_c{b}")
            cols.append(chunk[:, b])

    mat  = np.stack(cols, axis=1)
    corr = np.corrcoef(mat.T)

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor="#ddd",
        xticklabels=labels, yticklabels=labels,
        annot=False, cbar_kws={"label": "Pearson r", "shrink": 0.75},
    )
    ax.set_title("Feature Inter-Correlation Matrix\n(temporal-mean per sample — train set)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    group_sizes = [e - s for _, (s, e) in all_slices.items()]
    for b in np.cumsum(group_sizes)[:-1]:
        ax.axvline(b, color="black", lw=1.5)
        ax.axhline(b, color="black", lw=1.5)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "2_feature_correlation_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3-5. Confusion matrices — one image per scenario
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_single_cm(y_true, y_pred_prob, threshold, title, filepath):
    y_hat  = (y_pred_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true.astype(int), y_hat)
    total  = cm.sum()
    acc    = (y_hat == y_true.astype(int)).mean()

    group_names  = ["TN", "FP", "FN", "TP"]
    group_counts = [f"{v:,}" for v in cm.flatten()]
    group_pcts   = [f"{v / total:.1%}" for v in cm.flatten()]
    labels = np.array([f"{n}\n{c}\n{p}" for n, c, p in
                        zip(group_names, group_counts, group_pcts)]).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", ax=ax,
                linewidths=2, linecolor="white",
                xticklabels=["Normal", "Depressed"],
                yticklabels=["Normal", "Depressed"],
                cbar=False, annot_kws={"size": 15, "weight": "bold"})
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"{title}\nThreshold = {threshold:.4f}  |  Accuracy = {acc:.1%}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"    saved → {filepath}")


def plot_confusion_matrices(preds: dict, datasets: dict, opt_thr: float):
    print("  [3] Confusion matrix — validation (optimal threshold)...")
    _save_single_cm(
        datasets["val"][1], preds["val"], opt_thr,
        "Confusion Matrix — Validation Set",
        os.path.join(PLOTS_DIR, "3_confusion_matrix_val_optimal.png"),
    )

    print("  [4] Confusion matrix — test set (default threshold 0.50)...")
    _save_single_cm(
        datasets["test"][1], preds["test"], 0.5,
        "Confusion Matrix — Test Set",
        os.path.join(PLOTS_DIR, "4_confusion_matrix_test_default.png"),
    )

    print("  [5] Confusion matrix — test set (optimal threshold)...")
    _save_single_cm(
        datasets["test"][1], preds["test"], opt_thr,
        "Confusion Matrix — Test Set",
        os.path.join(PLOTS_DIR, "5_confusion_matrix_test_optimal.png"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Score distribution — test set only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_score_distribution(preds: dict, datasets: dict, opt_thr: float):
    print("  [6] Score distribution — test set...")

    _, y = datasets["test"]
    p    = preds["test"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(p[y == 0], bins=40, alpha=0.65, color=NORMAL_CLR,    label="Normal")
    ax.hist(p[y == 1], bins=40, alpha=0.65, color=DEPRESSED_CLR, label="Depressed")
    ax.axvline(0.5,     color="gray",  linestyle="--", lw=2,   label="Threshold = 0.50")
    ax.axvline(opt_thr, color="black", linestyle=":",  lw=2.5, label=f"Optimal threshold = {opt_thr:.3f}")
    ax.set_xlabel("Predicted probability of depression", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Score Distribution — Test Set (EATD + DS1)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, "6_score_distribution_test.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7-10. Feature distributions — one image per feature group
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_feature_distributions(X_train, y_train):
    print("  [7-10] Feature distributions per group...")

    dep_mask  = y_train == 1
    norm_mask = y_train == 0

    for file_idx, (fname, (s, e)) in enumerate(FEATURE_SLICES.items(), start=7):
        n_coeff   = e - s
        feat_mean = X_train[:, s:e, :].mean(axis=2)   # (N, bins)

        dep_data  = [feat_mean[dep_mask,  i] for i in range(n_coeff)]
        norm_data = [feat_mean[norm_mask, i] for i in range(n_coeff)]

        fig_w = max(9, n_coeff * 0.85)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        pos_norm = np.arange(n_coeff) * 2.5
        pos_dep  = pos_norm + 1.0

        bp_n = ax.boxplot(norm_data, positions=pos_norm, widths=0.8, patch_artist=True,
                          showfliers=False,
                          medianprops=dict(color="white", linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
        bp_d = ax.boxplot(dep_data,  positions=pos_dep,  widths=0.8, patch_artist=True,
                          showfliers=False,
                          medianprops=dict(color="white", linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))

        for box in bp_n["boxes"]:
            box.set_facecolor(NORMAL_CLR)
            box.set_alpha(0.85)
        for box in bp_d["boxes"]:
            box.set_facecolor(DEPRESSED_CLR)
            box.set_alpha(0.85)

        ax.set_xticks(pos_norm + 0.5)
        ax.set_xticklabels([f"c{i}" for i in range(n_coeff)], fontsize=10)
        ax.set_xlabel("Coefficient index", fontsize=11)
        ax.set_ylabel("Temporal mean (normalised)", fontsize=11)
        ax.set_title(f"Feature Distribution: {fname} — Normal vs Depressed",
                     fontsize=13, fontweight="bold")
        ax.legend(
            handles=[
                mpatches.Patch(facecolor=NORMAL_CLR,    alpha=0.85, label="Normal"),
                mpatches.Patch(facecolor=DEPRESSED_CLR, alpha=0.85, label="Depressed"),
            ],
            fontsize=11,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        safe = fname.lower().replace(" ", "_").replace("-", "_")
        out  = os.path.join(PLOTS_DIR, f"{file_idx}_feature_dist_{safe}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("  Generating visualisations → artifacts/plots/")
    print("=" * 60)

    print("\nLoading combined_v2 data splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
    X_rav,   y_rav   = load_split("ravdess")

    datasets = {
        "train"  : (X_train, y_train),
        "val"    : (X_val,   y_val),
        "test"   : (X_test,  y_test),
        "ravdess": (X_rav,   y_rav),
    }

    print("\nSplit sizes:")
    for k, (X, y) in datasets.items():
        print(f"  {k:8s}: {len(y):5d}  (dep={int(y.sum()):4d}, norm={int((y==0).sum()):5d})")

    # Load model once and predict for all splits
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf   # type: ignore
    model = tf.keras.models.load_model(
        os.path.join(MODEL_V2_DIR, "best.keras"), compile=False,
    )

    def predict(X):
        inp = X[..., np.newaxis].astype(np.float32)
        return model.predict(inp, batch_size=64, verbose=0).squeeze()

    preds = {name: predict(X) for name, (X, y) in datasets.items()}

    with open(os.path.join(MODEL_V2_DIR, "training_summary.json")) as f:
        opt_thr = json.load(f)["optimal_threshold"]

    print(f"\nOptimal threshold: {opt_thr:.4f}\n")

    # Generate all plots
    plot_class_distribution(datasets)
    plot_correlation_matrix(X_train)
    plot_confusion_matrices(preds, datasets, opt_thr)
    plot_score_distribution(preds, datasets, opt_thr)
    plot_feature_distributions(X_train, y_train)

    n = len([f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")])
    print("\n" + "=" * 60)
    print(f"  Done. {n} plots saved in artifacts/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
