"""
Generate all visualisation artefacts into artifacts/plots/.

Produces:
  1.  Feature spectrograms  — one sample per class for each of the 5 features
  2.  Feature correlation matrix — inter-feature Pearson correlations (train set)
  3.  Confusion matrices — v1 (multi_feature_combined) and v2 (multi_feature_v2)
      at both default and optimised thresholds
  4.  Feature distribution box-plots — dep vs norm per feature group
  5.  Class distribution bar chart across all splits
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PLOTS_DIR    = os.path.join(PROJECT_ROOT, "artifacts", "plots")
DATA_V2      = os.path.join(PROJECT_ROOT, "data", "processed", "combined_v2")
MODEL_V2_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models", "multi_feature_v2")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
NORMAL_CLR     = "#4C9BE8"
DEPRESSED_CLR  = "#E84C4C"
ACCENT         = "#2D2D2D"

sns.set_theme(style="whitegrid", font_scale=1.05)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_split(name: str):
    path = os.path.join(DATA_V2, f"{name}_features.npz")
    d    = np.load(path)
    return d["X"], d["y"]   # X: (N, 46, 313)  y: (N,)


FEATURE_SLICES = {
    "MFCC\n(13 coeffs)"         : (0,  13),
    "Delta-MFCC\n(13 coeffs)"   : (13, 26),
    "Chroma\n(12 bins)"         : (26, 38),
    "Spectral Contrast\n(7 bands)": (38, 45),
    "ZCR\n(1 dim)"              : (45, 46),
}
FEATURE_KEYS = list(FEATURE_SLICES.keys())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Feature spectrograms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_feature_spectrograms(X, y):
    print("  [1/5] Feature spectrograms...")
    dep_idx  = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(42)
    s_dep  = rng.choice(dep_idx)
    s_norm = rng.choice(norm_idx)

    n_features = len(FEATURE_SLICES)
    fig, axes = plt.subplots(n_features, 2,
                             figsize=(14, 3.5 * n_features),
                             constrained_layout=True)
    fig.suptitle("Acoustic Feature Maps\n(left: depressed  ·  right: normal)",
                 fontsize=14, fontweight="bold")

    for row, (fname, (s, e)) in enumerate(FEATURE_SLICES.items()):
        for col, (sample_idx, label, clr) in enumerate([
            (s_dep,  "Depressed", DEPRESSED_CLR),
            (s_norm, "Normal",    NORMAL_CLR),
        ]):
            ax  = axes[row, col]
            mat = X[sample_idx, s:e, :]   # (freq_bins, time)
            im  = ax.imshow(mat, aspect="auto", origin="lower",
                            cmap="magma", interpolation="nearest")
            ax.set_title(f"{label}  [{fname.split(chr(10))[0]}]",
                         color=clr, fontweight="bold", fontsize=10)
            ax.set_xlabel("Time frames (×5 s / 313)")
            ax.set_ylabel("Freq / coeff index")
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    out = os.path.join(PLOTS_DIR, "1_feature_spectrograms.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Feature correlation matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_correlation_matrix(X, y):
    print("  [2/5] Feature correlation matrix...")

    # Collapse each feature group to its temporal mean → per-sample scalar vector
    group_labels = []
    group_means  = []
    for fname, (s, e) in FEATURE_SLICES.items():
        chunk = X[:, s:e, :]           # (N, bins, T)
        mean_over_time = chunk.mean(axis=2)   # (N, bins)
        for b in range(e - s):
            group_labels.append(f"{fname.split(chr(10))[0]}_c{b}")
            group_means.append(mean_over_time[:, b])

    group_means  = np.stack(group_means, axis=1)   # (N, total_dims)
    group_labels_short = [l.replace("Spectral Contrast", "SC")
                            .replace("Delta-MFCC", "dMFCC")
                            .replace("Chroma", "Chr")
                            .replace("ZCR", "ZCR") for l in group_labels]

    corr = np.corrcoef(group_means.T)   # (total_dims, total_dims)

    fig, ax = plt.subplots(figsize=(16, 14), constrained_layout=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax,
        mask=mask,
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor="#ddd",
        xticklabels=group_labels_short,
        yticklabels=group_labels_short,
        annot=False, cbar_kws={"label": "Pearson r", "shrink": 0.7},
    )
    ax.set_title("Feature Inter-Correlation Matrix\n"
                 "(temporal-mean per sample, lower triangle)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    # draw group separator lines
    group_sizes = [e - s for _, (s, e) in FEATURE_SLICES.items()]
    boundaries  = np.cumsum(group_sizes)[:-1]
    for b in boundaries:
        ax.axvline(b, color="black", lw=1.5)
        ax.axhline(b, color="black", lw=1.5)

    # group name labels along diagonal
    group_names_clean = [k.split("\n")[0] for k in FEATURE_KEYS]
    prev = 0
    for gname, gsz in zip(group_names_clean, group_sizes):
        mid = prev + gsz / 2
        ax.text(mid, -0.8, gname, ha="center", va="top",
                fontsize=8, fontweight="bold", color=ACCENT,
                transform=ax.get_xaxis_transform())
        prev += gsz

    out = os.path.join(PLOTS_DIR, "2_feature_correlation_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Confusion matrices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_cm(ax, cm, title, subtitle=""):
    """Draw a single annotated confusion matrix on ax."""
    group_names  = ["TN", "FP", "FN", "TP"]
    group_counts = [f"{v:,}" for v in cm.flatten()]
    total        = cm.sum()
    group_pcts   = [f"{v / total:.1%}" for v in cm.flatten()]

    labels = [f"{n}\n{c}\n{p}" for n, c, p in
              zip(group_names, group_counts, group_pcts)]
    labels = np.array(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                ax=ax, linewidths=1, linecolor="white",
                xticklabels=["Normal", "Depressed"],
                yticklabels=["Normal", "Depressed"],
                cbar=False,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold")


def plot_confusion_matrices():
    print("  [3/5] Confusion matrices...")

    summary_path = os.path.join(MODEL_V2_DIR, "training_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    opt_thr = summary["optimal_threshold"]

    # Reload model for prediction
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf   # type: ignore
    model = tf.keras.models.load_model(
        os.path.join(MODEL_V2_DIR, "best.keras"),
        compile=False,
    )

    datasets = {}
    for name in ["train", "val", "test", "ravdess"]:
        X, y = load_split(name) if name != "ravdess" else load_split("ravdess")
        datasets[name] = (X, y)

    # --- get predictions ---
    def predict(X):
        inp = X[..., np.newaxis].astype(np.float32)
        return model.predict(inp, batch_size=64, verbose=0).squeeze()

    preds = {name: predict(X) for name, (X, y) in datasets.items()}

    split_titles = {
        "train"  : "Train (3,152 samples)",
        "val"    : "Validation (417 samples)\n[speaker-disjoint]",
        "test"   : "Test / EATD+DS1 (1,343 samples)",
        "ravdess": "RAVDESS (288 samples)\n[cross-domain, held-out]",
    }

    # ── 3a. Default threshold 0.5 ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(f"Confusion Matrices — multi_feature_v2\nThreshold = 0.50 (default)",
                 fontsize=13, fontweight="bold")
    for ax, (name, (X, y)) in zip(axes.flat, datasets.items()):
        y_hat = (preds[name] >= 0.5).astype(int)
        cm    = confusion_matrix(y.astype(int), y_hat)
        acc   = (y_hat == y.astype(int)).mean()
        _draw_cm(ax, cm, split_titles[name], f"Accuracy: {acc:.1%}")

    out05 = os.path.join(PLOTS_DIR, "3a_confusion_matrices_thr0.5.png")
    fig.savefig(out05, dpi=150)
    plt.close(fig)
    print(f"    saved → {out05}")

    # ── 3b. Optimised threshold ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle(f"Confusion Matrices — multi_feature_v2\n"
                 f"Optimised Threshold = {opt_thr:.4f} (Youden's J on val)",
                 fontsize=13, fontweight="bold")
    for ax, (name, (X, y)) in zip(axes.flat, datasets.items()):
        y_hat = (preds[name] >= opt_thr).astype(int)
        cm    = confusion_matrix(y.astype(int), y_hat)
        acc   = (y_hat == y.astype(int)).mean()
        _draw_cm(ax, cm, split_titles[name], f"Accuracy: {acc:.1%}")

    out_opt = os.path.join(PLOTS_DIR, "3b_confusion_matrices_optimised_thr.png")
    fig.savefig(out_opt, dpi=150)
    plt.close(fig)
    print(f"    saved → {out_opt}")

    # ── 3c. Side-by-side: test split only, both thresholds ───────────────────
    X_test, y_test = datasets["test"]
    p_test = preds["test"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Test Set — Threshold Comparison  (multi_feature_v2)",
                 fontsize=13, fontweight="bold")
    for ax, thr, label in [
        (axes[0], 0.5,     "Default threshold = 0.50"),
        (axes[1], opt_thr, f"Optimised threshold = {opt_thr:.4f}"),
    ]:
        y_hat = (p_test >= thr).astype(int)
        cm    = confusion_matrix(y_test.astype(int), y_hat)
        acc   = (y_hat == y_test.astype(int)).mean()
        _draw_cm(ax, cm, "Test / EATD+DS1", f"{label}\nAccuracy: {acc:.1%}")

    out_cmp = os.path.join(PLOTS_DIR, "3c_test_threshold_comparison.png")
    fig.savefig(out_cmp, dpi=150)
    plt.close(fig)
    print(f"    saved → {out_cmp}")

    return preds, datasets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Feature distribution box-plots (dep vs norm)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_feature_distributions(X, y, split_name="Train"):
    print(f"  [4/5] Feature distributions ({split_name})...")

    dep_mask  = y == 1
    norm_mask = y == 0

    group_names_clean = [k.split("\n")[0] for k in FEATURE_KEYS]
    n_groups = len(FEATURE_SLICES)
    fig, axes = plt.subplots(1, n_groups, figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f"Feature Distribution: Depressed vs Normal\n({split_name} set)",
                 fontsize=13, fontweight="bold")

    for ax, (fname, (s, e)), gname in zip(axes, FEATURE_SLICES.items(), group_names_clean):
        # temporal mean of each coefficient per sample
        chunk      = X[:, s:e, :]             # (N, bins, T)
        feat_mean  = chunk.mean(axis=2)        # (N, bins)
        global_mean = feat_mean.mean(axis=1)   # (N,)  — one scalar per sample

        dep_vals  = global_mean[dep_mask]
        norm_vals = global_mean[norm_mask]

        bp = ax.boxplot(
            [norm_vals, dep_vals],
            labels=["Normal", "Depressed"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker=".", alpha=0.3, markersize=4),
        )
        bp["boxes"][0].set_facecolor(NORMAL_CLR)
        bp["boxes"][0].set_alpha(0.85)
        bp["boxes"][1].set_facecolor(DEPRESSED_CLR)
        bp["boxes"][1].set_alpha(0.85)

        ax.set_title(gname, fontweight="bold")
        ax.set_ylabel("Temporal mean (normalised)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    out = os.path.join(PLOTS_DIR, f"4_feature_distributions_{split_name.lower()}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Class distribution bar chart across all splits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_class_distribution(datasets: dict):
    print("  [5/5] Class distribution chart...")

    labels_map = {
        "train"  : "Train",
        "val"    : "Validation",
        "test"   : "Test",
        "ravdess": "RAVDESS",
    }
    split_names = list(labels_map.values())
    dep_counts  = []
    norm_counts = []
    for name in labels_map:
        _, y = datasets[name]
        dep_counts.append(int(y.sum()))
        norm_counts.append(int((y == 0).sum()))

    x = np.arange(len(split_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars_n = ax.bar(x - width / 2, norm_counts, width,
                    label="Normal",     color=NORMAL_CLR,    alpha=0.9)
    bars_d = ax.bar(x + width / 2, dep_counts,  width,
                    label="Depressed",  color=DEPRESSED_CLR, alpha=0.9)

    for bar in list(bars_n) + list(bars_d):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h:,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(split_names, fontsize=11)
    ax.set_ylabel("Number of segments", fontsize=11)
    ax.set_title("Class Distribution Across Splits (combined_v2)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(norm_counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    out = os.path.join(PLOTS_DIR, "5_class_distribution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    saved → {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Score distribution (probability histograms)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_score_distributions(preds: dict, datasets: dict, opt_thr: float):
    print("  [6/6] Prediction score distributions...")

    split_order  = ["train", "val", "test", "ravdess"]
    split_titles = ["Train", "Validation", "Test / EATD+DS1", "RAVDESS"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Prediction Score Distributions\n"
                 "(model output probabilities, by true class)",
                 fontsize=13, fontweight="bold")

    for ax, name, title in zip(axes.flat, split_order, split_titles):
        _, y = datasets[name]
        p    = preds[name]

        ax.hist(p[y == 0], bins=40, alpha=0.6, color=NORMAL_CLR,    label="Normal")
        ax.hist(p[y == 1], bins=40, alpha=0.6, color=DEPRESSED_CLR, label="Depressed")
        ax.axvline(0.5,     color="gray",  linestyle="--", lw=1.5, label="thr=0.5")
        ax.axvline(opt_thr, color="black", linestyle=":",  lw=2.0,
                   label=f"thr={opt_thr:.3f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted probability of depression")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    out = os.path.join(PLOTS_DIR, "6_score_distributions.png")
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

    datasets_all = {
        "train"  : (X_train, y_train),
        "val"    : (X_val,   y_val),
        "test"   : (X_test,  y_test),
        "ravdess": (X_rav,   y_rav),
    }

    print(f"\nSplit sizes:")
    for k, (X, y) in datasets_all.items():
        print(f"  {k:8s}: {len(y):5d}  (dep={int(y.sum()):4d}, norm={int((y==0).sum()):5d})")

    # 1 – spectrograms (use training samples for variety)
    plot_feature_spectrograms(X_train, y_train)

    # 2 – correlation matrix (train set)
    plot_correlation_matrix(X_train, y_train)

    # 3 – confusion matrices (needs model)
    preds, _ = plot_confusion_matrices()

    # 4 – feature distributions (train and test)
    plot_feature_distributions(X_train, y_train, split_name="Train")
    plot_feature_distributions(X_test,  y_test,  split_name="Test")

    # 5 – class distribution
    plot_class_distribution(datasets_all)

    # 6 – score distributions
    with open(os.path.join(MODEL_V2_DIR, "training_summary.json")) as f:
        opt_thr = json.load(f)["optimal_threshold"]
    plot_score_distributions(preds, datasets_all, opt_thr)

    print("\n" + "=" * 60)
    print(f"  Done. {len(os.listdir(PLOTS_DIR))} plots saved in artifacts/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
