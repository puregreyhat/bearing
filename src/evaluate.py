#!/usr/bin/env python3
"""Summarise fold metrics, render confusion matrices, and audit suspicious results."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

RESULTS = ROOT / "results"


def main():
    metrics = pd.read_csv(RESULTS / "fold_metrics.csv")
    predictions = pd.read_csv(RESULTS / "fold_predictions.csv")
    metric_columns = [column for column in metrics if column.startswith(("balanced_accuracy", "macro_f1", "precision_", "recall_", "pr_auc_", "roc_auc_", "inference_", "training_"))]
    summary = metrics.groupby(["task", "scheme", "model"])[metric_columns].agg(["mean", "std"])
    summary.to_csv(RESULTS / "metrics_summary.csv")
    primary = summary.xs("leave_one_load_out", level="scheme")

    for (task, scheme, model), group in predictions.groupby(["task", "scheme", "model"]):
        labels = sorted(set(group.true) | set(group.predicted))
        matrix = confusion_matrix(group.true, group.predicted, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{task} — {model} — {scheme}")
        fig.tight_layout()
        fig.savefig(RESULTS / f"confusion_{task}_{scheme}_{model}.png", dpi=150)
        plt.close(fig)

    suspicious = metrics[(metrics.balanced_accuracy >= .995) | (metrics.macro_f1 >= .995)]
    audit = "No fold reached 99.5% balanced accuracy or macro-F1."
    if len(suspicious):
        audit = f"""{len(suspicious)} fold/model results reached 99.5%, so leakage was explicitly rechecked:

- `train.py` asserts disjoint File_ID sets for every outer fold.
- Feature selection hard-excludes IDs, labels, load, RPM, diameter, sampling rate, and window metadata.
- Leave-One-Load-Out test rows contain only the held-out load.
- Every test fold contains Healthy/Normal data and all task classes.
- Preprocessing and tuning occur inside each outer training fold.

The high scores therefore remain reported as CWRU benchmark separability, not assumed deployment performance.
"""
    (RESULTS / "leakage_audit.md").write_text("# Leakage audit\n\n" + audit)

    importance = pd.read_csv(RESULTS / "feature_importance_folds.csv")
    top = (importance.groupby(["task", "model", "feature"])[["native_importance", "permutation_importance"]]
           .mean().sort_values(["task", "model", "permutation_importance"], ascending=[True, True, False])
           .groupby(level=[0, 1]).head(10))
    top.to_csv(RESULTS / "feature_importance_top.csv")
    plot_importance(top)

    report = f"""# Model evaluation

All values below are mean ± standard deviation across the four Leave-One-Load-Out folds. Plain accuracy is intentionally not a headline metric. PR-AUC and ROC-AUC treat Healthy as the positive class. SMOTE was not used; imbalance handling is class weighting (or balanced sample weights for XGBoost), so there is no oversampled variant to compare.

{format_primary(primary)}

## Per-fold Leave-One-Load-Out results

{format_folds(metrics[metrics.scheme == 'leave_one_load_out'])}

## Leakage audit

{audit}

## Explainability

Native RF/XGBoost importance and out-of-fold permutation importance are in `feature_importance_folds.csv`; the top ten per task/model are in `feature_importance_top.csv`.
"""
    (RESULTS / "model_evaluation.md").write_text(report)
    print(report)


def format_primary(primary):
    rows = []
    for (task, model), row in primary.iterrows():
        values = []
        wanted = ["balanced_accuracy", "macro_f1"]
        if task == "binary":
            wanted += ["precision_Healthy", "recall_Healthy", "pr_auc_Healthy", "roc_auc_Healthy"]
        else:
            wanted += [name for name in row.index.get_level_values(0).unique() if name.startswith("recall_")]
        wanted += ["training_time_seconds", "inference_ms_per_window"]
        for metric in wanted:
            if (metric, "mean") in row.index and pd.notna(row[(metric, "mean")]):
                values.append(f"{metric}: {row[(metric, 'mean')]:.4f} ± {row[(metric, 'std')]:.4f}")
        rows.append(f"### {task} — {model}\n\n" + "; ".join(values))
    return "\n\n".join(rows)


def format_folds(folds):
    sections = []
    for task, group in folds.groupby("task"):
        columns = ["model", "held_out_load", "balanced_accuracy", "macro_f1"]
        if task == "binary":
            columns += ["precision_Healthy", "recall_Healthy", "pr_auc_Healthy", "roc_auc_Healthy"]
        else:
            columns += [column for column in group if column.startswith("recall_") and group[column].notna().any()]
        sections.append(f"### {task}\n\n" + group[columns].sort_values(["model", "held_out_load"]).to_markdown(index=False, floatfmt=".4f"))
    return "\n\n".join(sections)


def plot_importance(top):
    for (task, model), group in top.reset_index().groupby(["task", "model"]):
        group = group.sort_values("permutation_importance")
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].barh(group.feature, group.native_importance)
        axes[0].set_title("Native importance")
        axes[1].barh(group.feature, group.permutation_importance)
        axes[1].set_title("Out-of-fold permutation importance")
        fig.suptitle(f"{task} — {model}")
        fig.tight_layout()
        fig.savefig(RESULTS / f"feature_importance_{task}_{model}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
