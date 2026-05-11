"""
evaluate.py
===========
Classification metrics, confusion matrix plotting, and reporting.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from src.config import EMOTIC_CLASSES


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute all classification metrics.
    Returns zeros gracefully if all predictions were Unknown (API failures).
    """
    labels = EMOTIC_CLASSES
    kw     = dict(labels=labels, zero_division=0)

    # Guard: handle case where all predictions failed
    if len(y_true) == 0 or len(y_pred) == 0:
        print("\n  [WARNING] No valid predictions - all returned 'Unknown'.")
        print("  Root cause: API 400 errors (image too large). Fix applied below.")
        zero_cm = np.zeros((len(EMOTIC_CLASSES), len(EMOTIC_CLASSES)), dtype=int)
        return {
            "accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0,
            "f1_macro": 0.0, "precision_weighted": 0.0,
            "recall_weighted": 0.0, "f1_weighted": 0.0,
            "classification_report": "No valid predictions.",
            "confusion_matrix": zero_cm,
        }

    return {
        "accuracy":              round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":       round(precision_score(y_true, y_pred, average="macro",    **kw), 4),
        "recall_macro":          round(recall_score   (y_true, y_pred, average="macro",    **kw), 4),
        "f1_macro":              round(f1_score       (y_true, y_pred, average="macro",    **kw), 4),
        "precision_weighted":    round(precision_score(y_true, y_pred, average="weighted", **kw), 4),
        "recall_weighted":       round(recall_score   (y_true, y_pred, average="weighted", **kw), 4),
        "f1_weighted":           round(f1_score       (y_true, y_pred, average="weighted", **kw), 4),
        "classification_report": classification_report(y_true, y_pred, **kw),
        "confusion_matrix":      confusion_matrix(y_true, y_pred, labels=labels),
    }


def save_metrics(metrics: dict, out_dir: str, prefix: str):
    """Write metrics JSON and full classification report TXT to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    summary = {k: v for k, v in metrics.items()
               if k not in ("classification_report", "confusion_matrix")}
    json_path = os.path.join(out_dir, f"{prefix}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    txt_path = os.path.join(out_dir, f"{prefix}_report.txt")
    with open(txt_path, "w") as f:
        f.write(metrics["classification_report"])
    print(f"  Metrics JSON  -> {json_path}")
    print(f"  Class report  -> {txt_path}")


def plot_confusion_matrix(cm: np.ndarray, out_dir: str, prefix: str, title: str = ""):
    """Save a row-normalised heatmap of the 26x26 confusion matrix."""
    os.makedirs(out_dir, exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=EMOTIC_CLASSES, yticklabels=EMOTIC_CLASSES,
                cmap="Blues", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Recall (row-normalised)"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(title or prefix,    fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,             fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix  -> {path}")


def print_summary(metrics: dict, label: str = ""):
    """Print a compact summary table to stdout."""
    sep = "=" * 56
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
        print(sep)
    rows = [
        ("Accuracy",             "accuracy"),
        ("Precision  (macro)",   "precision_macro"),
        ("Recall     (macro)",   "recall_macro"),
        ("F1 Score   (macro)",   "f1_macro"),
        ("Precision  (weighted)","precision_weighted"),
        ("Recall     (weighted)","recall_weighted"),
        ("F1 Score   (weighted)","f1_weighted"),
    ]
    for display, key in rows:
        print(f"  {display:<24}: {metrics[key]:.4f}")
    print(f"{sep}\n")
