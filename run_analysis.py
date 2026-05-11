"""
run_analysis.py
===============
Generate all comparison plots, per-class recall charts,
failure-case mosaics, and sample images per label.

Run this AFTER both run_task1.py and run_task2.py have finished.

Usage
-----
    python run_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd

from src.config  import TASK1_OUT, TASK2_OUT, ANALYSIS_OUT, MODELS, EMOTIC_CLASSES
from src.analyze import (
    build_comparison_table,
    save_comparison_table,
    plot_f1_comparison,
    plot_per_class_recall,
    save_failure_cases,
    save_sample_images_per_label,
)


def _load_metrics(task_out: str, suffix: str) -> dict:
    """
    Reconstruct per-model metrics dicts from saved JSON + .npy files.
    Returns {model_key: metrics_dict}
    """
    out = {}
    for key in MODELS:
        json_path = os.path.join(task_out, f"{key}_{suffix}_metrics.json")
        cm_path   = os.path.join(task_out, f"{key}_{suffix}_cm.npy")

        if not os.path.isfile(json_path):
            print(f"  [SKIP] No metrics found for model '{key}' ({suffix})")
            continue

        with open(json_path) as f:
            m = json.load(f)

        m["confusion_matrix"] = (
            np.load(cm_path)
            if os.path.isfile(cm_path)
            else np.zeros((len(EMOTIC_CLASSES), len(EMOTIC_CLASSES)), dtype=int)
        )
        out[key] = m
    return out


def main():
    os.makedirs(ANALYSIS_OUT, exist_ok=True)

    print("=" * 60)
    print("  PROJECT 3  –  ANALYSIS & VISUALIZATIONS")
    print("=" * 60)

    # ── Load saved results ────────────────────────────────────────────────
    print("\nLoading saved metrics ...")
    t1 = _load_metrics(TASK1_OUT, "task1")
    t2 = _load_metrics(TASK2_OUT, "task2")

    if not t1:
        print("\nNo Task 1 results found. Run run_task1.py first.")
        return

    # ── 1. Comparison table ───────────────────────────────────────────────
    print("\nBuilding comparison table ...")
    comp_df = build_comparison_table(t1, t2)
    save_comparison_table(comp_df, ANALYSIS_OUT)

    # ── 2. F1 bar chart ───────────────────────────────────────────────────
    print("Generating F1 comparison bar chart ...")
    plot_f1_comparison(t1, t2, ANALYSIS_OUT)

    # ── 3. Per-class recall charts ────────────────────────────────────────
    print("Generating per-class recall charts ...")
    plot_per_class_recall(t1, t2, ANALYSIS_OUT)

    # ── 4. Failure / qualitative mosaics ──────────────────────────────────
    print("Generating failure case mosaics ...")
    fc_dir = os.path.join(ANALYSIS_OUT, "failure_cases")
    for key in MODELS:
        t1_csv = os.path.join(TASK1_OUT, f"{key}_task1_results.csv")
        t2_csv = os.path.join(TASK2_OUT, f"{key}_task2_results.csv")
        if not (os.path.isfile(t1_csv) and os.path.isfile(t2_csv)):
            print(f"  [SKIP] Missing results CSV for model: {key}")
            continue
        save_failure_cases(
            pd.read_csv(t1_csv),
            pd.read_csv(t2_csv),
            model_key = key,
            out_dir   = fc_dir,
        )

    # ── 5. Sample images per label ────────────────────────────────────────
    print("Saving sample images per EMOTIC label ...")
    sample_dir = os.path.join(ANALYSIS_OUT, "sample_images_per_label")
    for key in MODELS:
        csv = os.path.join(TASK2_OUT, f"{key}_task2_results.csv")
        if os.path.isfile(csv):
            save_sample_images_per_label(pd.read_csv(csv), sample_dir)
            break   # only need one model's results for this

    print(f"\nAnalysis complete.  All outputs → {ANALYSIS_OUT}")


if __name__ == "__main__":
    main()
