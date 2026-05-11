"""
run_task1.py
============
Task 1 – Full Image Expression Classification

Feeds complete scene images to the VLM and asks it to classify
the person's emotional state (forced choice across 26 EMOTIC classes).

Usage
-----
    python run_task1.py --model qwen
    python run_task1.py --model llava
    python run_task1.py --model internvl
    python run_task1.py --model all           # runs all 3 sequentially
    python run_task1.py --model qwen --sample 100   # quick 100-image test
"""

import argparse
import os
import json
import numpy as np

from src.dataset   import load_annotations
from src.models    import get_model
from src.inference import run_inference
from src.evaluate  import (
    compute_metrics, save_metrics, plot_confusion_matrix, print_summary
)
from src.config import TASK1_OUT, MODELS


def run_for_model(model_key: str, df):
    name = MODELS[model_key]["name"]
    print(f"\n{'='*60}")
    print(f"  TASK 1 – FULL IMAGE  |  {name}")
    print(f"{'='*60}")

    out_csv = os.path.join(TASK1_OUT, f"{model_key}_task1_results.csv")

    model = get_model(model_key)
    model.load()

    results_df = run_inference(
        model     = model,
        df        = df,
        image_col = "image_path",
        out_csv   = out_csv,
        context   = "full",
    )

    valid   = results_df[results_df["predicted_label"] != "Unknown"]
    metrics = compute_metrics(
        valid["label"].tolist(),
        valid["predicted_label"].tolist(),
    )

    print_summary(metrics, label=f"{name} – Full Image")
    save_metrics(metrics, TASK1_OUT, prefix=f"{model_key}_task1")

    # Save raw confusion matrix for run_analysis.py
    np.save(
        os.path.join(TASK1_OUT, f"{model_key}_task1_cm.npy"),
        metrics["confusion_matrix"],
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        out_dir = TASK1_OUT,
        prefix  = f"{model_key}_task1",
        title   = f"Task 1 Confusion Matrix – {name}",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Project 3 – Task 1: Full Image Classification"
    )
    parser.add_argument(
        "--model", default="all",
        help="Model key: qwen | llava | internvl | all  (default: all)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Random subset size for quick testing, e.g. --sample 100",
    )
    args = parser.parse_args()

    os.makedirs(TASK1_OUT, exist_ok=True)

    print("=" * 60)
    print("  PROJECT 3  –  TASK 1: FULL IMAGE CLASSIFICATION")
    print("=" * 60)

    df = load_annotations(sample=args.sample)
    print(f"\nDataset size : {len(df)} images\n")

    keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_metrics = {k: run_for_model(k, df) for k in keys}

    # Write combined JSON summary
    summary = {
        k: {mk: mv for mk, mv in m.items()
            if mk not in ("classification_report", "confusion_matrix")}
        for k, m in all_metrics.items()
    }
    with open(os.path.join(TASK1_OUT, "task1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTask 1 complete.  Outputs → {TASK1_OUT}")


if __name__ == "__main__":
    main()
