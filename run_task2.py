"""
run_task2.py
============
Task 2 – Face-Only Expression Classification

Uses ground-truth bounding boxes from annotations.csv to crop each
person from the scene, then classifies only the cropped face with
no background context visible.

Usage
-----
    python run_task2.py --model qwen
    python run_task2.py --model llava
    python run_task2.py --model internvl
    python run_task2.py --model all
    python run_task2.py --model qwen --sample 100
"""

import argparse
import os
import json
import numpy as np

from src.dataset   import load_annotations
from src.face_crop import crop_faces, save_failure_report
from src.models    import get_model
from src.inference import run_inference
from src.evaluate  import (
    compute_metrics, save_metrics, plot_confusion_matrix, print_summary
)
from src.config import TASK2_OUT, MODELS


def run_for_model(model_key: str, df_crops):
    name = MODELS[model_key]["name"]
    print(f"\n{'='*60}")
    print(f"  TASK 2 – FACE ONLY  |  {name}")
    print(f"{'='*60}")

    # Only process rows where cropping succeeded
    valid_df = df_crops[~df_crops["crop_failed"]].copy()
    print(f"  Usable cropped images : {len(valid_df)}")

    out_csv = os.path.join(TASK2_OUT, f"{model_key}_task2_results.csv")

    model = get_model(model_key)
    model.load()

    results_df = run_inference(
        model     = model,
        df        = valid_df,
        image_col = "cropped_path",
        out_csv   = out_csv,
        context   = "face",
    )

    valid   = results_df[results_df["predicted_label"] != "Unknown"]
    metrics = compute_metrics(
        valid["label"].tolist(),
        valid["predicted_label"].tolist(),
    )

    print_summary(metrics, label=f"{name} – Face Only")
    save_metrics(metrics, TASK2_OUT, prefix=f"{model_key}_task2")

    np.save(
        os.path.join(TASK2_OUT, f"{model_key}_task2_cm.npy"),
        metrics["confusion_matrix"],
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        out_dir = TASK2_OUT,
        prefix  = f"{model_key}_task2",
        title   = f"Task 2 Confusion Matrix – {name}",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Project 3 – Task 2: Face-Only Classification"
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

    os.makedirs(TASK2_OUT, exist_ok=True)

    print("=" * 60)
    print("  PROJECT 3  –  TASK 2: FACE-ONLY CLASSIFICATION")
    print("=" * 60)

    # ── Step 1: Load annotations ──────────────────────────────────────────
    df = load_annotations(sample=args.sample)
    print(f"\nDataset size : {len(df)} images")

    # ── Step 2: Crop faces using GT bounding boxes ────────────────────────
    df = crop_faces(df)

    failure_csv = os.path.join(TASK2_OUT, "crop_failures.csv")
    save_failure_report(df, failure_csv)

    n_ok  = (~df["crop_failed"]).sum()
    n_bad =   df["crop_failed"].sum()
    print(f"\n  Crop results:  OK={n_ok}   Failed={n_bad}\n")

    # ── Step 3: Run inference per requested model ─────────────────────────
    keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_metrics = {k: run_for_model(k, df) for k in keys}

    summary = {
        k: {mk: mv for mk, mv in m.items()
            if mk not in ("classification_report", "confusion_matrix")}
        for k, m in all_metrics.items()
    }
    with open(os.path.join(TASK2_OUT, "task2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTask 2 complete.  Outputs → {TASK2_OUT}")


if __name__ == "__main__":
    main()
