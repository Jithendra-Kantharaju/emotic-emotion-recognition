"""
inference.py
============
Core inference loop with automatic checkpoint / resume support.

If a results CSV already exists for a run, already-completed images are
skipped so interrupted runs can be resumed without re-processing.
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.prompt import build_prompt, parse_prediction


def run_inference(
    model,
    df: pd.DataFrame,
    image_col: str,      # "image_path" for Task 1 | "cropped_path" for Task 2
    out_csv: str,        # path where results are saved incrementally
    context: str = "full",
) -> pd.DataFrame:
    """
    Run forced-choice prediction for every row in df.

    Returns the full results DataFrame with two added columns:
        raw_output      – raw text returned by the model
        predicted_label – parsed EMOTIC class name (or 'Unknown')
    """
    prompt = build_prompt(context=context)

    # ── Resume support ────────────────────────────────────────────────────
    existing_df = None
    if os.path.isfile(out_csv):
        existing_df = pd.read_csv(out_csv)
        done_paths  = set(existing_df[image_col].astype(str))
        remaining   = df[~df[image_col].astype(str).isin(done_paths)].reset_index(drop=True)
        print(f"  Resume: {len(existing_df)} done, {len(remaining)} remaining")
    else:
        remaining = df.copy()

    new_rows = []
    for _, row in tqdm(
        remaining.iterrows(),
        total=len(remaining),
        desc=f"{model.model_name} [{context}]",
        unit="img",
    ):
        img_path = row[image_col]

        # Skip missing or failed-crop images
        if pd.isna(img_path) or not os.path.isfile(str(img_path)):
            new_rows.append({
                **row.to_dict(),
                "raw_output":      None,
                "predicted_label": "Unknown",
            })
            continue

        try:
            image = Image.open(str(img_path)).convert("RGB")
            raw   = model.predict(image, prompt)
            pred  = parse_prediction(raw)
        except Exception as e:
            print(f"\n  [ERROR] {img_path}: {e}")
            raw, pred = None, "Unknown"

        new_rows.append({
            **row.to_dict(),
            "raw_output":      raw,
            "predicted_label": pred,
        })

    # ── Merge with previously completed rows and save ─────────────────────
    new_df = pd.DataFrame(new_rows)
    if existing_df is not None and len(existing_df) > 0:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv}  ({len(final_df)} rows total)")

    return final_df
