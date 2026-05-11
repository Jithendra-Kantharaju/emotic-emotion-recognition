"""
face_crop.py
============
Crop person bounding boxes from full scene images using
ground-truth annotations.  Saves each crop to CROPPED_DIR
and skips images already cropped in a previous run.
"""

import os
import hashlib
import pandas as pd
from PIL import Image
from src.config import CROPPED_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────

def _crop_filename(image_path: str, x1: int, y1: int, x2: int, y2: int) -> str:
    """Generate a unique filename for a cropped face patch."""
    key = f"{image_path}_{x1}_{y1}_{x2}_{y2}"
    h   = hashlib.md5(key.encode()).hexdigest()[:12]
    ext = os.path.splitext(image_path)[-1] or ".jpg"
    return f"crop_{h}{ext}"


# ── Main function ─────────────────────────────────────────────────────────────

def crop_faces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over df rows and crop each bounding box.
    Skips crops that already exist on disk (resume-safe).

    Adds three columns to the returned DataFrame:
        cropped_path      – absolute path to the saved crop (None on failure)
        crop_failed       – True if the crop could not be produced
        crop_fail_reason  – error string or None
    """
    os.makedirs(CROPPED_DIR, exist_ok=True)

    paths, failed, reasons = [], [], []

    print(f"\nCropping {len(df)} person regions  →  {CROPPED_DIR}")

    for _, row in df.iterrows():
        img_path        = str(row["image_path"])
        x1, y1, x2, y2 = (int(row["bbox_x1"]), int(row["bbox_y1"]),
                           int(row["bbox_x2"]), int(row["bbox_y2"]))

        # ── Validate bounding box ─────────────────────────────────────────
        if x2 <= x1 or y2 <= y1:
            paths.append(None)
            failed.append(True)
            reasons.append("invalid_bbox")
            continue

        crop_path = os.path.join(CROPPED_DIR, _crop_filename(img_path, x1, y1, x2, y2))

        # Skip if already done in a previous run
        if os.path.isfile(crop_path):
            paths.append(crop_path)
            failed.append(False)
            reasons.append(None)
            continue

        # ── Perform crop ──────────────────────────────────────────────────
        try:
            img  = Image.open(img_path).convert("RGB")
            w, h = img.size
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 <= cx1 or cy2 <= cy1:
                raise ValueError(f"Bbox clamped to empty region ({cx1},{cy1},{cx2},{cy2})")
            img.crop((cx1, cy1, cx2, cy2)).save(crop_path)
            paths.append(crop_path)
            failed.append(False)
            reasons.append(None)
        except Exception as e:
            paths.append(None)
            failed.append(True)
            reasons.append(str(e))

    df = df.copy()
    df["cropped_path"]     = paths
    df["crop_failed"]      = failed
    df["crop_fail_reason"] = reasons

    ok  = sum(not f for f in failed)
    bad = sum(failed)
    print(f"  Successful crops : {ok}")
    print(f"  Failed crops     : {bad}")
    return df


def save_failure_report(df: pd.DataFrame, out_path: str):
    """Write a CSV listing all rows where cropping failed."""
    bad = df[df["crop_failed"]][
        ["image_path", "label", "crop_fail_reason",
         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bad.to_csv(out_path, index=False)
    print(f"  Crop failure report  → {out_path}")
