"""
dataset.py
==========
Load annotations.csv, resolve absolute image paths,
filter missing files and unknown labels,
and optionally return a reproducible random subset.
"""

import os
import pandas as pd
from src.config import (
    ANNOTATIONS_CSV, IMAGES_ROOT, EMOTIC_CLASSES, SAMPLE_SIZE, RANDOM_SEED
)


def load_annotations(sample: int = None) -> pd.DataFrame:
    """
    Returns a clean DataFrame with at minimum these columns:
        image_path  – absolute path to the full scene image
        label       – one of the 26 EMOTIC class strings
        bbox_x1/y1/x2/y2 – person bounding box (int)

    Parameters
    ----------
    sample : int or None
        If given, override SAMPLE_SIZE and return this many rows.
    """
    if not os.path.isfile(ANNOTATIONS_CSV):
        raise FileNotFoundError(
            f"annotations.csv not found at: {ANNOTATIONS_CSV}\n"
            "Please unzip emotic.zip into the emotic/ folder."
        )

    print(f"Loading annotations from: {ANNOTATIONS_CSV}")
    df = pd.read_csv(ANNOTATIONS_CSV)

    # ── Normalise column names ────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if "image" in lc and "path" in lc:
            rename_map[col] = "image_path"
        elif lc in ("label", "emotion", "category", "discrete_label"):
            rename_map[col] = "label"
        elif lc in ("x1", "bbox_x1"):
            rename_map[col] = "bbox_x1"
        elif lc in ("y1", "bbox_y1"):
            rename_map[col] = "bbox_y1"
        elif lc in ("x2", "bbox_x2"):
            rename_map[col] = "bbox_x2"
        elif lc in ("y2", "bbox_y2"):
            rename_map[col] = "bbox_y2"
    df = df.rename(columns=rename_map)

    print(f"  Raw rows         : {len(df)}")

    # ── Resolve absolute image paths ──────────────────────────────────────
    df["image_path"] = df["image_path"].apply(
        lambda p: os.path.join(IMAGES_ROOT, str(p).strip())
    )

    # ── Drop rows with missing image files ────────────────────────────────
    exists = df["image_path"].apply(os.path.isfile)
    print(f"  Missing images   : {(~exists).sum()}")
    df = df[exists].reset_index(drop=True)

    # ── Clean labels and drop unrecognized ones ───────────────────────────
    df["label"] = df["label"].astype(str).str.strip()
    valid = df["label"].isin(EMOTIC_CLASSES)
    print(f"  Unknown labels   : {(~valid).sum()}")
    df = df[valid].reset_index(drop=True)

    # ── Ensure bbox columns exist as integers ─────────────────────────────
    for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0

    print(f"  Usable rows      : {len(df)}")

    # ── Optional random subset ────────────────────────────────────────────
    n = sample if sample is not None else SAMPLE_SIZE
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  Subset size      : {n} rows (random seed={RANDOM_SEED})")

    return df
