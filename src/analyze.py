"""
analyze.py
==========
Cross-model comparison charts, per-class recall bar plots,
failure-case mosaics (context helped vs. hurt),
and one sample image pair per EMOTIC label.
Called by run_analysis.py after both tasks finish.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.config import EMOTIC_CLASSES, MODELS


# ── 1. Comparison table ───────────────────────────────────────────────────────

def build_comparison_table(t1: dict, t2: dict) -> pd.DataFrame:
    """
    t1 / t2 are dicts: {model_key: metrics_dict}
    Returns a tidy DataFrame for display and saving.
    """
    rows = []
    for key in MODELS:
        name = MODELS[key]["name"]
        for label, md in [("Full Image", t1), ("Face Only", t2)]:
            if key not in md:
                continue
            m = md[key]
            rows.append({
                "Model":    name,
                "Task":     label,
                "Accuracy": m["accuracy"],
                "Prec(M)":  m["precision_macro"],
                "Rec(M)":   m["recall_macro"],
                "F1(M)":    m["f1_macro"],
                "F1(W)":    m["f1_weighted"],
            })
    return pd.DataFrame(rows)


def save_comparison_table(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "comparison_table.csv")
    df.to_csv(path, index=False)
    print("\n" + "=" * 78)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78 + "\n")
    print(f"  Table saved → {path}")


# ── 2. F1 bar chart ───────────────────────────────────────────────────────────

def plot_f1_comparison(t1: dict, t2: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    keys        = [k for k in MODELS if k in t1]
    model_names = [MODELS[k]["name"] for k in keys]
    f1_full     = [t1[k]["f1_macro"]              for k in keys]
    f1_face     = [t2.get(k, {}).get("f1_macro", 0) for k in keys]

    x, w = np.arange(len(keys)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, f1_full, w, label="Full Image", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + w/2, f1_face, w, label="Face Only",  color="#C44E52", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("F1 Score (Macro)")
    ax.set_title("Full Image vs Face-Only F1 — All Models")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, "f1_comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  F1 bar chart → {path}")


# ── 3. Per-class recall charts ────────────────────────────────────────────────

def plot_per_class_recall(t1: dict, t2: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    def _recall(cm):
        s = cm.sum(axis=1).astype(float)
        s[s == 0] = 1
        return np.diag(cm) / s

    for key in MODELS:
        if key not in t1:
            continue
        name = MODELS[key]["name"]
        r1   = _recall(t1[key]["confusion_matrix"])
        r2   = (_recall(t2[key]["confusion_matrix"])
                if key in t2 else np.zeros(len(EMOTIC_CLASSES)))

        x, w = np.arange(len(EMOTIC_CLASSES)), 0.4
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.bar(x - w/2, r1, w, label="Full Image", color="#4C72B0", alpha=0.85)
        ax.bar(x + w/2, r2, w, label="Face Only",  color="#C44E52", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(EMOTIC_CLASSES, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Per-class Recall")
        ax.set_title(f"Per-class Recall – {name}")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        path = os.path.join(out_dir, f"{key}_per_class_recall.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Per-class recall → {path}")


# ── 4. Failure / qualitative case mosaics ────────────────────────────────────

def save_failure_cases(
    df_t1: pd.DataFrame,
    df_t2: pd.DataFrame,
    model_key: str,
    out_dir: str,
    n: int = 6,
):
    """
    Side-by-side mosaics showing examples where:
      • Context HELPED: full image correct, face-only wrong
      • Context HURT:   full image wrong,   face-only correct
    """
    os.makedirs(out_dir, exist_ok=True)
    name = MODELS[model_key]["name"]

    merged = df_t1.merge(
        df_t2[["image_path", "predicted_label", "cropped_path"]],
        on="image_path",
        suffixes=("_full", "_face"),
    )

    helped = merged[
        (merged["predicted_label_full"] == merged["label"]) &
        (merged["predicted_label_face"] != merged["label"])
    ].head(n)

    hurt = merged[
        (merged["predicted_label_full"] != merged["label"]) &
        (merged["predicted_label_face"] == merged["label"])
    ].head(n)

    def _mosaic(subset, title, fname):
        if len(subset) == 0:
            print(f"  No examples found for: {title}")
            return
        n_rows = len(subset)
        fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.5 * n_rows))
        if n_rows == 1:
            axes = [axes]
        for i, (_, row) in enumerate(subset.iterrows()):
            for j, (pcol, lbl_col, side) in enumerate([
                ("image_path",   "predicted_label_full", "Full image"),
                ("cropped_path", "predicted_label_face", "Face only"),
            ]):
                ax = axes[i][j]
                p  = row.get(pcol)
                if p and os.path.isfile(str(p)):
                    try:
                        ax.imshow(Image.open(str(p)).convert("RGB"))
                    except Exception:
                        ax.text(0.5, 0.5, "Load error", ha="center", va="center")
                else:
                    ax.text(0.5, 0.5, "No image", ha="center", va="center")
                ax.set_title(
                    f"{side}\nTrue: {row['label']}\nPred: {row[lbl_col]}",
                    fontsize=8,
                )
                ax.axis("off")
        fig.suptitle(f"{name}  –  {title}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Failure mosaic → {path}")

    _mosaic(helped, "Context HELPED  (full ✓  face ✗)", f"{model_key}_context_helped.png")
    _mosaic(hurt,   "Context HURT    (full ✗  face ✓)", f"{model_key}_context_hurt.png")


# ── 5. Sample image per EMOTIC label ─────────────────────────────────────────

def save_sample_images_per_label(df: pd.DataFrame, out_dir: str):
    """Save one full + cropped pair per EMOTIC class for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)

    for cls in EMOTIC_CLASSES:
        subset = df[df["label"] == cls]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        for ax, pcol, side in [
            (axes[0], "image_path",  "Full Image"),
            (axes[1], "cropped_path","Face Only"),
        ]:
            p = row.get(pcol)
            if p and os.path.isfile(str(p)):
                try:
                    ax.imshow(Image.open(str(p)).convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "Load error", ha="center")
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center")
            ax.set_title(side, fontsize=9)
            ax.axis("off")

        fig.suptitle(f"Label: {cls}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        safe_cls = cls.replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"sample_{safe_cls}.png"), dpi=100)
        plt.close()

    print(f"  Sample images per label → {out_dir}")
