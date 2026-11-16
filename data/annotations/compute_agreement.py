"""
MahaVarta: Inter-Annotator Agreement Calculator
===============================================
Computes Cohen's Kappa score and observed agreement percentage
between label_vishal and label_gufran in data/annotations/headlines.csv.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CSV_PATH = os.path.join(os.path.dirname(__file__), "headlines.csv")


def compute_agreement(csv_path: str = CSV_PATH):
    if not os.path.exists(csv_path):
        print(f"Error: Annotations file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Check that both columns are filled
    v_clean = df["label_vishal"].dropna().str.strip()
    g_clean = df["label_gufran"].dropna().str.strip()

    valid_mask = (v_clean != "") & (g_clean != "")
    valid_count = valid_mask.sum()

    if valid_count == 0:
        print("=" * 60)
        print("Inter-Annotator Agreement Status:")
        print("Both 'label_vishal' and 'label_gufran' are currently empty.")
        print("Please annotate the 200 headlines first according to ANNOTATION_GUIDE.md.")
        print("=" * 60)
        return

    v_labels = v_clean[valid_mask].tolist()
    g_labels = g_clean[valid_mask].tolist()

    total = len(v_labels)
    agreed = sum(1 for v, g in zip(v_labels, g_labels) if v == g)
    raw_agreement = (agreed / total) * 100

    kappa = cohen_kappa_score(v_labels, g_labels)

    classes = sorted(list(set(v_labels + g_labels)))
    cm = confusion_matrix(v_labels, g_labels, labels=classes)

    print("=" * 60)
    print("MahaVarta: Inter-Annotator Agreement Analysis")
    print("=" * 60)
    print(f"Total pairs evaluated:      {total} / {len(df)}")
    print(f"Identical agreements:      {agreed} ({raw_agreement:.2f}%)")
    print(f"Disagreements:             {total - agreed} ({(100 - raw_agreement):.2f}%)")
    print(f"Cohen's Kappa (κ):         {kappa:.4f}")
    print("\nConfusion Matrix (Rows: Vishal, Columns: Gufran):")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)

    if kappa >= 0.81:
        interpretation = "Almost perfect agreement"
    elif kappa >= 0.61:
        interpretation = "Substantial agreement"
    elif kappa >= 0.41:
        interpretation = "Moderate agreement"
    elif kappa >= 0.21:
        interpretation = "Fair agreement"
    else:
        interpretation = "Poor agreement"

    print(f"\nAgreement Level: {interpretation}")
    print("=" * 60)


if __name__ == "__main__":
    compute_agreement()
