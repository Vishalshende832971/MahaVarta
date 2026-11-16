"""
Compute document frequency and category entropy for Marathi vocabulary in MahaNews.
Identifies words that appear frequently and uniformly across all 12 news categories.
"""

import math
import json
import os
import sys
import pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_text, normalize_text, tokenize

def main():
    df = pd.read_csv("data/mahanews/train.csv")
    num_docs = len(df)
    print(f"Total training articles: {num_docs}")

    word_cat_counts = defaultdict(Counter)
    word_doc_freq = Counter()

    for text, cat in zip(df["Text"], df["Label"]):
        norm = normalize_text(str(text))
        cleaned = clean_text(norm)
        tokens = set(tokenize(cleaned))
        for t in tokens:
            if len(t) >= 2 and all("\u0900" <= c <= "\u097F" for c in t):
                word_cat_counts[t][cat] += 1
                word_doc_freq[t] += 1

    log2_12 = math.log2(12)
    candidate_metrics = []

    for w, total_df in word_doc_freq.items():
        if total_df < 200:
            continue
        cat_dist = word_cat_counts[w]
        total_occurrences = sum(cat_dist.values())
        entropy = 0.0
        for cat, cnt in cat_dist.items():
            p = cnt / total_occurrences
            entropy -= p * math.log2(p)
        norm_entropy = entropy / log2_12  # 1.0 = perfectly uniform across 12 categories

        df_pct = total_df / num_docs
        candidate_metrics.append({
            "word": w,
            "doc_freq": total_df,
            "df_pct": round(df_pct * 100, 2),
            "entropy": round(norm_entropy, 4),
            "score": round(df_pct * (norm_entropy ** 2), 4),
        })

    # Sort candidates by combined score: high frequency & high category entropy
    candidate_metrics.sort(key=lambda x: x["score"], reverse=True)

    top100 = candidate_metrics[:100]
    print("\nTOP 100 CANDIDATES (Ranked by Document Frequency * Normalized Category Entropy^2):")
    for rank, c in enumerate(top100, 1):
        print(f"{rank:3d}. {c['word']:<14} DF: {c['doc_freq']:5d} ({c['df_pct']:5.2f}%) | Entropy: {c['entropy']:.4f}")

    with open("data/top100_candidates.json", "w", encoding="utf-8") as f:
        json.dump(top100, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
