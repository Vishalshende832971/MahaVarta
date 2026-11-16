"""
MahaVarta: Stage 9 mT5 Generation Parameter Experiments Runner
==============================================================
Runs the 16 parameter sweep combinations over 5 sampled articles (80 total runs).
Saves to results/summarization/generation_experiments.csv and
results/summarization/parameter_summary_table.csv.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation_experiments import run_experiments

ARTICLES_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "summarization", "prompt_articles.csv")


def main():
    if not os.path.exists(ARTICLES_CSV):
        print(f"Error: {ARTICLES_CSV} not found! Run prompt experiments first.")
        sys.exit(1)

    articles_df = pd.read_csv(ARTICLES_CSV)
    print(f"Loaded {len(articles_df)} articles from {ARTICLES_CSV}")

    final_df, summary_table = run_experiments(articles_df, max_articles=5)

    print("\n" + "=" * 75)
    print("mT5 GENERATION PARAMETER EXPERIMENTS SUMMARY TABLE:")
    print("=" * 75)
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
