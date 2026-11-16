"""
MahaVarta: Stage 9 Qwen3 Prompt Comparison Script
=================================================
Selects 10 articles from MahaNews TEST split across 10 distinct categories (seed=42).
Saves them to results/summarization/prompt_articles.csv.
Runs all 3 prompt variants (zero_shot, role_based, role_constraints) on all 10 articles.
Computes objective checks:
  - has_three_items
  - devanagari_ratio
  - novel_word_ratio (stem-aware)
  - invented_name_candidates
  - words
Leaves manual evaluation columns completely empty for human review.
Saves progress incrementally to results/summarization/prompt_comparison.csv.
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import tokenize, clean_text
from src.prompting import (
    load_qwen,
    generate_brief,
    has_three_numbered_items,
    devanagari_ratio,
    novel_word_ratio,
    invented_name_candidates,
    output_length_words,
)

TEST_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "mahanews", "test.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "summarization")
ARTICLES_CSV = os.path.join(RESULTS_DIR, "prompt_articles.csv")
COMPARISON_CSV = os.path.join(RESULTS_DIR, "prompt_comparison.csv")


def sample_test_articles(n: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Sample n articles across distinct categories from MahaNews test split."""
    df = pd.read_csv(TEST_CSV)
    df = df[df["Label"] != "Label"].copy()  # Remove repeated header lines

    unique_categories = list(df["Label"].unique())[:n]
    sampled_rows = []

    for idx, cat in enumerate(unique_categories, 1):
        cat_df = df[df["Label"] == cat]
        row = cat_df.sample(n=1, random_state=random_state).iloc[0]
        text_clean = str(row["Text"]).strip()
        tokens_count = len(tokenize(clean_text(text_clean)))
        preview = text_clean[:100].replace("\n", " ") + "..."

        sampled_rows.append({
            "article_id": f"art_{idx:02d}_{cat.lower()}",
            "category": cat,
            "preview": preview,
            "token_count": tokens_count,
            "text": text_clean,
        })

    sampled_df = pd.DataFrame(sampled_rows)
    return sampled_df


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 75)
    print("MAHAVARTA STAGE 9: QWEN3 PROMPT COMPARISON EXPERIMENTS")
    print("=" * 75)

    # 1. Sample 10 articles across categories
    articles_df = sample_test_articles(n=10, random_state=42)
    articles_df.to_csv(ARTICLES_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(articles_df)} sampled test articles to: {ARTICLES_CSV}")

    # 2. Preload Qwen model
    print("Preloading Qwen3-0.6B...")
    load_qwen()

    # Check for existing progress
    completed_pairs = set()
    records = []
    if os.path.exists(COMPARISON_CSV):
        try:
            prev_df = pd.read_csv(COMPARISON_CSV)
            for _, r in prev_df.iterrows():
                completed_pairs.add((str(r["article_id"]), str(r["prompt_key"])))
            records = prev_df.to_dict("records")
            print(f"Found existing progress: {len(completed_pairs)} runs already completed.")
        except Exception as e:
            print(f"Could not read previous comparison CSV: {e}")

    prompt_keys = ["zero_shot", "role_based", "role_constraints"]
    total_runs = len(articles_df) * len(prompt_keys)
    run_idx = len(completed_pairs)
    t_start = time.time()

    # 3. Execute generation across all articles & prompt variants
    for _, art_row in articles_df.iterrows():
        art_id = art_row["article_id"]
        cat = art_row["category"]
        text = art_row["text"]

        for p_key in prompt_keys:
            if (art_id, p_key) in completed_pairs:
                continue

            print(f"[{run_idx + 1}/{total_runs}] Running {art_id} with '{p_key}'...")
            t0 = time.time()
            res = generate_brief(text, prompt_key=p_key, max_new_tokens=220)
            elapsed = round(time.time() - t0, 2)

            out_text = res.get("output", "")
            in_tokens = res.get("input_tokens", 0)
            out_tokens = res.get("output_tokens", 0)

            # Objective checks
            has_three = has_three_numbered_items(out_text)
            dev_ratio = devanagari_ratio(out_text)
            nov_ratio = novel_word_ratio(out_text, text)
            inv_names = invented_name_candidates(out_text, text)
            words = output_length_words(out_text)

            record = {
                "article_id": art_id,
                "category": cat,
                "prompt_key": p_key,
                "output": out_text,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "seconds": elapsed,
                "has_three_items": has_three,
                "devanagari_ratio": dev_ratio,
                "novel_word_ratio": nov_ratio,
                "invented_name_candidates": inv_names,
                "words": words,
                # Four empty columns for human review (Vishal and Gufran)
                "marathi_quality_1to5": "",
                "factual_accuracy_1to5": "",
                "follows_constraints_yes_no": "",
                "notes": "",
            }

            records.append(record)
            completed_pairs.add((art_id, p_key))
            run_idx += 1

            # Save incrementally after each run
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

            print(f"    Done in {elapsed}s | 3-items: {has_three} | Devanagari: {dev_ratio*100:.1f}% | Words: {words}")

    # Final save
    final_df = pd.DataFrame(records)
    final_df.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 75)
    print(f"All {len(final_df)} prompt comparison runs completed in {time.time() - t_start:.2f}s!")
    print(f"Saved results to: {COMPARISON_CSV}")
    print("=" * 75)

    # Compute summary table per prompt
    summary_data = []
    for p_key in prompt_keys:
        p_df = final_df[final_df["prompt_key"] == p_key]
        share_three_items = (p_df["has_three_items"].sum() / len(p_df)) * 100 if len(p_df) > 0 else 0.0
        avg_dev = p_df["devanagari_ratio"].mean() * 100 if len(p_df) > 0 else 0.0
        avg_nov = p_df["novel_word_ratio"].mean() if len(p_df) > 0 else 0.0
        avg_words = p_df["words"].mean() if len(p_df) > 0 else 0.0
        avg_sec = p_df["seconds"].mean() if len(p_df) > 0 else 0.0

        summary_data.append({
            "Prompt Variant": p_key,
            "3 Items Adherence (%)": f"{share_three_items:.1f}%",
            "Avg Devanagari Ratio (%)": f"{avg_dev:.1f}%",
            "Avg Novel Word Ratio": f"{avg_nov:.4f}",
            "Avg Word Count": f"{avg_words:.1f}",
            "Avg Latency (s)": f"{avg_sec:.2f}s",
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nPer-Prompt Objective Summary Table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
