"""
MahaVarta: mT5 Generation-Parameter Experiments Module
======================================================
Executes systematic one-at-a-time parameter perturbation experiments on mT5:
  - Defaults: num_beams=4, max_length=84, min_length=30, length_penalty=1.0, no_repeat_ngram_size=2
  - Perturbations (one parameter at a time, keeping rest at default):
      * num_beams in {1, 2, 4, 6}
      * max_length in {60, 84, 128}
      * min_length in {10, 30, 50}
      * length_penalty in {0.6, 1.0, 2.0}
      * no_repeat_ngram_size in {0, 2, 3}
    Total settings = 16 per article.
  - Excludes redundant default runs: the default setting is run ONCE per article and reused across sweeps.
  - Measures novel_word_ratio and new_capitalised_or_name_like_tokens (rough hint only; manual review is the real check).
  - Saves incremental progress after every run to results/summarization/generation_experiments.csv.
  - Computes average summary length (in words) for each setting and saves results/summarization/parameter_summary_table.csv.
"""

import os
import sys
import time
import logging
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.summarization import load_mt5, mt5_summary
from src.prompting import novel_word_ratio, invented_name_candidates, output_length_words

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "summarization")
CSV_PATH = os.path.join(RESULTS_DIR, "generation_experiments.csv")
SUMMARY_TABLE_PATH = os.path.join(RESULTS_DIR, "parameter_summary_table.csv")

DEFAULTS = {
    "num_beams": 4,
    "max_length": 84,
    "min_length": 30,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 2,
}

PARAMETER_SWEEPS = [
    ("num_beams", [1, 2, 4, 6]),
    ("max_length", [60, 84, 128]),
    ("min_length", [10, 30, 50]),
    ("length_penalty", [0.6, 1.0, 2.0]),
    ("no_repeat_ngram_size", [0, 2, 3]),
]


def get_all_16_settings():
    """Build list of all 16 parameter setting descriptors."""
    settings = []
    for param_name, values in PARAMETER_SWEEPS:
        for val in values:
            cfg = dict(DEFAULTS)
            cfg[param_name] = val
            is_default = (val == DEFAULTS[param_name])
            settings.append({
                "parameter": param_name,
                "value": val,
                "is_default": is_default,
                "kwargs": cfg,
            })
    return settings


def run_experiments(articles_df: pd.DataFrame, max_articles: int = 5):
    """
    Run mT5 generation parameter experiments over sampled articles.

    Args:
        articles_df: DataFrame containing at least 'article_id', 'category', and 'text'.
        max_articles: Number of articles to use (default: 5 to stay under 30 mins).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_settings = get_all_16_settings()

    sampled_articles = articles_df.head(max_articles)
    total_expected_runs = len(sampled_articles) * len(all_settings)
    print("=" * 75)
    print("MAHAVARTA STAGE 9: mT5 GENERATION PARAMETER EXPERIMENTS")
    print("=" * 75)
    print(f"Total articles: {len(sampled_articles)}")
    print(f"Settings per article: {len(all_settings)} (12 distinct evaluations, defaults reused)")
    print(f"Total records to produce: {total_expected_runs}")

    # Load model and run one warm-up generation (excluded from timing)
    print("\nPreloading mT5 model...")
    load_mt5()
    print("Executing warm-up run (excluded from timing)...")
    _ = mt5_summary("विराट कोहलीने शानदार शतक झळकावले.", **DEFAULTS)
    print("Warm-up complete!\n")

    # Load existing progress if available
    completed_keys = set()
    records = []
    if os.path.exists(CSV_PATH):
        try:
            prev_df = pd.read_csv(CSV_PATH)
            for _, r in prev_df.iterrows():
                key = (str(r["article_id"]), str(r["parameter_tested"]), str(r["parameter_value"]))
                completed_keys.add(key)
            records = prev_df.to_dict("records")
            print(f"Found existing progress: {len(records)} records ({len(completed_keys)} unique settings) already completed.")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")

    t_sweep_start = time.time()
    run_idx = len(records)

    for _, art_row in sampled_articles.iterrows():
        art_id = str(art_row["article_id"])
        cat = str(art_row.get("category", "General"))
        art_text = str(art_row["text"])

        # Check if default baseline is already computed for this article to reuse across other default slots
        cached_default_res = None
        for r in records:
            if str(r.get("article_id")) == art_id and r.get("parameter_tested") == "num_beams" and str(r.get("parameter_value")) == "4":
                cached_default_res = {
                    "summary": r["summary"],
                    "input_tokens": r["tokens"],
                    "seconds": r["seconds"],
                }
                break

        for setting in all_settings:
            param = setting["parameter"]
            val = setting["value"]
            kwargs = setting["kwargs"]
            is_def = setting["is_default"]

            run_key = (art_id, param, str(val))

            if run_key in completed_keys:
                continue

            if is_def and cached_default_res is not None:
                # Reuse cached default execution to avoid 5x redundant computation
                res = cached_default_res
                elapsed = res["seconds"]
            else:
                t0 = time.time()
                res = mt5_summary(art_text, **kwargs)
                elapsed = round(time.time() - t0, 2)
                res["seconds"] = elapsed
                if is_def and cached_default_res is None:
                    cached_default_res = res

            summary = res.get("summary", "")
            input_tokens = res.get("input_tokens", 0)
            words_count = output_length_words(summary)

            # Heuristic checks (rough hint only; manual review is the real check)
            novel_ratio = novel_word_ratio(summary, art_text)
            new_names = invented_name_candidates(summary, art_text)

            record = {
                "article_id": art_id,
                "category": cat,
                "num_beams": int(kwargs["num_beams"]),
                "max_length": int(kwargs["max_length"]),
                "min_length": int(kwargs["min_length"]),
                "length_penalty": float(kwargs["length_penalty"]),
                "no_repeat_ngram_size": int(kwargs["no_repeat_ngram_size"]),
                "summary": summary,
                "tokens": input_tokens,
                "seconds": elapsed,
                "novel_word_ratio": novel_ratio,
                "new_capitalised_or_name_like_tokens": new_names,
                "summary_words": words_count,
                "parameter_tested": param,
                "parameter_value": val,
            }

            records.append(record)
            completed_keys.add(run_key)
            run_idx += 1

            # Save incrementally after each run
            temp_df = pd.DataFrame(records)
            temp_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

            print(f"[{run_idx}/{total_expected_runs}] {art_id} | {param}={val} -> {words_count} words | {novel_ratio:.4f} novel | {elapsed}s")

    # Final save
    final_df = pd.DataFrame(records)
    final_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nAll experiments complete in {time.time() - t_sweep_start:.2f}s!")
    print(f"Saved {len(final_df)} rows to {CSV_PATH}")

    # Generate parameter summary table
    summary_table = (
        final_df.groupby(["parameter_tested", "parameter_value"])[["summary_words", "seconds", "novel_word_ratio"]]
        .mean()
        .round(2)
        .reset_index()
    )
    summary_table.columns = ["parameter", "value", "avg_summary_words", "avg_seconds", "avg_novel_word_ratio"]
    summary_table.to_csv(SUMMARY_TABLE_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved parameter summary table to {SUMMARY_TABLE_PATH}")
    return final_df, summary_table
