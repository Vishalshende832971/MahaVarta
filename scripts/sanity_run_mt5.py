"""
MahaVarta Stage 8 Sanity Run Script
Executes real mT5 summarisation on 3 sample articles, a deliberately long article for chunked mode,
and generation control comparison.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.summarization import (
    load_mt5,
    textrank_summary,
    mt5_summary,
    chunked_mt5_summary,
    summarize_auto,
    split_sentences,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_articles")


def main():
    print("=" * 70)
    print("MAHAVARTA STAGE 8: MT5 SUMMARISATION SANITY RUN")
    print("=" * 70)

    # Preload model
    t_load_start = time.time()
    tokenizer, model = load_mt5()
    print(f"Preloaded mT5 model on CPU in {time.time() - t_load_start:.2f}s\n")

    samples = [
        ("01_short_sports.txt", "1. Short - Sports"),
        ("02_medium_health.txt", "2. Medium - Health"),
        ("03_long_politics.txt", "3. Long - Politics"),
    ]

    sample_results = []

    # 1. Run on 3 sample articles
    print("--- 1. EVALUATION ON 3 SAMPLE ARTICLES ---")
    for filename, label in samples:
        filepath = os.path.join(SAMPLE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # TextRank Baseline
        tr_summary = textrank_summary(raw_text, n_sentences=3)

        # mT5 Abstractive Summary (auto mode)
        res = summarize_auto(raw_text, max_length=84, min_length=30, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=2)

        print(f"\n=======================================================")
        print(f"Sample: {label} ({filename})")
        print(f"=======================================================")
        print(f"Input Tokens: {res['input_tokens']}")
        print(f"Mode: {res['mode']} | Truncated: {res.get('truncated', False)} | Time: {res['seconds']}s")
        print(f"\n[TextRank Extractive Baseline]:\n{tr_summary}")
        print(f"\n[mT5 Abstractive Summary]:\n{res['summary']}")

        sample_results.append({
            "filename": filename,
            "label": label,
            "input_tokens": res["input_tokens"],
            "mode": res["mode"],
            "truncated": res.get("truncated", False),
            "seconds": res["seconds"],
            "textrank": tr_summary,
            "mt5": res["summary"],
        })

    # 2. Deliberately Long Article (Chunked Mode Demo)
    print("\n\n" + "=" * 70)
    print("--- 2. CHUNKED SUMMARISATION ON DELIBERATELY LONG ARTICLE ---")
    print("=" * 70)
    politics_file = os.path.join(SAMPLE_DIR, "03_long_politics.txt")
    with open(politics_file, "r", encoding="utf-8") as f:
        politics_text = f.read()

    # Concatenate politics text twice to create a long article (> 512 tokens)
    long_article = politics_text + "\n\n" + politics_text
    print(f"Created concatenated long article (original repeated twice).")

    res_long = summarize_auto(long_article, max_chunk_tokens=450, max_length=84, min_length=30, num_beams=4)

    print(f"Total Input Tokens: {res_long['input_tokens']} (> 512 -> chunking triggered)")
    print(f"Mode: {res_long['mode']}")
    print(f"Number of Chunks: {res_long.get('num_chunks', 1)}")
    print(f"Total Execution Time: {res_long['seconds']}s")

    print("\n[Intermediate Chunk Summaries]:")
    for i, c_sum in enumerate(res_long.get("chunk_summaries", []), 1):
        print(f"  Chunk #{i}: {c_sum}")

    print(f"\n[Final Cohesive Summary]:\n{res_long['summary']}")

    # 3. Generation Controls Comparison
    print("\n\n" + "=" * 70)
    print("--- 3. GENERATION CONTROLS DEMO (Default vs Alternate Settings) ---")
    print("=" * 70)
    sports_file = os.path.join(SAMPLE_DIR, "01_short_sports.txt")
    with open(sports_file, "r", encoding="utf-8") as f:
        sports_text = f.read()

    # Setting A: Default (num_beams=4, max_length=84, length_penalty=1.0)
    res_a = mt5_summary(sports_text, num_beams=4, max_length=84, min_length=30, length_penalty=1.0, no_repeat_ngram_size=2)
    # Setting B: Faster / Concise (num_beams=2, max_length=45, min_length=15, length_penalty=0.8, no_repeat_ngram_size=3)
    res_b = mt5_summary(sports_text, num_beams=2, max_length=45, min_length=15, length_penalty=0.8, no_repeat_ngram_size=3)

    print(f"Article: 01_short_sports.txt ({res_a['input_tokens']} tokens)")
    print(f"\nSetting A (Default: num_beams=4, max_len=84, min_len=30, len_pen=1.0, no_repeat=2):")
    print(f"  Time: {res_a['seconds']}s")
    print(f"  Summary: {res_a['summary']}")

    print(f"\nSetting B (Alternate: num_beams=2, max_len=45, min_len=15, len_pen=0.8, no_repeat=3):")
    print(f"  Time: {res_b['seconds']}s")
    print(f"  Summary: {res_b['summary']}")

    # Save results to JSON
    out_json = os.path.join(os.path.dirname(__file__), "..", "results", "summarization", "sanity_run_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "sample_runs": sample_results,
            "chunked_run": res_long,
            "controls_demo": {
                "setting_a_default": res_a,
                "setting_b_alternate": res_b,
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all sanity run results to: {out_json}")


if __name__ == "__main__":
    main()
