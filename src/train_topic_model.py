"""
MahaVarta: LDA Training and Evaluation Pipeline
===============================================
Executes the full pipeline for Stage 7:
  1. Loads MahaNews TRAINING split only.
  2. Preprocesses text with Indic NLP and Marathi stopwords.
  3. Builds Gensim Dictionary and Bag-of-Words corpus with extreme filtering.
  4. Runs coherence scan over topic counts [6, 8, 10, 12, 14, 16, 18, 20] using C_v.
  5. Selects best topic count and saves coherence table & plot.
  6. Trains final LDA model (10 passes, seed=42) and saves to models/.
  7. Saves topics.json, topic_vs_category.csv, and Devanagari bar chart.
"""

import os
import sys
import json
import tempfile
import logging
import pandas as pd

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.topic_modeling import (
    load_stopwords,
    preprocess_for_lda,
    build_corpus,
    coherence_scan,
    train_lda,
    save_lda_model,
    top_keywords,
    compute_topic_vs_category,
    plot_coherence,
    plot_topic_keywords_barchart,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "mahanews", "train.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "topic_modeling")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    logger.info(f"Loading MahaNews training dataset from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    logger.info(f"Loaded {len(df)} training articles.")

    stopwords = load_stopwords()
    logger.info(f"Loaded {len(stopwords)} Marathi stopwords.")

    # Fixed random subsample of 8,000 articles for coherence scan
    scan_sample_size = 8000
    scan_df_sample = df.sample(n=scan_sample_size, random_state=42)
    logger.info(f"Using fixed subsample of {scan_sample_size} articles (seed=42) for coherence scan and final training...")

    tokenized_scan_docs = [
        preprocess_for_lda(str(text), stopwords=stopwords)
        for text in scan_df_sample["Text"]
    ]
    # Filter out empty docs
    scan_pairs = [
        (doc, cat) for doc, cat in zip(tokenized_scan_docs, scan_df_sample["Label"])
        if len(doc) >= 5
    ]
    tokenized_scan_docs = [p[0] for p in scan_pairs]
    scan_categories = [p[1] for p in scan_pairs]
    logger.info(f"Valid tokenized scan documents (>=5 tokens): {len(tokenized_scan_docs)}")

    dictionary, corpus = build_corpus(
        tokenized_scan_docs,
        no_below=10,
        no_above=0.40,
        keep_n=30000,
    )

    # Coherence scan over 6 to 20 topics with step 2
    topic_candidates = list(range(6, 21, 2))  # [6, 8, 10, 12, 14, 16, 18, 20]
    scan_results = coherence_scan(
        corpus=corpus,
        dictionary=dictionary,
        tokenized_texts=tokenized_scan_docs,
        topic_range=topic_candidates,
        coherence="c_v",
        random_state=42,
        passes=5,
    )

    # Save coherence table and plot
    coherence_csv_path = os.path.join(RESULTS_DIR, "coherence_scan.csv")
    scan_results.to_csv(coherence_csv_path, index=False)
    logger.info(f"Saved coherence scan table to {coherence_csv_path}")

    coherence_plot_path = os.path.join(RESULTS_DIR, "coherence_plot.png")
    plot_coherence(scan_results, coherence_plot_path)

    # Select optimal topic count
    best_row = scan_results.loc[scan_results["coherence_cv"].idxmax()]
    best_k = int(best_row["num_topics"])
    best_coherence = float(best_row["coherence_cv"])

    # If another k is within 0.005 of max coherence and is smaller, prefer the smaller k
    for _, r in scan_results.iterrows():
        k = int(r["num_topics"])
        c = float(r["coherence_cv"])
        if k < best_k and (best_coherence - c) <= 0.005:
            logger.info(f"Topic count {k} (C_v={c:.4f}) is very close to {best_k} (C_v={best_coherence:.4f}); preferring parsimonious model.")
            best_k = k
            best_coherence = c
            break

    logger.info(f"Selected optimal number of topics: K={best_k} with C_v Coherence = {best_coherence:.4f}")

    # Train final LDA model
    logger.info(f"Training final LDA model (K={best_k}, passes=10, seed=42)...")
    final_lda = train_lda(
        corpus=corpus,
        dictionary=dictionary,
        num_topics=best_k,
        passes=10,
        random_state=42,
    )

    # Save final model and dictionary
    save_lda_model(final_lda, dictionary, MODELS_DIR)

    # Save topics.json
    topics_keywords = top_keywords(final_lda, n=10)
    topics_json_path = os.path.join(RESULTS_DIR, "topics.json")
    with open(topics_json_path, "w", encoding="utf-8") as f:
        json.dump(topics_keywords, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved topics keywords to {topics_json_path}")

    # Compute topic-vs-category alignment
    topic_cat_df = compute_topic_vs_category(
        lda=final_lda,
        dictionary=dictionary,
        tokenized_docs=tokenized_scan_docs,
        categories=scan_categories,
    )
    topic_cat_path = os.path.join(RESULTS_DIR, "topic_vs_category.csv")
    topic_cat_df.to_csv(topic_cat_path, index=False)
    logger.info(f"Saved topic vs category alignment to {topic_cat_path}")

    # Plot Devanagari bar chart
    barchart_path = os.path.join(RESULTS_DIR, "topic_keywords_barchart.png")
    plot_topic_keywords_barchart(final_lda, num_topics=best_k, top_n=8, output_path=barchart_path)

    print("\n" + "=" * 60)
    print("STAGE 7 TOPIC MODELLING COMPLETE")
    print("=" * 60)
    print(f"Optimal Topic Count: K = {best_k}")
    print(f"Coherence (C_v): {best_coherence:.4f}")
    print("\nCoherence Scan Table:")
    print(scan_results.to_string(index=False))
    print("\nTopic vs Dominant Category:")
    print(topic_cat_df[["topic_id", "dominant_category", "category_share_pct", "top_keywords"]].to_string(index=False))


if __name__ == "__main__":
    main()
