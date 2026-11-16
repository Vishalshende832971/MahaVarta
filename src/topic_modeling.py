"""
MahaVarta: Marathi News Topic Modelling Module (LDA)
====================================================
This module implements Latent Dirichlet Allocation (LDA) topic modelling
on Marathi news text using Gensim:
  - Text normalization, cleaning, Devanagari tokenization, and stopword filtering.
  - Dictionary creation with frequency boundary filtering (filter_extremes).
  - Coherence scan across a range of topic counts (C_v metric).
  - Final LDA model training and persistence.
  - Top topic keywords extraction and topic-vs-category alignment.
  - Article topic inference for new texts with robust edge-case handling.
"""

import os
import re
import json
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

from src.preprocessing import clean_text, normalize_text, tokenize

logger = logging.getLogger(__name__)

STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stopwords_mr.txt")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LDA_MODEL_FILE = os.path.join(MODEL_DIR, "lda_model.gensim")
LDA_DICT_FILE = os.path.join(MODEL_DIR, "lda_dictionary.gensim_dict")

# Lazy model cache
_cached_lda = None
_cached_dictionary = None
_cached_stopwords = None


def get_devanagari_font() -> str:
    """Find an installed font that correctly renders Devanagari text on Windows."""
    preferred = ["Nirmala UI", "Nirmala Text", "Mangal", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            return font
    return "sans-serif"


def load_stopwords(filepath: str = None) -> set[str]:
    """
    Load Marathi stopwords from data/stopwords_mr.txt.
    Caches the stopword set in memory after first load.
    """
    global _cached_stopwords
    if _cached_stopwords is not None and filepath is None:
        return _cached_stopwords

    path = filepath or STOPWORDS_PATH
    stopwords = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    else:
        logger.warning(f"Stopwords file not found at {path}. Using default fallback.")
        stopwords = {"आहे", "आहेत", "आणि", "किंवा", "या", "हे", "ती", "तो", "ते", "हा", "ही"}

    if filepath is None:
        _cached_stopwords = stopwords
    return stopwords


def preprocess_for_lda(text: str, stopwords: set[str] = None) -> list[str]:
    """
    Preprocess Marathi text specifically for LDA topic modelling:
      1. Unicode normalization (Indic NLP).
      2. Cleaning (remove HTML, URLs, keep Marathi text).
      3. Tokenization into Marathi words.
      4. Remove punctuation, symbols, non-Devanagari tokens, and tokens with length < 3.
      5. Remove domain-specific and general Marathi stopwords.

    Args:
        text: Raw Marathi text string.
        stopwords: Optional custom set of stopwords.

    Returns:
        List of informative Marathi word tokens.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    if stopwords is None:
        stopwords = load_stopwords()

    # Step 1: Normalize and clean
    norm_text = normalize_text(text)
    cleaned = clean_text(norm_text)

    # Step 2: Tokenize using Indic tokenizer
    raw_tokens = tokenize(cleaned)

    # Step 3: Filter tokens
    # Retain tokens that have at least 3 Devanagari characters and are not stopwords
    devanagari_pattern = re.compile(r"^[\u0900-\u097F]+$")
    filtered_tokens = []

    for tok in raw_tokens:
        tok = tok.strip()
        # Discard tokens shorter than 3 characters or non-Devanagari scripts
        if len(tok) < 3:
            continue
        if not devanagari_pattern.match(tok):
            continue
        if tok in stopwords:
            continue
        filtered_tokens.append(tok)

    return filtered_tokens


def build_corpus(
    tokenized_docs: list[list[str]],
    no_below: int = 15,
    no_above: float = 0.40,
    keep_n: int = 50000,
) -> tuple[Dictionary, list[list[tuple[int, int]]]]:
    """
    Build Gensim Dictionary and Bag-of-Words (BoW) corpus from tokenized documents.
    Applies filter_extremes to drop rare words and overly frequent generic words.

    Args:
        tokenized_docs: List of token lists for training documents.
        no_below: Minimum document frequency threshold (drop words appearing in fewer docs).
        no_above: Maximum document frequency fraction threshold (drop words appearing in > X% docs).
        keep_n: Max vocabulary size to retain after filtering.

    Returns:
        Tuple of (Gensim Dictionary, BoW Corpus).
    """
    dictionary = Dictionary(tokenized_docs)
    logger.info(f"Raw vocabulary size before filtering: {len(dictionary)}")

    # Filter out extremely rare and overly frequent tokens
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    logger.info(f"Filtered vocabulary size: {len(dictionary)}")

    # Build BoW representation for every document
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    return dictionary, corpus


def coherence_scan(
    corpus: list,
    dictionary: Dictionary,
    tokenized_texts: list[list[str]],
    topic_range = range(6, 21, 2),
    coherence: str = "c_v",
    random_state: int = 42,
    passes: int = 5,
) -> pd.DataFrame:
    """
    Train LDA models over a range of topic counts and evaluate coherence (C_v).

    Args:
        corpus: BoW corpus for training documents.
        dictionary: Gensim Dictionary mapping term IDs to words.
        tokenized_texts: Raw token lists for C_v sliding window coherence calculation.
        topic_range: Iterable of topic counts to scan (e.g. range(6, 21, 2)).
        coherence: Coherence metric ('c_v' preferred, 'u_mass' alternative).
        random_state: Fixed random seed for reproducible initialization.
        passes: Number of training passes over corpus per model.

    Returns:
        DataFrame with columns ['num_topics', 'coherence_cv'].
    """
    results = []
    logger.info(f"Starting coherence scan for topics: {list(topic_range)} using metric: {coherence}")

    for k in topic_range:
        logger.info(f"Training LDA with k={k} topics...")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=random_state,
            passes=passes,
            alpha="auto",
            eta="auto",
        )
        cm = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence=coherence,
        )
        score = float(cm.get_coherence())
        logger.info(f"Topics k={k} -> {coherence} Coherence: {score:.4f}")
        results.append({"num_topics": int(k), "coherence_cv": round(score, 4)})

    return pd.DataFrame(results)


def train_lda(
    corpus: list,
    dictionary: Dictionary,
    num_topics: int,
    passes: int = 10,
    random_state: int = 42,
) -> LdaModel:
    """
    Train final Latent Dirichlet Allocation (LDA) model.

    Args:
        corpus: BoW training corpus.
        dictionary: Filtered Gensim Dictionary.
        num_topics: Chosen optimal number of topics.
        passes: Training iterations over full corpus.
        random_state: Random seed for reproducibility.

    Returns:
        Trained Gensim LdaModel instance.
    """
    logger.info(f"Training final LDA model with {num_topics} topics, {passes} passes...")
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        alpha="auto",
        eta="auto",
    )
    logger.info("Final LDA model training complete.")
    return lda


def save_lda_model(lda: LdaModel, dictionary: Dictionary, model_dir: str = MODEL_DIR) -> None:
    """Save trained LDA model and Gensim dictionary to disk."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lda_model.gensim")
    dict_path = os.path.join(model_dir, "lda_dictionary.gensim_dict")

    lda.save(model_path)
    dictionary.save(dict_path)
    logger.info(f"Saved LDA model to {model_path} and dictionary to {dict_path}")


def load_lda_model(model_dir: str = MODEL_DIR) -> tuple[LdaModel | None, Dictionary | None]:
    """
    Load trained LDA model and Gensim dictionary lazily with memory caching.

    Returns:
        Tuple of (LdaModel, Dictionary) or (None, None) if files not found.
    """
    global _cached_lda, _cached_dictionary
    if _cached_lda is not None and _cached_dictionary is not None:
        return _cached_lda, _cached_dictionary

    model_path = os.path.join(model_dir, "lda_model.gensim")
    dict_path = os.path.join(model_dir, "lda_dictionary.gensim_dict")

    if os.path.exists(model_path) and os.path.exists(dict_path):
        logger.info(f"Loading LDA model from {model_path}...")
        _cached_lda = LdaModel.load(model_path)
        _cached_dictionary = Dictionary.load(dict_path)
        return _cached_lda, _cached_dictionary

    logger.warning(f"LDA model files not found in {model_dir}")
    return None, None


def top_keywords(lda: LdaModel, n: int = 10) -> dict[int, list[dict]]:
    """
    Extract top N keywords and weights for each discovered topic.

    Args:
        lda: Trained Gensim LdaModel.
        n: Number of top keywords per topic.

    Returns:
        dict mapping topic_id (int) -> list of {"word": str, "weight": float}.
    """
    topic_dict = {}
    for topic_id in range(lda.num_topics):
        terms = lda.show_topic(topic_id, topn=n)
        topic_dict[topic_id] = [
            {"word": str(word), "weight": round(float(weight), 4)}
            for word, weight in terms
        ]
    return topic_dict


def article_topics(
    lda: LdaModel,
    dictionary: Dictionary,
    tokens: list[str],
    min_prob: float = 0.01,
) -> list[tuple[int, float]]:
    """
    Infer topic distribution for a new article from its preprocessed tokens.

    Args:
        lda: Trained Gensim LdaModel.
        dictionary: Gensim Dictionary used during training.
        tokens: Preprocessed token list of the new article.
        min_prob: Minimum probability threshold for a topic to be included.

    Returns:
        Sorted list of (topic_id, probability) descending by probability.
        If input is empty or contains no vocabulary words, returns safe empty list.
    """
    if lda is None or dictionary is None or not tokens:
        return []

    bow = dictionary.doc2bow(tokens)
    if not bow:
        return []

    topic_dist = lda.get_document_topics(bow, minimum_probability=min_prob)
    # Sort topics descending by probability
    sorted_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)
    return [(int(t), round(float(p), 4)) for t, p in sorted_topics]


def compute_topic_vs_category(
    lda: LdaModel,
    dictionary: Dictionary,
    tokenized_docs: list[list[str]],
    categories: list[str],
) -> pd.DataFrame:
    """
    Compute which of the 12 MahaNews categories each topic is most strongly aligned with.

    Args:
        lda: Trained Gensim LdaModel.
        dictionary: Gensim Dictionary.
        tokenized_docs: Preprocessed document tokens.
        categories: Corresponding true category labels for each document.

    Returns:
        DataFrame summarizing Topic ID, Top Category, Association Share (%), and Top Keywords.
    """
    logger.info("Computing topic-vs-category cross-tabulation...")
    records = []

    for doc, cat in zip(tokenized_docs, categories):
        bow = dictionary.doc2bow(doc)
        if not bow:
            continue
        dist = lda.get_document_topics(bow)
        if dist:
            dominant_topic = max(dist, key=lambda x: x[1])[0]
            records.append({"topic_id": dominant_topic, "category": cat})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    contingency = pd.crosstab(df["topic_id"], df["category"])

    summary_rows = []
    for topic_id in range(lda.num_topics):
        if topic_id in contingency.index:
            row = contingency.loc[topic_id]
            top_cat = row.idxmax()
            top_count = int(row.max())
            total_count = int(row.sum())
            share = round((top_count / total_count) * 100, 2) if total_count > 0 else 0.0
        else:
            top_cat = "N/A"
            top_count = 0
            total_count = 0
            share = 0.0

        top_words = [w for w, _ in lda.show_topic(topic_id, topn=5)]
        summary_rows.append({
            "topic_id": topic_id,
            "dominant_category": top_cat,
            "category_share_pct": share,
            "topic_docs_count": total_count,
            "top_keywords": ", ".join(top_words),
        })

    return pd.DataFrame(summary_rows)


def plot_coherence(
    scan_df: pd.DataFrame,
    output_path: str = "results/topic_modeling/coherence_plot.png",
) -> None:
    """Plot coherence score vs topic count line chart and save to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(scan_df["num_topics"], scan_df["coherence_cv"], marker="o", color="#0066cc", linewidth=2.2, markersize=7)

    # Highlight optimal point
    best_idx = scan_df["coherence_cv"].idxmax()
    best_k = scan_df.loc[best_idx, "num_topics"]
    best_c = scan_df.loc[best_idx, "coherence_cv"]
    plt.scatter([best_k], [best_c], color="#d9534f", s=130, zorder=5, label=f"Optimal K={best_k} (C_v={best_c:.4f})")

    plt.title("LDA Topic Coherence Scan ($C_v$) across Topic Counts", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Topics ($K$)", fontsize=11)
    plt.ylabel("Coherence Score ($C_v$)", fontsize=11)
    plt.xticks(scan_df["num_topics"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved coherence plot to {output_path}")


def plot_topic_keywords_barchart(
    lda: LdaModel,
    num_topics: int,
    top_n: int = 8,
    output_path: str = "results/topic_modeling/topic_keywords_barchart.png",
) -> None:
    """
    Plot horizontal bar charts of top keywords for each topic.
    Uses Devanagari font (Nirmala UI) to ensure proper glyph rendering.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    devanagari_font = get_devanagari_font()

    # Determine grid layout
    cols = 3 if num_topics >= 9 else 2
    rows = (num_topics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharey=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for topic_id in range(num_topics):
        ax = axes[topic_id]
        terms = lda.show_topic(topic_id, topn=top_n)
        words = [t[0] for t in reversed(terms)]
        weights = [float(t[1]) for t in reversed(terms)]

        y_pos = range(len(words))
        bars = ax.barh(y_pos, weights, color="#2b7bba", alpha=0.85, edgecolor="#1c5583")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontfamily=devanagari_font, fontsize=11)
        ax.set_title(f"Topic {topic_id}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Term Weight", fontsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Hide unused subplots
    for j in range(num_topics, len(axes)):
        axes[j].axis("off")

    plt.suptitle("MahaVarta: Top Keywords per Discovered Topic (LDA)", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved topic keywords bar chart to {output_path}")
