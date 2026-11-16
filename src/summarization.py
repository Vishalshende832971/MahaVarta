"""
MahaVarta: Marathi Summarization Module (TextRank Baseline)
===========================================================
This module provides extractive summarisation for Marathi text using TextRank:
  - Marathi sentence segmentation aware of the Devanagari danda (।),
    full stops, question/exclamation marks, decimals, and abbreviations.
  - TF-IDF sentence vectorisation using the Devanagari Indic tokenizer.
  - Cosine similarity graph construction.
  - PageRank centrality scoring (via NetworkX with power-iteration fallback).
  - Chronological sentence reordering so summaries read smoothly.
  - Graceful edge-case handling (empty text, single sentence, fewer sentences).

Note: Pretrained mT5 abstractive summarisation will be added in Stage 8.
"""

import re
import logging
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import tokenize, clean_text

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """
    Split Marathi text into individual sentences.

    Sentence boundaries considered:
      - Devanagari danda (।) and double danda (॥)
      - Question mark (?) and exclamation mark (!)
      - Full stops (.) followed by space or newline
      - Newline boundaries (\n)

    Protections:
      - Decimal numbers (e.g., 3.14, ७.५) are NOT split.
      - Common Marathi abbreviations (डॉ., प्रा., श्री., सौ., इ., क्र., ता., दि.)
        are NOT treated as sentence boundaries.

    Args:
        text: Raw or cleaned Marathi text.

    Returns:
        List of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    processed = text.strip()

    # Step 1: Protect decimal numbers (both ASCII digits and Devanagari digits)
    # E.g., 3.14 -> 3<DECIMAL>14, ७.५ -> ७<DECIMAL>५
    processed = re.sub(r"([0-9०-९]+)\.([0-9०-९]+)", r"\1<DECIMAL>\2", processed)

    # Step 2: Protect common Marathi abbreviations ending with a dot
    abbreviations = [
        r"डॉ\.",    # Doctor
        r"प्रा\.",   # Professor
        r"श्री\.",   # Shri
        r"सौ\.",    # Sau
        r"इ\.",     # etc.
        r"क्र\.",    # No.
        r"ता\.",    # Taluka / Date
        r"दि\.",    # Date
        r"ना\.",    # Honorable
    ]
    for abbrev_pattern in abbreviations:
        # Replace the dot in abbreviation with a placeholder
        processed = re.sub(
            abbrev_pattern,
            lambda m: m.group(0)[:-1] + "<ABBDOT>",
            processed,
        )

    # Step 3: Split on sentence terminal characters:
    # - Devanagari danda (।) and double danda (॥)
    # - Question mark (?) and exclamation mark (!)
    # - Line breaks (\n)
    # - Periods (.) not inside decimals/abbreviations
    raw_sentences = re.split(r"[।॥!?\n]+|(?<=\s)\.(?=\s|$)|(?<=[^\d])\.(?=\s|$)", processed)

    # Step 4: Restore protected characters, trim, and filter out empty segments
    sentences = []
    for s in raw_sentences:
        restored = s.replace("<DECIMAL>", ".").replace("<ABBDOT>", ".").strip()
        # Clean any leftover punctuation or noise
        cleaned_sentence = re.sub(r"^\s*[,;:\-\.]+\s*", "", restored).strip()
        if cleaned_sentence and len(cleaned_sentence) > 1:
            sentences.append(cleaned_sentence)

    return sentences


def textrank_summary(text: str, n_sentences: int = 3) -> str:
    """
    Generate an extractive summary of a Marathi news article using TextRank.

    Algorithm:
      1. Segment text into sentences using Marathi-aware boundaries.
      2. If total sentences <= n_sentences, return all sentences in order.
      3. Vectorize sentences using TF-IDF with the Devanagari tokenizer.
      4. Compute pairwise cosine similarity between sentence vectors.
      5. Construct a weighted graph where edges represent sentence similarity.
      6. Run PageRank on the graph to rank sentences by centrality.
      7. Select top n_sentences and reorder them chronologically.
      8. Return the joined summary text with proper Devanagari danda punctuation.

    Args:
        text: Input Marathi article string.
        n_sentences: Number of summary sentences to extract (default: 3).

    Returns:
        Summary string, or empty string if input is empty.
    """
    if not text or not text.strip():
        return ""

    # Segment text into sentences
    sentences = split_sentences(text)

    # Edge cases:
    # 0 sentences -> return empty
    if not sentences:
        return ""

    # Fewer or equal sentences than requested -> return all sentences
    if len(sentences) <= n_sentences:
        return " । ".join(sentences) + " ।"

    # Build TF-IDF sentence matrix
    # Use the Devanagari-aware tokenizer from preprocessing
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except Exception as e:
        logger.warning(f"TF-IDF vectorisation failed for sentences: {e}")
        # Fallback: return first n_sentences
        return " । ".join(sentences[:n_sentences]) + " ।"

    # Compute pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Set diagonal to zero to eliminate self-loops in the graph
    np.fill_diagonal(similarity_matrix, 0.0)

    # If all similarities are zero (completely disjoint sentences),
    # fallback to leading sentences
    if np.all(similarity_matrix == 0):
        return " । ".join(sentences[:n_sentences]) + " ।"

    # Build NetworkX graph from similarity matrix
    try:
        graph = nx.from_numpy_array(similarity_matrix)
        # Compute PageRank scores (damping factor 0.85, standard for TextRank)
        scores = nx.pagerank(graph, alpha=0.85, max_iter=200, tol=1e-6)
    except Exception as e:
        logger.warning(f"NetworkX PageRank failed: {e}. Falling back to degree centrality.")
        # Fallback: sum of similarities (degree)
        degrees = similarity_matrix.sum(axis=1)
        scores = {i: float(deg) for i, deg in enumerate(degrees)}

    # Rank sentence indices by descending centrality score
    ranked_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)

    # Pick top n_sentences indices
    top_indices = ranked_indices[:n_sentences]

    # Reorder chronologically (by their original appearance in text)
    top_indices_sorted = sorted(top_indices)

    # Extract sentences and format with Devanagari danda
    summary_sentences = [sentences[idx] for idx in top_indices_sorted]
    summary_text = " । ".join(summary_sentences) + " ।"

    return summary_text
