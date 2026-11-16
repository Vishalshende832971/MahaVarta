"""
MahaVarta: Marathi Text Preprocessing Module
=============================================
This module handles all text preprocessing for Marathi news articles:
  - Unicode normalisation using the Indic NLP Library
  - Cleaning (remove HTML tags, URLs, extra whitespace)
  - Tokenisation using the Indic NLP tokenizer
  - Reading uploaded .txt files with UTF-8 decoding

Two text versions are maintained:
  - "original": untouched text for transformer models (they have their own tokenisers)
  - "cleaned": processed text for classical models (TF-IDF, LDA, TextRank)
"""

import re
import os

# ---------------------------------------------------------------------------
# Indic NLP Library setup
# The library needs its resources path set before importing processing modules.
# ---------------------------------------------------------------------------
from indicnlp import common as indic_common
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Try to find the Indic NLP resources directory.
# The indic_nlp_library pip package bundles resources in some versions,
# but if not found, the normalizer can still work without external resources
# for basic normalization.
_INDIC_RESOURCES_PATH = os.environ.get("INDIC_RESOURCES_PATH", "")
if _INDIC_RESOURCES_PATH:
    indic_common.set_resources_path(_INDIC_RESOURCES_PATH)

# Create a normalizer for Marathi ("mr")
_normalizer_factory = IndicNormalizerFactory()
_normalizer = _normalizer_factory.get_normalizer("mr")


def normalize_text(text: str) -> str:
    """
    Normalise Marathi (Devanagari) text using the Indic NLP Library.

    This ensures that visually identical Unicode characters share a single
    representation. For example, different ways of encoding the same
    Devanagari character are unified.

    Args:
        text: Raw Marathi text string.

    Returns:
        Normalised text string.
    """
    if not text or not text.strip():
        return ""
    return _normalizer.normalize(text)


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, URLs and extra whitespace.
    Keeps Devanagari characters and the Devanagari danda (।).

    Steps:
      1. Remove HTML tags (e.g. <p>, <br/>, <div>)
      2. Remove URLs (http, https, www)
      3. Collapse multiple whitespace characters into a single space
      4. Strip leading and trailing whitespace

    Args:
        text: Input text (may contain HTML, URLs, etc.)

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # Step 1: Remove HTML tags
    # Matches anything between < and >, including self-closing tags
    text = re.sub(r"<[^>]+>", "", text)

    # Step 2: Remove URLs
    # Matches http://, https://, and www. patterns
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Step 3: Collapse multiple whitespace (spaces, tabs, newlines) into one space
    text = re.sub(r"\s+", " ", text)

    # Step 4: Strip leading and trailing whitespace
    text = text.strip()

    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenise Marathi text into word tokens using the Indic NLP tokenizer.

    The Indic NLP tokenizer is designed for Indian languages and handles
    Devanagari script correctly, including splitting on punctuation like
    the Devanagari danda (।).

    Args:
        text: Cleaned Marathi text string.

    Returns:
        List of word tokens.
    """
    if not text or not text.strip():
        return []

    # indic_tokenize.trivial_tokenize splits on whitespace and punctuation
    # appropriate for Indian languages
    tokens = indic_tokenize.trivial_tokenize(text, lang="mr")

    return tokens


def preprocess(text: str) -> dict:
    """
    Run the full preprocessing pipeline on a Marathi text.

    Returns a dictionary with:
      - "original": the untouched input text (for transformer models)
      - "normalized": text after Unicode normalisation
      - "cleaned": text after normalisation + cleaning
      - "tokens": list of word tokens from the cleaned text

    Args:
        text: Raw input Marathi text.

    Returns:
        Dictionary with keys: original, normalized, cleaned, tokens
    """
    # Keep the original text untouched for transformer models
    original = text if text else ""

    # Step 1: Unicode normalisation
    normalized = normalize_text(original)

    # Step 2: Clean the normalised text
    cleaned = clean_text(normalized)

    # Step 3: Tokenise the cleaned text
    tokens = tokenize(cleaned)

    return {
        "original": original,
        "normalized": normalized,
        "cleaned": cleaned,
        "tokens": tokens,
    }


def read_uploaded_txt(path: str) -> str:
    """
    Read a .txt file uploaded by the user.

    The file must be UTF-8 encoded (standard for Marathi/Devanagari text).
    Returns a friendly error message if the file cannot be read.

    Args:
        path: File path to the .txt file.

    Returns:
        The text content of the file.

    Raises:
        ValueError: If the file cannot be decoded as UTF-8 or does not exist.
    """
    # Check if file exists
    if not os.path.exists(path):
        raise ValueError(f"फाइल सापडली नाही (File not found): {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        raise ValueError(
            "ही फाइल UTF-8 स्वरूपात नाही. "
            "कृपया UTF-8 एन्कोडिंग असलेली .txt फाइल अपलोड करा। "
            "(This file is not UTF-8 encoded. "
            "Please upload a .txt file with UTF-8 encoding.)"
        )
    except Exception as e:
        raise ValueError(
            f"फाइल वाचताना त्रुटी (Error reading file): {str(e)}"
        )
