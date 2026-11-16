"""
Unit tests for src/topic_modeling.py
====================================
Tests cover:
  - Stopwords loading from data/stopwords_mr.txt
  - Text preprocessing for LDA (Devanagari filtering, length threshold, stopwords)
  - Handling of empty, whitespace, and non-Devanagari inputs
  - Corpus and dictionary construction with frequency filtering
  - article_topics() output schema, edge cases (empty input, OOV words)
  - Topic weights sum to approximately 1.0
"""

import os
import sys
import pytest
from gensim.corpora import Dictionary
from gensim.models import LdaModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.topic_modeling import (
    load_stopwords,
    preprocess_for_lda,
    build_corpus,
    top_keywords,
    article_topics,
)


def test_load_stopwords():
    """Stopwords should load as a non-empty set containing Marathi words."""
    sw = load_stopwords()
    assert isinstance(sw, set)
    assert len(sw) > 50
    assert "आणि" in sw
    assert "आहे" in sw


def test_preprocess_for_lda_filtering():
    """Preprocessing should filter out punctuation, short words, English, and stopwords."""
    text = "भारताने क्रिकेट सामन्यात उत्कृष्ट खेळ केला आणि सामना जिंकला. http://news.com"
    tokens = preprocess_for_lda(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # Stopwords should not be present
    assert "आणि" not in tokens
    assert "केला" not in tokens
    # Short words (<3 chars) and URLs should not be present
    for tok in tokens:
        assert len(tok) >= 3
        # Must be Devanagari script only
        assert all("\u0900" <= c <= "\u097F" for c in tok)


def test_preprocess_for_lda_empty_inputs():
    """Empty or non-Devanagari inputs should return empty token list."""
    assert preprocess_for_lda("") == []
    assert preprocess_for_lda("   \n\t  ") == []
    assert preprocess_for_lda(None) == []
    assert preprocess_for_lda("12345 !@#$% English only text") == []


def test_build_corpus():
    """Corpus builder should return Gensim Dictionary and BoW corpus."""
    docs = [
        ["भारत", "क्रिकेट", "सामना", "खेळाडू", "विजय"],
        ["निवडणूक", "मतदान", "पक्ष", "उमेदवार", "विजय"],
        ["आरोग्य", "रुग्ण", "डॉक्टर", "औषध", "उपचार"],
    ]
    # Build with no_below=1 for toy test
    dictionary, corpus = build_corpus(docs, no_below=1, no_above=1.0)

    assert isinstance(dictionary, Dictionary)
    assert len(dictionary) > 0
    assert len(corpus) == len(docs)
    assert all(isinstance(bow, list) for bow in corpus)


def test_article_topics_and_weight_sum():
    """article_topics should return sorted (topic_id, prob) pairs summing to ~1.0."""
    docs = [
        ["भारत", "क्रिकेट", "सामना", "खेळाडू", "विजय"],
        ["निवडणूक", "मतदान", "पक्ष", "उमेदवार", "विजय"],
        ["आरोग्य", "रुग्ण", "डॉक्टर", "औषध", "उपचार"],
    ]
    dictionary, corpus = build_corpus(docs, no_below=1, no_above=1.0)
    lda = LdaModel(corpus, num_topics=3, id2word=dictionary, random_state=42, passes=3)

    query_tokens = ["क्रिकेट", "खेळाडू", "सामना"]
    topics = article_topics(lda, dictionary, query_tokens, min_prob=0.0)

    assert isinstance(topics, list)
    assert len(topics) == 3

    # Check structure
    for t_id, prob in topics:
        assert isinstance(t_id, int)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    # Probabilities should sum to approximately 1.0
    total_prob = sum(prob for _, prob in topics)
    assert abs(total_prob - 1.0) < 0.05

    # Should be sorted descending by probability
    probs = [p for _, p in topics]
    assert probs == sorted(probs, reverse=True)


def test_article_topics_edge_cases():
    """Edge cases: empty tokens, OOV tokens, None models should not crash."""
    docs = [["भारत", "क्रिकेट"]]
    dct, corpus = build_corpus(docs, no_below=1, no_above=1.0)
    lda = LdaModel(corpus, num_topics=2, id2word=dct, random_state=42, passes=2)

    # Empty tokens
    assert article_topics(lda, dct, []) == []

    # Out of vocabulary tokens
    assert article_topics(lda, dct, ["अज्ञातशब्द", "दुसराशब्द"]) == []

    # None model or None dictionary
    assert article_topics(None, dct, ["भारत"]) == []
    assert article_topics(lda, None, ["भारत"]) == []


def test_top_keywords():
    """top_keywords should return dict mapping topic_id to word-weight dicts."""
    docs = [["भारत", "क्रिकेट", "सामना", "खेळ"]]
    dct, corpus = build_corpus(docs, no_below=1, no_above=1.0)
    lda = LdaModel(corpus, num_topics=2, id2word=dct, random_state=42, passes=2)

    keywords = top_keywords(lda, n=2)
    assert isinstance(keywords, dict)
    assert len(keywords) == 2
    for t_id in keywords:
        assert isinstance(keywords[t_id], list)
        if len(keywords[t_id]) > 0:
            item = keywords[t_id][0]
            assert "word" in item
            assert "weight" in item
            assert isinstance(item["word"], str)
            assert isinstance(item["weight"], float)
