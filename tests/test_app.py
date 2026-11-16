"""
Test the Gradio analysis pipeline in app.py
===========================================
Tests analyze_article() on all 3 sample articles in data/sample_articles/:
  1. 01_short_sports.txt
  2. 02_medium_health.txt
  3. 03_long_politics.txt
Verifies that every tab (including Sentiment, NER, and Topics) produces valid outputs.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import analyze_article, load_sample_file


def test_analyze_short_sports_sample():
    """Test short sports sample article across all 7 tabs."""
    text = load_sample_file("01_short_sports.txt")
    assert len(text) > 0

    results = analyze_article(text, None, 2)
    (
        overview,
        clf_df,
        summary,
        sent_label_md,
        sent_scores_df,
        ner_highlight,
        ner_grouped_md,
        topics_df,
        orig,
        norm,
        clean,
        tokens,
        tfidf_df,
    ) = results

    # Tab 1: Overview
    assert "क्रीडा" in overview or "Sports" in overview
    assert "भावना" in overview

    # Tab 2: Classification
    assert not clf_df.empty
    assert "क्रीडा" in str(clf_df.values) or "Sports" in str(clf_df.values)

    # Tab 3: Summary
    assert len(summary) > 0

    # Tab 4: Sentiment
    assert "मुख्य भावना" in sent_label_md
    assert not sent_scores_df.empty
    assert len(sent_scores_df) == 3

    # Tab 5: NER
    assert isinstance(ner_highlight, list)
    assert len(ner_highlight) > 0
    assert isinstance(ner_grouped_md, str)

    # Tab 6: Topic Modelling
    assert not topics_df.empty
    assert "विषय" in str(topics_df.columns)

    # Tab 7: Preprocessing
    assert len(clean) > 0
    assert len(tokens) > 0
    assert not tfidf_df.empty


def test_analyze_medium_health_sample():
    """Test medium health sample article across all 7 tabs."""
    text = load_sample_file("02_medium_health.txt")
    assert len(text) > 0

    results = analyze_article(text, None, 3)
    (
        overview,
        clf_df,
        summary,
        sent_label_md,
        sent_scores_df,
        ner_highlight,
        ner_grouped_md,
        topics_df,
        orig,
        norm,
        clean,
        tokens,
        tfidf_df,
    ) = results

    assert not clf_df.empty
    assert "आरोग्य" in str(clf_df.values) or "Health" in str(clf_df.values)
    assert len(summary) > 0
    assert not sent_scores_df.empty
    assert isinstance(ner_highlight, list)
    assert not topics_df.empty
    assert not tfidf_df.empty


def test_analyze_long_politics_sample():
    """Test long politics sample article across all 7 tabs."""
    text = load_sample_file("03_long_politics.txt")
    assert len(text) > 0

    results = analyze_article(text, None, 3)
    (
        overview,
        clf_df,
        summary,
        sent_label_md,
        sent_scores_df,
        ner_highlight,
        ner_grouped_md,
        topics_df,
        orig,
        norm,
        clean,
        tokens,
        tfidf_df,
    ) = results

    assert not clf_df.empty
    assert "राजकारण" in str(clf_df.values) or "Politics" in str(clf_df.values)
    assert len(summary) > 0
    assert not sent_scores_df.empty
    assert isinstance(ner_highlight, list)
    assert not topics_df.empty
    assert not tfidf_df.empty


def test_empty_input_handling():
    """Test that empty input is handled cleanly without exceptions."""
    results = analyze_article("", None, 3)
    (
        overview,
        clf_df,
        summary,
        sent_label_md,
        sent_scores_df,
        ner_highlight,
        ner_grouped_md,
        topics_df,
        orig,
        norm,
        clean,
        tokens,
        tfidf_df,
    ) = results

    assert "कृपया मजकूर प्रविष्ट करा" in overview or "कृपया" in str(clf_df.values)
    assert ner_highlight == []
    assert not topics_df.empty or "विषय" in str(topics_df.columns)
