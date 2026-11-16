"""
Unit tests for src/ner.py
=========================
Tests cover:
  - Model lazy loading and supported labels mapping
  - extract_entities() output schema: text, label, score, start, end
  - Non-entity filtering ('Other', 'O' filtered out)
  - Empty and whitespace input handling
  - group_by_label() dictionary grouping and deduplication
  - Long text chunking at word boundaries without errors
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ner import load_model, get_supported_labels, extract_entities, group_by_label


def test_lazy_load_and_supported_labels():
    """NER pipeline should load and return dynamic id2label mapping."""
    labels = get_supported_labels()
    assert isinstance(labels, dict)
    assert len(labels) >= 7
    # Verify expected label categories are present in values
    label_values = set(labels.values())
    expected_categories = {"Person", "Location", "Organization", "Designation", "Date", "Time", "Measure"}
    assert expected_categories.issubset(label_values)


def test_extract_entities_format():
    """Extract entities should return dicts with text, label, score, start, end."""
    sample_text = "पंतप्रधान नरेंद्र मोदी यांनी नवी दिल्ली येथे बैठक घेतली."
    entities = extract_entities(sample_text)

    assert isinstance(entities, list)
    if len(entities) > 0:
        ent = entities[0]
        assert "text" in ent
        assert "label" in ent
        assert "score" in ent
        assert "start" in ent
        assert "end" in ent

        assert isinstance(ent["text"], str)
        assert isinstance(ent["label"], str)
        assert isinstance(ent["score"], float)
        assert 0.0 <= ent["score"] <= 1.0
        assert ent["start"] >= 0
        assert ent["end"] > ent["start"]
        # 'Other' or 'O' must never be emitted as an entity
        assert ent["label"] not in ("Other", "O", "LABEL_0")


def test_extract_entities_empty_input():
    """Extract entities should return [] for empty or whitespace-only inputs."""
    assert extract_entities("") == []
    assert extract_entities("   \n\t  ") == []
    assert extract_entities(None) == []


def test_group_by_label():
    """group_by_label should group entities by category and deduplicate."""
    mock_entities = [
        {"text": "नरेंद्र मोदी", "label": "Person", "score": 0.99, "start": 0, "end": 11},
        {"text": "नवी दिल्ली", "label": "Location", "score": 0.95, "start": 20, "end": 29},
        {"text": "नरेंद्र मोदी", "label": "Person", "score": 0.98, "start": 50, "end": 61},
        {"text": "मुंबई", "label": "Location", "score": 0.94, "start": 70, "end": 75},
    ]

    grouped = group_by_label(mock_entities)
    assert isinstance(grouped, dict)
    assert "Person" in grouped
    assert "Location" in grouped

    # Check deduplication
    assert grouped["Person"] == ["नरेंद्र मोदी"]
    assert grouped["Location"] == ["नवी दिल्ली", "मुंबई"]

    # Empty list should give empty dict
    assert group_by_label([]) == {}


def test_extract_entities_long_text():
    """Long text exceeding token limit should be processed in chunks without crash."""
    sentence = "सचिन तेंडुलकर यांनी मुंबई येथे पत्रकार परिषद घेतली. "
    long_text = sentence * 40  # ~320 words

    entities = extract_entities(long_text)
    assert isinstance(entities, list)
    assert len(entities) > 0
    # Spans should be ascending
    starts = [e["start"] for e in entities]
    assert starts == sorted(starts)
