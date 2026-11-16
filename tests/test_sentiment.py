"""
Unit tests for src/sentiment.py
===============================
Tests cover:
  - Model and tokenizer lazy loading
  - Dynamic id2label mapping from model.config
  - Output format: label, score, all_scores
  - Score labelled as 'model prediction score' (bounded in [0.0, 1.0])
  - Empty and whitespace input handling
  - Long input chunking (> 512 tokens)
  - evaluate_on_annotations refusal on empty/incomplete annotations
"""

import os
import sys
import tempfile
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentiment import load_model, predict, evaluate_on_annotations


def test_lazy_load_model():
    """Tokenizer and model should load and provide dynamic id2label."""
    tokenizer, model = load_model()
    assert tokenizer is not None
    assert model is not None
    assert hasattr(model.config, "id2label")
    assert len(model.config.id2label) == 3


def test_predict_format():
    """Predict should return dict with label, score, and all_scores."""
    text = "भारतीय खेळाडूंनी अतिशय उत्कृष्ट कामगिरी केली आणि विजय मिळवला."
    result = predict(text)

    assert isinstance(result, dict)
    assert "label" in result
    assert "score" in result
    assert "all_scores" in result

    assert isinstance(result["label"], str)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["score"] <= 1.0

    # Label must be one of the keys in all_scores
    assert result["label"] in result["all_scores"]
    assert len(result["all_scores"]) == 3


def test_predict_empty_input():
    """Predict should handle empty input without raising an exception."""
    res_empty = predict("")
    assert "label" in res_empty
    assert res_empty["score"] == 0.0

    res_ws = predict("   \n\t  ")
    assert "label" in res_ws
    assert res_ws["score"] == 0.0


def test_predict_long_input_chunking():
    """Long text exceeding 512 tokens should be chunked and averaged cleanly."""
    base_text = "हा एक अतिशय चांगला आणि सकारात्मक दिवस आहे. खेळाडूंचा उत्साह वाखाणण्याजोगा होता. "
    long_text = base_text * 60  # ~600 words / tokens

    result = predict(long_text)
    assert isinstance(result, dict)
    assert "label" in result
    assert 0.0 <= result["score"] <= 1.0


def test_evaluate_on_annotations_refuses_empty(tmp_path):
    """Evaluation must refuse to run if final_label is empty or incomplete."""
    csv_file = tmp_path / "test_headlines.csv"

    # Create CSV with empty final_label
    df = pd.DataFrame({
        "id": [1, 2],
        "headline": ["बातमी एक", "बातमी दोन"],
        "category": ["Sports", "Politics"],
        "label_vishal": ["", ""],
        "label_gufran": ["", ""],
        "final_label": ["", ""],
    })
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="empty labels|aborted"):
        evaluate_on_annotations(str(csv_file), str(tmp_path))


def test_evaluate_on_annotations_missing_column(tmp_path):
    """Evaluation must refuse if final_label column is missing."""
    csv_file = tmp_path / "missing_col.csv"
    df = pd.DataFrame({"headline": ["बातमी"]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        evaluate_on_annotations(str(csv_file), str(tmp_path))
