"""
Unit tests for src/classification.py
====================================
Tests cover:
  - Pipeline creation and training on small dummy dataset
  - predict() function returning top-k categories with scores
  - predict() handling empty or whitespace inputs
  - save() and load() persistence with joblib
  - evaluate() returning all required metrics
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification import train, evaluate, predict, save, load, CATEGORIES


@pytest.fixture
def dummy_data():
    """Small Marathi dummy corpus for fast testing."""
    train_texts = [
        "भारतीय क्रिकेट संघाने सामना जिंकला",
        "क्रिकेट खेळात फलंदाजाने शतक झळकावले",
        "निवडणुकीत राजकीय पक्षांची चुरस वाढली",
        "मंत्रिमंडळाची महत्त्वाची बैठक आज पार पडली",
        "आरोग्यासाठी दररोज व्यायाम आवश्यक आहे",
        "हॉस्पिटलमध्ये नवीन औषध उपलब्ध झाले आहे",
    ]
    train_labels = ["Sports", "Sports", "Politics", "Politics", "Health", "Health"]

    val_texts = [
        "खेळाडूने उत्तम कामगिरी केली",
        "मुख्यमंत्र्यांनी नवीन धोरण जाहीर केले",
        "डॉक्टरांनी रुग्णांना काळजी घेण्यास सांगितले",
    ]
    val_labels = ["Sports", "Politics", "Health"]

    return train_texts, train_labels, val_texts, val_labels


def test_train_and_predict(dummy_data, tmp_path):
    """Test training pipeline and predicting top-3 classes."""
    train_texts, train_labels, val_texts, val_labels = dummy_data

    model = train(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        results_dir=str(tmp_path / "results"),
    )

    assert model is not None
    assert hasattr(model, "predict")

    # Predict top 3
    preds = predict(model, "क्रिकेट सामन्यात विजय", top_k=3)
    assert isinstance(preds, list)
    assert len(preds) <= 3
    for cat, score in preds:
        assert isinstance(cat, str)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_predict_empty_input(dummy_data, tmp_path):
    """Test that predict gracefully handles empty or whitespace input."""
    train_texts, train_labels, val_texts, val_labels = dummy_data
    model = train(train_texts, train_labels, val_texts, val_labels, results_dir=str(tmp_path))

    assert predict(model, "") == []
    assert predict(model, "   \n\t  ") == []


def test_save_and_load(dummy_data, tmp_path):
    """Test saving and loading the model pipeline."""
    train_texts, train_labels, val_texts, val_labels = dummy_data
    model = train(train_texts, train_labels, val_texts, val_labels, results_dir=str(tmp_path))

    save_path = str(tmp_path / "model.joblib")
    save(model, save_path)
    assert os.path.exists(save_path)

    loaded = load(save_path)
    test_pred = loaded.predict(["क्रिकेट सामना"])
    assert len(test_pred) == 1


def test_evaluate(dummy_data, tmp_path):
    """Test evaluation output structure."""
    train_texts, train_labels, val_texts, val_labels = dummy_data
    results_dir = str(tmp_path / "eval_results")

    model = train(train_texts, train_labels, val_texts, val_labels, results_dir=results_dir)
    metrics = evaluate(model, val_texts, val_labels, results_dir=results_dir)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert os.path.exists(os.path.join(results_dir, "metrics.json"))
    assert os.path.exists(os.path.join(results_dir, "confusion_matrix.png"))
