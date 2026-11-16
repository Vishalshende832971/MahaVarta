"""
MahaVarta: Marathi Sentiment Analysis Module
=============================================
This module provides sentiment prediction using the pretrained
`l3cube-pune/marathi-sentiment-md` model:
  - Lazy, cached model and tokenizer loading (loaded only on first use).
  - Dynamically reads label mappings from model.config.id2label.
  - Returns "model prediction score" (never "confidence").
  - Handles long inputs by chunking and averaging softmax scores.
  - Handles empty and whitespace inputs gracefully.
  - Evaluation on human annotations (data/annotations/headlines.csv) with strict
    validation (refuses to run if final_label is empty or incomplete).

Model ID: l3cube-pune/marathi-sentiment-md
Note: The model was trained on reviews, subtitles, and tweets. When used on news,
scores are indicative.
"""

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

logger = logging.getLogger(__name__)

MODEL_ID = "l3cube-pune/marathi-sentiment-md"

# Global lazy cache
_tokenizer = None
_model = None


def load_model():
    """
    Load the pretrained Marathi sentiment tokenizer and model lazily.
    Caches the model in memory after first load.

    Returns:
        tuple of (tokenizer, model)
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Loading sentiment model and tokenizer: {MODEL_ID}...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        _model.eval()
        logger.info(f"Sentiment model loaded successfully. id2label: {_model.config.id2label}")
    return _tokenizer, _model


def predict(text: str) -> dict:
    """
    Predict sentiment for Marathi text.

    Args:
        text: Input Marathi text string.

    Returns:
        dict with keys:
          - "label": predicted sentiment label (from model.config.id2label)
          - "score": model prediction score (float between 0.0 and 1.0)
          - "all_scores": dict mapping every label name to its prediction score
    """
    if not text or not text.strip():
        # Handle empty/whitespace input gracefully
        tokenizer, model = load_model()
        id2label = model.config.id2label
        all_zero = {str(label): 0.0 for label in id2label.values()}
        # Default to Neutral or first label with 0 score
        default_label = id2label.get(1, list(id2label.values())[0])
        return {
            "label": str(default_label),
            "score": 0.0,
            "all_scores": all_zero,
        }

    tokenizer, model = load_model()
    id2label = model.config.id2label

    # Tokenize input without truncation first to check length
    tokens = tokenizer.encode(text, add_special_tokens=False)
    max_chunk_size = 510  # leave 2 tokens for [CLS] and [SEP]

    chunk_probs = []

    if len(tokens) <= max_chunk_size:
        # Single chunk inference
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            chunk_probs.append(probs)
    else:
        # Long text chunking: split into overlapping or contiguous windows
        logger.info(f"Text length ({len(tokens)} tokens) exceeds limit; chunking...")
        for i in range(0, len(tokens), max_chunk_size):
            chunk_token_ids = tokens[i : i + max_chunk_size]
            chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                chunk_probs.append(probs)

    # Average softmax probabilities across chunks
    avg_probs = np.mean(chunk_probs, axis=0)

    # Determine highest scoring index
    best_idx = int(np.argmax(avg_probs))
    best_label = str(id2label[best_idx])
    best_score = float(avg_probs[best_idx])

    # Build all_scores mapping
    all_scores = {
        str(id2label[i]): round(float(avg_probs[i]), 4)
        for i in range(len(avg_probs))
    }

    return {
        "label": best_label,
        "score": round(best_score, 4),
        "all_scores": all_scores,
    }


def evaluate_on_annotations(
    csv_path: str = "data/annotations/headlines.csv",
    results_dir: str = "results/sentiment",
) -> dict:
    """
    Evaluate the sentiment model on human-annotated headlines.

    Strict verification:
      Refuses to run if the file does not exist, if the 'final_label' column
      is missing, or if any entries in 'final_label' are empty or unlabelled.

    Args:
        csv_path: Path to annotations CSV file.
        results_dir: Directory where evaluation metrics will be written.

    Returns:
        dict containing accuracy, macro-F1, per-class report, and class distribution.

    Raises:
        ValueError: If annotations are empty, incomplete, or missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Annotations file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "final_label" not in df.columns:
        raise ValueError(
            f"Annotations CSV '{csv_path}' missing required column: 'final_label'."
        )

    # Check for empty, NaN, or whitespace-only labels
    is_empty_mask = df["final_label"].isna() | (df["final_label"].astype(object).fillna("").astype(str).str.strip() == "")
    empty_count = int(is_empty_mask.sum())
    total_count = len(df)

    if empty_count > 0:
        raise ValueError(
            f"Annotation evaluation aborted: 'final_label' column has {empty_count} / {total_count} "
            "empty labels. Annotators (Vishal and Gufran) must first label the headlines and "
            "resolve disagreements before running this evaluation."
        )

    cleaned_labels = df["final_label"].astype(str).str.strip()

    # Run predictions on all annotated headlines
    logger.info(f"Evaluating sentiment model on {total_count} annotated headlines...")
    headlines = df["headline"].astype(str).tolist()
    y_true = cleaned_labels.tolist()
    y_pred = [predict(h)["label"] for h in headlines]

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    classes = sorted(list(set(y_true + y_pred)))
    report_dict = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    class_distribution = pd.Series(y_true).value_counts().to_dict()

    summary_metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "total_evaluated": total_count,
        "class_distribution": class_distribution,
        "classes": classes,
        "detailed_report": report_dict,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "sentiment_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Sentiment evaluation complete. Results saved to: {out_json}")
    return summary_metrics
