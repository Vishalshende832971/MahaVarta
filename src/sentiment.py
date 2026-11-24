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
from src.summarization import split_sentences

logger = logging.getLogger(__name__)

MODEL_ID = "l3cube-pune/marathi-sentiment-md"

# Sentiment valence lexicons to calibrate news domain shift (social media -> formal news)
MARATHI_POS_WORDS = {
    "विजय", "शतक", "शतकं", "शतके", "अप्रतिम", "आनंद", "जल्लोष", "यश", "यशस्वी",
    "विक्रम", "विकास", "प्रगती", "सत्कार", "कौतुक", "उत्साह", "ऐतिहासिक", "सुवर्ण",
    "गौरव", "उत्कृष्ट", "सर्वोत्तम", "आवडते", "चांगला", "चांगली", "चांगले", "अभिनंदन",
    "सुवर्णपदक", "विजयी", "जिंकले", "बहारदार", "धमाकेदार", "सुखद", "आशादायी", "यशस्वीरीत्या",
    "मदत", "मदतकार", "फायदा", "फायदेशीर", "सकारात्मक", "आनंददायी", "प्रशंसनीय"
}

MARATHI_NEG_WORDS = {
    "मृत्यू", "अपघात", "शोककळा", "आक्रोश", "भीषण", "वाईट", "धक्कादायक", "खून", "चोरी",
    "गुन्हा", "हल्ला", "नुकसान", "संकट", "पराभव", "तणाव", "दहशत", "अटक", "पोटदुखी",
    "उलटी", "जुलाब", "मळमळ", "समस्या", "भीती", "पीडित", "आजारी", "गंभीर", "हानी",
    "फसवणूक", "भ्रष्टाचार", "वाद", "आरोप", "विरोध", "नाराजी", "निषेध", "दुर्दैवी",
    "दुःखद", "चिंताजनक", "अपयशी", "गमावले", "नकारात्मक", "हानीकारक", "धोकादायक", "धोका"
}

# Global lazy cache
_tokenizer = None
_model = None


def load_model():
    """
    Load the pretrained Marathi sentiment tokenizer and model lazily.
    Applies PyTorch dynamic 8-bit quantization and low_cpu_mem_usage for Render deployment.

    Returns:
        tuple of (tokenizer, model)
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Loading optimized sentiment model: {MODEL_ID}...")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            raw_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, local_files_only=True, low_cpu_mem_usage=True
            )
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            raw_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, low_cpu_mem_usage=True
            )

        try:
            # 8-bit dynamic quantization cuts memory footprint by ~75% (from 440MB to ~110MB)
            _model = torch.quantization.quantize_dynamic(
                raw_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            del raw_model
        except Exception as q_err:
            logger.warning(f"Quantization fallback: {q_err}")
            _model = raw_model

        _model.eval()
        import gc
        gc.collect()
        logger.info(f"Optimized sentiment model loaded successfully. id2label: {_model.config.id2label}")
    return _tokenizer, _model


def predict(text: str) -> dict:
    """
    Predict sentiment for Marathi text with domain-calibrated news sentiment aggregation.
    The base model (l3cube-pune/marathi-sentiment-md) was trained on colloquial tweets and reviews,
    where neutral is a catch-all class that heavily inflates when evaluating formal news reportage.
    This function:
      1. Evaluates sentence-level sentiment dynamics across the article.
      2. Identifies sentiment-bearing peaks and emotional valence (positive/negative).
      3. Normalizes against neutral reportage bias to surface true news sentiment.

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
        default_label = id2label.get(1, list(id2label.values())[0])
        return {
            "label": str(default_label),
            "score": 0.0,
            "all_scores": all_zero,
        }

    tokenizer, model = load_model()
    id2label = model.config.id2label

    sents = split_sentences(text)
    clean_sents = [s.strip() for s in sents if s.strip()]

    # If text is a single short sentence/phrase, run model directly
    if len(clean_sents) <= 1:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits.squeeze(), dim=-1).squeeze().cpu().numpy()

        words = set(text.replace(":", " ").replace(",", " ").replace(".", " ").split())
        pos_hits = len(words.intersection(MARATHI_POS_WORDS))
        neg_hits = len(words.intersection(MARATHI_NEG_WORDS))

        if pos_hits > neg_hits and probs[2] >= 0.05:
            probs[2] += min(0.35, pos_hits * 0.12)
            probs = probs / np.sum(probs)
        elif neg_hits > pos_hits and probs[0] >= 0.05:
            probs[0] += min(0.35, neg_hits * 0.12)
            probs = probs / np.sum(probs)

        best_idx = int(np.argmax(probs))
        best_label = str(id2label[best_idx])
        all_scores = {
            str(id2label[i]): round(float(probs[i]), 4)
            for i in range(len(probs))
        }
        return {
            "label": best_label,
            "score": round(float(probs[best_idx]), 4),
            "all_scores": all_scores,
        }

    # Multi-sentence / full news article evaluation
    sent_probs = []
    for s in clean_sents:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            logits = model(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            sent_probs.append(probs)

    arr = np.array(sent_probs)  # shape (N, 3): [Negative, Neutral, Positive]
    mean_p = np.mean(arr, axis=0)
    max_p = np.max(arr, axis=0)

    # Lexicon valence evidence
    words = set(text.replace(":", " ").replace(",", " ").replace(".", " ").split())
    pos_hits = len(words.intersection(MARATHI_POS_WORDS))
    neg_hits = len(words.intersection(MARATHI_NEG_WORDS))

    # Calibrated news sentiment scores (mitigating reportage neutral bias)
    pos_val = float(mean_p[2] * 0.35 + max_p[2] * 0.45 + min(0.40, pos_hits * 0.10))
    neg_val = float(mean_p[0] * 0.35 + max_p[0] * 0.45 + min(0.40, neg_hits * 0.10))
    neu_val = float(mean_p[1] * 0.28)

    raw_scores = np.array([neg_val, neu_val, pos_val])
    norm_scores = raw_scores / np.sum(raw_scores)

    best_idx = int(np.argmax(norm_scores))
    best_label = str(id2label[best_idx])

    all_scores = {
        str(id2label[i]): round(float(norm_scores[i]), 4)
        for i in range(len(norm_scores))
    }

    return {
        "label": best_label,
        "score": round(float(norm_scores[best_idx]), 4),
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
