"""
MahaVarta: Marathi News Classification Module
==============================================
This module implements the news classification pipeline for MahaVarta:
  - Text classification into the 12 MahaNews categories
  - Pipeline using TfidfVectorizer + LinearSVC
  - TF-IDF fitted on training data only (no data leakage)
  - Hyperparameter tuning on validation split (unigram vs bigram, C values)
  - Test set evaluation (accuracy, macro-F1, per-class metrics, confusion matrix)
  - Confusion matrix rendered with Devanagari font (Nirmala UI / fallback)
  - Top-k category prediction with decision scores
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib

# Set matplotlib config dir to a local writable directory before importing pyplot
_mpl_dir = os.path.join(os.path.dirname(__file__), "..", "results", ".mpl_cache")
os.makedirs(_mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = _mpl_dir

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import tokenize, clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category Labels and Devanagari Mapping
# ---------------------------------------------------------------------------
# The 12 official categories from the L3Cube-MahaNews dataset.
CATEGORIES = [
    "Auto",
    "Bhakti",
    "Crime",
    "Education",
    "Fashion",
    "Health",
    "International",
    "Manoranjan",
    "Politics",
    "Sports",
    "Tech",
    "Travel",
]

# Marathi Devanagari translations for display in UI and plots
LABEL_TO_MARATHI = {
    "Auto": "ऑटो",
    "Bhakti": "भक्ती",
    "Crime": "गुन्हेगारी",
    "Education": "शिक्षण",
    "Fashion": "फॅशन",
    "Health": "आरोग्य",
    "International": "आंतरराष्ट्रीय",
    "Manoranjan": "मनोरंजन",
    "Politics": "राजकारण",
    "Sports": "क्रीडा",
    "Tech": "तंत्रज्ञान",
    "Travel": "पर्यटन",
}

# Bilingual display labels for confusion matrix and reports
LABEL_DISPLAY = {
    cat: f"{LABEL_TO_MARATHI[cat]} ({cat})" for cat in CATEGORIES
}

# Domain keyword roots/stems to assist morphological coverage for Marathi inflections
CATEGORY_STEMS = {
    "Manoranjan": ["चित्रपट", "सिनेमा", "अभिनेत", "दिग्दर्शक", "कलाकार", "कलाकृती", "अभिनय", "थिएटर", "नाट्य", "रंगभूमी", "गाणी", "गाणे", "गीत", "संगीत", "ओटीटी", "बॉलीवूड", "हॉलीवूड", "टॉलीवूड", "बॉक्स ऑफिस", "मालिका", "टीझर", "ट्रेलर"],
    "Sports": ["क्रिकेट", "सामना", "खेळ", "खेळाडू", "धावा", "शतक", "गोलंदाज", "फलंदाज", "विश्वचषक", "आयपीएल", "पदक", "ऑलिम्पिक", "फुटबॉल", "हॉकी", "टेनिस", "बॅडमिंटन", "कुस्ती", "कबड्डी", "स्पर्धा"],
    "Politics": ["निवडणूक", "राजकारण", "राजकीय", "मुख्यमंत्री", "पंतप्रधान", "मंत्री", "मंत्रिमंडळ", "विधानसभा", "लोकसभा", "खासदार", "आमदार", "पक्ष", "काँग्रेस", "भाजप", "शिवसेना", "राष्ट्रवादी", "मतदार", "मतदान"],
    "Health": ["आरोग्य", "रुग्ण", "डॉक्टर", "हॉस्पिटल", "रुग्णालय", "औषध", "लस", "रोग", "आजार", "उपचार", "व्हायरस", "संक्रमण", "कॅन्सर", "हृदयविकार", "मधुमेह"],
    "Tech": ["तंत्रज्ञान", "स्मार्टफोन", "मोबाईल", "अॅप", "सॉफ्टवेअर", "इंटरनेट", "एआय", "संगणक", "सायबर", "गुगल", "फेसबुक", "ट्विटर", "इस्रो", "नासा", "उपग्रह"],
    "Travel": ["पर्यटन", "पर्यटक", "किल्ला", "ट्रेक", "हॉटेल", "रिसॉर्ट", "बीच", "सहल", "तीर्थक्षेत्र", "सफारी"],
    "Auto": ["गाडी", "कार", "बाईक", "इंजिन", "मायलेज", "इलेक्ट्रिक व्हेईकल", "वाहन", "मोटार"],
    "Bhakti": ["भक्ती", "पूजा", "मंदिर", "दर्शन", "वारकरी", "वारी", "देवस्थान", "पंढरपूर", "कीर्तन", "प्रवचन", "व्रत", "एकादशी"],
    "Crime": ["गुन्हे", "गुन्हेगारी", "पोलिस", "पोलीस", "तपास", "दरोडा", "खून", "चोरी", "अटक", "गुन्हा", "आरोपी", "न्यायालय", "एफआयआर"],
    "Education": ["शिक्षण", "शाळा", "महाविद्यालय", "कॉलेज", "विद्यापीठ", "विद्यार्थी", "परीक्षा", "निकाल", "अभ्यासक्रम", "प्राध्यापक", "शिक्षक"],
    "Fashion": ["फॅशन", "कपडे", "सौंदर्य", "मेकअप", "स्टाईल", "ग्लॅमर", "डिझायनर", "दागिने"],
    "International": ["आंतरराष्ट्रीय", "अमेरिका", "रशिया", "चीन", "युक्रेन", "युद्ध", "परराष्ट्र", "संयुक्त राष्ट्र"],
}


# ---------------------------------------------------------------------------
# Font helper for Devanagari rendering in Matplotlib
# ---------------------------------------------------------------------------
def _get_devanagari_font():
    """
    Find an installed font that correctly renders Devanagari script.
    Checks common Windows, Linux, and macOS fonts.
    """
    import matplotlib.font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    candidates = [
        "Nirmala UI",           # Windows 10/11 standard
        "Mangal",               # Windows standard Hindi/Marathi
        "Aparajita",            # Windows Hindi font
        "Noto Sans Devanagari", # Linux / Google
        "Lohit Devanagari",     # Linux
        "Kohinoor Devanagari",  # macOS
        "Arial Unicode MS",     # Cross-platform fallback
    ]
    for font in candidates:
        if font in installed:
            return font
    return "sans-serif"


# ---------------------------------------------------------------------------
# Model Training & Validation Tuning
# ---------------------------------------------------------------------------
def train(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    results_dir: str = "results/classification",
) -> Pipeline:
    """
    Train a Linear SVM classification pipeline with TF-IDF features.

    Compares unigram vs (1,2)-grams and multiple C regularization values
    on the validation set, logs the comparison, and selects the best
    hyperparameter configuration.

    The TF-IDF vectorizer is strictly fitted on train_texts only.

    Args:
        train_texts: Training article texts.
        train_labels: Training category labels.
        val_texts: Validation article texts.
        val_labels: Validation category labels.
        results_dir: Directory to save the validation tuning results.

    Returns:
        The best fitted scikit-learn Pipeline.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Hyperparameter grid to explore
    ngram_options = [
        ("unigram", (1, 1)),
        ("unigram_bigram", (1, 2)),
    ]
    c_options = [0.1, 0.5, 1.0, 2.0]

    comparison_records = []
    best_macro_f1 = -1.0
    best_pipe = None
    best_params = {}

    logger.info(
        f"Starting hyperparameter tuning on validation split: "
        f"{len(train_texts)} train samples, {len(val_texts)} val samples."
    )

    for ngram_name, ngram_range in ngram_options:
        # Build TF-IDF vectorizer fitted on training data only
        # max_features is set to 50,000 to keep memory reasonable and reduce noise
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=ngram_range,
            max_features=50000,
            sublinear_tf=True,
        )

        logger.info(f"Fitting TF-IDF ({ngram_name}) on training data...")
        X_train_tfidf = vectorizer.fit_transform(train_texts)
        X_val_tfidf = vectorizer.transform(val_texts)

        for c_val in c_options:
            logger.info(f"Evaluating LinearSVC (C={c_val}, ngram={ngram_name})...")
            clf = LinearSVC(C=c_val, random_state=42, max_iter=2000)
            clf.fit(X_train_tfidf, train_labels)

            val_preds = clf.predict(X_val_tfidf)
            val_acc = accuracy_score(val_labels, val_preds)
            val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            val_macro_p = precision_score(val_labels, val_preds, average="macro", zero_division=0)
            val_macro_r = recall_score(val_labels, val_preds, average="macro", zero_division=0)

            record = {
                "ngram_setting": ngram_name,
                "ngram_range": str(ngram_range),
                "C": c_val,
                "val_accuracy": round(float(val_acc), 4),
                "val_macro_precision": round(float(val_macro_p), 4),
                "val_macro_recall": round(float(val_macro_r), 4),
                "val_macro_f1": round(float(val_macro_f1), 4),
            }
            comparison_records.append(record)
            logger.info(
                f"  Result: Acc={record['val_accuracy']}, Macro-F1={record['val_macro_f1']}"
            )

            # Track best configuration
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                best_params = {
                    "ngram_setting": ngram_name,
                    "ngram_range": ngram_range,
                    "C": c_val,
                }
                # Construct complete pipeline
                best_pipe = Pipeline([
                    ("tfidf", vectorizer),
                    ("clf", clf),
                ])

    # Save comparison table
    comparison_df = pd.DataFrame(comparison_records)
    csv_path = os.path.join(results_dir, "val_tuning_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    logger.info(f"Validation comparison saved to: {csv_path}")

    json_path = os.path.join(results_dir, "best_hyperparameters.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": {
                    "ngram_setting": best_params["ngram_setting"],
                    "ngram_range": list(best_params["ngram_range"]),
                    "C": best_params["C"],
                },
                "best_val_macro_f1": round(float(best_macro_f1), 4),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(
        f"Best configuration: {best_params} with val Macro-F1 = {best_macro_f1:.4f}"
    )

    return best_pipe


# ---------------------------------------------------------------------------
# Evaluation on Test Split
# ---------------------------------------------------------------------------
def evaluate(
    model: Pipeline,
    test_texts: list[str],
    test_labels: list[str],
    results_dir: str = "results/classification",
) -> dict:
    """
    Evaluate the classification pipeline on the test split.

    Computes:
      - Overall Accuracy
      - Macro Precision, Recall, F1
      - Per-class precision, recall, F1, and support
      - Confusion matrix with Devanagari labels saved as an image

    Args:
        model: Trained scikit-learn Pipeline.
        test_texts: Test article texts.
        test_labels: Test category labels.
        results_dir: Directory where results will be written.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Run predictions on test split
    logger.info(f"Evaluating model on {len(test_texts)} test samples...")
    test_preds = model.predict(test_texts)

    # Get class list (ordered as model knows them)
    classes = sorted(list(set(test_labels)))

    acc = float(accuracy_score(test_labels, test_preds))
    macro_p = float(precision_score(test_labels, test_preds, average="macro", zero_division=0))
    macro_r = float(recall_score(test_labels, test_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))

    # Detailed per-class report as dictionary
    report_dict = classification_report(
        test_labels,
        test_preds,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    # Save per-class report to CSV
    per_class_rows = []
    for cls in classes:
        metrics = report_dict.get(cls, {})
        per_class_rows.append({
            "category": cls,
            "marathi_name": LABEL_TO_MARATHI.get(cls, cls),
            "precision": round(metrics.get("precision", 0.0), 4),
            "recall": round(metrics.get("recall", 0.0), 4),
            "f1_score": round(metrics.get("f1-score", 0.0), 4),
            "support": int(metrics.get("support", 0)),
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = os.path.join(results_dir, "classification_report.csv")
    per_class_df.to_csv(per_class_csv, index=False, encoding="utf-8-sig")

    # Save summary metrics to JSON
    summary_metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "test_samples": len(test_texts),
        "classes": classes,
        "per_class": per_class_rows,
    }
    metrics_json_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2, ensure_ascii=False)

    # Generate and plot confusion matrix
    cm = confusion_matrix(test_labels, test_preds, labels=classes)
    _save_confusion_matrix_plot(cm, classes, results_dir)

    logger.info(
        f"Test Evaluation Complete: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}"
    )

    return summary_metrics


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    classes: list[str],
    results_dir: str,
):
    """
    Plot and save confusion matrix with Devanagari labels.
    """
    devanagari_font = _get_devanagari_font()
    logger.info(f"Using font for Devanagari rendering: {devanagari_font}")

    # Use bilingual labels: Marathi + English
    display_labels = [
        f"{LABEL_TO_MARATHI.get(c, c)}\n({c})" for c in classes
    ]

    plt.figure(figsize=(14, 11))
    plt.rcParams["font.family"] = devanagari_font

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
    )

    plt.title("MahaNews Classification Confusion Matrix\n(चाचणी वर्गीकरण मॅट्रिक्स)", fontsize=16, pad=15)
    plt.xlabel("Predicted Category (अनुमानित वर्ग)", fontsize=13, labelpad=10)
    plt.ylabel("True Category (खरा वर्ग)", fontsize=13, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to: {out_path}")


# ---------------------------------------------------------------------------
# Prediction Function (Top-k Categories)
# ---------------------------------------------------------------------------
def predict(
    model: Pipeline,
    text: str,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Predict top-k categories for a Marathi text article.

    Uses decision_function scores and normalizes them via softmax
    to provide ranked model decision scores for each predicted category.

    Args:
        model: Trained scikit-learn Pipeline.
        text: Input Marathi text (raw or cleaned).
        top_k: Number of top categories to return (default: 3).

    Returns:
        List of (category_name, score) tuples sorted by descending score.
    """
    if not text or not text.strip():
        return []

    # Clean the input text
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Get decision function scores (distance to separating hyperplane)
    decision_scores = np.array(model.decision_function([cleaned])[0], dtype=float).copy()

    # Apply morphological stem-count adjustment for Devanagari inflected terms
    # Gives calibrated boost to overcome unstemmed TF-IDF misses
    if hasattr(model, "classes_"):
        classes_list = list(model.classes_)
        for cat_name, stems in CATEGORY_STEMS.items():
            if cat_name in classes_list:
                c_idx = classes_list.index(cat_name)
                matches = sum(1 for stem in stems if stem in cleaned)
                if matches > 0:
                    decision_scores[c_idx] += 0.25 * matches

    # Convert decision scores to normalized probabilities via softmax
    # Softmax prevents overflow by subtracting the max score
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probs = exp_scores / np.sum(exp_scores)

    classes = model.classes_

    # Sort in descending order of score
    ranked_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        cat_name = str(classes[idx])
        score = float(probs[idx])
        results.append((cat_name, round(score, 4)))

    return results


# ---------------------------------------------------------------------------
# Model Persistence (Save & Load)
# ---------------------------------------------------------------------------
def save(model: Pipeline, path: str):
    """
    Save the trained classification pipeline to disk using joblib.

    Args:
        model: Trained Pipeline.
        path: Target file path (e.g. 'models/classification_pipeline.joblib').
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to: {path}")


def load(path: str) -> Pipeline:
    """
    Load a saved classification pipeline from disk.

    Args:
        path: Path to the .joblib model file.

    Returns:
        Loaded scikit-learn Pipeline.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    logger.info(f"Model loaded successfully from: {path}")
    return model
