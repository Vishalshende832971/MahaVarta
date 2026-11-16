"""
MahaVarta: Training and Evaluation Runner for News Classification
=================================================================
Runs the full training, hyperparameter comparison, test evaluation,
and persistence workflow for the Marathi News Classifier.
"""

import os
import sys
import time
import pandas as pd

# Add repo root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification import train, evaluate, predict, save, CATEGORIES


def load_split(csv_path: str):
    """
    Load a dataset split and filter out any leaked header rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset split not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter out any repeated header lines from multi-file concatenation
    clean_df = df[df["Label"] != "Label"].copy()
    clean_df = clean_df.dropna(subset=["Text", "Label"])

    texts = clean_df["Text"].astype(str).tolist()
    labels = clean_df["Label"].astype(str).tolist()

    return texts, labels


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data", "mahanews")
    results_dir = os.path.join(repo_root, "results", "classification")
    models_dir = os.path.join(repo_root, "models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("MahaVarta: Marathi News Classification (Stage 3)")
    print("=" * 70)

    # Step 1: Load and verify dataset splits
    print("\n[1/5] Loading verified dataset splits from data/mahanews/...")
    train_texts, train_labels = load_split(os.path.join(data_dir, "train.csv"))
    val_texts, val_labels = load_split(os.path.join(data_dir, "valid.csv"))
    test_texts, test_labels = load_split(os.path.join(data_dir, "test.csv"))

    print(f"  - Train split size:      {len(train_texts):,} clean samples")
    print(f"  - Validation split size: {len(val_texts):,} clean samples")
    print(f"  - Test split size:       {len(test_texts):,} clean samples")
    print(f"  - Total dataset size:    {len(train_texts) + len(val_texts) + len(test_texts):,} clean samples")
    print(f"  - Number of classes:     {len(set(train_labels))} classes: {sorted(set(train_labels))}")

    # Step 2: Hyperparameter tuning and training
    print("\n[2/5] Running validation grid search (unigram vs bigram, C values)...")
    t0 = time.time()
    best_pipeline = train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        results_dir=results_dir,
    )
    t_train = time.time() - t0
    print(f"  Tuning & training completed in {t_train:.1f} seconds.")

    # Step 3: Save trained model
    model_save_path = os.path.join(models_dir, "classification_pipeline.joblib")
    print(f"\n[3/5] Saving best model to {model_save_path}...")
    save(best_pipeline, model_save_path)

    # Step 4: Evaluate on official test split
    print("\n[4/5] Evaluating best pipeline on official test split...")
    test_metrics = evaluate(
        model=best_pipeline,
        test_texts=test_texts,
        test_labels=test_labels,
        results_dir=results_dir,
    )

    print("\n--- TEST EVALUATION RESULTS ---")
    print(f"  Accuracy:        {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision: {test_metrics['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall:    {test_metrics['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1:        {test_metrics['macro_f1'] * 100:.2f}%")

    print("\n--- PER-CLASS PERFORMANCE ---")
    per_class_df = pd.DataFrame(test_metrics["per_class"])
    print(per_class_df.to_string(index=False))

    # Step 5: Test predict function on sample Marathi articles
    print("\n[5/5] Testing predict() function on sample Marathi articles...")
    samples = [
        (
            "क्रीडा बातमी (Sports)",
            "भारतीय क्रिकेट संघाने दुसऱ्या कसोटी सामन्यात ऑस्ट्रेलियाचा ६ गडी राखून पराभव केला. कर्णधार रोहित शर्माने शतकी खेळी केली."
        ),
        (
            "राजकारण बातमी (Politics)",
            "विधानसभा निवडणुकीच्या पार्श्वभूमीवर आज केंद्रीय मंत्रिमंडळाची महत्त्वाची बैठक पार पडली. मुख्यमंत्र्यांनी पत्रकार परिषदेत घोषणा केली."
        ),
        (
            "आरोग्य बातमी (Health)",
            "दररोज सकाळी व्यायाम करणे आणि संतुलित आहार घेणे आरोग्यासाठी अत्यंत फायदेशीर ठरते. डॉक्टरांनी नियमित तपासणीचा सल्ला दिला आहे."
        ),
        (
            "मनोरंजन बातमी (Manoranjan)",
            "मराठी चित्रपटसृष्टीतील प्रसिद्ध अभिनेत्याचा नवा चित्रपट पुढील महिन्यात सर्व चित्रपटगृहांमध्ये प्रदर्शित होणार आहे."
        ),
    ]

    for title, text in samples:
        top3 = predict(best_pipeline, text, top_k=3)
        print(f"\n  Article: {title}")
        print(f"  Text snippet: {text[:60]}...")
        print(f"  Top-3 Predictions: {top3}")

    print("\n" + "=" * 70)
    print("Stage 3 Completed Successfully!")
    print(f"All results saved under: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
