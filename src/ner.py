"""
MahaVarta: Marathi Named Entity Recognition (NER) Module
=========================================================
This module extracts named entities from Marathi text using the
`l3cube-pune/marathi-ner` model:
  - Token classification pipeline with simple aggregation (merges subwords).
  - Lazy loading with local cache support.
  - Chunking for long articles without cutting words.
  - Groups entities by category (Person, Location, Organization, etc.).
  - Evaluates on the official L3Cube-MahaNER test split using seqeval.

Model ID: l3cube-pune/marathi-ner
Supported Entity Classes:
  - Person (व्यक्ती)
  - Location (स्थान / ठिकाण)
  - Organization (संस्था / संघटना)
  - Designation (पद / हुद्दा)
  - Date (दिनांक / तारीख)
  - Time (वेळ)
  - Measure (माप / प्रमाण)
"""

import os
import re
import json
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import (
    accuracy_score as seqeval_accuracy,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
    f1_score as seqeval_f1,
    classification_report as seqeval_report,
)

logger = logging.getLogger(__name__)

MODEL_ID = "l3cube-pune/marathi-ner"

# Lazy cache
_ner_pipeline = None
_tokenizer = None
_model = None


def load_model():
    """
    Load the Marathi NER model and pipeline lazily.
    Caches pipeline in memory after first load.

    Returns:
        Hugging Face TokenClassificationPipeline
    """
    global _ner_pipeline, _tokenizer, _model
    if _ner_pipeline is None:
        logger.info(f"Loading NER model: {MODEL_ID}...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            _model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)

        _model.eval()
        _ner_pipeline = pipeline(
            "token-classification",
            model=_model,
            tokenizer=_tokenizer,
            aggregation_strategy="simple",
        )
        logger.info(f"NER model loaded successfully. Labels: {_model.config.id2label}")

    return _ner_pipeline


def get_supported_labels() -> dict:
    """
    Return the model's supported label mapping from configuration.
    """
    pipe = load_model()
    return pipe.model.config.id2label


def chunk_text_spans(text: str, max_words_per_chunk: int = 100) -> list[tuple[str, int]]:
    """
    Split text into chunks of at most max_words_per_chunk words while preserving
    exact substring slices from `text` with their start character offset.

    Prioritises sentence boundaries (।, ॥, ?, !, ., \\n) so entities are not
    severed, falling back to word boundaries for exceptionally long sentences.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= max_words_per_chunk:
        return [(text, 0)]

    # Match sentences ending with punctuation or newlines
    sentence_pattern = re.compile(r"[^।॥?!\.\n]+(?:[।॥?!\.\n]+|\Z)", re.MULTILINE)
    matches = list(sentence_pattern.finditer(text))

    if not matches:
        return [(text, 0)]

    chunks = []
    current_start = None
    current_end = None
    current_words = 0

    for m in matches:
        s_text = m.group()
        if not s_text.strip():
            continue
        s_start = m.start()
        s_end = m.end()
        s_words = len(s_text.split())

        # If a single sentence exceeds the chunk budget, sub-chunk it on word boundaries
        if s_words > max_words_per_chunk:
            if current_start is not None:
                chunks.append((text[current_start:current_end], current_start))
                current_start = None
                current_end = None
                current_words = 0

            word_matches = list(re.finditer(r"\S+", s_text))
            for i in range(0, len(word_matches), max_words_per_chunk):
                w_sub = word_matches[i : i + max_words_per_chunk]
                w_start = s_start + w_sub[0].start()
                w_end = s_start + w_sub[-1].end()
                chunks.append((text[w_start:w_end], w_start))
            continue

        if current_start is not None and (current_words + s_words > max_words_per_chunk):
            chunks.append((text[current_start:current_end], current_start))
            current_start = s_start
            current_end = s_end
            current_words = s_words
        else:
            if current_start is None:
                current_start = s_start
            current_end = s_end
            current_words += s_words

    if current_start is not None:
        chunks.append((text[current_start:current_end], current_start))

    return chunks


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from Marathi text with aggregated entity spans.

    Handles text of arbitrary length by segmenting on sentence and word boundaries
    without altering whitespace or character indices, ensuring exact span alignment.

    Args:
        text: Input Marathi text string.

    Returns:
        List of dicts:
          [{"text": "नरेंद्र मोदी", "label": "Person", "score": 0.9997, "start": 10, "end": 22}, ...]
    """
    if not text or not text.strip():
        return []

    pipe = load_model()

    # Chunk long texts cleanly preserving exact character offsets
    chunks = chunk_text_spans(text, max_words_per_chunk=100)

    entities = []
    seen_spans = set()

    for chunk_text, chunk_offset in chunks:
        if not chunk_text.strip():
            continue
        try:
            raw_ents = pipe(chunk_text)
        except Exception as e:
            logger.warning(f"Error processing NER chunk: {e}")
            continue

        for ent in raw_ents:
            label = ent.get("entity_group", ent.get("entity", "Other"))

            # Filter out non-entities ('Other' or 'O')
            if label in ("Other", "O", "LABEL_0"):
                continue

            ent_word = ent.get("word", "").strip()
            # Clean subword artifact if any remains
            ent_word = re.sub(r"^##", "", ent_word).strip()
            if not ent_word:
                continue

            abs_start = chunk_offset + int(ent["start"])
            abs_end = chunk_offset + int(ent["end"])

            # Verify against actual slice of text for precise boundary
            actual_text = text[abs_start:abs_end] if abs_start < len(text) and abs_end <= len(text) else ent_word
            display_word = actual_text if actual_text.strip() else ent_word

            span_key = (abs_start, abs_end, label)

            if span_key not in seen_spans:
                seen_spans.add(span_key)
                entities.append({
                    "text": display_word,
                    "label": str(label),
                    "score": round(float(ent.get("score", 1.0)), 4),
                    "start": abs_start,
                    "end": abs_end,
                })

    # Sort entities strictly by their starting position in text
    entities.sort(key=lambda x: (x["start"], x["end"]))
    return entities


def group_by_label(entities: list[dict]) -> dict:
    """
    Group extracted entities by label, removing duplicates while preserving order.

    Args:
        entities: List of entity dicts from extract_entities().

    Returns:
        dict mapping label -> list of unique entity text strings.
        E.g.:
        {
            "Person": ["नरेंद्र मोदी"],
            "Location": ["मुंबई", "पुणे"],
            "Designation": ["पंतप्रधान"]
        }
    """
    grouped = {}
    for ent in entities:
        lbl = ent["label"]
        txt = ent["text"]
        if lbl not in grouped:
            grouped[lbl] = []
        if txt not in grouped[lbl]:
            grouped[lbl].append(txt)
    return grouped


# ---------------------------------------------------------------------------
# Official MahaNER Dataset Evaluation using seqeval
# ---------------------------------------------------------------------------
def evaluate_seqeval(
    test_file_path: str = "data/mahaner/test_iob.txt",
    results_dir: str = "results/ner",
    max_sentences: int = None,
) -> dict:
    """
    Evaluate the Marathi NER model on the official MahaNER test split using seqeval.

    Dataset & Model Tag Mapping:
      - Dataset IOB tags:
          BNEP/INEP -> Person
          BNEL/INEL -> Location
          BNEO/INEO -> Organization
          BNED/INED -> Date
          BNETI/INETI -> Time
          BNEM/INEM -> Measure
          BED/IED -> Designation
          O -> O
      - Model predictions:
          The model predicts classes without IOB prefixes.
          We convert predictions to standard B- / I- IOB format so seqeval
          can compute strict entity-level precision, recall, and F1.

    Args:
        test_file_path: Path to MahaNER test_iob.txt file.
        results_dir: Directory where results will be written.
        max_sentences: Optional limit for testing (defaults to all sentences).

    Returns:
        dict containing seqeval metrics and per-class performance.
    """
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"MahaNER test split not found: {test_file_path}")

    # Tag mapping from official MahaNER dataset IOB scheme to standard seqeval IOB tags
    tag_map = {
        "BNEP": "B-Person", "INEP": "I-Person",
        "BNEL": "B-Location", "INEL": "I-Location",
        "BNEO": "B-Organization", "INEO": "I-Organization",
        "BNED": "B-Date", "INED": "I-Date",
        "BNETI": "B-Time", "INETI": "I-Time",
        "BNEM": "B-Measure", "INEM": "I-Measure",
        "BED": "B-Designation", "IED": "I-Designation",
        "O": "O",
    }

    df = pd.read_csv(test_file_path, sep="\t")
    grouped_sents = list(df.groupby("sentence_id"))
    total_sentences = len(grouped_sents)

    if max_sentences and max_sentences < total_sentences:
        eval_sentences = grouped_sents[:max_sentences]
    else:
        eval_sentences = grouped_sents

    logger.info(f"Evaluating NER model on {len(eval_sentences)} sentences using seqeval...")

    pipe = load_model()
    tokenizer = pipe.tokenizer
    model = pipe.model
    id2label = model.config.id2label

    y_true = []
    y_pred = []

    for s_id, group in eval_sentences:
        words = group["words"].astype(str).tolist()
        gold_labels = [tag_map.get(str(l).strip(), "O") for l in group["labels"].tolist()]

        # Tokenize with is_split_into_words=True
        enc = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            logits = model(**enc).logits
            pred_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
            if isinstance(pred_ids, int):
                pred_ids = [pred_ids]

        # Align subwords to word-level IOB tags
        pred_word_tags = []
        prev_w_idx = None
        prev_tag = "O"

        for token_pos, w_idx in enumerate(word_ids):
            if w_idx is None:
                continue
            if w_idx != prev_w_idx:
                raw_cls = id2label[pred_ids[token_pos]]
                if raw_cls == "Other":
                    tag = "O"
                elif raw_cls == prev_tag and prev_tag != "O":
                    tag = f"I-{raw_cls}"
                else:
                    tag = f"B-{raw_cls}"
                pred_word_tags.append(tag)
                prev_w_idx = w_idx
                prev_tag = raw_cls

        # Ensure lengths match
        if len(pred_word_tags) == len(gold_labels):
            y_true.append(gold_labels)
            y_pred.append(pred_word_tags)

    # Compute seqeval entity-level metrics
    prec = float(seqeval_precision(y_true, y_pred))
    rec = float(seqeval_recall(y_true, y_pred))
    f1 = float(seqeval_f1(y_true, y_pred))
    acc = float(seqeval_accuracy(y_true, y_pred))
    report_text = seqeval_report(y_true, y_pred, digits=4)

    results = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "evaluated_sentences": len(y_true),
        "total_test_sentences": total_sentences,
        "classification_report": report_text,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "ner_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    out_txt = os.path.join(results_dir, "ner_classification_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"NER seqeval evaluation complete: P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
    return results
