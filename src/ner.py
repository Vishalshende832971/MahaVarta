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


def get_available_memory_mb() -> float:
    """Return available system RAM in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:
            pass
    return 1024.0


def _extract_entities_fallback(text: str) -> list[dict]:
    """Fast, memory-safe entity extraction using gazetteer and pattern matching."""
    ents = []
    patterns = [
        ("Designation", r"\b(पंतप्रधान|मुख्यमंत्री|राष्ट्रपती|गृहमंत्री|अर्थमंत्री|राज्यपाल|खासदार|आमदार|कर्णधार|अध्यक्ष|उपाध्यक्ष|संचालक)\b"),
        ("Date", r"\b(सोमवारी|मंगळवारी|बुधवारी|गुरुवारी|शुक्रवारी|शनिवारी|रविवारी|जानेवारी|फेब्रुवारी|मार्च|एप्रिल|मे|जून|जुलै|ऑगस्ट|सप्टेंबर|ऑक्टोबर|नोव्हेंबर|डिसेंबर|[०-९0-9]{1,4}\s*(?:रोजी|साली|मध्ये)?)\b"),
        ("Location", r"\b(भारत|भारतात|नेपाळ|नेपाळमध्ये|महाराष्ट्र|महाराष्ट्रात|मुंबई|मुंबईत|पुणे|पुण्यात|नागपूर|दिल्ली|नवी दिल्ली|गुजरात|अमेरिका|चीन|पाकिस्तान|ऑस्ट्रेलिया|इंग्लंड|राष्ट्रपती भवन|राष्ट्रपती भवनात)\b"),
        ("Person", r"\b(नरेंद्र मोदी|मोदींनी|मोदी|बालेन शाह|राहुल गांधी|अमित शाह|शरद पवार|देवेंद्र फडणवीस|एकनाथ शिंदे|विराट कोहली|रोहित शर्मा|ऋषभ पंत|पंत)\b"),
        ("Measure", r"\b([०-९0-9]+(?:\.[०-९0-9]+)?\s*(?:कोटी|लाख|हजार|टक्के|%|किमी|मीटर|धावा|विकेट|षटक|रुपये))\b"),
    ]
    import re
    seen_spans = set()
    for label, pat in patterns:
        for m in re.finditer(pat, text):
            s, e = m.start(), m.end()
            if not any(s >= es and e <= ee for es, ee in seen_spans):
                seen_spans.add((s, e))
                ents.append({
                    "text": text[s:e],
                    "label": label,
                    "score": 0.9500,
                    "start": s,
                    "end": e,
                })
    return sorted(ents, key=lambda x: x["start"])


def load_model():
    """
    Load the Marathi NER model and pipeline lazily.
    Applies PyTorch dynamic 8-bit quantization and low_cpu_mem_usage for Render deployment.

    Returns:
        Hugging Face TokenClassificationPipeline
    """
    global _ner_pipeline, _tokenizer, _model
    if _ner_pipeline is None:
        avail_mb = get_available_memory_mb()
        if avail_mb < 280.0:
            logger.warning(f"Low memory detected ({avail_mb:.1f} MB available). Skipping heavy NER BERT loading.")
            return None

        logger.info(f"Loading optimized NER model: {MODEL_ID}...")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            raw_model = AutoModelForTokenClassification.from_pretrained(
                MODEL_ID, local_files_only=True, low_cpu_mem_usage=True
            )
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            raw_model = AutoModelForTokenClassification.from_pretrained(
                MODEL_ID, low_cpu_mem_usage=True
            )

        try:
            # 8-bit dynamic quantization: cuts memory footprint by ~75%
            _model = torch.quantization.quantize_dynamic(
                raw_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            del raw_model
        except Exception as q_err:
            logger.warning(f"NER Quantization fallback: {q_err}")
            _model = raw_model

        _model.eval()
        _ner_pipeline = pipeline(
            "token-classification",
            model=_model,
            tokenizer=_tokenizer,
            aggregation_strategy="simple",
        )
        import gc
        gc.collect()
        logger.info(f"Optimized NER model loaded successfully. Labels: {_model.config.id2label}")

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


# Punctuation set for word-boundary snapping
PUNCT_SPLIT = set(" \t\r\n,।॥?!.:;'\"()[]{}<>-/")

# Common Marathi function words / stopwords that should never be marked as entities
MARATHI_STOPWORDS = {
    "दरम्यान", "दरम्यानच्या", "आणि", "यांनी", "म्हणाले", "आहेत", "नाही", "केले",
    "यांच्या", "यांना", "याने", "त्यांनी", "त्यांचे", "आपली", "आपले", "केली", "झाले", "होते"
}

# Common Marathi inflectional and postpositional suffixes to strip for grouped entity display.
# Single consonants like 'त' are omitted to prevent truncating proper nouns like 'भारत'.
MARATHI_SUFFIX_STRIP = [
    "च्यासाठी", "कडच्या", "कडचे", "पासून", "बद्दल", "विषयी", "पर्यंत",
    "यांच्या", "यांना", "याने", "याचा", "याची", "याचे", "याच्या",
    "हूनही", "तीलही", "साठी", "कडे", "कडून", "मध्ये", "तील", "समोर", "जवळ",
    "शिवाय", "देखील", "च्या", "चा", "ची", "चे", "वरून", "वर", "हून",
    "यांच्या", "यांचे", "यांचा", "यांची", "करांना", "करांचे", "करांनी", "करांचा",
    "तील", "मधून", "तील", "ात", "ांना", "ांनी", "ंनी"
]


def expand_to_word_boundary(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Snap subword character offsets to full whitespace/punctuation-delimited word boundaries.
    Prevents subword fragment artifacts (e.g. 'सी' from 'सीता', 'ंतप्रधान' from 'पंतप्रधान').
    """
    while start > 0 and text[start - 1] not in PUNCT_SPLIT:
        start -= 1
    while end < len(text) and text[end] not in PUNCT_SPLIT:
        end += 1
    return start, end


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from Marathi text with word-level boundary snapping
    and adjacent multi-word entity merging.

    Handles text of arbitrary length by segmenting on sentence boundaries,
    snaps subword predictions to full words, removes isolated subword noise,
    and merges adjacent tokens of the same entity class (e.g., 'बालेन शाह').

    Args:
        text: Input Marathi text string.

    Returns:
        List of dicts:
          [{"text": "नरेंद्र मोदी", "label": "Person", "score": 0.9997, "start": 10, "end": 22}, ...]
    """
    if not text or not text.strip():
        return []

    try:
        pipe = load_model()
    except Exception as e:
        logger.warning(f"Failed to load NER model: {e}. Using fallback.")
        return _extract_entities_fallback(text)

    if pipe is None:
        return _extract_entities_fallback(text)

    # Chunk long texts cleanly preserving exact character offsets
    chunks = chunk_text_spans(text, max_words_per_chunk=100)

    snapped_entities = []

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

            abs_start = chunk_offset + int(ent["start"])
            abs_end = chunk_offset + int(ent["end"])
            score = float(ent.get("score", 1.0))

            # Snap subword offsets to full word boundaries in source text
            w_start, w_end = expand_to_word_boundary(text, abs_start, abs_end)
            w_text = text[w_start:w_end].strip(" \t\r\n,।॥?!.:;'\"()[]{}<>-/")

            # Reject empty, single-character artifacts, or common Marathi stopwords
            if not w_text or len(w_text) <= 1 or w_text in MARATHI_STOPWORDS:
                continue

            snapped_entities.append({
                "text": w_text,
                "label": str(label),
                "score": round(score, 4),
                "start": w_start,
                "end": w_end,
            })

    # Sort strictly by starting position
    snapped_entities.sort(key=lambda x: (x["start"], x["end"]))

    # Step 1: Deduplicate identical or overlapping word spans
    deduped = []
    for ent in snapped_entities:
        if not deduped:
            deduped.append(ent)
            continue
        last = deduped[-1]
        if ent["start"] == last["start"] and ent["end"] == last["end"]:
            # Same word: keep higher confidence label
            if ent["score"] > last["score"]:
                deduped[-1] = ent
        elif ent["start"] < last["end"]:
            # Overlapping spans: expand to outer boundary
            last["end"] = max(last["end"], ent["end"])
            last["text"] = text[last["start"]:last["end"]].strip(" \t\r\n,।॥?!.:;'\"()[]{}<>-/")
            last["score"] = max(last["score"], ent["score"])
        else:
            deduped.append(ent)

    # Step 2: Merge adjacent entities of the same class (e.g. बालेन + शाह -> बालेन शाह)
    merged_entities = []
    i = 0
    while i < len(deduped):
        curr = deduped[i]
        j = i + 1
        curr_start = curr["start"]
        curr_end = curr["end"]
        curr_label = curr["label"]
        max_score = curr["score"]

        while j < len(deduped):
            nxt = deduped[j]
            gap = text[curr_end:nxt["start"]]
            gap_clean = gap.strip(" \t\r\n,।॥?!.:;'\"()[]{}<>-/")
            # Merge if adjacent with same label and separated only by spaces/hyphens (<= 3 chars)
            if nxt["label"] == curr_label and len(gap_clean) == 0 and len(gap) <= 3:
                curr_end = nxt["end"]
                max_score = max(max_score, nxt["score"])
                j += 1
            else:
                break

        full_text = text[curr_start:curr_end].strip(" \t\r\n,।॥?!.:;'\"()[]{}<>-/")
        if full_text and full_text not in MARATHI_STOPWORDS:
            merged_entities.append({
                "text": full_text,
                "label": curr_label,
                "score": round(max_score, 4),
                "start": curr_start,
                "end": curr_end,
            })
        i = j

    return merged_entities


def group_by_label(entities: list[dict]) -> dict:
    """
    Group extracted entities by label, removing duplicates while preserving order.
    Normalises trailing inflectional suffixes for clean grouped list display.

    Args:
        entities: List of entity dicts from extract_entities().

    Returns:
        dict mapping label -> list of unique entity text strings.
        E.g.:
        {
            "Person": ["नरेंद्र मोदी", "बालेन शाह"],
            "Location": ["नेपाळ", "भारत"],
            "Designation": ["पंतप्रधान"]
        }
    """
    grouped = {}
    for ent in entities:
        lbl = ent["label"]
        raw_txt = ent["text"].strip()
        
        # Clean trailing postpositional suffix for grouped display
        clean_txt = raw_txt
        for suff in MARATHI_SUFFIX_STRIP:
            if clean_txt.endswith(suff) and len(clean_txt) - len(suff) >= 3:
                clean_txt = clean_txt[:-len(suff)].strip()
                break

        display_name = clean_txt if clean_txt else raw_txt

        if lbl not in grouped:
            grouped[lbl] = []
        if display_name not in grouped[lbl]:
            grouped[lbl].append(display_name)
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
