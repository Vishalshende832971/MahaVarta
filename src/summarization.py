"""
MahaVarta: Marathi Summarization Module (TextRank Baseline)
===========================================================
This module provides extractive summarisation for Marathi text using TextRank:
  - Marathi sentence segmentation aware of the Devanagari danda (।),
    full stops, question/exclamation marks, decimals, and abbreviations.
  - TF-IDF sentence vectorisation using the Devanagari Indic tokenizer.
  - Cosine similarity graph construction.
  - PageRank centrality scoring (via NetworkX with power-iteration fallback).
  - Chronological sentence reordering so summaries read smoothly.
  - Graceful edge-case handling (empty text, single sentence, fewer sentences).

Note: Pretrained mT5 abstractive summarisation will be added in Stage 8.
"""

import re
import time
import logging
import numpy as np
import networkx as nx
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.preprocessing import tokenize, clean_text

logger = logging.getLogger(__name__)

# mT5 multilingual model fine-tuned on XL-Sum (including Marathi)
MT5_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# Official whitespace normalisation handler from csebuetnlp/mT5_multilingual_XLSum model card
WHITESPACE_HANDLER = lambda k: re.sub(r"\s+", " ", re.sub(r"\n+", " ", k.strip()))

# Lazy model cache for mT5
_cached_mt5_tokenizer = None
_cached_mt5_model = None


def split_sentences(text: str) -> list[str]:
    """
    Split Marathi text into individual sentences.

    Sentence boundaries considered:
      - Devanagari danda (।) and double danda (॥)
      - Question mark (?) and exclamation mark (!)
      - Full stops (.) followed by space or newline
      - Newline boundaries (\n)

    Protections:
      - Decimal numbers (e.g., 3.14, ७.५) are NOT split.
      - Common Marathi abbreviations (डॉ., प्रा., श्री., सौ., इ., क्र., ता., दि.)
        are NOT treated as sentence boundaries.

    Args:
        text: Raw or cleaned Marathi text.

    Returns:
        List of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    processed = text.strip()

    # Step 1: Protect decimal numbers (both ASCII digits and Devanagari digits)
    # E.g., 3.14 -> 3<DECIMAL>14, ७.५ -> ७<DECIMAL>५
    processed = re.sub(r"([0-9०-९]+)\.([0-9०-९]+)", r"\1<DECIMAL>\2", processed)

    # Step 2: Protect common Marathi abbreviations ending with a dot
    abbreviations = [
        r"डॉ\.",    # Doctor
        r"प्रा\.",   # Professor
        r"श्री\.",   # Shri
        r"सौ\.",    # Sau
        r"इ\.",     # etc.
        r"क्र\.",    # No.
        r"ता\.",    # Taluka / Date
        r"दि\.",    # Date
        r"ना\.",    # Honorable
    ]
    for abbrev_pattern in abbreviations:
        # Replace the dot in abbreviation with a placeholder
        processed = re.sub(
            abbrev_pattern,
            lambda m: m.group(0)[:-1] + "<ABBDOT>",
            processed,
        )

    # Step 3: Split on sentence terminal characters:
    # - Devanagari danda (।) and double danda (॥)
    # - Question mark (?) and exclamation mark (!)
    # - Line breaks (\n)
    # - Periods (.) not inside decimals/abbreviations
    raw_sentences = re.split(r"[।॥!?\n]+|(?<=\s)\.(?=\s|$)|(?<=[^\d])\.(?=\s|$)", processed)

    # Step 4: Restore protected characters, trim, and filter out empty segments
    sentences = []
    for s in raw_sentences:
        restored = s.replace("<DECIMAL>", ".").replace("<ABBDOT>", ".").strip()
        # Clean any leftover punctuation or noise
        cleaned_sentence = re.sub(r"^\s*[,;:\-\.]+\s*", "", restored).strip()
        if cleaned_sentence and len(cleaned_sentence) > 1:
            sentences.append(cleaned_sentence)

    return sentences


def textrank_summary(text: str, n_sentences: int = 3) -> str:
    """
    Generate an extractive summary of a Marathi news article using TextRank.

    Algorithm:
      1. Segment text into sentences using Marathi-aware boundaries.
      2. If total sentences <= n_sentences, return all sentences in order.
      3. Vectorize sentences using TF-IDF with the Devanagari tokenizer.
      4. Compute pairwise cosine similarity between sentence vectors.
      5. Construct a weighted graph where edges represent sentence similarity.
      6. Run PageRank on the graph to rank sentences by centrality.
      7. Select top n_sentences and reorder them chronologically.
      8. Return the joined summary text with proper Devanagari danda punctuation.

    Args:
        text: Input Marathi article string.
        n_sentences: Number of summary sentences to extract (default: 3).

    Returns:
        Summary string, or empty string if input is empty.
    """
    if not text or not text.strip():
        return ""

    # Segment text into sentences
    sentences = split_sentences(text)

    # Edge cases:
    # 0 sentences -> return empty
    if not sentences:
        return ""

    # Fewer or equal sentences than requested -> return all sentences
    if len(sentences) <= n_sentences:
        return " । ".join(sentences) + " ।"

    # Build TF-IDF sentence matrix
    # Use the Devanagari-aware tokenizer from preprocessing
    try:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except Exception as e:
        logger.warning(f"TF-IDF vectorisation failed for sentences: {e}")
        # Fallback: return first n_sentences
        return " । ".join(sentences[:n_sentences]) + " ।"

    # Compute pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Set diagonal to zero to eliminate self-loops in the graph
    np.fill_diagonal(similarity_matrix, 0.0)

    # If all similarities are zero (completely disjoint sentences),
    # fallback to leading sentences
    if np.all(similarity_matrix == 0):
        return " । ".join(sentences[:n_sentences]) + " ।"

    # Build NetworkX graph from similarity matrix
    try:
        graph = nx.from_numpy_array(similarity_matrix)
        # Compute PageRank scores (damping factor 0.85, standard for TextRank)
        scores = nx.pagerank(graph, alpha=0.85, max_iter=200, tol=1e-6)
    except Exception as e:
        logger.warning(f"NetworkX PageRank failed: {e}. Falling back to degree centrality.")
        # Fallback: sum of similarities (degree)
        degrees = similarity_matrix.sum(axis=1)
        scores = {i: float(deg) for i, deg in enumerate(degrees)}

    # Rank sentence indices by descending centrality score
    ranked_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)

    # Pick top n_sentences indices
    top_indices = ranked_indices[:n_sentences]

    # Reorder chronologically (by their original appearance in text)
    top_indices_sorted = sorted(top_indices)

    # Extract sentences and format with Devanagari danda
    summary_sentences = [sentences[idx] for idx in top_indices_sorted]
    summary_text = " । ".join(summary_sentences) + " ।"

    return summary_text


# ===========================================================================
# Stage 8: Abstractive Summarisation using mT5 Multilingual XL-Sum
# ===========================================================================

def load_mt5(model_name: str = MT5_MODEL_NAME) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Lazy and cached loader for the pretrained mT5 abstractive summarisation model.
    Loads AutoTokenizer and AutoModelForSeq2SeqLM directly without pipeline().
    Uses CUDA GPU if available, otherwise CPU.

    Args:
        model_name: Hugging Face model identifier (default: csebuetnlp/mT5_multilingual_XLSum).

    Returns:
        Tuple of (AutoTokenizer, AutoModelForSeq2SeqLM).
    """
    global _cached_mt5_tokenizer, _cached_mt5_model
    if _cached_mt5_tokenizer is not None and _cached_mt5_model is not None:
        return _cached_mt5_tokenizer, _cached_mt5_model

    logger.info(f"Loading optimized mT5 summariser from {model_name}...")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        raw_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, local_files_only=True, low_cpu_mem_usage=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        raw_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, low_cpu_mem_usage=True
        )

    try:
        # Quantize linear layers to 8-bit to fit in low-memory environments
        model = torch.quantization.quantize_dynamic(
            raw_model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except Exception as q_err:
        logger.warning(f"mT5 quantization fallback: {q_err}")
        model = raw_model

    model.eval()
    import gc
    gc.collect()

    _cached_mt5_tokenizer = tokenizer
    _cached_mt5_model = model
    return _cached_mt5_tokenizer, _cached_mt5_model


def chunk_sentences(
    sentences: list[str],
    max_chunk_tokens: int = 450,
    tokenizer_fn: callable = None,
) -> list[str]:
    """
    Group Marathi sentences into chunks of at most `max_chunk_tokens` tokens,
    ensuring no sentence is ever cut in the middle.

    Args:
        sentences: List of individual sentence strings.
        max_chunk_tokens: Maximum allowed token count per chunk (default: 450).
        tokenizer_fn: Optional callable taking a sentence string and returning its token count.
                      If None, uses the mT5 tokenizer.

    Returns:
        List of text chunks (each chunk containing joined sentences).
    """
    if not sentences:
        return []

    if tokenizer_fn is None:
        tokenizer, _ = load_mt5()
        tokenizer_fn = lambda s: len(tokenizer.encode(WHITESPACE_HANDLER(s), add_special_tokens=False))

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        s_clean = sentence.strip()
        if not s_clean:
            continue
        s_token_count = max(1, tokenizer_fn(s_clean))

        # If single sentence alone exceeds max_chunk_tokens, put it in its own chunk
        if s_token_count >= max_chunk_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(s_clean)
            continue

        # If adding this sentence exceeds max_chunk_tokens, flush current chunk
        if current_chunk and (current_tokens + s_token_count > max_chunk_tokens):
            chunks.append(" ".join(current_chunk))
            current_chunk = [s_clean]
            current_tokens = s_token_count
        else:
            current_chunk.append(s_clean)
            current_tokens += s_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def mt5_summary(
    text: str,
    max_length: int = 84,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 2,
    model_name: str = MT5_MODEL_NAME,
) -> dict:
    """
    Generate an abstractive summary of Marathi text using mT5 Multilingual XL-Sum.
    Truncates inputs to 512 tokens as required by the model architecture.

    Args:
        text: Input Marathi text string.
        max_length: Maximum summary length in tokens (default: 84).
        min_length: Minimum summary length in tokens (default: 30).
        num_beams: Beam search width (default: 4).
        length_penalty: Exponential penalty to the length (default: 1.0).
        no_repeat_ngram_size: Size of n-grams that cannot occur repeatedly (default: 2).
        model_name: Hugging Face model identifier.

    Returns:
        dict with:
            - summary: Generated abstractive summary string.
            - input_tokens: Number of tokens in input article.
            - truncated: bool, True if article exceeded 512 tokens.
            - mode: "single"
            - seconds: Execution time in seconds.
    """
    t0 = time.time()

    # Handle empty or very short input gracefully
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return {
            "summary": "मजकूर खूप लहान किंवा रिक्त आहे (Text too short or empty for summarisation).",
            "input_tokens": 0,
            "truncated": False,
            "mode": "single",
            "seconds": 0.0,
        }

    tokenizer, model = load_mt5(model_name)
    device = next(model.parameters()).device

    # Exact whitespace normalisation from mT5 model card
    cleaned_text = WHITESPACE_HANDLER(text)

    # Count raw tokens
    raw_tokens = tokenizer.encode(cleaned_text, add_special_tokens=False)
    input_token_count = len(raw_tokens)
    truncated = input_token_count > 512

    # Tokenize with 512 max length and padding
    inputs = tokenizer(
        [cleaned_text],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    ).to(device)

    # Generate summary with beam search
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    elapsed = round(time.time() - t0, 2)
    return {
        "summary": summary,
        "input_tokens": input_token_count,
        "truncated": truncated,
        "mode": "single",
        "seconds": elapsed,
    }


def chunked_mt5_summary(
    text: str,
    max_length: int = 84,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 2,
    max_chunk_tokens: int = 450,
    model_name: str = MT5_MODEL_NAME,
) -> dict:
    """
    Generate an abstractive summary of long Marathi articles using chunked summarisation:
      1. Split article into sentences using Marathi sentence segmentation.
      2. Group sentences into chunks of at most ~450 tokens without splitting sentences.
      3. Summarise each chunk individually with mT5.
      4. Join intermediate chunk summaries. If joined text is under 512 tokens,
         summarise once more to produce a cohesive final summary. If still too long,
         repeat at most 2 times before falling back to joined summaries.

    Args:
        text: Input long Marathi text string.
        max_length: Maximum summary length in tokens.
        min_length: Minimum summary length in tokens.
        num_beams: Beam search width.
        length_penalty: Length penalty.
        no_repeat_ngram_size: Size of repeated n-grams forbidden.
        max_chunk_tokens: Chunk size threshold in tokens (default: 450).
        model_name: Hugging Face model identifier.

    Returns:
        dict with:
            - summary: Final abstractive summary.
            - input_tokens: Total token count of the original long article.
            - num_chunks: Number of chunks created.
            - mode: "chunked"
            - seconds: Total execution time.
            - chunk_summaries: List of intermediate chunk summaries.
    """
    t0 = time.time()

    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return {
            "summary": "मजकूर खूप लहान किंवा रिक्त आहे (Text too short or empty for summarisation).",
            "input_tokens": 0,
            "num_chunks": 0,
            "mode": "chunked",
            "seconds": 0.0,
            "chunk_summaries": [],
        }

    tokenizer, model = load_mt5(model_name)
    cleaned_text = WHITESPACE_HANDLER(text)
    total_input_tokens = len(tokenizer.encode(cleaned_text, add_special_tokens=False))

    # Segment into sentences
    sentences = split_sentences(text)
    if not sentences:
        sentences = [cleaned_text]

    # Group into chunks
    chunks = chunk_sentences(sentences, max_chunk_tokens=max_chunk_tokens)

    # If only 1 chunk, run single summary
    if len(chunks) <= 1:
        res = mt5_summary(
            text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            model_name=model_name,
        )
        return {
            "summary": res["summary"],
            "input_tokens": total_input_tokens,
            "num_chunks": 1,
            "mode": "chunked",
            "seconds": res["seconds"],
            "chunk_summaries": [res["summary"]],
        }

    logger.info(f"Article split into {len(chunks)} chunks for chunked summarisation.")

    # Summarise each chunk
    chunk_summaries = []
    chunk_max_len = max(max_length, 84)
    chunk_min_len = max(min_length // 2, 25)

    for i, chunk in enumerate(chunks, 1):
        chunk_res = mt5_summary(
            chunk,
            max_length=chunk_max_len,
            min_length=chunk_min_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            model_name=model_name,
        )
        s_text = chunk_res["summary"].strip()
        if s_text and "लहान किंवा रिक्त" not in s_text:
            chunk_summaries.append(s_text)

    if not chunk_summaries:
        chunk_summaries = [chunk_res["summary"]]

    # For multi-chunk long articles, concatenating the individual chunk summaries
    # ensures full coverage across every section of the article without collapsing
    # into a brief 1-2 sentence snippet.
    joined_summary = " ".join(chunk_summaries)
    tokens_in_joined = len(tokenizer.encode(WHITESPACE_HANDLER(joined_summary), add_special_tokens=False))

    if tokens_in_joined > 512:
        # If joined chunk summaries exceed 512 tokens, run a hierarchical reduction
        # with a scaled token budget proportional to article length.
        scaled_max = min(350, max(max_length, len(chunks) * 60))
        scaled_min = min(180, max(min_length, len(chunks) * 30))
        final_res = mt5_summary(
            joined_summary,
            max_length=scaled_max,
            min_length=scaled_min,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            model_name=model_name,
        )
        final_summary = final_res["summary"]
    else:
        # Full multi-sentence abstractive summary preserved across all chunks
        final_summary = joined_summary

    elapsed = round(time.time() - t0, 2)
    return {
        "summary": final_summary,
        "input_tokens": total_input_tokens,
        "num_chunks": len(chunks),
        "mode": "chunked",
        "seconds": elapsed,
        "chunk_summaries": chunk_summaries,
    }


def summarize_auto(
    text: str,
    max_chunk_tokens: int = 450,
    **generation_kwargs,
) -> dict:
    """
    Automatic summarisation router:
      - Uses mt5_summary for short articles (<= 512 tokens).
      - Uses chunked_mt5_summary for long articles (> 512 tokens).
      - Handles empty/short inputs safely without crashing.

    Args:
        text: Raw Marathi article text.
        max_chunk_tokens: Maximum tokens per chunk if chunking is triggered.
        **generation_kwargs: Generation hyperparameters (max_length, min_length, num_beams, etc.)

    Returns:
        dict containing summary, input_tokens, mode, seconds, etc.
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return {
            "summary": "मजकूर खूप लहान किंवा रिक्त आहे (Text too short or empty for summarisation).",
            "input_tokens": 0,
            "truncated": False,
            "mode": "single",
            "seconds": 0.0,
        }

    tokenizer, _ = load_mt5()
    cleaned = WHITESPACE_HANDLER(text)
    token_count = len(tokenizer.encode(cleaned, add_special_tokens=False))

    if token_count <= 512:
        return mt5_summary(text, **generation_kwargs)
    else:
        return chunked_mt5_summary(text, max_chunk_tokens=max_chunk_tokens, **generation_kwargs)
