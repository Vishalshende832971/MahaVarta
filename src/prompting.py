"""
MahaVarta: Prompt-Engineered Marathi News Brief Module (Qwen3-0.6B)
===================================================================
This module provides instruction-tuned LLM news brief generation using Qwen/Qwen3-0.6B:
  - Lazy loading of Qwen3-0.6B with native FP32 precision on CPU.
  - Three distinct prompting strategies (Zero-shot, Role-based, Role-based constrained).
  - Explicit disabling of thinking mode (enable_thinking=False) with <think> strip guard.
  - Deterministic generation (do_sample=False) across all runs.
  - Context-aware input truncation at sentence boundaries (~1,500 tokens ceiling).
  - Objective metrics (no quality judgement):
      * has_three_numbered_items
      * devanagari_ratio
      * novel_word_ratio (stem/prefix-aware to avoid penalizing inflections)
      * invented_name_candidates (NER entity check + stem absence)
      * output_length_words
"""

import os
import re
import sys
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.preprocessing import tokenize, clean_text
from src.summarization import split_sentences
from src.ner import extract_entities

logger = logging.getLogger(__name__)

QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"

# Common Marathi inflectional and postpositional suffixes
MARATHI_SUFFIXES = [
    "च्यासाठी", "कडच्या", "कडचे", "पासून", "बद्दल", "विषयी", "पर्यंत",
    "यांच्या", "यांना", "याने", "याचा", "याची", "याचे", "याच्या",
    "हूनही", "तीलही", "साठी", "कडे", "कडून", "मध्ये", "तील", "समोर", "जवळ",
    "च्या", "चा", "ची", "चे", "ना", "नी", "ने", "ला", "वर", "हून", "त"
]

# Exact prompt templates from Part B, section B3 of MahaVarta_Project_Spec.md
PROMPTS = {
    "zero_shot": (
        "Give a short Marathi headline and three key takeaways for this Marathi news article.\n\n"
        "Article:\n"
        "{article}"
    ),
    "role_based": (
        "You are a professional Marathi news editor.\n"
        "Read the following Marathi news article and write a short Marathi headline and three key takeaways.\n\n"
        "Article:\n"
        "{article}"
    ),
    "role_constraints": (
        "You are a professional Marathi news editor.\n\n"
        "Analyze the following Marathi news article.\n\n"
        "Instructions:\n"
        "1. Write one short headline.\n"
        "2. Generate exactly 3 key takeaways as a numbered list.\n"
        "3. Write only in Marathi.\n"
        "4. Use simple and clear language.\n"
        "5. Use only information present in the article. Do not add outside facts, names or numbers.\n"
        "6. Keep the output concise.\n\n"
        "Article:\n"
        "{article}"
    ),
}

# Lazy cache for Qwen model
_cached_qwen_tokenizer = None
_cached_qwen_model = None


def load_qwen(model_name: str = QWEN_MODEL_NAME) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Lazy and cached loader for Qwen/Qwen3-0.6B.
    Uses torch.float32 precision for native and stable CPU execution on Windows.
    Uses CUDA if available.

    Returns:
        tuple of (AutoTokenizer, AutoModelForCausalLM)
    """
    global _cached_qwen_tokenizer, _cached_qwen_model
    if _cached_qwen_tokenizer is not None and _cached_qwen_model is not None:
        return _cached_qwen_tokenizer, _cached_qwen_model

    logger.info(f"Loading Qwen model from {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FP32 is native on x86 CPU and avoids missing bfloat16 kernel issues on Windows
    chosen_dtype = torch.float32

    # Try local cache first to avoid network timeout delays when offline
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=chosen_dtype,
            local_files_only=True,
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=chosen_dtype,
        ).to(device)

    model.eval()
    _cached_qwen_tokenizer = tokenizer
    _cached_qwen_model = model
    return _cached_qwen_tokenizer, _cached_qwen_model


def strip_thinking_tokens(text: str) -> str:
    """Strip any <think>...</think> reasoning blocks if model emitted them."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned)
    return cleaned.strip()


def truncate_article_at_sentence(
    article: str,
    tokenizer,
    max_tokens: int = 1500,
) -> tuple[str, bool]:
    """
    Truncate article text at a sentence boundary if it exceeds max_tokens.

    Args:
        article: Raw input text.
        tokenizer: Hugging Face tokenizer.
        max_tokens: Maximum allowed token threshold (default: 1500).

    Returns:
        tuple: (truncated_text, is_truncated)
    """
    tokens = tokenizer.encode(article, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return article, False

    sentences = split_sentences(article)
    accumulated_sentences = []
    current_tokens = 0

    for s in sentences:
        s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
        if current_tokens + s_tokens <= max_tokens:
            accumulated_sentences.append(s)
            current_tokens += s_tokens
        else:
            break

    if not accumulated_sentences:
        # Fallback if first sentence alone exceeds max_tokens
        truncated_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    else:
        truncated_text = " ".join(accumulated_sentences)

    return truncated_text, True


def generate_brief(
    article: str,
    prompt_key: str,
    max_new_tokens: int = 300,
    model_name: str = QWEN_MODEL_NAME,
) -> dict:
    """
    Generate a prompt-engineered Marathi news brief using Qwen3-0.6B.
    Deterministic generation (do_sample=False) is strictly enforced.

    Args:
        article: Input Marathi news text.
        prompt_key: Key from PROMPTS ("zero_shot", "role_based", "role_constraints").
        max_new_tokens: Maximum tokens to generate (default: 300).
        model_name: Model identifier.

    Returns:
        dict: {output, prompt_key, input_tokens, output_tokens, seconds, truncated_input}
    """
    t0 = time.time()

    if prompt_key not in PROMPTS:
        raise ValueError(f"Unknown prompt_key '{prompt_key}'. Expected one of: {list(PROMPTS.keys())}")

    # Handle empty or very short input without calling model
    if not article or not isinstance(article, str) or len(article.strip()) < 15:
        return {
            "output": "मजकूर खूप लहान किंवा रिक्त आहे (Article text is too short or empty for brief generation).",
            "prompt_key": prompt_key,
            "input_tokens": 0,
            "output_tokens": 0,
            "seconds": 0.0,
            "truncated_input": False,
        }

    tokenizer, model = load_qwen(model_name)
    device = next(model.parameters()).device

    # Truncate at sentence boundary if article > 1500 tokens
    proc_article, truncated_input = truncate_article_at_sentence(article, tokenizer, max_tokens=1500)

    # Format prompt text
    prompt_template = PROMPTS[prompt_key]
    formatted_prompt = prompt_template.format(article=proc_article)

    # Build chat message with thinking mode explicitly turned off
    messages = [{"role": "user", "content": formatted_prompt}]
    try:
        # Qwen-specific apply_chat_template with enable_thinking=False
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback if tokenizer version doesn't accept enable_thinking
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Tokenize input prompt
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    input_token_count = model_inputs["input_ids"].shape[1]

    # Generate with deterministic settings (do_sample=False)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    # Slice off the input tokens to get strictly new generated tokens
    new_ids = generated_ids[0][input_token_count:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Strip thinking tags if any leaked
    clean_output = strip_thinking_tokens(raw_output)
    output_token_count = len(new_ids)

    elapsed = round(time.time() - t0, 2)
    return {
        "output": clean_output,
        "prompt_key": prompt_key,
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "seconds": elapsed,
        "truncated_input": truncated_input,
    }


def compare_prompts(article: str, max_new_tokens: int = 300) -> dict[str, dict]:
    """
    Run all three prompt variants on the same article.

    Args:
        article: Marathi news text.
        max_new_tokens: Generation length limit.

    Returns:
        dict mapping prompt_key -> generation result dict.
    """
    results = {}
    for p_key in ["zero_shot", "role_based", "role_constraints"]:
        results[p_key] = generate_brief(article, prompt_key=p_key, max_new_tokens=max_new_tokens)
    return results


# ===========================================================================
# Part 2: Objective Checks (Pure Measurement - No Quality Judgement)
# ===========================================================================

def has_three_numbered_items(output: str) -> bool:
    """
    Check if the output contains exactly 3 numbered items.
    Recognises ASCII (1, 2, 3) and Devanagari numerals (१, २, ३) with dots/brackets.
    """
    if not output or not output.strip():
        return False

    lines = output.splitlines()
    item_pattern = re.compile(
        r"^\s*(?:[1-9०-९]+[\.\)]|\[[1-9०-९]+\]|\([1-9०-९]+\))\s+"
    )
    numbered_lines = [line for line in lines if item_pattern.match(line)]
    return len(numbered_lines) == 3


def devanagari_ratio(output: str) -> float:
    """
    Calculate the share of alphabetic letter characters that are Devanagari.
    Formula: count(Devanagari letters) / (count(Devanagari letters) + count(Latin letters)).
    Returns 0.0 if no letters are present.
    """
    if not output or not output.strip():
        return 0.0

    devanagari_letters = len(re.findall(r"[\u0900-\u097F]", output))
    latin_letters = len(re.findall(r"[a-zA-Z]", output))
    total_letters = devanagari_letters + latin_letters

    if total_letters == 0:
        return 0.0
    return round(devanagari_letters / total_letters, 4)


def _marathi_stem(word: str) -> str:
    """Helper to remove common Marathi inflectional suffixes from a word."""
    w = word.strip().lower()
    for suff in MARATHI_SUFFIXES:
        if w.endswith(suff) and len(w) - len(suff) >= 3:
            return w[:-len(suff)]
    return w


def _is_stem_present(word: str, article_stems: set[str], article_words: set[str]) -> bool:
    """
    Check if a word's stem or root is present in the source article.
    Matches via suffix stripping OR shared prefix of >= 3-4 characters
    so inflected forms (e.g. कोहली vs कोहलीची) are NOT counted as new.
    """
    w_low = word.lower()
    # 1. Exact match
    if w_low in article_words:
        return True

    # 2. Suffix-stripped stem match
    w_stem = _marathi_stem(w_low)
    if w_stem in article_stems or w_stem in article_words:
        return True

    # 3. Prefix match: shared prefix of at least 4 characters (or 3 if word length <= 4)
    min_prefix_len = 3 if len(w_low) <= 4 else 4
    for a_w in article_words:
        # Check common prefix
        prefix_len = 0
        for c1, c2 in zip(w_low, a_w):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        if prefix_len >= min_prefix_len:
            return True

    return False


def novel_word_ratio(output: str, article: str) -> float:
    """
    Share of output words whose STEM is not found in the source article.
    Note: this is only a rewording hint, not a hallucination measure.
    """
    if not output or not article:
        return 0.0

    out_tokens = tokenize(clean_text(output))
    art_tokens = tokenize(clean_text(article))

    word_pattern = re.compile(r"[\u0900-\u097Fa-zA-Z0-9]")
    out_words = [w for w in out_tokens if word_pattern.search(w)]
    art_words = set(w.lower() for w in art_tokens if word_pattern.search(w))
    art_stems = set(_marathi_stem(w) for w in art_words)

    if not out_words:
        return 0.0

    novel_count = 0
    for w in out_words:
        if not _is_stem_present(w, art_stems, art_words):
            novel_count += 1

    return round(novel_count / len(out_words), 4)


def invented_name_candidates(output: str, article: str) -> str:
    """
    Identify words in the output that look like names (flagged by the NER model as
    Person, Location, Organization, or capitalised Latin words) AND whose stem is absent
    from the article.

    Note: this is only a rough hint to flag potential hallucinations;
    manual review is the real check.
    """
    if not output or not article:
        return "None"

    art_tokens = tokenize(clean_text(article))
    word_pattern = re.compile(r"[\u0900-\u097Fa-zA-Z0-9]")
    art_words = set(w.lower() for w in art_tokens if word_pattern.search(w))
    art_stems = set(_marathi_stem(w) for w in art_words)

    invented_candidates = set()

    # 1. Run NER on output
    try:
        ents = extract_entities(output)
        for ent in ents:
            if ent.get("label") in ["Person", "Location", "Organization"]:
                e_tokens = tokenize(ent.get("text", ""))
                for tok in e_tokens:
                    tok_clean = tok.strip()
                    if len(tok_clean) > 1 and word_pattern.search(tok_clean):
                        if not _is_stem_present(tok_clean, art_stems, art_words):
                            invented_candidates.add(tok_clean)
    except Exception as e:
        logger.warning(f"NER extraction failed during invented name detection: {e}")

    # 2. Capitalised Latin words
    out_tokens = tokenize(clean_text(output))
    for w in out_tokens:
        if re.match(r"^[A-Z][a-zA-Z]+$", w):
            if not _is_stem_present(w, art_stems, art_words):
                invented_candidates.add(w)

    if not invented_candidates:
        return "None"
    return ", ".join(sorted(invented_candidates))


def output_length_words(output: str) -> int:
    """Return total number of words in output text."""
    if not output or not output.strip():
        return 0
    words = [w for w in tokenize(clean_text(output)) if re.search(r"[\u0900-\u097Fa-zA-Z0-9]", w)]
    return len(words)
