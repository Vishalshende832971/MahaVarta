import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
_tok = _model = None


def _load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
    return _tok, _model


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]


def chunk_text(text, tok, max_tokens=450):
    """Pack whole sentences into chunks that fit mT5's 512-token input."""
    chunks, cur, cur_len = [], [], 0
    for s in split_sentences(text):
        n = len(tok.encode(s, add_special_tokens=False))
        if cur and cur_len + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@torch.no_grad()
def _generate(text, max_len=84, min_len=20):
    tok, model = _load()
    inp = tok(text, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inp, max_length=max_len, min_length=min_len,
                         num_beams=4, no_repeat_ngram_size=2)
    return tok.decode(out[0], skip_special_tokens=True)


def summarize(text, max_len=84):
    tok, _ = _load()
    text = " ".join(text.split())
    chunks = chunk_text(text, tok)
    if len(chunks) <= 1:
        return _generate(text, max_len)
    partials = [_generate(c, max_len) for c in chunks]      # summarize each chunk
    return _generate(" ".join(partials), max_len + 40)      # second pass to merge


if __name__ == "__main__":
    import sys
    print(summarize(open(sys.argv[1], encoding="utf-8").read()))