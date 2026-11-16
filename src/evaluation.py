"""
MahaVarta: ROUGE evaluation of the summarisers (Stage 10)
Run from the project root:
    python -m src.evaluation              # all methods
    python -m src.evaluation --no-qwen    # TextRank + mT5 only
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

from rouge_score import rouge_scorer

from src.summarization import textrank_summary, summarize_auto, split_sentences
from src.prompting import generate_brief, PROMPTS

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = ROOT / "data" / "sample_articles"
REFS_DIR = ARTICLES_DIR / "references"
OUT_DIR = ROOT / "results" / "summarization"


class MarathiTokenizer:
    """rouge-score's default tokenizer removes Devanagari, which would give all zeros."""

    def tokenize(self, text):
        text = re.sub(r"[\u0964\u0965]", " ", text)
        text = re.sub(r"[^\u0900-\u097FA-Za-z0-9]+", " ", text)
        return text.lower().split()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def get_methods(use_qwen: bool = True) -> dict:
    methods = {
        "TextRank": lambda t: textrank_summary(t, 3),
        "mT5": lambda t: summarize_auto(t)["summary"],
    }
    if use_qwen:
        for key in PROMPTS:
            methods[f"Qwen3 {key}"] = lambda t, k=key: generate_brief(t, k)["output"]
    return methods


def evaluate(use_qwen: bool = True) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=MarathiTokenizer())
    methods = get_methods(use_qwen)
    rows, outputs = [], {}

    for article_path in sorted(ARTICLES_DIR.glob("*.txt")):
        text = read_text(article_path)
        ref_path = REFS_DIR / article_path.name
        if ref_path.exists():
            reference, ref_type = read_text(ref_path), "human"
        else:
            reference, ref_type = " ".join(split_sentences(text)[:3]), "lead3_proxy"
            print(f"WARNING: no reference for {article_path.name}; using lead-3 proxy.")
        outputs[article_path.name] = {"reference": reference}

        for name, fn in methods.items():
            t0 = time.time()
            try:
                output = str(fn(text))
            except Exception as exc:
                print(f"{article_path.name} / {name} FAILED: {exc}")
                continue
            seconds = round(time.time() - t0, 1)
            scores = scorer.score(reference, output)
            outputs[article_path.name][name] = output
            row = {
                "article": article_path.name,
                "method": name,
                "rouge1": round(scores["rouge1"].fmeasure, 4),
                "rouge2": round(scores["rouge2"].fmeasure, 4),
                "rougeL": round(scores["rougeL"].fmeasure, 4),
                "seconds": seconds,
                "reference": ref_type,
            }
            rows.append(row)
            print(f"{article_path.name:24s} {name:22s} R1={row['rouge1']:.3f} "
                  f"R2={row['rouge2']:.3f} RL={row['rougeL']:.3f} {seconds}s [{ref_type}]")

    (OUT_DIR / "rouge_outputs.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def save_results(rows: list[dict]) -> None:
    if not rows:
        print("No results to save.")
        return
    with open(OUT_DIR / "rouge_scores.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for r in rows:
        grouped.setdefault(r["method"], []).append(r)
    with open(OUT_DIR / "rouge_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "rouge1", "rouge2", "rougeL", "avg_seconds", "n_articles"])
        for method, rs in grouped.items():
            mean = lambda k: round(sum(x[k] for x in rs) / len(rs), 4)
            writer.writerow([method, mean("rouge1"), mean("rouge2"), mean("rougeL"),
                             mean("seconds"), len(rs)])
    print(f"\nSaved to {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qwen", action="store_true", help="skip the Qwen3 prompt variants")
    args = parser.parse_args()
    save_results(evaluate(use_qwen=not args.no_qwen))


if __name__ == "__main__":
    main()