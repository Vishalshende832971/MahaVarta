# Model Configuration

This directory holds model configuration files and notes.
Trained model weights (`.joblib`, `.pkl`, `.bin`) are excluded from Git.

## Models Used

| Model | ID | Task | Loaded from |
|---|---|---|---|
| TF-IDF + LinearSVC | local | Classification | Trained from MahaNews |
| mT5 XL-Sum | `csebuetnlp/mT5_multilingual_XLSum` | Summarisation | Hugging Face Hub |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | News brief | Hugging Face Hub |
| MahaSent-MD | `l3cube-pune/marathi-sentiment-md` | Sentiment | Hugging Face Hub |
| MahaNER | `l3cube-pune/marathi-ner` | NER | Hugging Face Hub |
| LDA | local | Topic modelling | Trained from MahaNews |
