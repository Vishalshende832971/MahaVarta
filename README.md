# MahaVarta: Marathi News Classification & Summarization

**NLP Mini Project — MIT Academy of Engineering, Alandi, Pune**

**Team:** Vishal Gorakh Shende (PRN 202401110034), Gufran Shah (PRN [TODO])

---

## About

MahaVarta is an NLP application that analyses Marathi news articles. A user pastes a Marathi news article or uploads a `.txt` file, and the application returns:

- **News classification** into 12 categories (TF-IDF + Linear SVM)
- **Abstractive summary** (mT5 multilingual XL-Sum) with an extractive TextRank baseline
- **Prompt-engineered news brief** (Qwen3-0.6B with three prompt strategies)
- **Sentiment analysis** (MahaSent-MD pretrained model)
- **Named Entity Recognition** (MahaNER pretrained model)
- **Topic discovery** (LDA)
- **Preprocessing view** showing normalisation, cleaning, tokenisation and TF-IDF

## Setup

**Recommended Python Version:** **Python 3.12** (Python 3.12.8 64-bit recommended; fully supports PyTorch, Hugging Face Transformers, Gradio, and Gensim binary wheels without requiring C++ build tools).

```bash
# Clone the repository
git clone https://github.com/Vishalshende832971/MahaVarta.git
cd MahaVarta

# Create virtual environment with Python 3.12
py -3.12 -m venv venv312

# Activate virtual environment
# Windows (PowerShell):
.\venv312\Scripts\Activate.ps1
# Windows (cmd):
.\venv312\Scripts\activate.bat
# Linux/Mac:
source venv312/bin/activate

# Install verified pinned dependencies
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

The Gradio interface opens in your browser. Paste a Marathi news article or upload a `.txt` file and click **Analyse**.

## Project Structure

```
marathi-news-intelligence/
├── app.py                         # Gradio dashboard
├── requirements.txt               # Pinned dependencies
├── README.md
├── LICENSE
├── DATASET_AND_MODEL_LICENSES.md
├── data/
│   ├── sample_articles/           # Sample articles for testing
│   ├── annotations/               # 200 labelled headlines
│   └── dataset_sources.md
├── notebooks/                     # Exploration and experiment notebooks
├── src/
│   ├── preprocessing.py
│   ├── classification.py
│   ├── sentiment.py
│   ├── ner.py
│   ├── summarization.py
│   ├── prompting.py
│   ├── topic_modeling.py
│   └── evaluation.py
├── models/model_config.md
├── results/                       # All evaluation outputs
├── screenshots/
├── poster/
└── tests/
```

## Datasets and Models

| Task | Resource | Source | Split Sizes (Verified) |
|---|---|---|---|
| Classification | MahaNews (12 categories) | L3Cube MarathiNLP (LPC) | Train: 42,870 \| Val: 5,357 \| Test: 5,366 (Total: 53,593) |
| Sentiment | MahaSent-MD | `l3cube-pune/marathi-sentiment-md` | Pretrained inference |
| NER | MahaNER | `l3cube-pune/marathi-ner` | Train: 21,500 \| Val: 1,500 \| Test: 2,000 |
| Summarisation | XL-Sum Marathi | `csebuetnlp/mT5_multilingual_XLSum` | XL-Sum Marathi test subset |
| News brief | Qwen3-0.6B | `Qwen/Qwen3-0.6B` | Evaluation sample articles |

Raw datasets and model weights are **not included** in this repository. They are downloaded at runtime.

## Evaluation

Results are saved under `results/` and are generated from actual code runs. See [DATASET_AND_MODEL_LICENSES.md](DATASET_AND_MODEL_LICENSES.md) for licence details.

## Limitations

- The sentiment model was not trained on news data; results are validated on manually labelled headlines.
- Abstractive and LLM-generated text may contain factual errors; outputs are marked as machine-generated.
- Free hosting limits speed and availability.

## Team

- **Vishal Gorakh Shende** (PRN 202401110034)
- **Gufran Shah** (PRN [TODO])

## Acknowledgements

- L3Cube Pune for the MarathiNLP datasets and models
- Hugging Face for model hosting and Spaces
- csebuetnlp for the mT5 XL-Sum model
- Qwen Team for Qwen3-0.6B
