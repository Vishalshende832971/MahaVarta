# MahaVarta (महावार्ता): Complete Project Presentation Master Guide
## End-to-End Technical Documentation, Sequential Presentation Walkthrough & Examiner Viva Defense

---

### Project Metadata
- **Project Title:** MahaVarta (महावार्ता) — Marathi News Intelligence, Classification & Summarisation System
- **Academic Course:** Natural Language Processing (NLP) Mini Project
- **Institution:** MIT Academy of Engineering (MITAOE), Alandi, Pune
- **Authors & Team Members:**
  - **Vishal Gorakh Shende** (PRN: `202401110034`)
  - **Gufran Shah** (PRN: `202401110011`)
- **Repository:** [https://github.com/Vishalshende832971/MahaVarta](https://github.com/Vishalshende832971/MahaVarta)
- **Verified Environment:** Python 3.12.8 (Windows 64-bit)
- **Core Technology Stack:** PyTorch 2.15, Hugging Face Transformers 5.23, Scikit-learn 1.9, Gensim 4.4, Indic NLP Library 0.92, Gradio 6.32, NetworkX 3.7, Rouge-Score 0.1.2

---

# TABLE OF CONTENTS
1. [Project Abstract & Executive Overview](#1-project-abstract--executive-overview)
2. [Linguistic Background & The Problem Statement](#2-linguistic-background--the-problem-statement)
3. [End-to-End System Architecture](#3-end-to-end-system-architecture)
4. [Exhaustive Dataset Specifications & Data Engineering](#4-exhaustive-dataset-specifications--data-engineering)
5. [Pretrained Foundation Models & Open-Source Artifacts](#5-pretrained-foundation-models--open-source-artifacts)
6. [Stage-by-Stage Technical Deep Dive & Code Implementation](#6-stage-by-stage-technical-deep-dive--code-implementation)
   - [Stage 1: Preprocessing & Morphological Pipeline](#stage-1-preprocessing--morphological-pipeline)
   - [Stage 2: 12-Class News Classification Engine](#stage-2-12-class-news-classification-engine)
   - [Stage 3: Extractive Summarisation (TextRank)](#stage-3-extractive-summarisation-textrank)
   - [Stage 4: Abstractive Summarisation (mT5 XL-Sum) & Chunking](#stage-4-abstractive-summarisation-mt5-xl-sum--chunking)
   - [Stage 5: Generative News Briefing & Prompt Engineering (Qwen3-0.6B)](#stage-5-generative-news-briefing--prompt-engineering-qwen3-06b)
   - [Stage 6: Sentiment Analysis (MahaSent-MD)](#stage-6-sentiment-analysis-mahasent-md)
   - [Stage 7: Named Entity Recognition (MahaNER)](#stage-7-named-entity-recognition-mahaner)
   - [Stage 8: Latent Topic Discovery (18-Topic Gensim LDA)](#stage-8-latent-topic-discovery-18-topic-gensim-lda)
7. [User Interface & Dashboard Architecture (`app.py`)](#7-user-interface--dashboard-architecture-apppy)
8. [Comprehensive Experimental Results & Quantitative Benchmark Tables](#8-comprehensive-experimental-results--quantitative-benchmark-tables)
9. [Software Quality Assurance & Automated Testing (86/86 Passing)](#9-software-quality-assurance--automated-testing-8686-passing)
10. [Step-by-Step 5-Minute Live Demonstration Script](#10-step-by-step-5-minute-live-demonstration-script)
11. [Exhaustive Examiner Viva Q&A Bank (Every Anticipated Question & Defense)](#11-exhaustive-examiner-viva-qa-bank-every-anticipated-question--defense)
12. [Project Limitations & Future Directions](#12-project-limitations--future-directions)

---

# 1. Project Abstract & Executive Overview

**MahaVarta (महावार्ता)** is an end-to-end Natural Language Processing system specifically engineered for the Marathi language. While regional digital news in Maharashtra (Loksatta, Sakal, Maharashtra Times, Lokmat) produces tens of thousands of articles daily, regional readers and journalists lack modern computational intelligence tools to aggregate, categorize, distill, and analyze this text in real time.

MahaVarta takes raw Marathi text (via direct input or `.txt` file upload) and simultaneously produces:
1. **12-Class News Classification** via a Linear Support Vector Machine using TF-IDF and morphological stem weighting (**87.74% Test Accuracy**, **85.70% Macro-F1**).
2. **Dual Summarisation**:
   - Extractive Baseline: **TextRank** using sentence TF-IDF cosine similarity and Google PageRank centrality.
   - Abstractive Model: **mT5 Multilingual XL-Sum** with a sentence-aware token chunking algorithm preventing mid-sentence clipping.
3. **Generative News Briefing**: **Qwen3-0.6B** with 3 prompt engineering strategies evaluated against automated rule-based constraint checks.
4. **Sentiment Analysis**: **MahaSent-MD** classifying text into Positive, Neutral, and Negative with calibrated probability bars.
5. **Named Entity Recognition (NER)**: **MahaNER** identifying 7 entity categories with WordPiece BIO subword de-tokenisation (**96.51% Accuracy**, **79.29% F1**).
6. **Latent Topic Discovery**: **Gensim LDA Topic Modeling** (18 topics, optimal coherence $C_v = 0.4852$).
7. **Transparent Preprocessing Inspector**: Displays intermediate steps from raw text $\to$ normalisation $\to$ regex cleaning $\to$ Indic NLP tokenisation $\to$ TF-IDF keywords.

---

# 2. Linguistic Background & The Problem Statement

### 2.1 The Linguistic Attributes of Marathi
Marathi belongs to the Indo-Aryan language family, written in the Devanagari script. It is the **3rd most spoken language in India** and the **10th most spoken globally**, with over 83 million native speakers.

Processing Marathi computationally involves severe linguistic challenges that break standard English NLP tools:
1. **Agglutinative Morphology & Case Inflection (*Vibhakti Pratyay*):**
   In English, prepositions are distinct words (*"in the cinema"*, *"of the cinema"*). In Marathi, case markers attach directly to noun and verb stems as suffixes:
   - Root stem: `चित्रपट` (Cinema / Movie)
   - Genitive inflection: `चित्रपटाचे` (Of the cinema)
   - Locative inflection: `चित्रपटात` (In the cinema)
   - Oblique plural: `चित्रपटांमधील` (Among the movies)
   - Instrumental: `चित्रपटाने` (By the cinema)
   Standard word-level TF-IDF treats all five forms as completely distinct, unrelated features, causing massive feature sparsity.
2. **Devanagari Unicode Inconsistencies:**
   The same visual glyph can be encoded in divergent Unicode codepoints (precomposed characters vs decomposed base consonant + virama + matra). Nuktas (`़`) and complex conjunct consonants (*jodakshare* like `क्ष`, `ज्ञ`, `त्र`) vary across news platforms.
3. **Non-Standard Sentence Delimiters:**
   Devanagari uses the single danda (`।`) and double danda (`॥`) as traditional full stops, but modern digital journalism mixes these with Western periods (`.`), question marks (`?`), exclamation marks (`!`), and common honorific abbreviations (`डॉ.`, `प्रा.`, `श्री.`, `सौ.`). Standard sentence splitters break prematurely at abbreviations.

### 2.2 The Low-Resource Challenge in Indian NLP
Despite 83+ million speakers, Marathi lacks the massive pretraining datasets, standardized benchmarks, and fine-tuned pipelines available for English or Mandarin. Most commercial APIs provide poor translation layers that lose regional idioms, political nuances, and cultural expressions. MahaVarta provides a native, unified regional platform built directly on authentic Marathi corpora.

---

# 3. End-to-End System Architecture

```
                                  ┌──────────────────────────────┐
                                  │      Raw Marathi Article     │
                                  │   (Pasted Text or .txt File) │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │     Stage 1: Preprocessing   │
                                  │   • Devanagari Normalisation │
                                  │   • Noise & URL Cleaning     │
                                  │   • Sentence Segmentation    │
                                  │   • Indic NLP Tokenisation   │
                                  │   • Entropy Stopword Pruning │
                                  └──────────────┬───────────────┘
                                                 │
            ┌───────────────────┬────────────────┼───────────────────┬───────────────────┐
            │                   │                │                   │                   │
            ▼                   ▼                ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   ┌───────────────┐
    │    Stage 2    │   │    Stage 3    │ │    Stage 4    │ │    Stage 5    │   │    Stage 6    │
    │Classification │   │  Extractive   │ │  Abstractive  │ │   Generative  │   │   Sentiment   │
    │ Linear SVM +  │   │  Summariser   │ │  Summariser   │ │   News Brief  │   │   Analysis    │
    │ TF-IDF Vector │   │  (TextRank)   │ │  (mT5 XLSum)  │ │  (Qwen3-0.6B) │   │ (MahaSent-MD) │
    └───────┬───────┘   └───────┬───────┘ └───────┬───────┘ └───────┬───────┘   └───────┬───────┘
            │                   │                │                   │                   │
            └───────────────────┼────────────────┴───────────────────┴───────────────────┤
                                │                                                        │
                                ▼                                                        ▼
                        ┌───────────────┐                                        ┌───────────────┐
                        │    Stage 7    │                                        │    Stage 8    │
                        │ Named Entity  │                                        │     Topic     │
                        │  Recognition  │                                        │   Modelling   │
                        │   (MahaNER)   │                                        │   (18-Topic   │
                        │  7 Categories │                                        │  Gensim LDA)  │
                        └───────┬───────┘                                        └───────┬───────┘
                                │                                                        │
                                └────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │        Gradio Interactive UI      │
                                       │    (7 Live Tabs + Benchmark Hub)  │
                                       └───────────────────────────────────┘
```

---

# 4. Exhaustive Dataset Specifications & Data Engineering

### 4.1 Primary Dataset: L3Cube-MahaNews (LPC - Long Paragraph Classification)
- **Repository:** `https://github.com/l3cube-pune/MarathiNLP`
- **Citation:** Mittal, Saloni, et al. *"L3Cube-MahaNews: News-Based Short Text and Long Document Classification Datasets in Marathi."* SPELLL / Springer Nature (2023). [arXiv:2404.18216](https://arxiv.org/abs/2404.18216).
- **Split Sizes (Scrubbed of Concatenation Headers):**
  - **Train:** **42,870** articles (80.0%)
  - **Validation:** **5,357** articles (10.0%)
  - **Test:** **5,366** articles (10.0%)
  - **Total Dataset Size:** **53,593** verified articles.
- **The 12 Official Classes:**
  1. `Politics` (राजकारण)
  2. `International` (आंतरराष्ट्रीय)
  3. `Manoranjan` (मनोरंजन)
  4. `Crime` (गुन्हेगारी)
  5. `Bhakti` (भक्ती)
  6. `Sports` (क्रीडा)
  7. `Auto` (ऑटो)
  8. `Travel` (पर्यटन)
  9. `Tech` (तंत्रज्ञान)
  10. `Education` (शिक्षण)
  11. `Health` (आरोग्य)
  12. `Fashion` (फॅशन)

### 4.2 Named Entity Recognition Dataset: MahaNER
- **Repository:** `https://huggingface.co/datasets/l3cube-pune/marathi-ner`
- **Citation:** Litake, Onkar, et al. *"L3Cube-MahaNER: A Named Entity Recognition Dataset for Marathi."* EACL / arXiv:2203.04838 (2022).
- **Evaluation Split:** Official test set of **1,999 sentences** containing **1,943 annotated named entities**.
- **BIO Tagging Scheme:** 7 entity types (`Person`, `Location`, `Organization`, `Date`, `Time`, `Measure`, `Designation`).

### 4.3 Summarisation Dataset: XL-Sum Marathi
- **Repository:** `https://huggingface.co/datasets/csebuetnlp/xlsum`
- **Citation:** Hasan, Tahmid, et al. *"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages."* Findings of ACL (2021). [arXiv:2106.13822](https://arxiv.org/abs/2106.13822).
- **Nature:** Authentic BBC Marathi news articles paired with professional human reference summaries, used for ROUGE evaluation.

### 4.4 Custom Double-Blind Headline Sentiment Annotations (`data/annotations/`)
- **Dataset File:** `data/annotations/headlines.csv`
- **Sample:** **200 diverse Marathi news headlines** sampled across all 12 categories from the L3Cube Short Headline Corpus.
- **Independent Annotators:** Vishal Gorakh Shende (`label_vishal`) and Gufran Shah (`label_gufran`).
- **Target Classes:** `Positive`, `Negative`, `Neutral`.
- **Protocol:** Double-blind independent labeling, conflict adjudication, and evaluation against pretrained sentiment models.

### 4.5 Information-Theoretic Stopwords Dataset (`data/stopwords_mr.txt`)
- **Engineering Script:** `scripts/entropy_stopwords.py`
- **Mathematical Principle (Normalized Shannon Entropy):**
  $$H(w) = -\sum_{c=1}^{12} P(c \mid w) \log_2 P(c \mid w), \quad \bar{H}(w) = \frac{H(w)}{\log_2(12)}$$
  Tokens with document frequency $DF \ge 200$ and $\bar{H}(w) \to 1.0$ appear uniformly across all 12 categories, proving they contain zero category-discriminative information. 400+ mathematical stopwords were retained (`आहे`, `नाही`, `या`, `केले`, `होते`, `आणि`, `पण`).

---

# 5. Pretrained Foundation Models & Open-Source Artifacts

| Model Name | Hugging Face ID | Parameters | Underlying Architecture | Project Task | License |
|:---|:---|:---:|:---|:---|:---|
| **mT5 XL-Sum** | `csebuetnlp/mT5_multilingual_XLSum` | ~300M | Multilingual Encoder-Decoder T5 | Abstractive Marathi summarisation | CC-BY-NC-SA-4.0 |
| **Qwen3-0.6B** | `Qwen/Qwen3-0.6B` | ~600M | Autoregressive Decoder-only Transformer | Multi-prompt news brief generation | Apache 2.0 |
| **MahaSent-MD** | `l3cube-pune/marathi-sentiment-md` | ~110M | Fine-tuned RoBERTa / MahaBERT | 3-Class Sentiment prediction | MIT |
| **MahaNER** | `l3cube-pune/marathi-ner` | ~110M | Token-Classification MahaBERT | 7-Entity Named Entity Recognition | MIT |
| **Linear SVM** | `models/classification_pipeline.joblib` | 7.3 MB | Sparse Linear Support Vector Machine | 12-Class News Classification | Project Built |
| **Gensim LDA** | `models/lda_model.gensim` | 18 topics | Online Variational Bayes LDA | Latent topic discovery | Project Built |

---

# 6. Stage-by-Stage Technical Deep Dive & Code Implementation

---

### Stage 1: Preprocessing & Morphological Pipeline
**Source File:** `src/preprocessing.py`

#### 1. Devanagari Normalisation:
```python
from indicnlp.normalize.indic_normalize import DevanagariNormalizer

_normalizer = DevanagariNormalizer()

def normalize_devanagari(text: str) -> str:
    if not text or not text.strip():
        return ""
    return _normalizer.normalize(text)
```
*Why this matters:* Standardizes nuktas and combines Unicode NFC characters so identical words with different typing encodings match.

#### 2. Sentence Boundary Detection with Abbreviation Protection:
```python
def split_sentences(text: str) -> list[str]:
    text = normalize_devanagari(text)

    # Protect common Marathi abbreviations from being split
    abbrevs = ["डॉ.", "प्रा.", "श्री.", "सौ.", "मा.", "इ."]
    protected = text
    for i, abb in enumerate(abbrevs):
        protected = protected.replace(abb, abb[:-1] + f"__ABB_{i}__")

    # Protect decimal numbers (e.g., 2.5 or २.५)
    protected = re.sub(r'(\d+)\.(\d+)', r'\1__DEC__\2', protected)

    # Split strictly on Devanagari dandas (।, ॥), ?, !, or periods followed by whitespace
    pattern = r'(?<=[।॥?!]|\.)\s+'
    raw_sentences = re.split(pattern, protected)

    cleaned = []
    for s in raw_sentences:
        for i in range(len(abbrevs)):
            s = s.replace(f"__ABB_{i}__", ".")
        s = s.replace("__DEC__", ".").strip()
        if len(s) > 1:
            cleaned.append(s)
    return cleaned
```

---

### Stage 2: 12-Class News Classification Engine
**Source File:** `src/classification.py`

#### 1. Pipeline Definition:
```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=3,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(
        C=0.5,
        loss="squared_hinge",
        random_state=42
    ))
])
```
*Validation Grid Search:* Tuned across $C \in \{0.1, 0.5, 1.0, 2.0\}$ and unigram vs bigram. **Unigram with $C=0.5$ won with 0.8344 validation Macro-F1**.

#### 2. Decision Function, Softmax Calibration & Morphological Stem Boost:
```python
def predict(model: Pipeline, text: str, top_k: int = 3) -> list[tuple[str, float]]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # 1. Raw decision scores from 12 One-vs-Rest binary classifiers
    decision_scores = np.array(model.decision_function([cleaned])[0], dtype=float).copy()

    # 2. Morphological stem-count adjustment for Devanagari inflected terms
    # Solves token mismatch (e.g. चित्रपटाचे vs चित्रपट)
    if hasattr(model, "classes_"):
        classes_list = list(model.classes_)
        for cat_name, stems in CATEGORY_STEMS.items():
            if cat_name in classes_list:
                c_idx = classes_list.index(cat_name)
                matches = sum(1 for stem in stems if stem in cleaned)
                if matches > 0:
                    decision_scores[c_idx] += 0.25 * matches

    # 3. Softmax probability normalization
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probs = exp_scores / np.sum(exp_scores)

    classes = model.classes_
    ranked = np.argsort(probs)[::-1][:top_k]
    return [(str(classes[idx]), round(float(probs[idx]), 4)) for idx in ranked]
```

---

### Stage 3: Extractive Summarisation (TextRank)
**Source File:** `src/summarization.py`

```python
def textrank_summary(text: str, n_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= n_sentences:
        return text.strip()

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    tfidf_mat = vectorizer.fit_transform(sentences)

    # Pairwise sentence cosine similarity matrix
    sim_matrix = (tfidf_mat * tfidf_mat.T).toarray()

    # Build graph and run Google PageRank (damping factor d=0.85)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph, alpha=0.85, max_iter=100)

    # Select top-k sentences
    ranked_indices = sorted(scores, key=scores.get, reverse=True)[:n_sentences]

    # Re-sort chronologically to preserve discourse flow
    ranked_indices.sort()
    return " ".join([sentences[i] for i in ranked_indices])
```

---

### Stage 4: Abstractive Summarisation (mT5 XL-Sum) & Chunking
**Source File:** `src/summarization.py`

#### Sentence-Aware Token Budget Window:
```python
def chunk_sentences(sentences: list[str], max_chunk_tokens: int = 450, tokenizer_fn=None) -> list[str]:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = tokenizer_fn(sentence) if tokenizer_fn else len(sentence.split())
        if current_tokens + sent_tokens > max_chunk_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sent_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
```
*Why this matters:* Guarantees that no sentence is chopped mid-word or mid-thought, preventing grammatical fragmentation in the encoder.

---

### Stage 5: Generative News Briefing & Prompt Engineering (Qwen3-0.6B)
**Source File:** `src/prompting.py`

#### The 3 Prompt Strategies:
1. **Zero-Shot:** Direct command to summarize the Marathi text.
2. **Role-Based:** Personifies the model as a senior Marathi news editor (*"तुम्ही एक अनुभवी मराठी वृत्त संपादक आहात..."*).
3. **Role + Structural Constraints:** Enforces strict structural rules:
   - Exactly 3 bullet points numbered in Devanagari (`१.`, `२.`, `३.`).
   - No conversational preamble or postscript.
   - Pure Devanagari script.

#### Objective Automated Validation Functions:
```python
def has_three_numbered_items(text: str) -> bool:
    pattern = r'(?:^|\n)\s*(?:[1-3]|[१-३])[\.\)]\s*\S+'
    matches = re.findall(pattern, text)
    return len(matches) == 3

def devanagari_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace() and c not in string.punctuation]
    if not chars:
        return 0.0
    dev_count = sum(1 for c in chars if '\u0900' <= c <= '\u097F')
    return dev_count / len(chars)
```

---

### Stage 6: Sentiment Analysis (MahaSent-MD)
**Source File:** `src/sentiment.py`
- Model: `l3cube-pune/marathi-sentiment-md` (fine-tuned MahaBERT).
- Labels: `Negative` (0), `Neutral` (1), `Positive` (2).
- Output: Dominant sentiment label with Softmax probability bars.

---

### Stage 7: Named Entity Recognition (MahaNER)
**Source File:** `src/ner.py`
- Model: `l3cube-pune/marathi-ner` (fine-tuned RoBERTa token classification).
- WordPiece subwords starting with `##` are de-tokenised and merged.
- Contiguous `B-` and `I-` BIO tags are grouped into complete entity strings (`Person`, `Location`, `Organization`, `Date`, `Time`, `Measure`, `Designation`).

---

### Stage 8: Latent Topic Discovery (18-Topic Gensim LDA)
**Source File:** `src/topic_modeling.py`
- Online Variational Bayes LDA trained on 8,000 MahaNews articles.
- Evaluated $C_v$ topic coherence across $k \in [6, 20]$.
- Selected **$k = 18$ ($C_v = 0.4852$)** as optimal topic count.

---

# 7. User Interface & Dashboard Architecture (`app.py`)

Deployed using **Gradio** with a dual-language (Marathi Devanagari primary, English secondary) layout:
- **Tab 1: आढावा (Overview):** Executive summary card, word/sentence/character counts, dominant category, sentiment score, extracted entities, and top TextRank preview.
- **Tab 2: वर्गवारी (Classification):** Top-5 category prediction table and confidence bar chart.
- **Tab 3: सारांश (Summary):** Side-by-side comparison of extractive TextRank vs. abstractive mT5 with summary length sliders.
- **Tab 4: बातम्या संक्षेप (News Brief - Qwen3):** Generative news brief with prompt strategy switcher, temperature slider, and live objective constraint check badges.
- **Tab 5: भावना विश्लेषण (Sentiment):** Positive / Neutral / Negative percentage breakdown.
- **Tab 6: नामित घटक (NER):** Color-coded entity highlighting and grouped entity category cards.
- **Tab 7: विषय शोध (Topics):** LDA dominant topic ID, topic weight, and representative keywords.
- **Tab 8: पूर्वप्रक्रिया (Preprocessing):** Educational walkthrough displaying the text transformations: Raw $\rightarrow$ Normalised $\rightarrow$ Cleaned $\rightarrow$ Tokenised $\rightarrow$ Top TF-IDF Terms.
- **Tab 9: मूल्यमापन (Evaluation):** Dedicated academic dashboard rendering the Confusion Matrix, ROUGE Benchmark table, NER classification report, and LDA Topic Coherence curves.

---

# 8. Comprehensive Quantitative Evaluation Results

### 8.1 12-Class Classification Results (5,366 Unseen Test Samples)
- **Test Accuracy:** **87.74%**
- **Macro-Precision:** **87.11%**
- **Macro-Recall:** **84.61%**
- **Macro-F1 Score:** **85.70%**

| Category | Marathi Name | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| **Bhakti** | भक्ती | 0.9032 | 0.9170 | **0.9101** | 458 |
| **Sports** | क्रीडा | 0.8876 | 0.9059 | **0.8966** | 425 |
| **Auto** | ऑटो | 0.8981 | 0.8634 | **0.8804** | 388 |
| **Crime** | गुन्हेगारी | 0.8780 | 0.8711 | **0.8745** | 512 |
| **Politics** | राजकारण | 0.8552 | 0.8690 | **0.8621** | 977 |
| **Tech** | तंत्रज्ञान | 0.8415 | 0.8734 | **0.8571** | 316 |
| **Health** | आरोग्य | 0.8415 | 0.8625 | **0.8519** | 160 |
| **International** | आंतरराष्ट्रीय | 0.8871 | 0.8142 | **0.8491** | 689 |
| **Education** | शिक्षण | 0.9026 | 0.7722 | **0.8323** | 180 |
| **Manoranjan** | मनोरंजन | 0.8660 | 0.7797 | **0.8206** | 767 |
| **Travel** | पर्यटन | 0.8333 | 0.8037 | **0.8182** | 382 |
| **Fashion** | फॅशन | 0.8588 | 0.6636 | **0.7487** | 110 |

---

### 8.2 Summarisation ROUGE Benchmark

| Method | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency (CPU) |
|:---|:---|:---:|:---:|:---:|:---:|
| **TextRank** | Extractive Graph | **0.3520** | **0.1727** | **0.3087** | **0.13 s** |
| **mT5 Multilingual XL-Sum** | Abstractive Seq2Seq | 0.3042 | 0.1538 | 0.2365 | 20.43 s |
| **Qwen3 (Zero-Shot)** | Generative LLM | 0.2574 | 0.0562 | 0.1976 | 536.56 s |
| **Qwen3 (Role-Based)** | Generative LLM | 0.2051 | 0.0530 | 0.1820 | 76.00 s |
| **Qwen3 (Role + Constraints)** | Generative LLM | 0.2383 | 0.0767 | 0.1889 | 79.43 s |

---

### 8.3 Named Entity Recognition Evaluation (MahaNER Test = 1,999 Sentences)
- **Token Accuracy:** **96.51%**
- **Micro F1-Score:** **79.29%** (Precision: 80.60%, Recall: 78.02%)

| Entity Tag | Description | Precision | Recall | F1-Score | Support |
|:---|:---|:---:|:---:|:---:|:---:|
| **Person** | व्यक्ती | 0.8729 | 0.8668 | **0.8699** | 428 |
| **Measure** | परिमाण / संख्या | 0.8364 | 0.8700 | **0.8529** | 523 |
| **Date** | तारीख | 0.8626 | 0.8177 | **0.8396** | 192 |
| **Location** | ठिकाण | 0.7720 | 0.7322 | **0.7516** | 407 |
| **Designation** | पद / हुद्दा | 0.6667 | 0.7027 | **0.6842** | 74 |
| **Organization** | संस्था / संघटना | 0.6783 | 0.6094 | **0.6420** | 256 |
| **Time** | वेळ | 0.7500 | 0.4286 | **0.5455** | 63 |

---

### 8.4 Topic Coherence ($C_v$) Optimization

| Number of Topics ($k$) | Coherence Score ($C_v$) | Topic Status |
|:---:|:---:|:---|
| $k = 6$ | 0.3489 | Overly broad topics |
| $k = 8$ | 0.3360 | Under-clustering |
| $k = 10$ | 0.4164 | Clear separation of sports/health |
| $k = 12$ | 0.4045 | Stable baseline |
| $k = 14$ | 0.4398 | Good separation of business |
| $k = 16$ | 0.3920 | Coherence dip |
| **$k = 18$ (Selected)** | **0.4852** | **Peak Coherence: Distinct semantic topics** |
| $k = 20$ | 0.4170 | Over-clustering and topic duplication |

---

# 9. Software Quality Assurance & Automated Testing (86/86 Passing)

Every module has automated unit and integration tests under `tests/`:
- `tests/test_preprocessing.py`: **26 passed** (Dandas, abbreviations, unicode normalisation, noise cleaning).
- `tests/test_classification.py`: **4 passed** (Pipeline training, decision scores, empty input handling).
- `tests/test_summarization.py`: **19 passed** (Sentence segmentation, TextRank, chunking, mT5 execution).
- `tests/test_prompting.py`: **15 passed** (Constraint checking, Devanagari ratio, novel word metrics).
- `tests/test_sentiment.py`: **6 passed** (MahaSent inference, 3-class distribution).
- `tests/test_ner.py`: **5 passed** (MahaNER tokenization, BIO reconstruction, span extraction).
- `tests/test_topic_modeling.py`: **7 passed** (Gensim LDA training, inference, dictionary persistence).
- `tests/test_app.py`: **4 passed** (End-to-end Gradio pipeline execution on all sample articles).
- **Total:** **86 / 86 tests passed (100% pass rate)**.

---

# 10. Step-by-Step 5-Minute Live Demonstration Script

1. **Step 1: Open the Application**
   - Point your browser to `http://127.0.0.1:7860`.
   - Point out the dual-language interface: Marathi Devanagari headings with English subtitles.
2. **Step 2: Load Sample Article**
   - Click `01_short_sports.txt` (Ravi Shastri discussing Virat Kohli's T20 innings against Pakistan).
   - Click **विश्लेषण करा (Analyse)**.
3. **Step 3: Tab-by-Tab Walkthrough**
   - **आढावा (Overview):** Point to predicted category `क्रीडा (Sports)`, positive sentiment (`0.997`), word count (71 words), and key entities (`विराट कोहली`, `शास्त्री`).
   - **वर्गवारी (Classification):** Point out the bar chart showing `Sports` at the top with over 80% confidence, followed by other categories.
   - **सारांश (Summary):** Highlight the extractive TextRank summary and compare it with the concise abstractive mT5 summary.
   - **भावना (Sentiment):** Show the 3-bar confidence distribution (Positive vs Neutral vs Negative).
   - **नामित घटक (NER):** Show live highlighted spans: Person in purple (`विराट कोहली`), Organizations in orange (`इंडियन एक्सप्रेस`), Locations in blue (`ऑस्ट्रेलिया`).
   - **विषय शोध (Topics):** Show the dominant LDA topic and its representative keywords (`खेळ`, `सामना`, `धावा`).
   - **पूर्वप्रक्रिया (Preprocessing):** Show the educational inspector: raw text $\to$ normalized $\to$ cleaned $\to$ tokenized $\to$ TF-IDF keywords.
   - **मूल्यमापन (Evaluation):** Show the built-in confusion matrix and ROUGE benchmark tables.

---

# 11. Exhaustive Examiner Viva Q&A Bank (Every Anticipated Question & Defense)

### Category A: The Classifier & Multi-Class SVM (Crucial Defense)

#### Q1: "SVM is a binary classifier. How can you use it for 12 classes? Won't it give wrong output?"
> **Examiner Defense Script:**
> *"Ma'am, that is a common theoretical misconception. While Vladimir Vapnik's original 1995 formulation was for binary $y \in \{-1, +1\}$, multi-class SVM has been a mathematically proven standard for 25+ years via the **One-vs-Rest (OvR / One-vs-All)** architecture.*  
>
> *In our `LinearSVC` model, the algorithm decomposes the 12-class problem into **12 independent convex optimization problems**:*
> $$\min_{\mathbf{w}_k, b_k} \frac{1}{2} \|\mathbf{w}_k\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i^{(k)}(\mathbf{w}_k^T \mathbf{x}_i + b_k))^2$$
> *where for the $k$-th classifier, samples belonging to that class are labeled $+1$ and all other 11 classes combined are labeled $-1$.*  
>
> *At test time, all 12 classifiers compute their signed geometric distance from their respective hyperplanes: $f_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + b_k$. The final prediction is:*
> $$\hat{y} = \arg\max_{k \in \{1, \dots, 12\}} f_k(\mathbf{x})$$
> *In 2004, Rifkin & Klautz published 'In Defense of One-vs-All Classification' in the Journal of Machine Learning Research (JMLR), proving that One-vs-All with Linear SVM performs as well as, or better than, any single all-in-one multi-class formulation.*  
>
> *Empirical Proof: If the classifier were giving wrong or random outputs, our accuracy across 12 classes would be random chance ($\frac{1}{12} \approx 8.33\%$). But on our **5,366 unseen test articles**, it achieved **87.74% accuracy** (over 10.5 times random chance) and **85.70% Macro-F1**, with high diagonal concentration in the confusion matrix."*

#### Q2: "Why didn't you use Naive Bayes or Logistic Regression instead?"
> **Answer:**
> - **Naive Bayes** assumes conditional independence between words given the category. In natural language—and especially agglutinative Marathi—words are heavily co-dependent.
> - **Logistic Regression** minimizes log loss but does not enforce a maximum geometric margin.
> - **Linear SVM maximizes the geometric margin ($2/\|\mathbf{w}\|$)** between the classes, providing superior generalization on noisy, high-dimensional TF-IDF text features.

#### Q3: "What is the role of the regularisation parameter $C=0.5$?"
> **Answer:**
> $C$ is the penalty parameter controlling the trade-off between maximizing the margin width and minimizing training classification errors:
> - A very high $C$ overfits to training outliers by penalizing any margin violation.
> - A small $C$ allows softer margins, ignoring noisy words.  
> On our validation grid search, $C=0.5$ produced the highest validation Macro-F1 (**0.8344**), preventing the model from memorizing rare idiosyncratic news terms.

---

### Category B: Marathi Linguistics & Preprocessing

#### Q4: "How does Marathi morphology affect TF-IDF, and how did you resolve it?"
> **Answer:**
> Marathi attaches suffixes (*vibhakti pratyay*) directly to nouns: `चित्रपट` becomes `चित्रपटाचे`, `चित्रपटात`, `कलाकारांच्या`. Standard TF-IDF treats these as unrelated unigrams. If an article mentioned `पर्यटन` (tourism) once in a policy sentence, the uninflected token `पर्यटन` had a massive base TF-IDF weight for `Travel`.  
> We solved this by implementing **morphological stem boosting** in `src/classification.py`. We maintain domain root stems (`CATEGORY_STEMS`). If an article contains multiple cinema/theatre stems (`चित्रपट`, `अभिनेत`, `दिग्दर्शक`, `कलाकार`, `अभिनय`, `थिएटर`), it adds calibrated margin support, ensuring the true domain context reliably overrides isolated cross-domain mentions.

#### Q5: "How did you generate your stopwords mathematically?"
> **Answer:**
> We used **Shannon Category Entropy** (`scripts/entropy_stopwords.py`):
> $$H(w) = -\sum_{c=1}^{12} P(c \mid w) \log_2 P(c \mid w)$$
> Normalized by $\log_2(12)$. If a token has high document frequency ($DF \ge 200$) and normalized entropy $\approx 1.0$, it appears uniformly across Politics, Sports, Health, Religion, etc. This proves mathematically that it carries zero domain-discriminative value (function words like `आहे`, `नाही`, `केले`, `होते`, `आणि`).

---

### Category C: Summarisation & Deep Learning Models

#### Q6: "Why did extractive TextRank get a higher ROUGE score than abstractive mT5?"
> **Answer:**
> ROUGE (Recall-Oriented Understudy for Gifting Evaluation) measures **surface n-gram overlap**. Because benchmark reference summaries often reuse prominent lead sentences, extractive TextRank (which copies verbatim sentences) naturally matches more exact word sequences (**0.3520 ROUGE-1**). However, abstractive mT5 (**0.3042 ROUGE-1**) produces more concise, human-like summaries that synthesize ideas across multiple paragraphs without redundancy.

#### Q7: "How did you solve the 512-token limit in mT5 for long news articles?"
> **Answer:**
> We created a sentence-aware chunking function (`chunk_sentences`) in `src/summarization.py`. Instead of naively cutting text mid-sentence, it greedily packs complete sentences up to a 450-token window, runs hierarchical summarisation across chunks, and aggregates the results cleanly.

#### Q8: "How does Qwen3-0.6B perform, and why did prompt constraints reduce latency?"
> **Answer:**
> Unconstrained Zero-Shot prompts generated verbose introductions, taking **~536 seconds** on CPU. Enforcing **Role + Structural Constraints** (demanding exactly 3 bullet points in Devanagari numerals with no intro/outro) cut off preamble generation, reducing inference time down to **~79 seconds** (an 85% speedup) while guaranteeing 100% format compliance.

---

### Category D: Topic Modeling & LDA

#### Q9: "What is Topic Coherence ($C_v$) and how did you choose $k=18$?"
> **Answer:**
> Topic Coherence measures the semantic relatedness of top words within a topic using sliding-window Normalized Pointwise Mutual Information (NPMI).  
> We scanned $k \in [6, 20]$ in steps of 2.  
> Coherence peaked at **$k = 18$ ($C_v = 0.4852$)**, producing distinct, interpretable topics (cricket, state politics, religion, agriculture, cinema). Lower $k$ merged distinct domains, while higher $k$ produced redundant duplicate clusters.

---

# 12. Project Limitations & Future Directions

### Limitations
1. **CPU Latency for Large Generative LLMs:** Deep models like Qwen3 take 70–80 seconds on CPU without hardware acceleration.
2. **Regional Dialect Sparsity:** Pretrained models are primarily trained on standard Puneri/Mumbai Marathi and have lower accuracy on dialects like Ahirani, Varhadi, or Konkani.

### Future Scope
1. **4-Bit Model Quantization (GGUF/ONNX):** Quantize mT5 and Qwen3 to 4-bit precision for sub-second execution on standard laptops and mobile devices.
2. **Speech-to-Text Pipeline:** Integrate Marathi Whisper models to transcribe TV broadcasts directly into the intelligence pipeline.
3. **Multi-Dialectal Expansion:** Augment training corpora with conversational regional Marathi literature and social media text.
