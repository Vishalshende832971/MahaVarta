# MahaVarta (महावार्ता): Comprehensive Project Explanation & Technical Documentation

---

## 1. Project Title & Metadata

- **Project Title:** MahaVarta (महावार्ता) — Marathi News Intelligence, Classification & Summarisation System
- **Academic Context:** NLP Mini Project — Natural Language Processing
- **Institution:** MIT Academy of Engineering (MITAOE), Alandi, Pune
- **Authors & Contributors:**
  - **Vishal Gorakh Shende** (PRN: `202401110034`)
  - **Gufran Shah** (PRN: `202401110011`)
- **Repository:** [https://github.com/Vishalshende832971/MahaVarta](https://github.com/Vishalshende832971/MahaVarta)
- **Primary Programming Language:** Python 3.12 (Verified on Python 3.12.8 64-bit)
- **Primary Frameworks:** PyTorch, Hugging Face Transformers, Scikit-learn, Gensim, Indic NLP Library, Gradio, NetworkX

---

## 2. Motivation, Background & Problem Statement

### 2.1 The Linguistic Landscape of Marathi
Marathi is an Indo-Aryan language spoken predominantly by over 83 million people in Maharashtra, India. It is ranked as the **3rd most spoken language in India** and the **10th most spoken in the world**. Marathi is written in the **Devanagari script** and possesses distinct linguistic attributes:
- **Agglutinative Morphology:** Words are composed by adding bound grammatical morphemes (*vibhakti pratyay*) directly to noun and verb stems. For example:
  - Base stem: `चित्रपट` (Cinema / Movie)
  - Genitive inflection: `चित्रपटाचे` (Of the cinema)
  - Locative inflection: `चित्रपटात` (In the cinema)
  - Oblique plural: `चित्रपटांमधील` (Among the movies)
- **Free Word Order & SOV Typology:** Subject-Object-Verb (SOV) is canonical, but poetic and journalistic Marathi frequently shifts phrasal order for emphasis.
- **Script Nuances:** Distinct punctuation like the Devanagari single danda (`।`) and double danda (`॥`), pervasive conjunct consonants (*jodakshare* like `क्ष`, `ज्ञ`, `त्र`), and nuktas.

### 2.2 The Low-Resource Challenge in Indian Language NLP
Despite its massive real-world speaker base, Marathi is classified as a **low-resource language** in computational linguistics:
1. **Lack of Clean Curated News Pipelines:** Most open-source NLP pipelines target English, Mandarin, or Spanish. Commercial tools fail on Devanagari encoding or provide rudimentary translation layers.
2. **Lack of Unified Intelligence Platforms:** Journalists, researchers, and regional readers typically have to consult disparate scripts or models to achieve classification, extractive summary, deep abstractive brief, sentiment scoring, and entity tracking.
3. **Severe Morphological Sparsity:** Naive tokenization creates out-of-vocabulary (OOV) bottlenecks and bag-of-words mismatch across inflected word forms.

### 2.3 Objectives of MahaVarta
MahaVarta addresses these limitations by providing an end-to-end, multi-task NLP platform that accepts raw Marathi text (or uploaded text files) and executes an 8-stage intelligent pipeline in real time.

---

## 3. High-Level System Architecture

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

## 4. Exhaustive Dataset Specifications & Data Engineering

MahaVarta utilizes four distinct benchmark corpora, custom human annotations, and an information-theoretically curated stopwords dataset.

### 4.1 Primary Dataset: L3Cube-MahaNews (LPC - Long Paragraph Classification)

- **Source Repository:** `https://github.com/l3cube-pune/MarathiNLP/tree/main/L3Cube-MahaNews/LPC`
- **Academic Citation:**  
  Mittal, Saloni, et al. *"L3Cube-MahaNews: News-Based Short Text and Long Document Classification Datasets in Marathi."* SPELLL / Springer Nature (2023). [arXiv:2404.18216](https://arxiv.org/abs/2404.18216).
- **Domain:** Mainstream Marathi journalism collected from major digital portals (e.g. Loksatta, Maharashtra Times, Sakal).
- **Task:** 12-class Single-Label News Classification and LDA Topic Model training corpus.
- **Verified Split Sizes (after filtering out multi-file concatenation header artifacts):**
  - **Training Split (`train.csv`):** **42,870** clean samples (80.0%)
  - **Validation Split (`valid.csv`):** **5,357** clean samples (10.0%)
  - **Testing Split (`test.csv`):** **5,366** clean samples (10.0%)
  - **Total Dataset Size:** **53,593** verified Marathi news articles.
- **Dataset Schema:**
  - `Text` (string): Full body text of the Marathi news article.
  - `Label` (string): Ground-truth category name in English (e.g., `Politics`, `Sports`).
- **Data Quality & Preprocessing Sanitization:**
  - Standard online dumps of MahaNews contain 11 duplicate CSV header rows (`Text,Label`) due to batch file concatenations.
  - MahaVarta's data loader (`load_split` in `src/train_classifier.py`) programmatically scrubs these duplicate header rows and null values, preventing train-test contamination.
- **The 12 Categories and Distribution:**
  1. **`Politics` (राजकारण):** ~19% of corpus (highest support, state and national government).
  2. **`International` (आंतरराष्ट्रीय):** ~13% (global conflicts, geopolitical summits).
  3. **`Manoranjan` (मनोरंजन):** ~14% (Marathi cinema, Bollywood, theater, television serials).
  4. **`Crime` (गुन्हेगारी):** ~10% (police investigations, court rulings, accidents).
  5. **`Bhakti` (भक्ती):** ~9% (Warkari sampradaya, Pandharpur wari, temple rituals, spirituality).
  6. **`Sports` (क्रीडा):** ~8% (cricket, kabaddi, Olympics, international football).
  7. **`Auto` (ऑटो):** ~7% (automobile launches, vehicle reviews, EV infrastructure).
  8. **`Travel` (पर्यटन):** ~7% (Maharashtra forts, Konkan beaches, hill stations, eco-tourism).
  9. **`Tech` (तंत्रज्ञान):** ~6% (smartphones, ISRO launches, AI developments, cyber security).
  10. **`Education` (शिक्षण):** ~3% (SSC/HSC board results, university admissions, scholarships).
  11. **`Health` (आरोग्य):** ~3% (nutrition, public health warnings, diseases, medical advancements).
  12. **`Fashion` (फॅशन):** ~2% (lifestyle, traditional attire, beauty and design).

---

### 4.2 Named Entity Recognition Dataset: MahaNER

- **Source Repository:** `https://huggingface.co/datasets/l3cube-pune/marathi-ner` / L3Cube Pune
- **Paper Citation:**  
  Litake, Onkar, et al. *"L3Cube-MahaNER: A Named Entity Recognition Dataset for Marathi."* EACL / arXiv:2203.04838 (2022).
- **Split Sizes:**
  - **Train:** 21,500 sentences
  - **Validation:** 1,500 sentences
  - **Test (Used for Final Evaluation in MahaVarta):** **1,999 sentences** (containing **1,943 annotated named entities**).
- **Annotation Format:** Word-level BIO tagging schema (Beginning, Inside, Outside).
- **7 Target Entity Classes:**
  - `Person` (B-Person, I-Person): e.g., *नरेंद्र मोदी, अशोक सराफ, लता मंगेशकर*
  - `Location` (B-Location, I-Location): e.g., *मुंबई, पुणे, श्रीहरिकोटा, नागपूर*
  - `Organization` (B-Organization, I-Organization): e.g., *इस्रो, राज्य सरकार, शिवसेना, बीसीसीआय*
  - `Date` (B-Date, I-Date): e.g., *१५ ऑगस्ट, नोव्हेंबरच्या पहिल्या आठवड्यात*
  - `Time` (B-Time, I-Time): e.g., *पहाटे ५ वाजता, संध्याकाळी*
  - `Measure` (B-Measure, I-Measure): e.g., *६५० अंकांनी, १ किमी, ५० टक्के*
  - `Designation` (B-Designation, I-Designation): e.g., *मुख्यमंत्री, पंतप्रधान, संचालक, पोलीस अधीक्षक*

---

### 4.3 Summarisation Dataset: XL-Sum Marathi

- **Source Repository:** `https://huggingface.co/datasets/csebuetnlp/xlsum`
- **Paper Citation:**  
  Hasan, Tahmid, et al. *"XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages."* Findings of ACL (2021). [arXiv:2106.13822](https://arxiv.org/abs/2106.13822).
- **Dataset Nature:** High-quality professionally written BBC Marathi news articles paired with single-sentence human-written reference summaries (*bhashantar* / executive abstractive summaries).
- **Usage in MahaVarta:** Used to benchmark and compute objective ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) across TextRank, mT5, and Qwen3 prompt variations.

---

### 4.4 Custom Human-Annotated Dataset: News Headlines Sentiment (`data/annotations/`)

- **Dataset File:** [`data/annotations/headlines.csv`](data/annotations/headlines.csv)
- **Sample Size:** **200 diverse Marathi news headlines** sampled across all 12 categories from the L3Cube-MahaNews Short Headline Corpus (SHC).
- **Annotators:** 
  - Annotator 1: **Vishal Gorakh Shende** (`label_vishal`)
  - Annotator 2: **Gufran Shah** (`label_gufran`)
- **Annotation Methodology & Agreement Protocol:**
  - Conducted strictly independent double-blind annotations without cross-consultation.
  - Three target classes: `Positive`, `Negative`, `Neutral`.
  - Guidelines codified in [`data/annotations/ANNOTATION_GUIDE.md`](data/annotations/ANNOTATION_GUIDE.md):
    - **Positive:** Victories, scientific breakthroughs, awards, recovery, constructive policies.
    - **Negative:** Deaths, road accidents, criminal violence, inflation, natural calamities.
    - **Neutral:** Routine administrative circulars, election schedules, weather reports without severe damage.
  - Conflicts adjudicated to produce `final_label` for ground-truth sentiment verification.

---

### 4.5 Information-Theoretic Curated Stopwords Dataset (`data/stopwords_mr.txt`)

- **Script:** [`scripts/entropy_stopwords.py`](scripts/entropy_stopwords.py)
- **Methodology (Category-Entropy Pruning):**
  - Unlike arbitrary manual stopword lists, MahaVarta computed normalized Shannon entropy across the 12 news categories for every token in the 42,870 training articles:
    $$H(w) = -\sum_{c=1}^{12} P(c \mid w) \log_2 P(c \mid w)$$
    $$\text{Normalized Entropy } \bar{H}(w) = \frac{H(w)}{\log_2(12)}$$
  - A word that appears with high document frequency ($DF \ge 200$) and high normalized entropy ($\bar{H}(w) \to 1.0$) is distributed uniformly across all 12 genres, proving it is a purely functional grammatical word carrying zero category-discriminative information.
  - **Resulting Stopword List:** 400+ mathematical stopwords (`आहे`, `नाही`, `या`, `केले`, `होते`, `आणि`, `पण`, `म्हणून`, `झाले`, `त्यांनी`).

---

### 4.6 Benchmark Articles Suite (`data/sample_articles/`)

Three curated news articles stored as reference samples:
1. **`01_short_sports.txt` (71 words, 3 sentences):**
   - *Topic:* Ravi Shastri's remarks on Virat Kohli's T20 World Cup innings against Pakistan in Australia.
   - *Target Domain:* `क्रीडा` (Sports) | Sentiment: `Positive`.
2. **`02_medium_health.txt` (47 words, 4 sentences):**
   - *Topic:* Medical effects of excessive Vitamin C and lemon water on acid secretion and gastroesophageal reflux.
   - *Target Domain:* `आरोग्य` (Health) | Sentiment: `Negative`.
3. **`03_long_politics.txt` (140 words, 10 sentences):**
   - *Topic:* Election Commission dispute between Eknath Shinde and Uddhav Thackeray factions, freezing of the bow-and-arrow symbol, and allotment of the flaming torch symbol for the Andheri by-election.
   - *Target Domain:* `राजकारण` (Politics) | Sentiment: `Neutral`.

---

## 5. Pretrained Deep Learning Models & Open-Source Artifacts

| Model Name | Hugging Face Repository ID | Parameter Count | Underlying Architecture | Primary Function in Project | Open-Source Licence |
|:---|:---|:---:|:---|:---|:---|
| **mT5 Multilingual XL-Sum** | `csebuetnlp/mT5_multilingual_XLSum` | ~300M (Base) | Encoder-Decoder Sequence-to-Sequence Transformer | Abstractive summarisation for Marathi articles | CC-BY-NC-SA-4.0 |
| **Qwen3-0.6B** | `Qwen/Qwen3-0.6B` | ~600M | Autoregressive Decoder-only Transformer | Multi-prompt generative news briefs & bullet points | Apache 2.0 |
| **MahaSent-MD** | `l3cube-pune/marathi-sentiment-md` | ~110M | Fine-tuned MahaBERT (RoBERTa architecture) | 3-Class Sentiment prediction (Pos/Neu/Neg) | MIT License |
| **MahaNER** | `l3cube-pune/marathi-ner` | ~110M | Token-Classification MahaBERT | 7-Entity Named Entity Recognition | MIT License |
| **TF-IDF + LinearSVC** | Local artifact: `models/classification_pipeline.joblib` | 7.3 MB serialized | Sparse Linear Support Vector Classifier | 12-Class News Classification | Project Built |
| **Gensim LDA Model** | Local artifact: `models/lda_model.gensim` | 18 topics | Online Variational Bayes Dirichlet Allocation | Latent topic discovery & keyword clustering | Project Built |

---

## 6. Module-by-Module Technical Deep Dive

### 4.1 Module 1: Preprocessing & Morphological Handling (`src/preprocessing.py`)

1. **Devanagari Normalisation (`normalize_devanagari`):**
   - Implemented via `indicnlp.normalize.indic_normalize.DevanagariNormalizer`.
   - Replaces composite glyphs with canonical Unicode NFC forms, resolves divergent matra sequences, and standardises nukta characters so identical words with divergent input encodings match accurately.
2. **Text Cleaning (`clean_text`):**
   - Strips HTML markup (`<[^>]+>`).
   - Removes HTTP/HTTPS web links and `www.*` domain URLs.
   - Collapses contiguous whitespace, linefeeds, and tab characters into a single standard space.
3. **Marathi Sentence Segmentation (`split_sentences`):**
   - Standard sentence segmenters (such as NLTK's English Punkt) fail in Marathi because they cannot parse Devanagari dandas (`।`, `॥`) and falsely split on honorific abbreviations (`डॉ.`, `प्रा.`, `श्री.`, `इ.`) or decimal digits (`२.५`, `10.5`).
   - MahaVarta uses custom regex lookbehinds and lookaheads that protect abbreviations and split strictly at sentence-terminating dandas (`।`, `॥`), question marks (`?`), exclamation marks (`!`), and terminal periods (`.`).
4. **Tokenisation (`tokenize`):**
   - Powered by the Indic NLP Library tokeniser specifically engineered for Indian scripts.
5. **Stopwords Pruning (`data/stopwords_mr.txt`):**
   - A curated dictionary of 400+ high-frequency Marathi grammatical words (`आहे`, `नाही`, `या`, `त्या`, `केले`, `होते`, `म्हणून`, `आणि`, `पण`). Words were ranked and pruned using entropy thresholding across corpus frequencies.

---

### 4.2 Module 2: News Classification Engine (`src/classification.py`)

- **Dataset:** L3Cube-MahaNews Single-Label Multi-Class corpus containing 53,593 articles categorized into 12 classes:
  - `Auto` (ऑटो), `Bhakti` (भक्ती), `Crime` (गुन्हेगारी), `Education` (शिक्षण), `Fashion` (फॅशन), `Health` (आरोग्य), `International` (आंतरराष्ट्रीय), `Manoranjan` (मनोरंजन), `Politics` (राजकारण), `Sports` (क्रीडा), `Tech` (तंत्रज्ञान), `Travel` (पर्यटन).
- **Split Sizes (Zero Leakage):**
  - **Train:** 42,870 articles (80%)
  - **Validation:** 5,357 articles (10%)
  - **Test:** 5,366 articles (10%)
- **Machine Learning Architecture:**
  - `TfidfVectorizer`: Sublinear term-frequency scaling (`sublinear_tf=True`), minimum document frequency `min_df=3`, unigram token ranges.
  - `LinearSVC`: One-vs-Rest Linear Support Vector Classifier with squared hinge loss, tuned regularization penalty $C = 0.5$.
- **Validation Tuning Comparison:**
  - Unigram vs Bigram exploration: Unigram ($C=0.5$) achieved the superior validation Macro-F1 (**0.8344**) compared to Bigram ($C=1.0$, F1: 0.8192) due to lower parameter variance and reduced feature sparsity.
- **Inference & Decision Normalisation:**
  - Raw hyperplane distances $d_k(x)$ from `decision_function` are mapped to well-calibrated class confidence distributions using the Softmax function:
    $$P(y = c_k \mid x) = \frac{\exp(d_k(x) - \max_j d_j(x))}{\sum_{i=1}^{12} \exp(d_i(x) - \max_j d_j(x))}$$
- **Morphological Stem Boost:**
  - Implements domain-specific Devanagari root stem detection (`CATEGORY_STEMS`) in `predict()` to overcome unstemmed TF-IDF misses on inflected terms (e.g., ensuring `चित्रपटाचे`, `दिग्दर्शकाने`, `कलाकारांच्या` boost `Manoranjan` over single stray occurrences of uninflected keywords like `पर्यटन`).

---

### 4.3 Module 3: Extractive Summarisation (TextRank) (`src/summarization.py`)

- **Theory:** Graph-based ranking model adapted from Mihalcea & Tarau (2004).
- **Algorithm Execution:**
  1. The Marathi text is segmented into sentences: $S = [s_1, s_2, \dots, s_N]$.
  2. If $N \le k$, the entire text is returned directly.
  3. Sentences are vectorized into TF-IDF space.
  4. An undirected graph $G = (V, E)$ is constructed where vertices $V$ represent sentences, and edge weights $W_{ij}$ represent pairwise sentence cosine similarity:
     $$W_{ij} = \frac{\vec{v}(s_i) \cdot \vec{v}(s_j)}{\|\vec{v}(s_i)\| \|\vec{v}(s_j)\|}$$
  5. The PageRank iterative algorithm computes vertex centrality scores until convergence (damping factor $d = 0.85$, max iterations = 100).
  6. The top-$k$ highest-ranked sentences are extracted and reordered chronologically as they appeared in the original news article to preserve logical narrative flow.

---

### 4.4 Module 4: Abstractive Summarisation (mT5 XL-Sum) (`src/summarization.py`)

- **Pretrained Foundation Model:** `csebuetnlp/mT5_multilingual_XLSum`
  - Architecture: Sequence-to-sequence Encoder-Decoder Transformer based on Google's mT5 architecture.
  - Training: Fine-tuned on the multilingual XL-Sum dataset (BBC news articles across 44 languages, including Marathi).
- **Sentence-Aware Chunking Strategy (`chunk_sentences`):**
  - Standard Transformers truncate text at 512 tokens, discarding late-breaking details in long articles.
  - MahaVarta segments the article by complete sentences and fills token budgets incrementally up to a safe window of 450 tokens.
  - Chunks are summarized separately and aggregated hierarchically without cutting mid-sentence.
- **Decoding Configuration:**
  - Beam search with `num_beams=2`
  - Minimum summary length = 15 tokens; maximum summary length = 80 tokens
  - Repetition penalty = 2.5
  - Length penalty = 1.0

---

### 4.5 Module 5: Prompt-Engineered News Briefing (Qwen3-0.6B) (`src/prompting.py`)

- **LLM Engine:** `Qwen/Qwen3-0.6B` (Alibaba Cloud's modern 600M parameter multilingual instruction-tuned model running locally).
- **Prompt Strategy Matrix:**
  1. **Zero-Shot:** Direct command instruction without framing.
  2. **Role-Based:** Assigns a specialized editorial persona:
     > *"तुम्ही एक अनुभवी मराठी वृत्त संपादक आहात. खालील बातमी काळजीपूर्वक वाचा आणि महत्त्वाचा सारांश तयार करा."*
  3. **Role + Structural Constraints:** Combines editorial role with strict output requirements:
     > *"१. बातमीचा सारांश बरोबर ३ मुद्द्यांमध्ये तयार करा.*  
     > *२. प्रत्येक मुद्द्याची सुरुवात मराठी अंकातील क्रमांकाने करा (१., २., ३.).*  
     > *३. कोणतीही बाहेरील माहिती किंवा प्रस्तावना जोडू नका."*
- **Programmatic Evaluation & Validation Functions:**
  - `has_three_numbered_items(text)`: Validates format compliance via regex matching across Devanagari (`१.`, `२.`, `३.`) and ASCII digits.
  - `devanagari_ratio(text)`: Computes the script purity score:
    $$\text{Ratio} = \frac{\text{Count of Devanagari Unicode characters}}{\text{Total non-whitespace characters}}$$
  - `novel_word_ratio(source, summary)`: Analyzes abstractive vocabulary novelty using stem-sensitive set intersections.
  - `invented_name_candidates(source, summary)`: Detects hallucinated capitalized entities or proper nouns that do not appear in the source context.

---

### 4.6 Module 6: Sentiment Analysis (`src/sentiment.py`)

- **Model:** `l3cube-pune/marathi-sentiment-md` (MahaBERT model fine-tuned on Marathi sentiment benchmarks).
- **Classes:** 3 categories:
  - `Positive` (सकारात्मक)
  - `Neutral` (तटस्थ)
  - `Negative` (नकारात्मक)
- **Output:** Returns dominant label alongside full probability distribution rendered as graphical confidence bars.

---

### 4.7 Module 7: Named Entity Recognition (`src/ner.py`)

- **Model:** `l3cube-pune/marathi-ner` (BERT token classification fine-tuned on the MahaNER dataset).
- **Recognized Entity Types (7 Classes):**
  1. `Person` (व्यक्ती)
  2. `Location` (ठिकाण)
  3. `Organization` (संघटना)
  4. `Date` (तारीख)
  5. `Time` (वेळ)
  6. `Measure` (परिमाण / संख्यात्मक मूल्ये)
  7. `Designation` (पद / हुद्दा)
- **Subword & BIO Reconstruction:**
  - Handles Hugging Face subword tokens (marked by `##`) and recombines Devanagari split characters back into cohesive entity phrases.
  - Converts token-level predictions into character span tuples `(start_idx, end_idx, label)` for direct color-coded visual rendering in Gradio's `gr.HighlightedText`.

---

### 4.8 Module 8: Latent Topic Modeling (`src/topic_modeling.py`)

- **Algorithm:** Gensim Latent Dirichlet Allocation (LDA) with online variational inference.
- **Preprocessing Pipeline:** Stopword filtering using `stopwords_mr.txt`, length filtering ($\ge 3$ characters), dictionary pruning (`no_below=10`, `no_above=0.5`).
- **Optimal Topic Coherence Scanning:**
  - Scanned topic numbers from $k=6$ to $k=20$.
  - Evaluated using $C_v$ coherence (measuring co-occurrence frequency in sliding windows).
  - Selected $k = 18$ topics with peak coherence **$C_v = 0.4852$**.
- **Outputs:** Shows document topic distribution vector $\theta_d$ and the top-6 representative Marathi keywords per topic.

---

## 7. Web Interface Architecture (`app.py`)

The application is deployed as an interactive web dashboard using **Gradio**:
- **Dual-Language UI Design:** Primary Marathi Devanagari headers with English technical subtitles for academic accessibility.
- **Input Channels:** Direct text box paste or instant `.txt` file upload (with automatic UTF-8 fallback handling).
- **Tab Layout (7 Analysis Views + Benchmark Hub):**
  1. **आढावा (Overview):** Executive summary card, word/sentence/character counts, dominant category, sentiment score, extracted key entities, and top TextRank preview.
  2. **वर्गवारी (Classification):** Top-5 category prediction table and confidence bar chart.
  3. **सारांश (Summary):** Side-by-side comparison of extractive TextRank vs. abstractive mT5 with summary length sliders.
  4. **बातम्या संक्षेप (News Brief - Qwen3):** Generative news brief with prompt strategy switcher, temperature slider, and live objective constraint check badges.
  5. **भावना विश्लेषण (Sentiment):** Positive / Neutral / Negative percentage breakdown.
  6. **नामित घटक (NER):** Color-coded entity highlighting and grouped entity category cards.
  7. **विषय शोध (Topics):** LDA dominant topic ID, topic weight, and representative keywords.
  8. **पूर्वप्रक्रिया (Preprocessing):** Educational walkthrough displaying the text transformations: Raw $\rightarrow$ Normalised $\rightarrow$ Cleaned $\rightarrow$ Tokenised $\rightarrow$ Top TF-IDF Terms.
  9. **मूल्यमापन (Evaluation):** Dedicated academic dashboard rendering the Confusion Matrix, ROUGE Benchmark table, NER classification report, and LDA Topic Coherence curves.

---

## 8. Comprehensive Quantitative Evaluation Results

### 6.1 Classification Test Set Results (5,366 Unseen Samples)

- **Test Accuracy:** **87.74%** (`0.8774`)
- **Macro-Precision:** **87.11%** (`0.8711`)
- **Macro-Recall:** **84.61%** (`0.8461`)
- **Macro-F1 Score:** **85.70%** (`0.8570`)

#### Per-Class Detailed Performance:
| Category | Marathi Name | Precision | Recall | F1-Score | Test Support |
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

### 6.2 Summarisation ROUGE Benchmark

Evaluated against reference Marathi human summaries on XL-Sum test articles:

| Method | Summarisation Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency (CPU) |
|:---|:---|:---:|:---:|:---:|:---:|
| **TextRank** | Extractive Graph | **0.3520** | **0.1727** | **0.3087** | **0.13 s** |
| **mT5 Multilingual XL-Sum** | Abstractive Seq2Seq | 0.3042 | 0.1538 | 0.2365 | 20.43 s |
| **Qwen3 (Zero-Shot)** | Generative LLM | 0.2574 | 0.0562 | 0.1976 | 536.56 s |
| **Qwen3 (Role-Based)** | Generative LLM | 0.2051 | 0.0530 | 0.1820 | 76.00 s |
| **Qwen3 (Role + Constraints)** | Generative LLM | 0.2383 | 0.0767 | 0.1889 | 79.43 s |

**Key Academic Observations:**
1. TextRank achieves the highest ROUGE scores because ROUGE evaluates strict n-gram surface overlap, and extractive methods copy exact verbatim sentences.
2. mT5 produces superior human readability and concise abstractions, synthesizing multi-sentence context into novel summary phrases without redundancy.
3. Adding structural constraints to Qwen3 dramatically reduces CPU inference latency (from ~536s down to ~79s) by halting verbose introductory text.

---

### 6.3 Named Entity Recognition Evaluation (MahaNER Test = 1,999 Sentences)

- **Overall Token Accuracy:** **81.0%**
- **Micro Precision:** **81.2%**
- **Micro Recall:** **80.8%**
- **Micro F1-Score:** **81.0%**

#### Entity Breakdown:
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

### 6.4 LDA Topic Coherence Optimization Scan

| Number of Topics ($k$) | Coherence Score ($C_v$) | Topic Interpretability Assessment |
|:---:|:---:|:---|
| $k = 6$ | 0.3489 | Topics are overly broad; politics and crime overlap heavily. |
| $k = 8$ | 0.3360 | Under-clustering; entertainment terms merge with general sports. |
| $k = 10$ | 0.4164 | Substantial improvement; clear separation of sports and health. |
| $k = 12$ | 0.4045 | Stable coherence. |
| $k = 14$ | 0.4398 | Strong separation of business/economy and education. |
| $k = 16$ | 0.3920 | Coherence dips due to slight term fragmentation. |
| **$k = 18$ (Optimal)** | **0.4852** | **Peak coherence: distinct topics for cricket, agriculture, elections, religion, movies.** |
| $k = 20$ | 0.4170 | Over-clustering; redundant duplicate topics emerge. |

---

## 9. Software Engineering & Quality Assurance

- **Automated Test Suite:** Comprehensive pytest testing suite covering 86 unit and integration test cases across all modules:
  - `tests/test_preprocessing.py`: 26 tests (dandas, abbreviations, unicode normalisation, cleaning).
  - `tests/test_classification.py`: 4 tests (pipeline training, decision scores, empty input handling).
  - `tests/test_summarization.py`: 19 tests (sentence segmentation, TextRank, chunking, mT5 execution).
  - `tests/test_prompting.py`: 15 tests (constraint checking, Devanagari ratio, novel word metrics).
  - `tests/test_sentiment.py`: 6 tests (MahaSent inference, 3-class distribution).
  - `tests/test_ner.py`: 5 tests (MahaNER tokenization, BIO reconstruction, span extraction).
  - `tests/test_topic_modeling.py`: 7 tests (Gensim LDA training, inference, dictionary persistence).
  - `tests/test_app.py`: 4 tests (end-to-end Gradio pipeline execution on all sample articles).
- **Test Pass Rate:** **100% (86 passed / 86 tests)**.

---

## 10. Limitations & Future Scope

### 10.1 Limitations
1. **CPU Latency for Deep Generative LLMs:** While TextRank runs in 0.13 seconds and Linear SVM runs in milliseconds, deep autoregressive models like Qwen3 take 70–80 seconds on CPU without hardware acceleration.
2. **Polysemy & Mixed-Domain News:** In short articles featuring cross-domain vocabulary (e.g., government policies promoting film tourism), bag-of-words classifiers can encounter ambiguity without morphological stem boosting.
3. **Dialectal Variation:** Pretrained models are primarily trained on standard Puneri/Mumbai Marathi news register and may experience lower accuracy on regional dialects like Varhadi, Ahirani, or Konkani.

### 10.2 Future Scope
1. **Lightweight Edge Fine-Tuning:** Quantize mT5 and Qwen3 into 4-bit GGUF or ONNX representations for sub-second edge deployment on standard consumer laptops and smartphones.
2. **Audio / Speech-to-Text Integration:** Incorporate fine-tuned Marathi Whisper models to allow direct voice input from television broadcasts or podcasts.
3. **Multi-Dialectal Expansion:** Augment training corpora with conversational regional Marathi literature and social media text.

---

## 11. How to Run and Verify the System

### 9.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/Vishalshende832971/MahaVarta.git
cd MahaVarta

# Activate Python 3.12 environment
# Windows PowerShell:
.\venv312\Scripts\Activate.ps1
# Linux / macOS:
source venv312/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Running Unit Tests
```bash
# Run all fast tests (excluding slow heavy LLM generation)
pytest -m "not slow"

# Run complete test suite (all 86 tests)
pytest
```

### 9.3 Launching the Application
```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in any web browser to interact with the dashboard.
