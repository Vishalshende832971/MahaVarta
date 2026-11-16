# Dataset Sources

## MahaNews (News Classification)
- **Official Repository:** https://github.com/l3cube-pune/MarathiNLP
- **Sub-directory:** `L3Cube-MahaNews/LPC/` (Long Paragraph Classification corpus)
- **Paper Citation:** Mittal, Saloni, et al. *"L3Cube-MahaNews: News-Based Short Text and Long Document Classification Datasets in Marathi."* SPELLL / Springer Nature (2023). [arXiv:2404.18216](https://arxiv.org/abs/2404.18216)
- **Download URLs:**
  - Train: `https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNews/LPC/LPC_Train.csv`
  - Validation: `https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNews/LPC/LPC_Valid.csv`
  - Test: `https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNews/LPC/LPC_Test.csv`
- **Verified Split Sizes (clean samples after removing 11 repeated headers from multi-file concatenation):**
  - **Train:** 42,870 records (80%)
  - **Validation:** 5,357 records (10%)
  - **Test:** 5,366 records (10%)
  - **Total:** 53,593 records
- **Number of Classes:** 12 classes
  - `Auto` (ऑटो)
  - `Bhakti` (भक्ती)
  - `Crime` (गुन्हेगारी)
  - `Education` (शिक्षण)
  - `Fashion` (फॅशन)
  - `Health` (आरोग्य)
  - `International` (आंतरराष्ट्रीय)
  - `Manoranjan` (मनोरंजन)
  - `Politics` (राजकारण)
  - `Sports` (क्रीडा)
  - `Tech` (तंत्रज्ञान)
  - `Travel` (पर्यटन)

---

## MahaSent-MD (Sentiment Analysis)
- **Model:** `l3cube-pune/marathi-sentiment-md` on Hugging Face
- **Usage:** Pretrained inference only (no additional training data needed)

## MahaNER (Named Entity Recognition)
- **Model:** `l3cube-pune/marathi-ner` on Hugging Face
- **Test data:** Official MahaNER test split (25,000 samples: 21,500 train, 1,500 val, 2,000 test)

## XL-Sum Marathi (Summarisation)
- **Dataset:** `csebuetnlp/xlsum` on Hugging Face (Marathi subset)
- **Model:** `csebuetnlp/mT5_multilingual_XLSum` on Hugging Face

## Qwen3-0.6B (News Brief)
- **Model:** `Qwen/Qwen3-0.6B` on Hugging Face
- **Usage:** Prompt-engineered Marathi news brief generation
