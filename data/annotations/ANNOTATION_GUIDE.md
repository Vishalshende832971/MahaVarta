# Marathi News Headline Sentiment Annotation Guide
## (मराठी बातम्या मथळा भावना विश्लेषण मार्गदर्शक)

**Project:** MahaVarta (Marathi News Classification & Summarization)  
**Team Members / Annotators:** Vishal Gorakh Shende & Gufran Shah  
**File to Annotate:** [`data/annotations/headlines.csv`](headlines.csv) (200 Marathi news headlines sampled from L3Cube-MahaNews SHC)

---

## 1. Sentiment Classes & Guidelines

Every headline must be annotated with exactly one of the three following labels (case-sensitive):
- **`Positive`**
- **`Negative`**
- **`Neutral`**

| Class | Description | Marathi Examples |
|---|---|---|
| **`Positive`** | Good news, victories, accomplishments, awards, celebrations, recovery, price cuts, constructive policies, positive health tips. | • भारताचा ऑस्ट्रेलियाविरुद्ध रोमहर्षक विजय!<br>• मराठी शास्त्रज्ञाचा आंतरराष्ट्रीय पातळीवर सन्मान.<br>• शेतकऱ्यांसाठी नवीन सिंचन योजना मंजूर. |
| **`Negative`** | Accidents, deaths, crimes, losses, conflicts, disasters, corruption, disease spread, price hikes, inflation. | • महामार्गावर भीषण अपघात, तिघांचा जागीच मृत्यू.<br>• महागाईचा भडका; पेट्रोल आणि डिझेलच्या दरात मोठी वाढ.<br>• बँकेत लाखो रुपयांचा अपहार उघडकीस. |
| **`Neutral`** | Objective factual reporting, routine announcements, government circulars, meeting schedules, weather updates without severe damage. | • उद्या मुंबईत मंत्रिमंडळाची महत्त्वाची बैठक होणार.<br>• दहावीच्या परीक्षांचे वेळापत्रक जाहीर करण्यात आले.<br>• सोने आणि चांदीचे आजचे ताजे बाजारभाव. |

### Decision Rules:
1. **Focus on the headline's primary message:** Evaluate how a general Marathi reader would perceive the event being reported.
2. **Context over isolated words:** A word like "सरकार" (Government) or "निवडणूक" (Election) is neutral unless framed positively or negatively.
3. **Sports victories vs defeats:** If a team won, label `Positive`; if defeated or knocked out, label `Negative`.
4. **When in doubt:** If an article merely describes an event without clear positive or negative framing, choose `Neutral`.

---

## 2. Independent Annotation Procedure

1. **Independent Labelling:**
   - **Vishal** fills the `label_vishal` column for all 200 rows without consulting Gufran.
   - **Gufran** fills the `label_gufran` column for all 200 rows without consulting Vishal.
2. **Agreement Measurement:**
   - Once both columns are complete, run the agreement calculation script:
     ```powershell
     python data/annotations/compute_agreement.py
     ```
   - The script reports:
     - **Observed Percentage Agreement (%)**
     - **Cohen's Kappa Score ($\kappa$)**
3. **Adjudication & Consensus:**
   - Filter rows where `label_vishal != label_gufran`.
   - Discuss disagreements and decide on the agreed final value in the `final_label` column.
4. **Model Evaluation:**
   - Once `final_label` has all 200 rows completed, run:
     ```powershell
     python -c "from src.sentiment import evaluate_on_annotations; evaluate_on_annotations('data/annotations/headlines.csv')"
     ```

---

## 3. Interpreting Cohen's Kappa ($\kappa$)

| Kappa Range ($\kappa$) | Level of Agreement |
|---|---|
| `< 0.20` | Poor agreement |
| `0.21 – 0.40` | Fair agreement |
| `0.41 – 0.60` | Moderate agreement |
| `0.61 – 0.80` | Substantial agreement |
| `0.81 – 1.00` | Almost perfect agreement |
