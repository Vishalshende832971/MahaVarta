"""
MahaVarta: Marathi News Classification & Summarization Web Application
=======================================================================
Gradio dashboard supporting:
  - Tab 1: Overview (Article stats, predicted category badge, sentiment, key entities, dominant topic, summary)
  - Tab 2: Classification (Top 3 categories with category scores, trained SVM)
  - Tab 3: Summary (Extractive TextRank summary + Stage 8 mT5 abstractive summary)
  - Tab 4: News brief (Qwen3 prompt-engineered news brief + prompt variants comparison)
  - Tab 5: Sentiment (Sentiment label + model prediction score, 3 class scores, news caveat note)
  - Tab 6: Entities (gr.HighlightedText in context + grouped entities by category)
  - Tab 7: Topics (Discovered topics, top keywords, topic weights via LDA)
  - Tab 8: Preprocessing (CO1 demonstration: original, normalized, cleaned, tokens, TF-IDF)

Authors: Vishal Gorakh Shende & Gufran Shah
Course: Natural Language Processing, MIT Academy of Engineering, Alandi, Pune
"""

import os
import sys
import logging
import pandas as pd
import gradio as gr

# Ensure src modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from pathlib import Path
from src.preprocessing import preprocess, read_uploaded_txt
from src.classification import load as load_classifier, predict as predict_category, LABEL_TO_MARATHI
from src.summarization import (
    textrank_summary,
    split_sentences,
    summarize_auto,
    mt5_summary,
    chunked_mt5_summary,
)
from src.sentiment import predict as predict_sentiment
from src.ner import extract_entities, group_by_label
from src.topic_modeling import load_lda_model, article_topics, preprocess_for_lda
from src.prompting import (
    generate_brief,
    compare_prompts,
    has_three_numbered_items,
    devanagari_ratio,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global State & Model Loading (Loaded Once at Startup)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classification_pipeline.joblib")
classifier_model = None
model_load_error = None

lda_model = None
lda_dict = None
lda_load_error = None

# Load Classifier
try:
    if os.path.exists(MODEL_PATH):
        classifier_model = load_classifier(MODEL_PATH)
        logger.info("Classification model loaded successfully at startup.")
    else:
        model_load_error = (
            f"मॉडेल फाइल सापडली नाही: '{MODEL_PATH}'\n"
            "कृपया मॉडेल प्रशिक्षित करण्यासाठी प्रथम हा आदेश चालवा:\n"
            "python src/train_classifier.py\n"
        )
        logger.warning(model_load_error)
except Exception as e:
    model_load_error = f"मॉडेल लोड करताना त्रुटी (Error loading model): {str(e)}"
    logger.error(model_load_error)

# Load LDA Topic Model
try:
    lda_model, lda_dict = load_lda_model()
    if lda_model is not None:
        logger.info(f"LDA model loaded successfully at startup with {lda_model.num_topics} topics.")
    else:
        lda_load_error = "LDA मॉडेल सापडले नाही. कृपया प्रथम 'python src/train_topic_model.py' चालवा."
        logger.warning(lda_load_error)
except Exception as e:
    lda_load_error = f"LDA मॉडेल लोड करताना त्रुटी: {str(e)}"
    logger.error(lda_load_error)


# ---------------------------------------------------------------------------
# Helper: Sample Articles & Highlighting
# ---------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_articles")


def load_sample_file(filename: str) -> str:
    """Helper to read sample article text from data/sample_articles/."""
    path = os.path.join(SAMPLES_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def format_for_highlighted_text(text: str, entities: list[dict]) -> list[tuple[str, str | None]]:
    """Convert text and entity spans into list of (text_segment, label_or_None) for gr.HighlightedText."""
    if not text:
        return []
    if not entities:
        return [(text, None)]

    sorted_ents = sorted(entities, key=lambda x: (x["start"], x["end"]))
    segments = []
    current_idx = 0

    for ent in sorted_ents:
        start = max(0, int(ent["start"]))
        end = min(len(text), int(ent["end"]))
        label = ent["label"]

        if start > current_idx:
            segments.append((text[current_idx:start], None))

        if start >= current_idx and end > start:
            segments.append((text[start:end], label))
            current_idx = end
        elif end > current_idx and end > start:
            actual_start = max(current_idx, start)
            segments.append((text[actual_start:end], label))
            current_idx = end

    if current_idx < len(text):
        segments.append((text[current_idx:], None))

    return segments


# ---------------------------------------------------------------------------
# Analysis Pipeline Function (with Tab-Level Error Resilience)
# ---------------------------------------------------------------------------
def analyze_article(input_text: str, uploaded_file, n_summary_sentences: int):
    """
    Main controller function that processes the Marathi article across all tabs.
    Each tab calculation is isolated inside try/except.
    """
    article_text = ""
    if uploaded_file is not None:
        try:
            file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            article_text = read_uploaded_txt(file_path)
        except Exception as e:
            err_msg = f"फाइल वाचताना अडचण आली: {str(e)}"
            return (
                err_msg,
                pd.DataFrame({"त्रुटी (Error)": [err_msg]}),
                err_msg,
                err_msg,
                pd.DataFrame({"त्रुटी (Error)": [err_msg]}),
                [(err_msg, None)],
                err_msg,
                pd.DataFrame({"त्रुटी (Error)": [err_msg]}),
                err_msg, "", "", "", pd.DataFrame({"माहिती": ["काहीही नाही"]})
            )
    else:
        article_text = input_text if input_text else ""

    article_text = article_text.strip()
    if not article_text:
        msg = "कृपया मजकूर प्रविष्ट करा किंवा .txt फाइल अपलोड करा। (Please enter Marathi text or upload a .txt file.)"
        return (
            msg,
            pd.DataFrame({"माहिती (Info)": [msg]}),
            msg,
            "**माहिती:** मजकूर उपलब्ध नाही.",
            pd.DataFrame(columns=["भावना वर्ग (Sentiment Class)", "मॉडेल भाकीत गुण (Model Prediction Score)"]),
            [],
            "कोणतेही घटक आढळले नाहीत.",
            pd.DataFrame(columns=["क्रमांक (Rank)", "विषय (Topic)", "विषय भार (Topic Weight)", "महत्त्वाचे शब्द (Top Keywords)"]),
            "", "", "", "0", pd.DataFrame(columns=["शब्द (Word)", "TF-IDF Weight"])
        )

    # -----------------------------------------------------------------------
    # Tab 7: Preprocessing (CO1 Demonstration)
    # -----------------------------------------------------------------------
    original_disp = article_text
    norm_disp = ""
    clean_disp = ""
    tokens_disp = ""
    vocab_size_disp = "0"
    tfidf_sample_df = pd.DataFrame(columns=["शब्द (Word)", "TF-IDF Weight"])

    try:
        prep_res = preprocess(article_text)
        original_disp = prep_res["original"]
        norm_disp = prep_res["normalized"]
        clean_disp = prep_res["cleaned"]
        tokens = prep_res["tokens"]
        tokens_disp = f"एकूण टोकन्स (Total tokens): {len(tokens)}\n\n" + ", ".join(tokens[:100])
        if len(tokens) > 100:
            tokens_disp += " ... [आणखी टोकन्स उर्वरित आहेत]"
        vocab_size_disp = str(len(set(tokens)))

        if classifier_model is not None and hasattr(classifier_model, "named_steps"):
            vectorizer = classifier_model.named_steps.get("tfidf")
            if vectorizer is not None:
                vec = vectorizer.transform([clean_disp])
                feature_names = vectorizer.get_feature_names_out()
                cx = vec.tocoo()
                word_weights = sorted(
                    [(feature_names[col], round(float(val), 4)) for col, val in zip(cx.col, cx.data)],
                    key=lambda x: x[1],
                    reverse=True
                )[:15]
                if word_weights:
                    tfidf_sample_df = pd.DataFrame(word_weights, columns=["शब्द (Word)", "TF-IDF Weight"])
    except Exception as e:
        logger.error(f"Preprocessing tab error: {e}")
        clean_disp = f"Preprocessing Error: {str(e)}"

    # -----------------------------------------------------------------------
    # Tab 2: Classification (Top 3 Categories with Category Scores)
    # -----------------------------------------------------------------------
    classification_df = pd.DataFrame()
    top_category_name = "अज्ञात (Unknown)"

    try:
        if classifier_model is not None:
            top3 = predict_category(classifier_model, article_text, top_k=3)
            if top3:
                top_category_name = f"{LABEL_TO_MARATHI.get(top3[0][0], top3[0][0])} ({top3[0][0]})"
                rows = []
                for rank, (cat, score) in enumerate(top3, 1):
                    rows.append({
                        "क्रमांक (Rank)": f"#{rank}",
                        "वर्ग (Category)": cat,
                        "मराठी नाव (Marathi Name)": LABEL_TO_MARATHI.get(cat, cat),
                        "वर्गवारी गुण (Category Score)": f"{score:.4f}",
                    })
                classification_df = pd.DataFrame(rows)
            else:
                classification_df = pd.DataFrame({"संदेश": ["कोणताही वर्ग ओळखता आला नाही."]})
        else:
            classification_df = pd.DataFrame({"त्रुटी (Error)": [model_load_error]})
    except Exception as e:
        logger.error(f"Classification tab error: {e}")
        classification_df = pd.DataFrame({"त्रुटी (Error)": [f"Classification Error: {str(e)}"]})

    # -----------------------------------------------------------------------
    # Tab 3: Summary (Extractive TextRank)
    # -----------------------------------------------------------------------
    summary_text = ""
    try:
        n_sents = int(n_summary_sentences) if n_summary_sentences else 3
        summary_text = textrank_summary(article_text, n_sentences=n_sents)
        if not summary_text:
            summary_text = "सारांश तयार करता आला नाही."
    except Exception as e:
        logger.error(f"Summarization tab error: {e}")
        summary_text = f"सारांश तयार करताना त्रुटी (Summary Error): {str(e)}"

    # -----------------------------------------------------------------------
    # Tab 4: Sentiment Analysis (l3cube-pune/marathi-sentiment-md)
    # -----------------------------------------------------------------------
    sentiment_label_md = ""
    sentiment_scores_df = pd.DataFrame()
    sentiment_result = None

    try:
        sentiment_result = predict_sentiment(article_text)
        pred_label = sentiment_result.get("label", "Unknown")
        pred_score = sentiment_result.get("score", 0.0)
        all_scores = sentiment_result.get("all_scores", {})

        marathi_sent_map = {
            "Positive": "सकारात्मक (Positive) 😊",
            "Negative": "नकारात्मक (Negative) 😞",
            "Neutral": "तटस्थ (Neutral) 😐",
        }
        marathi_label = marathi_sent_map.get(pred_label, pred_label)

        sentiment_label_md = f"""
### 🎭 मुख्य भावना: **`{marathi_label}`**
- **मॉडेल भाकीत गुण (Model Prediction Score):** **`{pred_score:.4f}`**
*(टीप: हा स्कोअर मॉडेलचे भाकीत गुण दर्शवतो, ही संभाव्यता किंवा 'confidence' नाही.)*
"""
        sent_rows = []
        for lbl, sc in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            sent_rows.append({
                "भावना वर्ग (Sentiment Class)": marathi_sent_map.get(lbl, lbl),
                "इंग्रजी नाव (English Label)": lbl,
                "मॉडेल भाकीत गुण (Model Prediction Score)": f"{float(sc):.4f}",
            })
        sentiment_scores_df = pd.DataFrame(sent_rows)
    except Exception as e:
        logger.error(f"Sentiment tab error: {e}")
        sentiment_label_md = f"**भावना विश्लेषण त्रुटी (Sentiment Error):** {str(e)}"
        sentiment_scores_df = pd.DataFrame({"त्रुटी (Error)": [str(e)]})

    # -----------------------------------------------------------------------
    # Tab 5: Named Entity Recognition (l3cube-pune/marathi-ner)
    # -----------------------------------------------------------------------
    highlighted_entities = []
    grouped_entities_md = ""
    extracted_ents = []

    try:
        extracted_ents = extract_entities(article_text)
        highlighted_entities = format_for_highlighted_text(article_text, extracted_ents)
        grouped = group_by_label(extracted_ents)

        label_marathi_names = {
            "Person": "व्यक्ती (Person)",
            "Location": "स्थान / ठिकाण (Location)",
            "Organization": "संस्था / संघटना (Organization)",
            "Designation": "पद / हुद्दा (Designation)",
            "Date": "दिनांक / तारीख (Date)",
            "Time": "वेळ (Time)",
            "Measure": "माप / प्रमाण (Measure)",
        }

        if grouped:
            md_parts = ["### 📂 वर्गवारीनुसार घटक (Entities Grouped by Category)\n"]
            for label, items in grouped.items():
                label_name = label_marathi_names.get(label, label)
                items_str = ", ".join([f"`{it}`" for it in items])
                md_parts.append(f"- **{label_name}:** {items_str}")
            grouped_entities_md = "\n".join(md_parts)
        else:
            grouped_entities_md = "या बातमीत कोणतेही नामित घटक आढळले नाहीत."
    except Exception as e:
        logger.error(f"NER tab error: {e}")
        highlighted_entities = [(article_text, None)]
        grouped_entities_md = f"**नामित घटक ओळखताना त्रुटी (NER Error):** {str(e)}"

    # -----------------------------------------------------------------------
    # Tab 6: Topic Modelling (LDA)
    # -----------------------------------------------------------------------
    topics_df = pd.DataFrame()
    top_topic_str = "काहीही नाही (None)"

    try:
        if lda_model is not None and lda_dict is not None:
            tokens_lda = preprocess_for_lda(article_text)
            art_topics = article_topics(lda_model, lda_dict, tokens_lda, min_prob=0.01)
            if art_topics:
                top3_topics = art_topics[:3]
                rows = []
                for rank, (tid, prob) in enumerate(top3_topics, 1):
                    kw_list = [w for w, _ in lda_model.show_topic(tid, topn=6)]
                    rows.append({
                        "क्रमांक (Rank)": f"#{rank}",
                        "विषय क्रमांक (Topic ID)": f"Topic {tid}",
                        "विषय भार (Topic Weight)": f"{prob:.4f}",
                        "महत्त्वाचे शब्द (Top Keywords)": ", ".join(kw_list),
                    })
                topics_df = pd.DataFrame(rows)
                top_topic_str = f"Topic {top3_topics[0][0]} (भार: {top3_topics[0][1]:.4f})"
            else:
                topics_df = pd.DataFrame({"संदेश (Info)": ["या लेखासाठी पुरेसे विषय घटक आढळले नाहीत."]})
        else:
            topics_df = pd.DataFrame({"त्रुटी (Error)": [lda_load_error or "LDA मॉडेल उपलब्ध नाही."]})
    except Exception as e:
        logger.error(f"Topics tab error: {e}")
        topics_df = pd.DataFrame({"त्रुटी (Error)": [f"Topic Modeling Error: {str(e)}"]})

    # -----------------------------------------------------------------------
    # Tab 1: Overview
    # -----------------------------------------------------------------------
    overview_md = ""
    try:
        sents = split_sentences(article_text)
        words = article_text.split()

        sent_brief = sentiment_result.get("label", "N/A") if sentiment_result else "N/A"
        sent_brief_score = f"{sentiment_result.get('score', 0.0):.4f}" if sentiment_result else "0.0000"

        sent_color_cls = "sentiment-neu"
        if "positive" in sent_brief.lower():
            sent_color_cls = "sentiment-pos"
        elif "negative" in sent_brief.lower():
            sent_color_cls = "sentiment-neg"

        # Structured entity badges
        entity_badges_html = []
        if extracted_ents:
            grouped_quick = group_by_label(extracted_ents)
            if "Person" in grouped_quick:
                for p in grouped_quick["Person"][:4]:
                    entity_badges_html.append(f'<span class="entity-pill pill-person">👤 {p}</span>')
            if "Location" in grouped_quick:
                for loc in grouped_quick["Location"][:4]:
                    entity_badges_html.append(f'<span class="entity-pill pill-location">📍 {loc}</span>')
            if "Organization" in grouped_quick:
                for org in grouped_quick["Organization"][:4]:
                    entity_badges_html.append(f'<span class="entity-pill pill-org">🏢 {org}</span>')

        entity_pills_disp = " ".join(entity_badges_html) if entity_badges_html else '<span style="color:#1e293b; font-weight:700; font-size:0.92rem;">या बातमीत ठळक घटक आढळले नाहीत.</span>'

        overview_md = f"""
<div class="kpi-grid">
    <div class="kpi-card card-cat">
        <div class="kpi-title">🏷️ अनुमानित मुख्य वर्ग (Predicted Category)</div>
        <div class="kpi-value"><span class="category-badge">{top_category_name}</span></div>
        <div class="kpi-sub" style="color:#1e293b; font-weight:700;">Linear SVM (C=0.5, 87.74% Acc)</div>
    </div>
    <div class="kpi-card card-sent">
        <div class="kpi-title">🎭 अनुमानित भावना (Predicted Sentiment)</div>
        <div class="kpi-value"><span class="{sent_color_cls}">● {sent_brief}</span></div>
        <div class="kpi-sub" style="color:#1e293b; font-weight:700;">मॉडेल भाकीत गुण: <b>{sent_brief_score}</b></div>
    </div>
    <div class="kpi-card card-topic">
        <div class="kpi-title">🎯 प्रमुख शोधलेला विषय (Dominant Topic)</div>
        <div class="kpi-value" style="font-size:1.05rem; color:#047857; font-weight:800;">{top_topic_str}</div>
        <div class="kpi-sub" style="color:#1e293b; font-weight:700;">18-Topic Unsupervised LDA Model</div>
    </div>
    <div class="kpi-card card-stats">
        <div class="kpi-title">📊 बातमी सांख्यिकी (Article Stats)</div>
        <div class="kpi-value" style="font-size:1.05rem; color:#0f172a; font-weight:800;"><b>{len(words)}</b> शब्द | <b>{len(sents)}</b> वाक्ये</div>
        <div class="kpi-sub" style="color:#1e293b; font-weight:700;">एकूण अक्षरे: {len(article_text)} | UTF-8 Devanagari</div>
    </div>
</div>

<div class="executive-box">
    <div class="box-title" style="color:#0f172a; font-weight:800;">👥 शोधलेले प्रमुख घटक (Key Extracted Entities)</div>
    <div class="entity-container">{entity_pills_disp}</div>
</div>

<div class="executive-box" style="margin-top: 14px;">
    <div class="box-title" style="color:#0f172a; font-weight:800;">📝 संक्षिप्त मजकूर सारांश (TextRank Extractive Summary)</div>
    <div class="summary-body" style="color:#0f172a; font-weight:600;">{summary_text}</div>
</div>
"""
    except Exception as e:
        overview_md = f"Overview Error: {str(e)}"

    return (
        overview_md,
        classification_df,
        summary_text,
        sentiment_label_md,
        sentiment_scores_df,
        highlighted_entities,
        grouped_entities_md,
        topics_df,
        original_disp,
        norm_disp,
        clean_disp,
        tokens_disp,
        tfidf_sample_df,
    )


def handle_mt5_summary(
    text: str,
    max_length: int,
    min_length: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
):
    """
    Generate an abstractive Marathi summary using csebuetnlp/mT5_multilingual_XLSum.
    Uses chunked summarisation for articles longer than 512 tokens.
    """
    if not text or not text.strip():
        return (
            "कृपया विश्लेषणासाठी बातमीचा मजकूर प्रविष्ट करा (Please input article text).",
            "⚠️ मजकूर रिक्त आहे.",
        )

    try:
        logger.info("Starting mT5 abstractive summarisation...")
        res = summarize_auto(
            text,
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )

        summary = res.get("summary", "")
        tokens = res.get("input_tokens", 0)
        seconds = res.get("seconds", 0.0)
        mode = res.get("mode", "single")
        truncated = res.get("truncated", False)
        num_chunks = res.get("num_chunks", 1)

        warning_note = ""
        if tokens > 512:
            warning_note = f"\n\n> ⚠️ **Long article ({tokens} tokens > 512): using chunked summarisation ({num_chunks} chunks).**"
        elif truncated:
            warning_note = f"\n\n> ⚠️ **सूचना:** लेख ५१२ टोकन्सपेक्षा मोठा असल्याने ५१२ टोकन्सनंतरचा भाग कापला गेला."

        status_info = f"""
**📊 mT5 सारांश मेट्रिक्स (Summary Metrics):**
- **इनपुट टोकन्स (Input Tokens):** `{tokens}`
- **लागलेला वेळ (Time Taken):** `{seconds} सेकंद`
- **पद्धती (Mode):** `{mode.capitalize()}` ({num_chunks} chunk(s))
{warning_note}

---
*📢 **महत्त्वाची नोंद:** हा सारांश `csebuetnlp/mT5_multilingual_XLSum` डीप लर्निंग मॉडेलद्वारे मशीन-निर्मित (machine-generated, may contain errors) आहे; यात भाषिक अथवा वस्तुस्थितीजन्य त्रुटी असण्याची शक्यता असू शकते.*
"""
        return summary, status_info
    except Exception as e:
        logger.error(f"mT5 generation error: {e}")
        return (
            f"mT5 सारांश तयार करताना त्रुटी आली: {str(e)}",
            f"❌ त्रुटी (Error): {str(e)}",
        )


def handle_news_brief(text: str, prompt_variant: str):
    """
    Generate news brief using Qwen/Qwen3-0.6B with prompt engineering variants.
    Keeps try/except isolation so any failure does not crash the application.
    """
    if not text or not text.strip():
        empty_msg = "कृपया बातमीचा मजकूर प्रविष्ट करा (Please input article text)."
        empty_badge = "⚠️ मजकूर रिक्त आहे."
        return empty_msg, empty_badge, empty_msg, empty_badge, empty_msg, empty_badge

    try:
        logger.info(f"Generating news brief with variant: {prompt_variant}")

        def format_card(res_item):
            output_txt = res_item.get("output", "")
            has_three = "✅ होय (Yes)" if has_three_numbered_items(output_txt) else "❌ नाही (No)"
            dev = devanagari_ratio(output_txt) * 100
            sec = res_item.get("seconds", 0.0)
            words = len(output_txt.split())
            badge = f"""**📊 उद्दिष्ट तपासणी (Objective Checks):**
- **३ क्रमांकित मुद्दे (3 items):** {has_three}
- **देवनागरी प्रमाण (Devanagari):** `{dev:.1f}%`
- **शब्द संख्या (Words):** `{words}` | **वेळ:** `{sec}s`"""
            return output_txt, badge

        if prompt_variant == "Compare all three":
            res_all = compare_prompts(text)
            txt1, badge1 = format_card(res_all.get("zero_shot", {}))
            txt2, badge2 = format_card(res_all.get("role_based", {}))
            txt3, badge3 = format_card(res_all.get("role_constraints", {}))
            return txt1, badge1, txt2, badge2, txt3, badge3
        else:
            res = generate_brief(text, prompt_key=prompt_variant, max_new_tokens=250)
            out_txt, badge = format_card(res)
            not_sel = "*(या पर्यायाची निवड केलेली नाही - Not selected)*"
            not_sel_b = ""
            if prompt_variant == "zero_shot":
                return out_txt, badge, not_sel, not_sel_b, not_sel, not_sel_b
            elif prompt_variant == "role_based":
                return not_sel, not_sel_b, out_txt, badge, not_sel, not_sel_b
            else:  # role_constraints
                return not_sel, not_sel_b, not_sel, not_sel_b, out_txt, badge
    except Exception as e:
        logger.error(f"News brief generation error: {e}")
        err_msg = f"Qwen3 बातमी संक्षेप तयार करताना त्रुटी आली: {str(e)}"
        err_badge = f"❌ त्रुटी (Error): {str(e)}"
        return err_msg, err_badge, err_msg, err_badge, err_msg, err_badge


# ---------------------------------------------------------------------------
# Gradio Blocks UI Construction
# ---------------------------------------------------------------------------
def load_rouge_tables():
    """Load ROUGE results written by src/evaluation.py."""
    base = Path(__file__).resolve().parent / "results" / "summarization"
    summary_path = base / "rouge_summary.csv"
    scores_path = base / "rouge_scores.csv"
    if not summary_path.exists() or not scores_path.exists():
        empty = pd.DataFrame({"message": ["Run `python -m src.evaluation` first."]})
        return empty, empty
    summary = pd.read_csv(summary_path).rename(columns={
        "method": "Method", "rouge1": "ROUGE-1", "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L", "avg_seconds": "Avg seconds", "n_articles": "Articles"})
    per_article = pd.read_csv(scores_path).drop(columns=["reference"]).rename(columns={
        "article": "Article", "method": "Method", "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "seconds": "Seconds"})
    return summary, per_article

def build_app():
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Yantramanav:wght@400;500;600;700;900&display=swap');

    /* Global Typography & Light Aesthetic */
    * {
        font-family: 'Plus Jakarta Sans', 'Yantramanav', system-ui, -apple-system, sans-serif !important;
    }
    /* Force Dark, Crisp High-Contrast Text Everywhere */
    :root, .gradio-container, .dark, body.dark {
        --body-text-color: #0f172a !important;
        --body-text-color-subdued: #1e293b !important;
        --block-label-text-color: #0f172a !important;
        --block-title-text-color: #0f172a !important;
        --input-text-color: #0f172a !important;
        --input-placeholder-color: #475569 !important;
        --table-text-color: #0f172a !important;
        --checkbox-label-text-color: #0f172a !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    .dark .gradio-container, .dark .block, .dark .panel {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* All Text, Paragraphs, Headings and Markdown */
    p, span, div, li, strong, b, h1, h2, h3, h4, h5, h6, .prose, .prose * {
        color: #0f172a !important;
    }
    em, i, .prose em, .prose i {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    small, .caption, .info, .subtext {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* Form Labels, Accordions & Block Titles */
    label, label span, .block-title, .block-label, .wrap label, .form label {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }
    details, details summary, .label-accordion, .accordion-title {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }

    /* Dropdowns, Selects, and Choices */
    select, option, .dropdown, div[data-testid="dropdown"], .secondary-wrap, span.single-select, .wrap-inner, .options, .item, .dropdown-choice, .dropdown-arrow, input[role="combobox"] {
        color: #0f172a !important;
        font-weight: 700 !important;
        background-color: #ffffff !important;
    }
    .dropdown input, .dropdown span, .dropdown div {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Radio Buttons & Labels */
    .radio-group, .radio-group *, .radio-group label, .radio-group span, input[type="radio"] + span, label.selected {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Inputs, Textareas, Numbers & Placeholders */
    input, textarea, input[type="text"], input[type="number"], .input-wrap {
        color: #0f172a !important;
        font-weight: 600 !important;
        background-color: #ffffff !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 10px !important;
    }
    ::placeholder, input::placeholder, textarea::placeholder {
        color: #475569 !important;
        opacity: 1 !important;
        font-weight: 500 !important;
    }

    /* Sliders and Range Indicators */
    input[type="range"], .range-slider, .slider-value, .range-slider-value {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* Executive Top Navigation Header */
    .top-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.08), 0 2px 6px -1px rgba(0, 0, 0, 0.04);
        border-top: 4px solid #4f46e5;
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon {
        font-size: 2.2rem;
        line-height: 1;
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .brand-devanagari {
        color: #4f46e5;
        font-weight: 800;
    }
    .brand-tagline {
        font-size: 0.85rem;
        color: #1e293b;
        font-weight: 600;
        margin-top: 2px;
    }
    .nav-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .nav-pill {
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .badge-primary { background: #e0e7ff; color: #312e81 !important; border-color: #a5b4fc; }
    .badge-info { background: #e0f2fe; color: #075985 !important; border-color: #7dd3fc; }
    .badge-accent { background: #fef3c7; color: #78350f !important; border-color: #fcd34d; }
    .badge-purple { background: #f3e8ff; color: #581c87 !important; border-color: #d8b4fe; }
    .badge-author { background: #e2e8f0; color: #0f172a !important; border-color: #94a3b8; font-weight: 800; }

    /* High-Impact Primary Action Button */
    .primary-btn {
        background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        padding: 12px 24px !important;
    }
    .primary-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.55) !important;
    }

    /* Executive KPI Dashboard Cards (2x2 Fixed Grid) */
    .kpi-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 16px;
    }
    @media (max-width: 600px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    .kpi-card, .dark .kpi-card {
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08) !important;
    }
    .card-cat { border-left: 5px solid #4f46e5 !important; }
    .card-sent { border-left: 5px solid #0284c7 !important; }
    .card-topic { border-left: 5px solid #0d9488 !important; }
    .card-stats { border-left: 5px solid #475569 !important; }

    .kpi-title, .dark .kpi-title {
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
    }
    .kpi-value, .dark .kpi-value {
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        line-height: 1.3;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #1e293b !important;
        margin-top: 4px;
        font-weight: 700;
    }
    .category-badge, .dark .category-badge {
        display: inline-block;
        background: #eef2ff !important;
        color: #312e81 !important;
        padding: 4px 14px;
        border-radius: 8px;
        font-weight: 800;
        border: 1.5px solid #818cf8 !important;
    }
    .sentiment-pos {
        color: #15803d !important;
        font-weight: 800;
    }
    .sentiment-neu {
        color: #b45309 !important;
        font-weight: 800;
    }
    .sentiment-neg {
        color: #b91c1c !important;
        font-weight: 800;
    }

    /* Executive Box & Entity Pills */
    .executive-box {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04);
    }
    .box-title {
        font-size: 0.92rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .entity-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .entity-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.86rem;
        font-weight: 700;
    }
    .pill-person { background: #dbeafe !important; color: #1e3a8a !important; border: 1px solid #93c5fd !important; }
    .pill-location { background: #d1fae5 !important; color: #064e3b !important; border: 1px solid #6ee7b7 !important; }
    .pill-org { background: #ede9fe !important; color: #581c87 !important; border: 1px solid #c4b5fd !important; }
    .summary-body {
        font-size: 1.02rem;
        line-height: 1.65;
        color: #0f172a !important;
        font-weight: 500;
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
    }

    /* Clean, Visible Tab Navigation Buttons */
    .tabs > .tab-nav, div[role="tablist"] {
        border-bottom: 2px solid #cbd5e1 !important;
        gap: 6px !important;
        background: #ffffff !important;
        padding: 6px 8px !important;
        border-radius: 12px 12px 0 0 !important;
        display: flex !important;
        flex-wrap: wrap !important;
    }
    .tabs > .tab-nav button, div[role="tablist"] button {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 0.9rem !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        background: #e2e8f0 !important;
        transition: all 0.15s ease !important;
        white-space: nowrap !important;
    }
    .tabs > .tab-nav button:hover, div[role="tablist"] button:hover {
        color: #0f172a !important;
        background: #cbd5e1 !important;
    }
    .tabs > .tab-nav button.selected, div[role="tablist"] button[aria-selected="true"], div[role="tablist"] button.selected {
        color: #ffffff !important;
        background: #4f46e5 !important;
        border-color: #3730a3 !important;
        font-weight: 900 !important;
        box-shadow: 0 3px 10px rgba(79, 70, 229, 0.4) !important;
    }

    /* Crisp White Dataframes & Tables (Overrides Dark Mode) */
    .dataframe, table, .table-wrap, .table-wrap * {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    table th, .table-wrap th {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #94a3b8 !important;
        font-size: 0.88rem !important;
    }
    table td, .table-wrap td {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        font-size: 0.88rem !important;
    }
    table tr:hover td {
        background-color: #f8fafc !important;
    }

    /* Clean Sample Buttons */
    .sample-btn {
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 7px 14px !important;
    }
    .sample-btn:hover {
        background: #eef2ff !important;
        color: #4338ca !important;
        border-color: #6366f1 !important;
    }

    /* Pipeline Guide Under Analyze Button */
    .pipeline-guide {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #0f172a !important;
        margin-top: 10px;
        padding: 8px 12px;
        background: #e2e8f0;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-weight: 700;
    }
    .pipeline-guide span {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* High-Contrast Readability for All Markdown, Labels & Disclaimers */
    p, span, div, li, strong, b, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    em, i {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    blockquote {
        color: #0f172a !important;
        font-weight: 600 !important;
        background: #f1f5f9 !important;
        border-left: 4px solid #4f46e5 !important;
        padding: 12px 16px !important;
        border-radius: 6px !important;
    }
    label span, .block label, .form label, .wrap label {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }

    /* Inputs, Textareas and Tables */
    .dark input, .dark textarea, .dark select, input, textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #94a3b8 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    .dark table, table {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    .dark tr, tr {
        background-color: #ffffff !important;
    }

    /* Alert and Note Callout Boxes */
    .alert-box {
        background: #dcfce7 !important;
        border-left: 5px solid #16a34a !important;
        padding: 14px 18px;
        margin-top: 15px;
        border-radius: 10px;
        font-size: 0.95em;
        color: #14532d !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(22, 163, 74, 0.15);
        border: 1px solid #86efac;
    }
    .warning-box {
        background: #fef3c7 !important;
        border-left: 5px solid #d97706 !important;
        padding: 14px 18px;
        margin-top: 15px;
        border-radius: 10px;
        font-size: 0.92em;
        box-shadow: 0 1px 3px rgba(217, 119, 6, 0.15);
        color: #78350f !important;
        font-weight: 600;
        border: 1px solid #fcd34d;
    }
    """

    soft_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(title="MahaVarta - Marathi News Intelligence", theme=soft_theme, css=custom_css) as demo:
        gr.HTML(
            f"""
            <style>{custom_css}</style>
            <div class="top-nav-bar">
                <div class="nav-brand">
                    <div class="brand-icon">📰</div>
                    <div class="brand-text">
                        <div class="brand-title">MahaVarta <span class="brand-devanagari">(महावार्ता)</span></div>
                        <div class="brand-tagline">मराठी बातमी बुद्धिमत्ता आणि बहु-मॉडेल विश्लेषण प्रणाली | MITAOE Alandi, Pune</div>
                    </div>
                </div>
                <div class="nav-badges">
                    <span class="nav-pill badge-primary">⚡ Linear SVM <b>87.74%</b></span>
                    <span class="nav-pill badge-info">🏷️ MahaNER <b>96.51%</b></span>
                    <span class="nav-pill badge-accent">🤖 mT5 XL-Sum</span>
                    <span class="nav-pill badge-purple">🧠 Qwen3-0.6B</span>
                    <span class="nav-pill badge-author">👨‍💻 <b>Vishal Shende</b> &amp; <b>Gufran Shah</b></span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left Input Column
            with gr.Column(scale=4):
                gr.Markdown("### ✍️ नमुना बातमी निवडा (Quick Samples)")
                with gr.Row():
                    btn_sample_short = gr.Button("🏏 १. क्रीडा (Sports)", size="sm", elem_classes=["sample-btn"])
                    btn_sample_med = gr.Button("🩺 २. आरोग्य (Health)", size="sm", elem_classes=["sample-btn"])
                    btn_sample_long = gr.Button("🏛️ ३. राजकारण (Politics)", size="sm", elem_classes=["sample-btn"])

                input_textbox = gr.Textbox(
                    label="मराठी बातमीचा मजकूर पेस्ट करा (Paste Marathi News Article)",
                    placeholder="येथे बातमीचा मजकूर पेस्ट करा किंवा वरील १/२/३ नमुना बातमी बटण दाबा...",
                    lines=9,
                )

                with gr.Accordion("📁 किंवा .txt फाइल अपलोड करा (Or Upload .txt file)", open=False):
                    file_upload = gr.File(
                        label="Upload .txt file (UTF-8)",
                        file_types=[".txt"],
                        type="filepath",
                    )

                summary_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="सारांश वाक्यांची संख्या (TextRank Summary Sentence Count)",
                )

                analyze_btn = gr.Button(
                    "🚀 संपूर्ण बातमीचे विश्लेषण करा (Analyse Article)",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
                )

                gr.HTML(
                    """
                    <div class="pipeline-guide">
                        <span>⚡ 12-वर्ग वर्गीकरण</span> • 
                        <span>🏷️ नामित घटक (NER)</span> • 
                        <span>🎭 भावना विश्लेषण</span> • 
                        <span>📋 TextRank &amp; mT5</span> • 
                        <span>🎯 18-विषय LDA</span>
                    </div>
                    """
                )

            # Right Output Tabs Column
            with gr.Column(scale=6):
                with gr.Tabs():
                    # Tab 1: Overview
                    with gr.Tab("📌 आढावा (Overview)"):
                        overview_output = gr.Markdown("बातमीचे विश्लेषण करण्यासाठी डावीकडील **Analyse Article** बटण दाबा.")

                    # Tab 2: Classification
                    with gr.Tab("🏷️ वर्गवारी (Classification)"):
                        gr.Markdown("### 🏆 सर्वोच्च ३ संभाव्य वर्ग आणि वर्गवारी गुण (Top-3 Categories & Scores)")
                        gr.Markdown(
                            "*टीप: हे गुण Linear SVM निर्णय फलन (decision function) वर आधारित मॉडेल वर्गवारी गुण आहेत (हे संभाव्यता किंवा कॉन्फिडन्स नाहीत).*"
                        )
                        classification_output = gr.Dataframe(
                            label="Classification Decisions",
                            headers=["क्रमांक (Rank)", "वर्ग (Category)", "मराठी नाव (Marathi Name)", "वर्गवारी गुण (Category Score)"],
                        )
                        gr.Markdown(
                            """
                            ---
                            **मॉडेल तपशील:**
                            - **मॉडेल:** TF-IDF + Linear Support Vector Machine (`LinearSVC`, $C=0.5$, unigram)
                            - **प्रशिक्षण डेटा:** L3Cube-MahaNews अधिकृत डेटासेट (४२,८७० लेख, १२ वर्ग)
                            - **चाचणी अचूकता (Test Accuracy):** **८७.७४%** | **Macro-F1:** **८५.७०%**
                            """
                        )

                    # Tab 3: Summary (Full-Width Tabs: TextRank & mT5)
                    with gr.Tab("📋 सारांश (Summary)"):
                        gr.Markdown("### 📑 बातमीचा बहु-मॉडेल सारांश (Multi-Model Summarization)")
                        with gr.Tabs():
                            # Sub-tab 1: Extractive TextRank
                            with gr.Tab("🔹 १. TextRank एक्स्ट्रॅक्टिव्ह सारांश (Extractive)"):
                                gr.Markdown("*लेखातील सर्वाधिक मध्यवर्ती/महत्त्वाच्या मूळ वाक्यांची निवड (TF-IDF + Cosine Graph + PageRank Centrality):*")
                                summary_output = gr.Textbox(
                                    label="TextRank Summary (Extractive)",
                                    lines=8,
                                )

                            # Sub-tab 2: Abstractive mT5
                            with gr.Tab("🔸 २. mT5 ॲब्स्ट्रॅक्टिव्ह सारांश (Deep Learning Abstractive)"):
                                gr.Markdown("*`csebuetnlp/mT5_multilingual_XLSum` डीप लर्निंग मॉडेलद्वारे स्वतःच्या शब्दांत तयार केलेला सारांश:*")
                                btn_generate_mt5 = gr.Button(
                                    "✨ Generate mT5 Summary",
                                    variant="primary",
                                    size="md",
                                    elem_classes=["primary-btn"],
                                )
                                mt5_summary_output = gr.Textbox(
                                    label="mT5 Abstractive Summary (Machine-Generated, May Contain Errors)",
                                    lines=8,
                                    placeholder="mT5 सारांशासाठी वरील बटण दाबा...",
                                )
                                mt5_status_output = gr.Markdown("💡 *टीप: पहिल्या वेळी क्लिक केल्यावर मॉडेल कॅशमधून लोड होईल.*")

                                with gr.Accordion("⚙️ mT5 जनरेशन पॅरामीटर्स (Generation Controls)", open=False):
                                    with gr.Row():
                                        mt5_max_len = gr.Slider(minimum=50, maximum=300, value=150, step=5, label="Max Length (कमाल लांबी)")
                                        mt5_min_len = gr.Slider(minimum=20, maximum=150, value=50, step=5, label="Min Length (किमान लांबी)")
                                    with gr.Row():
                                        mt5_beams = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Num Beams")
                                        mt5_len_penalty = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Length Penalty")
                                        mt5_ngram_size = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="No Repeat N-gram")

                    # Tab 4: News Brief (Qwen3 Prompt-Engineered Brief with Sub-tabs)
                    with gr.Tab("📰 बातम्या संक्षेप (News Brief - Qwen3)"):
                        gr.Markdown("### ⚡ Qwen3-0.6B प्रॉम्प्ट-इंजिनिअर्ड बातम्या संक्षेप (News Brief)")
                        gr.Markdown(
                            "*लहान भाषा मॉडेल (`Qwen/Qwen3-0.6B`) वापरून बातमीचे शीर्षक आणि ३ महत्त्वाचे मुद्दे संक्षेप रूपात तयार करा:*"
                        )
                        with gr.Row():
                            with gr.Column(scale=3):
                                brief_prompt_variant = gr.Radio(
                                    choices=["zero_shot", "role_based", "role_constraints", "Compare all three"],
                                    value="role_constraints",
                                    label="प्रॉम्प्ट पर्याय (Prompt Variant)",
                                    info="शून्य-शॉट (Zero-shot), भूमिका-आधारित (Role-based), भूमिका + अटी (Role + Constraints), किंवा तिन्हींची तुलना",
                                )
                            with gr.Column(scale=2):
                                btn_generate_brief = gr.Button("✨ Generate News Brief", variant="primary", size="lg", elem_classes=["primary-btn"])

                        with gr.Tabs():
                            # Sub-tab 1: Zero-shot
                            with gr.Tab("१. शून्य-शॉट (Zero-shot)"):
                                brief_out_zs = gr.Textbox(
                                    label="Zero-shot Output",
                                    lines=7,
                                    placeholder="प्रॉम्प्ट जनरेशन निकाल येथे दिसेल...",
                                )
                                brief_badge_zs = gr.Markdown("")

                            # Sub-tab 2: Role-based
                            with gr.Tab("२. भूमिका-आधारित (Role-based)"):
                                brief_out_rb = gr.Textbox(
                                    label="Role-based Output",
                                    lines=7,
                                    placeholder="प्रॉम्प्ट जनरेशन निकाल येथे दिसेल...",
                                )
                                brief_badge_rb = gr.Markdown("")

                            # Sub-tab 3: Role + Constraints
                            with gr.Tab("३. भूमिका + अटी (Role + Constraints)"):
                                brief_out_rc = gr.Textbox(
                                    label="Role + Constraints Output",
                                    lines=7,
                                    placeholder="प्रॉम्प्ट जनरेशन निकाल येथे दिसेल...",
                                )
                                brief_badge_rc = gr.Markdown("")

                        gr.HTML(
                            """
                            <div class="warning-box">
                                ⚠️ <b>महत्त्वाची नोंद (Caveat Note):</b><br>
                                <i>Generated by a small 0.6B model. Marathi quality is limited and it may add information that is not in the article. Machine-generated, please verify.</i><br>
                                (हे ०.६ अब्ज पॅरामीटर्सच्या लहान मॉडेलद्वारे (<code>Qwen/Qwen3-0.6B</code>) तयार केले आहे. मराठी भाषेची व्याकरण व वाक्यरचना गुणवत्ता मर्यादित असू शकते आणि मॉडेल लेखात नसलेली माहिती स्वतःहून जोडू शकते (hallucination). हे मशीन-निर्मित असून कृपया पडताळणी करावी.)
                            </div>
                            """
                        )

                    # Tab 5: Sentiment
                    with gr.Tab("🎭 भावना विश्लेषण (Sentiment)"):
                        sentiment_label_out = gr.Markdown("भावना विश्लेषणासाठी लेख प्रविष्ट करा.")
                        gr.Markdown("#### सर्व भावनांचे मॉडेल भाकीत गुण (Model Prediction Scores by Class):")
                        sentiment_scores_out = gr.Dataframe(
                            label="All Sentiment Scores",
                            headers=["भावना वर्ग (Sentiment Class)", "इंग्रजी नाव (English Label)", "मॉडेल भाकीत गुण (Model Prediction Score)"],
                        )
                        gr.HTML(
                            """
                            <div class="warning-box">
                                ⚠️ <b>महत्त्वाची नोंद (Caveat Note):</b><br>
                                <i>This model was trained on reviews and tweets, not news. Treat results on news as indicative.</i><br>
                                (हे मॉडेल (<code>l3cube-pune/marathi-sentiment-md</code>) सोशल मीडिया ट्विट्स आणि पुनरावलोकनांवर (reviews) प्रशिक्षित केलेले आहे, बातम्यांवर नाही. त्यामुळे बातम्यांवरील भावनांचे निकाल केवळ सूचक म्हणून विचारात घ्यावेत.)
                            </div>
                            """
                        )

                    # Tab 5: Named Entity Recognition
                    with gr.Tab("👥 नामित घटक (Entities - NER)"):
                        gr.Markdown("### 🔍 लेखातील नामित घटक (Extracted Named Entities in Context)")
                        gr.Markdown("*मजकुरात ओळखलेले घटक वेगवेगळ्या रंगांमध्ये हायलाइट केलेले आहेत:*")
                        ner_highlight_out = gr.HighlightedText(
                            label="Entities in Context",
                            combine_adjacent=False,
                        )
                        gr.Markdown("---")
                        ner_grouped_out = gr.Markdown("घटक गट येथे दिसतील.")
                        gr.Markdown(
                            """
                            ---
                            **मॉडेल तपशील:**
                            - **मॉडेल:** `l3cube-pune/marathi-ner` (Token Classification)
                            - **समर्थित वर्ग:** Person, Location, Organization, Designation, Date, Time, Measure
                            - **MahaNER टेस्ट स्प्लिट चाचणी F1:** **७९.२९%** | अचूकता: **९६.५१%**
                            """
                        )

                    # Tab 6: Topic Modelling (LDA)
                    with gr.Tab("🎯 विषय शोध (Topics - LDA)"):
                        gr.Markdown("### 🔍 बातमीचे प्रमुख विषय (Discovered Topics & Topic Weights)")
                        gr.Markdown(
                            "*टीप: हे गुण LDA (Latent Dirichlet Allocation) मॉडेलनुसार लेखातील संभाव्यता विषय भार (Topic Weight) दर्शवतात.*"
                        )
                        topics_output = gr.Dataframe(
                            label="Top 3 Topics",
                            headers=["क्रमांक (Rank)", "विषय क्रमांक (Topic ID)", "विषय भार (Topic Weight)", "महत्त्वाचे शब्द (Top Keywords)"],
                        )
                        gr.Markdown(
                            """
                            ---
                            **मॉडेल तपशील:**
                            - **मॉडेल:** Gensim Latent Dirichlet Allocation (LDA)
                            - **इष्टतम विषय संख्या (Optimal K):** १८ विषय ($C_v$ सुसंगतता: ०.४८५२)
                            - **प्रशिक्षण कॉर्पस:** L3Cube-MahaNews प्रशिक्षण डेटासेट (मराठी स्टॉपवर्ड्स फिल्टरिंगसह)
                            """
                        )

                    # Tab 7: Preprocessing (CO1 Demonstration)
                    with gr.Tab("⚙️ मजकूर पूर्वप्रक्रिया (Preprocessing - CO1)"):
                        gr.Markdown("### 🔬 CO1 प्रात्यक्षिक: युनिकोड नॉर्मलायझेशन, क्लीनिंग, टोकनायझेशन आणि TF-IDF")
                        with gr.Accordion("१. मूळ मजकूर (Original Input Text)", open=False):
                            orig_text_out = gr.Textbox(label="Original Text (Untouched for Transformers)", lines=4)
                        with gr.Accordion("२. युनिकोड नॉर्मलाइझ केलेला मजकूर (Unicode Normalized - Indic NLP)", open=False):
                            norm_text_out = gr.Textbox(label="Normalized Text", lines=4)
                        with gr.Accordion("३. स्वच्छ केलेला मजकूर (Cleaned Text)", open=True):
                            clean_text_out = gr.Textbox(label="Cleaned Text (HTML, URLs removed, danda kept)", lines=4)
                        with gr.Accordion("४. शब्द टोकन्स (Word Tokens - Indic Tokenizer)", open=True):
                            tokens_out = gr.Textbox(label="Token List", lines=4)
                        with gr.Accordion("५. शब्दसंग्रह सांख्यिकी (Vocabulary Statistics & TF-IDF)", open=True):
                            tfidf_sample_out = gr.Dataframe(
                                label="Top Informative Terms by TF-IDF Weight (या लेखातील महत्त्वाचे शब्द)",
                                headers=["शब्द (Word)", "TF-IDF Weight"],
                            )

                    # Tab 8: Evaluation (ROUGE results)
                    with gr.Tab("📊 मूल्यमापन (Evaluation)"):
                        gr.Markdown(
                            "### ROUGE evaluation of summarisation methods\n"
                            "Scores are F-measures against human-written Marathi reference "
                            "summaries for 3 sample articles. Results come from "
                            "`python -m src.evaluation`."
                        )
                        _summary_df, _per_article_df = load_rouge_tables()
                        gr.Markdown("**Average across articles**")
                        gr.Dataframe(value=_summary_df, interactive=False)
                        gr.Markdown("**Per article**")
                        gr.Dataframe(value=_per_article_df, interactive=False)
                        gr.Markdown(
                            "**How to read this**\n"
                            "- Only 3 articles were evaluated, so differences are indicative, "
                            "not statistically significant.\n"
                            "- ROUGE rewards word overlap. TextRank copies article sentences, "
                            "so it is favoured. mT5 paraphrases, which ROUGE under-rewards.\n"
                            "- Qwen3 writes a headline plus three takeaways, not a short "
                            "summary, so its comparison is not like for like.\n"
                            "- The first Qwen3 zero_shot run includes the one-time model "
                            "download, so its time is inflated."
                        )

        # Wire Up Analyze Button
        analyze_btn.click(
            fn=analyze_article,
            inputs=[input_textbox, file_upload, summary_slider],
            outputs=[
                overview_output,
                classification_output,
                summary_output,
                sentiment_label_out,
                sentiment_scores_out,
                ner_highlight_out,
                ner_grouped_out,
                topics_output,
                orig_text_out,
                norm_text_out,
                clean_text_out,
                tokens_out,
                tfidf_sample_out,
            ],
        )

        # Wire Up mT5 Generate Button
        btn_generate_mt5.click(
            fn=handle_mt5_summary,
            inputs=[
                input_textbox,
                mt5_max_len,
                mt5_min_len,
                mt5_beams,
                mt5_len_penalty,
                mt5_ngram_size,
            ],
            outputs=[
                mt5_summary_output,
                mt5_status_output,
            ],
        )

        # Wire Up News Brief Generate Button
        btn_generate_brief.click(
            fn=handle_news_brief,
            inputs=[input_textbox, brief_prompt_variant],
            outputs=[
                brief_out_zs,
                brief_badge_zs,
                brief_out_rb,
                brief_badge_rb,
                brief_out_rc,
                brief_badge_rc,
            ],
        )

        # Wire Up Sample Article Buttons
        btn_sample_short.click(
            fn=lambda: load_sample_file("01_short_sports.txt"),
            outputs=[input_textbox],
        )
        btn_sample_med.click(
            fn=lambda: load_sample_file("02_medium_health.txt"),
            outputs=[input_textbox],
        )
        btn_sample_long.click(
            fn=lambda: load_sample_file("03_long_politics.txt"),
            outputs=[input_textbox],
        )

    return demo


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
