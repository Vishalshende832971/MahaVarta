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

        key_ents_summary = []
        if extracted_ents:
            grouped_quick = group_by_label(extracted_ents)
            if "Person" in grouped_quick:
                key_ents_summary.append(f"**व्यक्ती:** {', '.join(grouped_quick['Person'][:3])}")
            if "Location" in grouped_quick:
                key_ents_summary.append(f"**ठिकाणे:** {', '.join(grouped_quick['Location'][:3])}")
            if "Organization" in grouped_quick:
                key_ents_summary.append(f"**संघटना:** {', '.join(grouped_quick['Organization'][:3])}")

        key_ents_str = " | ".join(key_ents_summary) if key_ents_summary else "काहीही नाही"
        sent_brief = sentiment_result.get("label", "N/A") if sentiment_result else "N/A"
        sent_brief_score = f"{sentiment_result.get('score', 0.0):.4f}" if sentiment_result else "0.0000"

        overview_md = f"""
### 📊 बातमीचे विश्लेषण (Executive Summary)

- **अनुमानित मुख्य वर्ग (Predicted Category):** **`{top_category_name}`**
- **अनुमानित भावना (Predicted Sentiment):** **`{sent_brief}`** *(भाकीत गुण: `{sent_brief_score}`)*
- **प्रमुख शोधलेला विषय (Dominant LDA Topic):** **`{top_topic_str}`**
- **एकूण शब्द (Word Count):** `{len(words)}` | **वाक्ये (Sentences):** `{len(sents)}` | **अक्षरे (Characters):** `{len(article_text)}`
- **प्रमुख घटक (Key Entities):** {key_ents_str}

---

#### 📝 संक्षिप्त मजकूर सारांश (TextRank Summary):
> {summary_text}
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
def build_app():
    custom_css = """
    .main-header { text-align: center; margin-bottom: 20px; }
    .alert-box {
        background-color: #f0f7ff;
        border-left: 5px solid #0066cc;
        padding: 12px 16px;
        margin-top: 15px;
        border-radius: 4px;
        font-size: 0.95em;
    }
    .warning-box {
        background-color: #fff9e6;
        border-left: 5px solid #ff9900;
        padding: 12px 16px;
        margin-top: 15px;
        border-radius: 4px;
        font-size: 0.92em;
    }
    """

    with gr.Blocks(title="MahaVarta - Marathi News Intelligence", css=custom_css) as demo:
        gr.Markdown(
            """
            # 📰 MahaVarta (महावार्ता): Marathi News Intelligence
            ### मराठी बातमी विश्लेषण, वर्गीकरण आणि सारांशीकरण प्रणाली
            **NLP Mini Project — MIT Academy of Engineering, Alandi, Pune**
            **Team:** Vishal Gorakh Shende (PRN 202401110034) & Gufran Shah | **Course:** Natural Language Processing
            """,
            elem_classes=["main-header"],
        )

        with gr.Row():
            # Left Input Column
            with gr.Column(scale=4):
                gr.Markdown("### ✍️ बातमी प्रविष्ट करा (Input Article)")
                input_textbox = gr.Textbox(
                    label="मराठी बातमीचा मजकूर पेस्ट करा (Paste Marathi News Article)",
                    placeholder="येथे बातमीचा मजकूर पेस्ट करा...",
                    lines=10,
                )
                file_upload = gr.File(
                    label="किंवा .txt फाइल अपलोड करा (Or Upload .txt file - UTF-8)",
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

                analyze_btn = gr.Button("🔍 बातमीचे विश्लेषण करा (Analyse Article)", variant="primary", size="lg")

                # Sample Article Quick Load Buttons
                gr.Markdown("#### 📂 नमुना बातम्या (Sample Articles)")
                with gr.Row():
                    btn_sample_short = gr.Button("१. क्रीडा (Short - Sports)", size="sm")
                    btn_sample_med = gr.Button("२. आरोग्य (Medium - Health)", size="sm")
                    btn_sample_long = gr.Button("३. राजकारण (Long - Politics)", size="sm")

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

                    # Tab 3: Summary (Side-by-Side: TextRank & mT5)
                    with gr.Tab("📋 सारांश (Summary)"):
                        gr.Markdown("### 📑 बातमीचा सारांश: एक्स्ट्रॅक्टिव्ह (TextRank) आणि ॲब्स्ट्रॅक्टिव्ह (mT5)")
                        with gr.Row():
                            # Left Column: Extractive TextRank
                            with gr.Column(scale=1):
                                gr.Markdown("#### 🔹 १. TextRank एक्स्ट्रॅक्टिव्ह सारांश (Extractive)")
                                gr.Markdown("*लेखातील सर्वाधिक मध्यवर्ती/महत्त्वाच्या मूळ वाक्यांची निवड:*")
                                summary_output = gr.Textbox(
                                    label="TextRank Summary (Extractive)",
                                    lines=8,
                                )
                                gr.Markdown(
                                    """
                                    **पद्धती तपशील:**
                                    - मराठी वाक्य विभाजन (दंडा `।`, पूर्णविराम `.`, प्रश्न `?`)
                                    - TF-IDF + Cosine Graph + PageRank Centrality
                                    """
                                )

                            # Right Column: Abstractive mT5
                            with gr.Column(scale=1):
                                gr.Markdown("#### 🔸 २. mT5 ॲब्स्ट्रॅक्टिव्ह सारांश (Abstractive)")
                                gr.Markdown("*`csebuetnlp/mT5_multilingual_XLSum` डीप लर्निंग मॉडेलद्वारे स्वतःच्या शब्दांत सारांश:*")
                                btn_generate_mt5 = gr.Button(
                                    "✨ Generate mT5 Summary",
                                    variant="primary",
                                )
                                mt5_summary_output = gr.Textbox(
                                    label="mT5 Abstractive Summary (Machine-Generated, May Contain Errors)",
                                    lines=8,
                                    placeholder="mT5 सारांशासाठी वरील बटण दाबा...",
                                )
                                mt5_status_output = gr.Markdown("💡 *टीप: पहिल्या वेळी क्लिक केल्यावर मॉडेल कॅशमधून लोड होईल.*")

                                with gr.Accordion("⚙️ mT5 जनरेशन पॅरामीटर्स (Generation Controls)", open=False):
                                    with gr.Row():
                                        mt5_max_len = gr.Slider(minimum=30, maximum=150, value=84, step=2, label="Max Length")
                                        mt5_min_len = gr.Slider(minimum=10, maximum=80, value=30, step=2, label="Min Length")
                                    with gr.Row():
                                        mt5_beams = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Num Beams")
                                        mt5_len_penalty = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Length Penalty")
                                        mt5_ngram_size = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="No Repeat N-gram")

                    # Tab 4: News Brief (Qwen3 Prompt-Engineered Brief)
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
                                btn_generate_brief = gr.Button("✨ Generate News Brief", variant="primary", size="lg")

                        with gr.Row():
                            # Column 1: Zero-shot
                            with gr.Column(scale=1):
                                gr.Markdown("#### १. शून्य-शॉट (Zero-shot)")
                                brief_out_zs = gr.Textbox(
                                    label="Zero-shot Output",
                                    lines=7,
                                    placeholder="प्रॉम्प्ट जनरेशन निकाल येथे दिसेल...",
                                )
                                brief_badge_zs = gr.Markdown("")

                            # Column 2: Role-based
                            with gr.Column(scale=1):
                                gr.Markdown("#### २. भूमिका-आधारित (Role-based)")
                                brief_out_rb = gr.Textbox(
                                    label="Role-based Output",
                                    lines=7,
                                    placeholder="प्रॉम्प्ट जनरेशन निकाल येथे दिसेल...",
                                )
                                brief_badge_rb = gr.Markdown("")

                            # Column 3: Role + Constraints
                            with gr.Column(scale=1):
                                gr.Markdown("#### ३. भूमिका + अटी (Role + Constraints)")
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
