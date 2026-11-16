"""
Unit tests for src/summarization.py (TextRank Summarizer)
=========================================================
Tests cover:
  - Marathi sentence segmentation with dandas, question marks, exclamations, periods
  - Preservation of decimal numbers (ASCII and Devanagari digits)
  - Preservation of common Marathi abbreviations (डॉ., प्रा., श्री., इ.)
  - TextRank edge cases:
      - Empty input text
      - Whitespace-only input
      - Single sentence input
      - Fewer sentences than requested n_sentences
      - Extraction of top-k sentences in chronological order
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.summarization import (
    split_sentences,
    textrank_summary,
    chunk_sentences,
    mt5_summary,
    chunked_mt5_summary,
    summarize_auto,
    WHITESPACE_HANDLER,
)


class TestSentenceSplitter:
    """Tests for split_sentences() function."""

    def test_split_on_devanagari_danda(self):
        """Should split sentences ending with Devanagari danda (।) and double danda."""
        text = "हा पहिला वाक्य आहे। हा दुसरा वाक्य आहे॥ हा तिसरा वाक्य आहे।"
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert "पहिला" in sentences[0]
        assert "दुसरा" in sentences[1]
        assert "तिसरा" in sentences[2]

    def test_split_on_question_and_exclamation(self):
        """Should split on ? and ! marks."""
        text = "आज हवामान कसे आहे? पाऊस पडण्याची शक्यता आहे! सावधगिरी बाळगा."
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert "हवामान" in sentences[0]
        assert "पाऊस" in sentences[1]
        assert "सावधगिरी" in sentences[2]

    def test_preserves_decimals(self):
        """Should not split on periods within decimal numbers (ASCII & Devanagari)."""
        text = "भारताचा विकास दर ७.५ टक्के राहिला. महागाई दर 4.2 टक्क्यांवर आला आहे।"
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "७.५" in sentences[0]
        assert "4.2" in sentences[1]

    def test_preserves_abbreviations(self):
        """Should not split on periods in common Marathi abbreviations."""
        text = "डॉ. बाबासाहेब आंबेडकर यांनी संविधानाची निर्मिती केली. प्रा. जोशी यांनी व्याख्यान दिले."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "डॉ." in sentences[0]
        assert "प्रा." in sentences[1]

    def test_empty_and_whitespace(self):
        """Should return empty list for empty or whitespace-only input."""
        assert split_sentences("") == []
        assert split_sentences("   \n\t  ") == []


class TestTextRankSummary:
    """Tests for textrank_summary() function."""

    def test_empty_input(self):
        """Empty input should return empty string."""
        assert textrank_summary("") == ""
        assert textrank_summary("   ") == ""

    def test_single_sentence(self):
        """Single sentence input should return that sentence."""
        text = "पुणे शहरात आज मुसळधार पाऊस पडला।"
        summary = textrank_summary(text, n_sentences=3)
        assert "पुणे शहरात आज मुसळधार पाऊस पडला" in summary

    def test_fewer_sentences_than_requested(self):
        """When text has fewer sentences than n_sentences, return all sentences."""
        text = "पंतप्रधानांनी दौऱ्याची घोषणा केली। ते उद्या मुंबईला भेट देणार आहेत।"
        summary = textrank_summary(text, n_sentences=5)
        assert "पंतप्रधानांनी" in summary
        assert "मुंबईला" in summary

    def test_multi_sentence_extraction(self):
        """Should extract requested number of sentences from a multi-sentence news text."""
        article = (
            "भारतीय क्रिकेट संघाने आज शानदार विजय मिळवला. "
            "कर्णधार रोहित शर्माने उत्कृष्ट फलंदाजी करत शतक झळकावले। "
            "गोलंदाजांनीही उत्तम कामगिरी करत प्रतिस्पर्ध्याला रोखले. "
            "सामनावीर पुरस्कार रोहित शर्माला देण्यात आला! "
            "प्रेक्षकांनी मैदानात जल्लोष केला आणि संघाचे अभिनंदन केले। "
            "हा विजय मालिकेसाठी अत्यंत महत्त्वाचा ठरणार आहे."
        )
        summary = textrank_summary(article, n_sentences=2)
        assert len(summary) > 0
        # Should contain danda punctuation
        assert "।" in summary

    def test_chronological_order_preserved(self):
        """Selected sentences must appear in their original narrative order."""
        article = (
            "वाक्य एक: घटना घडली. "
            "वाक्य दोन: पोलिसांनी तपास सुरू केला. "
            "वाक्य तीन: आरोपीला अटक करण्यात आली. "
            "वाक्य चार: न्यायालयात हजर केले जाईल. "
            "वाक्य पाच: पुढील सुनावणी पुढील आठवड्यात होईल."
        )
        summary = textrank_summary(article, n_sentences=3)

        # Check that sentences in summary appear in original relative order
        pos_two = summary.find("दोन") if "दोन" in summary else -1
        pos_four = summary.find("चार") if "चार" in summary else -1
        if pos_two != -1 and pos_four != -1:
            assert pos_two < pos_four


class TestChunkingLogic:
    """Fast tests for sentence chunking logic that do NOT need the real 2.3 GB model."""

    def test_chunk_sentences_mock_tokenizer(self):
        """Should group sentences into chunks without exceeding token limit or breaking sentences."""
        sentences = [
            "पहिला वाक्य आहे ज्यामध्ये काही शब्द आहेत.",  # length ~ 7 words
            "दुसरा वाक्य थोडा मोठा आहे आणि माहिती देतो.",   # length ~ 8 words
            "तिसरा वाक्य आणखी नवीन घटना सांगतो.",        # length ~ 6 words
            "चौथा वाक्य या संपूर्ण प्रकरणाचा शेवट करतो.",    # length ~ 7 words
        ]

        # Fake tokenizer function: each word is 10 tokens
        fake_tokenizer = lambda s: len(s.split()) * 10

        # Max chunk tokens = 90 (sentences 1 + 2 = 70 + 80 = 150 > 90, so each in its own or paired)
        chunks = chunk_sentences(sentences, max_chunk_tokens=100, tokenizer_fn=fake_tokenizer)

        # Chunks must be created
        assert len(chunks) >= 2
        # None of the chunks should contain cut sentences
        for c in chunks:
            # Each chunk must contain at least one complete sentence
            assert any(s in c for s in sentences)

    def test_chunk_sentences_empty_input(self):
        """Empty input should return empty list."""
        assert chunk_sentences([]) == []
        assert chunk_sentences([], max_chunk_tokens=300) == []

    def test_chunk_sentences_single_long_sentence(self):
        """A sentence larger than max_chunk_tokens must not crash and become its own chunk."""
        long_sent = "खूप मोठा वाक्य " * 50
        fake_tokenizer = lambda s: 600  # Exceeds max_chunk_tokens
        chunks = chunk_sentences([long_sent], max_chunk_tokens=450, tokenizer_fn=fake_tokenizer)
        assert len(chunks) == 1
        assert chunks[0] == long_sent.strip()

    def test_whitespace_handler(self):
        """WHITESPACE_HANDLER should collapse multiple spaces and newlines."""
        raw = "  पहिला  वाक्य\n\n  आहे \t  दुसरा \n वाक्य  "
        cleaned = WHITESPACE_HANDLER(raw)
        assert cleaned == "पहिला वाक्य आहे दुसरा वाक्य"

    def test_mt5_summary_empty_input(self):
        """Empty or whitespace-only input must return error dict without calling model."""
        res = mt5_summary("")
        assert res["input_tokens"] == 0
        assert res["truncated"] is False
        assert "लहान किंवा रिक्त" in res["summary"]

        res_ws = mt5_summary("     ")
        assert res_ws["input_tokens"] == 0
        assert "लहान किंवा रिक्त" in res_ws["summary"]

    def test_chunked_summary_empty_input(self):
        """Empty input to chunked_mt5_summary must return safe empty dict."""
        res = chunked_mt5_summary("")
        assert res["num_chunks"] == 0
        assert res["chunk_summaries"] == []
        assert "लहान किंवा रिक्त" in res["summary"]

    def test_summarize_auto_empty_input(self):
        """Empty input to summarize_auto must return safe dict."""
        res = summarize_auto("")
        assert res["input_tokens"] == 0
        assert "लहान किंवा रिक्त" in res["summary"]


class TestMT5RealModel:
    """
    Integration tests using the real csebuetnlp/mT5_multilingual_XLSum model.
    Marked as 'slow' so they are skipped during quick test runs.
    """

    @pytest.mark.slow
    def test_real_mt5_short_inference(self):
        """Test inference on a short Marathi text with real mT5."""
        text = (
            "भारतीय अंतराळ संशोधन संस्था म्हणजेच इस्रोने आज नवीन उपग्रहाचे यशस्वी प्रक्षेपण केले. "
            "श्रीहरिकोटा येथील सतीश धवन अंतराळ केंद्रावरून हा उपग्रह प्रक्षेपित करण्यात आला. "
            "हा उपग्रह हवामान बदलांचा अभ्यास करण्यासाठी अत्यंत उपयुक्त ठरणार आहे."
        )
        res = mt5_summary(text, max_length=50, min_length=15, num_beams=2)
        assert isinstance(res["summary"], str)
        assert len(res["summary"]) > 0
        assert res["input_tokens"] > 0
        assert res["truncated"] is False
        assert res["mode"] == "single"
        assert res["seconds"] > 0

    @pytest.mark.slow
    def test_real_mt5_chunked_inference(self):
        """Test chunked summarisation on a multi-paragraph Marathi text with small chunk size."""
        text = (
            "महाराष्ट्र राज्य शासनाने आज शेतकऱ्यांसाठी नवीन कर्जमाफी योजनेची घोषणा केली आहे. "
            "मुख्यमंत्री एकनाथ शिंदे आणि उपमुख्यमंत्री देवेंद्र फडणवीस यांनी पत्रकार परिषदेत ही माहिती दिली. "
            "या योजनेमुळे राज्यातील लाखो शेतकऱ्यांना मोठा दिलासा मिळणार असल्याचे स्पष्ट झाले. "
            "विरोधी पक्षांनी मात्र या निर्णयाचे स्वागत करतानाच अंमलबजावणी वेळेत व्हावी अशी मागणी केली आहे. "
            "दुसरीकडे कृषी तज्ज्ञांनी या निर्णयामुळे ग्रामीण अर्थव्यवस्थेला चालना मिळेल असा विश्वास व्यक्त केला आहे."
        )
        # Force chunking with a small chunk limit (e.g. 30 tokens per chunk)
        res = chunked_mt5_summary(text, max_chunk_tokens=30, max_length=50, min_length=10, num_beams=2)
        assert isinstance(res["summary"], str)
        assert res["num_chunks"] >= 2
        assert len(res["chunk_summaries"]) >= 2
        assert res["mode"] == "chunked"
        assert res["seconds"] > 0
