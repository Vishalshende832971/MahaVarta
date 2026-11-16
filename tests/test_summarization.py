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

from src.summarization import split_sentences, textrank_summary


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
