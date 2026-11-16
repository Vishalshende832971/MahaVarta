"""
Unit tests for src/prompting.py (Qwen3 Prompting & Objective Checks)
=====================================================================
Tests cover:
  - has_three_numbered_items with ASCII and Devanagari numerals
  - devanagari_ratio calculation on pure Marathi, English, and mixed text
  - novel_word_ratio stem matching (inflected forms like कोहली vs कोहलीची must NOT be novel)
  - invented_name_candidates (detects novel entity names, ignores inflected source names)
  - output_length_words
  - Prompt dictionary structure and placeholder integrity
  - Empty input handling
  - Integration test for real Qwen3 generation (marked slow)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompting import (
    PROMPTS,
    has_three_numbered_items,
    devanagari_ratio,
    novel_word_ratio,
    invented_name_candidates,
    output_length_words,
    generate_brief,
    compare_prompts,
    strip_thinking_tokens,
)


class TestObjectiveChecks:
    """Fast tests for objective checks that do NOT require the real LLM."""

    def test_has_three_numbered_items_ascii(self):
        """Should return True for exactly 3 items numbered with ASCII digits."""
        text = (
            "मुख्य बातमी: पुण्यात मुसळधार पाऊस.\n"
            "1. शाळांना सुट्टी जाहीर झाली आहे.\n"
            "2. वाहतूक कोंडी निर्माण झाली आहे.\n"
            "3. नागरिकांनी घरातच राहावे असे आवाहन केले आहे."
        )
        assert has_three_numbered_items(text) is True

    def test_has_three_numbered_items_devanagari(self):
        """Should return True for exactly 3 items numbered with Devanagari numerals."""
        text = (
            "महत्त्वाचे मुद्दे:\n"
            "१. पहिली ठळक बातमी.\n"
            "२. दुसरी ठळक बातमी.\n"
            "३. तिसरी ठळक बातमी."
        )
        assert has_three_numbered_items(text) is True

    def test_has_three_numbered_items_false_when_four(self):
        """Should return False if there are 4 numbered items."""
        text = "1. मुद्दा एक\n2. मुद्दा दोन\n3. मुद्दा तीन\n4. मुद्दा चार"
        assert has_three_numbered_items(text) is False

    def test_has_three_numbered_items_false_when_two(self):
        """Should return False if there are fewer than 3 items."""
        text = "1. मुद्दा एक\n2. मुद्दा दोन"
        assert has_three_numbered_items(text) is False

    def test_devanagari_ratio_pure_marathi(self):
        """Pure Marathi text should have devanagari_ratio == 1.0."""
        text = "महाराष्ट्र राज्य निवडणूक आयोग"
        assert devanagari_ratio(text) == 1.0

    def test_devanagari_ratio_mixed(self):
        """Mixed English and Marathi text should compute accurate letter ratio."""
        # 4 Devanagari letters (भारत), 5 Latin letters (India)
        text = "भारत India"
        ratio = devanagari_ratio(text)
        assert ratio == round(4 / 9, 4)

    def test_devanagari_ratio_empty(self):
        """Empty or numeric-only text should return 0.0."""
        assert devanagari_ratio("") == 0.0
        assert devanagari_ratio("12345 67890") == 0.0

    def test_novel_word_ratio_inflections_not_flagged(self):
        """
        Crucial test: inflected forms of source words (e.g. कोहली vs कोहलीची, शास्त्री vs शास्त्रींनी)
        must share stems and NOT be counted as novel words!
        """
        article = "विराट कोहली आणि शास्त्री यांनी मैदानात उत्तम कामगिरी केली."
        # Summary uses inflected forms: कोहलीची (genitive), शास्त्रींनी (agentive), मैदानात
        summary = "कोहलीची खेळी शास्त्रींनी पाहिली."

        # कोहलीची matches stem of कोहली -> NOT novel
        # शास्त्रींनी matches stem of शास्त्री -> NOT novel
        # खेळी and पाहिली are novel
        ratio = novel_word_ratio(summary, article)
        # Summary has 4 words: कोहलीची, खेळी, शास्त्रींनी, पाहिली
        # 2 novel (खेळी, पाहिली) / 4 total = 0.50
        assert ratio <= 0.55

    def test_invented_name_candidates_inflection_ignored(self):
        """Inflected forms of known source names should NOT be flagged as invented."""
        article = "पंतप्रधान नरेंद्र मोदी यांनी भाषण केले."
        summary = "नरेंद्र मोदींनी नवीन योजनेची घोषणा केली."
        # मोदींनी is an inflected form of मोदी -> should not be flagged as invented
        res = invented_name_candidates(summary, article)
        assert "मोदी" not in res
        assert "नरेंद्र" not in res

    def test_invented_name_candidates_flags_hallucination(self):
        """Names completely absent from article must be flagged as invented."""
        article = "क्रिकेट सामन्यात भारतीय संघाने विजय मिळवला."
        summary = "कर्णधार जयंत पाटील यांनी शतक केले."
        # जयंत पाटील is not in article -> should be flagged
        res = invented_name_candidates(summary, article)
        assert "जयंत" in res or "पाटील" in res

    def test_output_length_words(self):
        """Should accurately count Devanagari words."""
        text = "हा एक साधा मराठी वाक्य आहे."
        assert output_length_words(text) == 6

    def test_strip_thinking_tokens(self):
        """Should strip <think> blocks if model generated them."""
        raw = "<think>Let me analyze the news article...</think>हा खरा सारांश आहे."
        cleaned = strip_thinking_tokens(raw)
        assert cleaned == "हा खरा सारांश आहे."

    def test_prompts_dict_integrity(self):
        """Verify that PROMPTS dict contains the three required keys and {article} placeholder."""
        assert "zero_shot" in PROMPTS
        assert "role_based" in PROMPTS
        assert "role_constraints" in PROMPTS
        for key, p in PROMPTS.items():
            assert "{article}" in p

    def test_generate_brief_empty_input(self):
        """Empty input should return safe fallback dict without calling model."""
        res = generate_brief("", prompt_key="zero_shot")
        assert res["input_tokens"] == 0
        assert res["output_tokens"] == 0
        assert res["truncated_input"] is False
        assert "लहान किंवा रिक्त" in res["output"]


class TestRealQwenModel:
    """Integration tests with the real Qwen3-0.6B model (marked slow)."""

    @pytest.mark.slow
    def test_real_qwen_generation(self):
        """Test inference with real Qwen3-0.6B on a short article."""
        article = (
            "भारतीय अंतराळ संशोधन संस्था (इस्रो) ने नवीन उपग्रहाचे यशस्वी प्रक्षेपण केले. "
            "हा उपग्रह देशाच्या हवामान अंदाजासाठी अत्यंत मोलाचा ठरणार आहे. "
            "पंतप्रधानांनी सर्व वैज्ञानिकांचे अभिनंदन केले."
        )
        res = generate_brief(article, prompt_key="role_constraints", max_new_tokens=150)
        assert isinstance(res["output"], str)
        assert len(res["output"]) > 0
        assert res["input_tokens"] > 0
        assert res["output_tokens"] > 0
        assert res["seconds"] > 0
        assert "<think>" not in res["output"]
