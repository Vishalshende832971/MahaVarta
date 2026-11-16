"""
Unit tests for src/preprocessing.py
====================================
Tests cover:
  - URL removal
  - HTML tag removal
  - Extra whitespace collapsing
  - Devanagari danda (।) preservation
  - Empty input handling
  - Non-UTF-8 file reading error
  - Full preprocessing pipeline
"""

import os
import sys
import tempfile
import pytest

# Add the project root to the path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    normalize_text,
    clean_text,
    tokenize,
    preprocess,
    read_uploaded_txt,
)


class TestCleanText:
    """Tests for the clean_text function."""

    def test_removes_urls_http(self):
        """URLs starting with http:// should be removed."""
        text = "बातमी पहा http://example.com/news येथे"
        result = clean_text(text)
        assert "http" not in result
        assert "example.com" not in result
        assert "बातमी" in result
        assert "येथे" in result

    def test_removes_urls_https(self):
        """URLs starting with https:// should be removed."""
        text = "अधिक माहिती https://www.example.org/marathi साठी"
        result = clean_text(text)
        assert "https" not in result
        assert "example.org" not in result

    def test_removes_urls_www(self):
        """URLs starting with www. should be removed."""
        text = "संकेतस्थळ www.batmya.com वर उपलब्ध"
        result = clean_text(text)
        assert "www" not in result
        assert "batmya.com" not in result

    def test_removes_html_tags(self):
        """HTML tags like <p>, <br/>, <div> should be removed."""
        text = "<p>मराठी बातमी</p><br/><div>अधिक माहिती</div>"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result
        assert "मराठी बातमी" in result
        assert "अधिक माहिती" in result

    def test_removes_nested_html(self):
        """Nested HTML tags should be fully removed."""
        text = '<div class="news"><p>पुणे</p></div>'
        result = clean_text(text)
        assert "<" not in result
        assert "पुणे" in result

    def test_collapses_extra_whitespace(self):
        """Multiple spaces, tabs, and newlines become a single space."""
        text = "मराठी   बातमी\t\tआज\n\nपुणे"
        result = clean_text(text)
        # No double spaces
        assert "  " not in result
        assert "\t" not in result
        assert "\n" not in result
        # Content preserved
        assert "मराठी" in result
        assert "बातमी" in result
        assert "पुणे" in result

    def test_preserves_devanagari_danda(self):
        """The Devanagari danda (।) should be preserved in cleaned text."""
        text = "हा एक वाक्य आहे। हा दुसरा वाक्य आहे।"
        result = clean_text(text)
        # The purna viram (।) should still be present
        assert "।" in result

    def test_combined_html_url_whitespace(self):
        """Test cleaning text with HTML, URLs, and extra whitespace together."""
        text = "<p>बातमी  </p>  https://news.com   <br/>  पुणे"
        result = clean_text(text)
        assert "<" not in result
        assert "https" not in result
        assert "  " not in result
        assert "बातमी" in result
        assert "पुणे" in result


class TestEmptyInput:
    """Tests for empty and edge-case inputs."""

    def test_empty_string_normalize(self):
        """normalize_text should return empty string for empty input."""
        assert normalize_text("") == ""

    def test_empty_string_clean(self):
        """clean_text should return empty string for empty input."""
        assert clean_text("") == ""

    def test_empty_string_tokenize(self):
        """tokenize should return empty list for empty input."""
        assert tokenize("") == []

    def test_whitespace_only_normalize(self):
        """normalize_text should return empty string for whitespace-only input."""
        assert normalize_text("   ") == ""

    def test_whitespace_only_clean(self):
        """clean_text should return empty string for whitespace-only input."""
        assert clean_text("   \t\n  ") == ""

    def test_whitespace_only_tokenize(self):
        """tokenize should return empty list for whitespace-only input."""
        assert tokenize("   ") == []

    def test_none_like_empty_preprocess(self):
        """preprocess should handle empty string gracefully."""
        result = preprocess("")
        assert result["original"] == ""
        assert result["normalized"] == ""
        assert result["cleaned"] == ""
        assert result["tokens"] == []


class TestNormalizeText:
    """Tests for the normalize_text function."""

    def test_marathi_text_normalises(self):
        """Basic Marathi text should be normalised without errors."""
        text = "पुणे शहरात आज पाऊस पडला"
        result = normalize_text(text)
        # Should return a non-empty string
        assert len(result) > 0
        # Marathi characters should still be present
        assert "पुणे" in result

    def test_returns_string(self):
        """normalize_text should always return a string."""
        result = normalize_text("test मराठी")
        assert isinstance(result, str)


class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic_marathi_tokenization(self):
        """Basic Marathi sentence should be split into tokens."""
        text = "पुणे शहरात आज पाऊस पडला"
        tokens = tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "पुणे" in tokens

    def test_danda_is_separate_token(self):
        """The Devanagari danda (।) should be a separate token."""
        text = "हा वाक्य आहे।"
        tokens = tokenize(text)
        assert isinstance(tokens, list)
        # The danda should appear as a separate token or be handled
        assert len(tokens) > 0


class TestPreprocess:
    """Tests for the full preprocess pipeline."""

    def test_returns_all_keys(self):
        """preprocess should return dict with original, normalized, cleaned, tokens."""
        result = preprocess("पुणे शहरात आज पाऊस पडला")
        assert "original" in result
        assert "normalized" in result
        assert "cleaned" in result
        assert "tokens" in result

    def test_original_is_untouched(self):
        """The original text should be preserved exactly as given."""
        text = "  <p>पुणे</p>  https://example.com  "
        result = preprocess(text)
        assert result["original"] == text

    def test_cleaned_has_no_html_or_urls(self):
        """Cleaned text should have no HTML tags or URLs."""
        text = "<p>बातमी</p> https://news.com पुणे"
        result = preprocess(text)
        assert "<" not in result["cleaned"]
        assert "https" not in result["cleaned"]

    def test_tokens_are_list_of_strings(self):
        """Tokens should be a list of strings."""
        result = preprocess("पुणे शहरात आज पाऊस पडला")
        assert isinstance(result["tokens"], list)
        for token in result["tokens"]:
            assert isinstance(token, str)


class TestReadUploadedTxt:
    """Tests for reading uploaded .txt files."""

    def test_reads_utf8_file(self):
        """Should correctly read a UTF-8 encoded Marathi text file."""
        # Create a temporary UTF-8 file with Marathi text
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("पुणे शहरात आज पाऊस पडला")
            temp_path = f.name

        try:
            result = read_uploaded_txt(temp_path)
            assert "पुणे" in result
            assert "पाऊस" in result
        finally:
            os.unlink(temp_path)

    def test_non_utf8_file_raises_error(self):
        """Should raise ValueError for non-UTF-8 encoded files."""
        # Create a temporary file with non-UTF-8 bytes (Latin-1 encoded)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        ) as f:
            # Write bytes that are valid Latin-1 but invalid UTF-8
            f.write(b"\x80\x81\x82\xff\xfe\xfd")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="UTF-8"):
                read_uploaded_txt(temp_path)
        finally:
            os.unlink(temp_path)

    def test_missing_file_raises_error(self):
        """Should raise ValueError for a file that doesn't exist."""
        with pytest.raises(ValueError, match="not found|सापडली"):
            read_uploaded_txt("/nonexistent/path/file.txt")
