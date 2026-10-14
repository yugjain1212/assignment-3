"""Tests for simple whitespace normalization."""

import unittest

from src.text_processor import clean_text


class CleanTextTests(unittest.TestCase):
    def test_collapses_line_breaks_and_repeated_spaces(self) -> None:
        text = "Cloud computing\n\nprovides     on-demand\nresources."

        self.assertEqual(
            clean_text(text),
            "Cloud computing provides on-demand resources.",
        )

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        self.assertEqual(clean_text("  hello world \n"), "hello world")

    def test_empty_or_whitespace_text_becomes_empty(self) -> None:
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(" \n\t "), "")


if __name__ == "__main__":
    unittest.main()
