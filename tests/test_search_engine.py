"""Tests for page-wise TF-IDF and cosine-similarity search."""

import unittest

from src.search_engine import (
    EmptyQueryError,
    NoMatchingTermsError,
    PDFSearchEngine,
    PDFSearchError,
)


class PDFSearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pages = [
            {
                "page_number": 1,
                "text": "Cloud computing provides scalable infrastructure.",
            },
            {"page_number": 2, "text": "   \n  "},
            {
                "page_number": 3,
                "text": "Cloud security protects cloud infrastructure.",
            },
            {
                "page_number": 4,
                "text": "Machine learning uses algorithms to learn patterns.",
            },
        ]
        self.engine = PDFSearchEngine(self.pages)

    def test_search_finds_most_relevant_original_page(self) -> None:
        result = self.engine.search("cloud security")

        self.assertEqual(result["page_number"], 3)
        self.assertGreater(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertIn("Cloud security", result["text"])

    def test_blank_pages_are_skipped_without_renumbering(self) -> None:
        results = self.engine.search_top_k("cloud", top_k=3)

        self.assertEqual([result["page_number"] for result in results], [3, 1, 4])
        self.assertNotIn(2, [result["page_number"] for result in results])

    def test_top_k_results_are_ranked_by_score(self) -> None:
        results = self.engine.search_top_k("cloud security", top_k=3)
        scores = [result["score"] for result in results]

        self.assertEqual(len(results), 3)
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_and_whitespace_queries_raise_clear_errors(self) -> None:
        for query in ("", "   \n"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(EmptyQueryError, "Query cannot be empty"):
                    self.engine.search(query)

    def test_unknown_terms_raise_a_clear_error(self) -> None:
        with self.assertRaisesRegex(
            NoMatchingTermsError,
            "No meaningful matching terms",
        ):
            self.engine.search("quasarxyz")

    def test_single_page_input_can_be_searched(self) -> None:
        engine = PDFSearchEngine(
            [{"page_number": 9, "text": "Ocean ecosystems contain diverse life."}]
        )

        result = engine.search("ocean ecosystems")

        self.assertEqual(result["page_number"], 9)
        self.assertGreater(result["score"], 0.0)

    def test_pages_without_usable_terms_are_rejected(self) -> None:
        with self.assertRaises(PDFSearchError):
            PDFSearchEngine([{"page_number": 1, "text": "the and is"}])


if __name__ == "__main__":
    unittest.main()
