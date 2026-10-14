"""Tests for PDF text extraction and page numbering."""

import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.pdf_reader import (
    NoExtractableTextError,
    PDFReaderError,
    extract_pages,
)


class ExtractPagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temporary_directory.name) / "sample.pdf"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_extracts_text_and_preserves_blank_page_number(self) -> None:
        document = pymupdf.open()
        first_page = document.new_page()
        first_page.insert_text((72, 72), "Text on the first page")
        document.new_page()
        third_page = document.new_page()
        third_page.insert_text((72, 72), "Text on the third page")
        document.save(self.pdf_path)
        document.close()

        pages = extract_pages(self.pdf_path)

        self.assertEqual(len(pages), 3)
        self.assertEqual([page["page_number"] for page in pages], [1, 2, 3])
        self.assertIn("Text on the first page", pages[0]["text"])
        self.assertEqual(pages[1]["text"], "")
        self.assertIn("Text on the third page", pages[2]["text"])

    def test_missing_file_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "PDF file not found"):
            extract_pages(self.pdf_path)

    def test_invalid_pdf_has_a_clear_error(self) -> None:
        self.pdf_path.write_text("This is not a PDF", encoding="utf-8")

        with self.assertRaises(PDFReaderError):
            extract_pages(self.pdf_path)

    def test_pdf_without_text_has_a_clear_error(self) -> None:
        document = pymupdf.open()
        document.new_page()
        document.save(self.pdf_path)
        document.close()

        with self.assertRaises(NoExtractableTextError):
            extract_pages(self.pdf_path)


if __name__ == "__main__":
    unittest.main()
