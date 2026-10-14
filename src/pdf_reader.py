"""Extract selectable text from each page of a PDF."""

from pathlib import Path
from typing import TypedDict, cast

import pymupdf


class ExtractedPage(TypedDict):
    """Text and original, one-based page number from a PDF."""

    page_number: int
    text: str


class PDFReaderError(RuntimeError):
    """Raised when a PDF cannot be read or processed."""


class NoExtractableTextError(PDFReaderError):
    """Raised when a PDF has no selectable text on any page."""


def extract_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extract text from every page, preserving its one-based PDF number.

    Blank pages are included in the returned list so callers can decide how
    to handle them without losing their original page numbering.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
        raise PDFReaderError(f"Could not open PDF '{path}': {error}") from error

    with document:
        if document.is_encrypted:
            raise PDFReaderError(
                f"PDF '{path}' is password-protected and cannot be read."
            )

        pages: list[ExtractedPage] = []
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                text = cast(str, page.get_text("text"))
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )
        except (RuntimeError, ValueError) as error:
            raise PDFReaderError(
                f"Could not extract text from PDF '{path}': {error}"
            ) from error

    if not any(page["text"].strip() for page in pages):
        raise NoExtractableTextError(
            f"PDF '{path}' contains no extractable selectable text."
        )

    return pages
