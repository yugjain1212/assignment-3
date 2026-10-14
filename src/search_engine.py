"""Page-wise TF-IDF search with cosine-similarity ranking."""

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.pdf_reader import ExtractedPage
from src.text_processor import clean_text


class SearchResult(TypedDict):
    """The best-matching page and its cosine-similarity score."""

    page_number: int
    score: float
    text: str


class PDFSearchError(ValueError):
    """Raised when the pages cannot be searched."""


class EmptyQueryError(PDFSearchError):
    """Raised when a query is empty or contains only whitespace."""


class NoMatchingTermsError(PDFSearchError):
    """Raised when the query has no terms in the fitted TF-IDF vocabulary."""


class PDFSearchEngine:
    """Fit one TF-IDF index over PDF pages and reuse it for each query."""

    def __init__(self, pages: Sequence[Mapping[str, object]]) -> None:
        """Clean and index non-empty pages, keeping their original numbers."""
        self._pages: list[ExtractedPage] = []
        for page in pages:
            page_number = page.get("page_number")
            raw_text = page.get("text")
            if not isinstance(page_number, int) or not isinstance(raw_text, str):
                raise PDFSearchError(
                    "Each page must have an integer page_number and string text."
                )

            text = clean_text(raw_text)
            if text:
                self._pages.append({"page_number": page_number, "text": text})

        if not self._pages:
            raise PDFSearchError("No pages with extractable text are available.")

        self._vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self._page_vectors = self._vectorizer.fit_transform(
                [page["text"] for page in self._pages]
            )
        except ValueError as error:
            raise PDFSearchError(
                "The PDF pages contain no usable terms for TF-IDF search."
            ) from error

    def search(self, query: str) -> SearchResult:
        """Return the single page with the highest cosine similarity."""
        return self.search_top_k(query, top_k=1)[0]

    def search_top_k(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to ``top_k`` pages, ordered by descending similarity."""
        cleaned_query = clean_text(query)
        if not cleaned_query:
            raise EmptyQueryError("Query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self._vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self._page_vectors).ravel()
        if not np.any(scores):
            raise NoMatchingTermsError(
                "No meaningful matching terms were found in the PDF for this query."
            )
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            {
                "page_number": self._pages[index]["page_number"],
                "score": float(scores[index]),
                "text": self._pages[index]["text"],
            }
            for index in ranked_indices
        ]
