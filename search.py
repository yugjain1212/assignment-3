"""Command-line interface for PDF retrieval-augmented generation."""

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

from src.llm import LLMError, generate_answer
from src.pdf_reader import PDFReaderError, extract_pages
from src.search_engine import PDFSearchEngine, PDFSearchError

try:
    import_module("dotenv").load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DEFAULT_MODEL = os.getenv("MODEL") or "nvidia/nemotron-3.5-lightning-30b-a3b"


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve relevant PDF pages with TF-IDF and use an LLM to "
            "generate a context-grounded answer."
        )
    )
    parser.add_argument("pdf_path", help="path to a selectable-text PDF")
    parser.add_argument("query", help="question to ask about the PDF")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="N",
        help="number of relevant pages to provide as context (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"NVIDIA model to use (default: {DEFAULT_MODEL})",
    )
    return parser


def _print_results(pdf_path: str, query: str, prompt: str, answer: str) -> None:
    print("========================================")
    print("          PDF RAG ASSISTANT")
    print("========================================")
    print(f"\nPDF:\n{pdf_path}")
    print(f"\nQuery:\n{query}")
    print("\nAugmented Prompt")
    print("----------------------------------------")
    print(prompt)
    print("----------------------------------------")
    print("\nGenerated Response")
    print("----------------------------------------")
    print(answer)
    print("----------------------------------------")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the search, and return a process exit code."""
    parser = _create_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        return 2
    if not arguments.query.strip():
        print("Error: Query cannot be empty.", file=sys.stderr)
        return 1

    try:
        pages = extract_pages(arguments.pdf_path)
        search_engine = PDFSearchEngine(pages)
        chunks = search_engine.search_top_k(arguments.query, arguments.top_k)
        prompt, answer = generate_answer(arguments.query, chunks, arguments.model)
    except (FileNotFoundError, PDFReaderError, PDFSearchError, LLMError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    _print_results(arguments.pdf_path, arguments.query, prompt, answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
