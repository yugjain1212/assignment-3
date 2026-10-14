# PDF Retrieval-Augmented Generation (RAG)

A command-line RAG application that answers questions about a selectable-text PDF. It extracts PDF pages, retrieves the most relevant pages with TF-IDF and cosine similarity, and sends the question together with those pages to NVIDIA's OpenAI-compatible chat API. The CLI prints both the augmented prompt and the generated answer.

Provide your own PDF when running the command. Local PDFs under `data/` are ignored by Git by default; only publish documents you have permission to redistribute.

## How it works

```text
PDF
 ↓
Extract text from each page
 ↓
Clean text and build a TF-IDF index
 ↓
Rank pages for the question with cosine similarity
 ↓
Use the top-k pages as retrieved context
 ↓
Build an augmented prompt (question + context)
 ↓
Send the prompt to NVIDIA NIM
 ↓
Print the augmented prompt and answer
```

Each non-empty PDF page is one retrieval chunk. The existing TF-IDF index is fitted once and reused for each query. Similarity scores are used internally to rank context; they are not displayed as the final answer.

The prompt instructs the model to answer only from retrieved context, say when the context is insufficient, and cite the original PDF page numbers. Retrieved context is still model input, so the model may make mistakes; verify important answers against the cited PDF pages.

## Requirements

- Python 3.10+
- PyMuPDF for PDF text extraction
- scikit-learn and NumPy for TF-IDF retrieval
- OpenAI Python SDK (used with NVIDIA's compatible endpoint)
- An NVIDIA API key

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then put your NVIDIA key in `.env`. `.env` is ignored by Git. The settings look like this:

```dotenv
NVIDIA_API_KEY=your-nvidia-api-key
BASE_URL=https://integrate.api.nvidia.com/v1
MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

`BASE_URL` and `MODEL` default to these values; `--model` overrides `MODEL`. Shell environment variables take precedence over values in `.env`. `OPENAI_API_KEY` and `API_KEY` are also accepted as key variable names, but `NVIDIA_API_KEY` is recommended.

Do not put the API key in source code or commit `.env` to the repository. The question and retrieved PDF text are sent to NVIDIA for generation; do not use this tool with documents you are not allowed to share with that service. API usage may incur charges.

## Usage

Ask a question about a PDF you are allowed to use:

```bash
python search.py path/to/your-document.pdf "How does backpropagation work?"
```

By default, the three most relevant pages are included as context. Change the number of retrieved pages with `--top-k` and the model with `--model`:

```bash
python search.py path/to/your-document.pdf "What is a neural network?" --top-k 5 --model nvidia/nemotron-3.5-lightning-30b-a3b
```

The CLI validates that the query is non-empty and that `--top-k` is at least 1. It reports a clear error if the PDF cannot be read, the query has no matching terms, the API key is unavailable, or the model request fails.

## Example output shape

The prompt content and answer vary with the question, retrieved pages, and model response. Each run prints:

```text
========================================
          PDF RAG ASSISTANT
========================================

PDF:
path/to/your-document.pdf

Query:
How does backpropagation work?

Augmented Prompt
----------------------------------------
Answer the question using only the context below...

Context:
[Page 42]
...retrieved PDF text...

Question:
How does backpropagation work?
----------------------------------------

Generated Response
----------------------------------------
...answer grounded in the retrieved context...
----------------------------------------
```

## Project structure

```text
.
├── data/                   # local PDFs; ignored by Git
├── .env.example            # safe environment-variable template
├── src/
│   ├── __init__.py
│   ├── pdf_reader.py       # page-wise PDF text extraction
│   ├── text_processor.py   # whitespace normalization
│   ├── search_engine.py    # TF-IDF retrieval and page ranking
│   └── llm.py              # augmented prompt and NVIDIA chat generation
├── tests/
│   ├── test_pdf_reader.py
│   ├── test_text_processor.py
│   ├── test_search_engine.py
│   └── test_llm.py
├── search.py               # command-line RAG application
└── requirements.txt
```

## Tests

Run the unit suite without making an API call:

```bash
python -m unittest discover -v
```

## Limitations

- TF-IDF retrieves based on shared words, so it may miss semantically related text that uses different wording.
- A PDF page is the retrieval unit; a relevant passage can be buried in a long page.
- Scanned or image-only PDFs need OCR, which is not included.
- PDF text extraction may not preserve tables, columns, or reading order.
- The answer can only be as complete as the retrieved context and may still be inaccurate; check its cited pages.
- NVIDIA's API requires network access and receives the question and retrieved text.
