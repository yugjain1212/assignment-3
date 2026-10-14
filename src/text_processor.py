"""Simple, predictable text cleanup for extracted PDF page text."""


def clean_text(text: str) -> str:
    """Collapse whitespace and trim the text without changing its wording.

    ``str.split`` treats spaces, tabs, and line breaks as separators, so
    joining its result with one space normalizes all of them consistently.
    """
    return " ".join(text.split())
