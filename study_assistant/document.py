"""Load one short PDF or text document and return clean text."""
from __future__ import annotations

import io
import re
from pathlib import Path

MAX_WORDS = 6000  # "short document" guard-rail; keeps prompts small and cheap
MIN_WORDS = 80


class DocumentError(Exception):
    pass


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newlines -> spaces
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_document(source, filename: str | None = None) -> str:
    """source: a file path (str/Path) or raw bytes (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")
        filename = filename or path.name
        data = path.read_bytes()
    else:
        data = bytes(source)
        filename = filename or "upload.txt"

    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentError("pypdf is required for PDFs: pip install pypdf") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as exc:
            raise DocumentError(f"Could not read PDF: {exc}") from exc
    elif ext in {".txt", ".md", ""}:
        text = data.decode("utf-8", errors="ignore")
    else:
        raise DocumentError(f"Unsupported file type '{ext}'. Use .pdf or .txt")

    text = _clean(text)
    n = len(text.split())
    if n < MIN_WORDS:
        raise DocumentError(
            f"Document has only {n} words (minimum {MIN_WORDS}). "
            "If it is a scanned PDF, it needs OCR first."
        )
    if n > MAX_WORDS:
        raise DocumentError(f"Document has {n} words; please upload a shorter one (max {MAX_WORDS}).")
    return text
