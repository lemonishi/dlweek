"""Utilities for processing uploaded documents and generating skill JSON."""

import os
import re
from pathlib import Path
from typing import Tuple

import pytesseract
from PIL import Image
from pdfminer.high_level import extract_text as extract_pdf_text
import docx


def extract_text_from_doc(file_path: Path) -> str:
    """Extract text from a variety of document formats."""
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        # try PDF text extraction; if empty, fall back to OCR
        text = extract_pdf_text(str(file_path))
        if text.strip():
            return text
        # convert pages to images and OCR
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise RuntimeError("Install pdf2image to OCR scanned PDFs")
        images = convert_from_path(str(file_path))
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    elif suffix in {".docx", ".doc"}:
        doc = docx.Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")


def detect_handwriting(text: str) -> bool:
    """Naively detect handwriting by looking for /\n characters or gibberish patterns.

    A real solution would OCR and inspect font, but as a heuristic we simply
    look for sequences of characters that OCR tends to produce. This stub can be
    replaced later with a dedicated classifier.
    """
    # if text contains many lines shorter than 3 characters or many non-alphanumeric
    # lines -> assume handwriting
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l) < 4)
    gib = sum(1 for l in lines if re.fullmatch(r"[^\w\s]+", l))
    return short > len(lines) * 0.3 or gib > 2


def classify_document(text: str, user_tag: str = None) -> str:
    """Return a document type: 'lecture' or 'homework'.

    - If user_tag provided, use it (must contain word lecture/homework).
    - Otherwise, look for handwriting patterns or keywords.
    """
    if user_tag:
        tag = user_tag.lower()
        if "lecture" in tag:
            return "lecture"
        if "homework" in tag or "assignment" in tag:
            return "homework"
    # fallback to handwriting detection
    if detect_handwriting(text):
        return "homework"
    # look for keywords
    if re.search(r"lecture|slide|ppt|notes", text, re.I):
        return "lecture"
    return "lecture"
