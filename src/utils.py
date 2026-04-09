"""
utils.py
--------
Text extraction (PDF / TXT) and preprocessing utilities.
All operations are pure Python + pdfplumber — no network calls.
"""

import re
from collections import Counter

import pdfplumber
import streamlit as st


# ─────────────────────────────────────────────────────────────
#  TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from a PDF file page by page using pdfplumber.
    Returns an empty string and shows an error if extraction fails.
    """
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")
    return text.strip()


def extract_text_from_txt(uploaded_file) -> str:
    """
    Decode a plain-text file as UTF-8.
    Silently replaces undecodable bytes rather than crashing.
    """
    try:
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        st.error(f"Could not read text file: {exc}")
        return ""


def extract_text(uploaded_file) -> str:
    """
    Route an uploaded Streamlit file to the correct extractor
    based on its extension (.pdf or .txt).
    """
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    st.warning(f"Unsupported file type: {uploaded_file.name}")
    return ""


# ─────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Normalise text for vectorization:
      1. Lowercase
      2. Remove everything that isn't a letter, digit, or space
      3. Collapse runs of whitespace to a single space
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list:
    """
    Split text into sentences using a lookbehind on sentence-ending
    punctuation.  Only sentences with at least 5 words are kept to
    filter out headings, numbering artefacts, etc.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.split()) >= 5]


# ─────────────────────────────────────────────────────────────
#  N-GRAM UTILITIES
# ─────────────────────────────────────────────────────────────

def get_ngrams(text: str, n: int) -> Counter:
    """
    Return a Counter of n-gram tuples from preprocessed text.
    Example: get_ngrams("the cat sat", 2)
             → Counter({('the', 'cat'): 1, ('cat', 'sat'): 1})
    """
    words = preprocess(text).split()
    return Counter(zip(*[words[i:] for i in range(n)]))


# ─────────────────────────────────────────────────────────────
#  JACCARD SIMILARITY
# ─────────────────────────────────────────────────────────────

def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Jaccard similarity between the unique word sets of two documents.
    Formula: |A ∩ B| / |A ∪ B|
    Returns a percentage (0–100).
    """
    words1 = set(preprocess(text1).split())
    words2 = set(preprocess(text2).split())
    if not words1 or not words2:
        return 0.0
    return round(len(words1 & words2) / len(words1 | words2) * 100, 1)
