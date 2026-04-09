"""
analyzer.py
-----------
All NLP analysis logic.  No UI code lives here.

Exported functions
------------------
compute_all_scores(text1, text2) -> dict
    Returns the complete scores dictionary used by the Streamlit UI.

find_matching_sentences(text1, text2, top_n) -> list[dict]
    Returns the top N most similar sentence pairs.

get_top_shared_keywords(text1, text2, top_n) -> list[tuple]
    Returns (word, tfidf_doc1, tfidf_doc2) tuples for the most
    meaningful shared terms.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import (
    preprocess,
    split_sentences,
    get_ngrams,
    jaccard_similarity,
)


# ─────────────────────────────────────────────────────────────
#  INTERNAL AXIS SCORERS
# ─────────────────────────────────────────────────────────────

def _score_sentence_overlap(text1: str, text2: str) -> float:
    """
    SENTENCE OVERLAP
    ────────────────
    For every sentence in each document, check whether a near-duplicate
    exists in the other document (TF-IDF cosine ≥ 0.60).  Report the
    average matched fraction across both documents.

    Symmetric: swapping doc1 and doc2 gives the same score.
    """
    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)
    if not sents1 or not sents2:
        return 0.0

    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(
            [preprocess(s) for s in sents1 + sents2]
        )
    except ValueError:
        return 0.0

    n1 = len(sents1)
    sim = cosine_similarity(tfidf[:n1], tfidf[n1:])   # (n1 × n2)

    THRESHOLD = 0.60
    matched1 = sum(1 for i in range(n1)           if sim[i].max()    >= THRESHOLD)
    matched2 = sum(1 for j in range(len(sents2))  if sim[:, j].max() >= THRESHOLD)
    return round(((matched1 / n1) + (matched2 / len(sents2))) / 2 * 100, 1)


def _score_structure_similarity(text1: str, text2: str) -> float:
    """
    STRUCTURE SIMILARITY
    ────────────────────
    Cosine similarity of trigram (3-word phrase) frequency vectors.
    More sensitive to paraphrasing than word-level Jaccard because it
    captures shared phrase patterns even when individual words differ.
    """
    ng1 = get_ngrams(text1, 3)
    ng2 = get_ngrams(text2, 3)
    all_keys = list(set(ng1) | set(ng2))
    if not all_keys:
        return 0.0

    v1 = np.array([ng1.get(k, 0) for k in all_keys], dtype=float)
    v2 = np.array([ng2.get(k, 0) for k in all_keys], dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(float(np.dot(v1, v2) / (norm1 * norm2)) * 100, 1)


def _score_uniqueness(text1: str, text2: str) -> float:
    """
    UNIQUENESS
    ──────────
    Percentage of Document 1's bigrams that do NOT appear in Document 2.
    100 = fully unique content.
    0   = every bigram in doc1 also appears in doc2.
    """
    ng1 = get_ngrams(text1, 2)
    ng2 = get_ngrams(text2, 2)
    if not ng1:
        return 100.0
    shared  = sum(min(ng1[k], ng2.get(k, 0)) for k in ng1)
    total1  = sum(ng1.values())
    overlap = (shared / total1) * 100
    return round(max(0.0, 100.0 - overlap), 1)


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

def compute_all_scores(text1: str, text2: str) -> dict:
    """
    Compute all five similarity scores for the two documents.

    Returns
    -------
    {
        "overall":              float  — TF-IDF (1–3 gram) cosine similarity %
        "lexical_match":        float  — Jaccard word-set similarity %
        "sentence_overlap":     float  — fraction of near-duplicate sentences %
        "structure_similarity": float  — trigram cosine similarity %
        "uniqueness":           float  — bigram uniqueness of doc1 vs doc2 %
    }
    """
    # Overall score uses 1–3 gram TF-IDF to capture multi-word patterns
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
        mat = vec.fit_transform([preprocess(text1), preprocess(text2)])
        overall = float(cosine_similarity(mat[0:1], mat[1:2])[0][0]) * 100
    except ValueError:
        overall = 0.0

    return {
        "overall":              round(overall, 1),
        "lexical_match":        jaccard_similarity(text1, text2),
        "sentence_overlap":     _score_sentence_overlap(text1, text2),
        "structure_similarity": _score_structure_similarity(text1, text2),
        "uniqueness":           _score_uniqueness(text1, text2),
    }


def find_matching_sentences(text1: str, text2: str, top_n: int = 5) -> list:
    """
    Find the top N most similar sentence pairs across both documents.

    Returns a list of dicts:
        [{"score": float, "sent1": str, "sent2": str}, ...]

    Deduplication ensures each sentence appears in at most one pair.
    """
    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)
    if not sents1 or not sents2:
        return []

    try:
        vec   = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(
            [preprocess(s) for s in sents1 + sents2]
        )
    except ValueError:
        return []

    n1  = len(sents1)
    sim = cosine_similarity(tfidf[:n1], tfidf[n1:])

    # Collect all pairs above noise floor
    pairs = [
        {"score": sim[i][j], "sent1": sents1[i], "sent2": sents2[j]}
        for i in range(sim.shape[0])
        for j in range(sim.shape[1])
        if sim[i][j] > 0.10
    ]
    pairs.sort(key=lambda x: x["score"], reverse=True)

    # Greedy deduplication
    seen1, seen2, out = set(), set(), []
    for p in pairs:
        if p["sent1"] not in seen1 and p["sent2"] not in seen2:
            out.append(p)
            seen1.add(p["sent1"])
            seen2.add(p["sent2"])
        if len(out) >= top_n:
            break
    return out


def get_top_shared_keywords(text1: str, text2: str, top_n: int = 10) -> list:
    """
    Return the top N keywords that appear in BOTH documents,
    ranked by their average TF-IDF weight across the two documents.

    Returns a list of (word, score_doc1, score_doc2) tuples.
    """
    p1, p2 = preprocess(text1), preprocess(text2)
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            max_features=5000,
        )
        mat = vec.fit_transform([p1, p2])
    except ValueError:
        return []

    vocab   = vec.get_feature_names_out()
    scores1 = mat[0].toarray()[0]
    scores2 = mat[1].toarray()[0]

    shared = [
        (vocab[i], round(float(scores1[i]), 4), round(float(scores2[i]), 4))
        for i in range(len(vocab))
        if scores1[i] > 0 and scores2[i] > 0
    ]
    shared.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
    return shared[:top_n]


# ─────────────────────────────────────────────────────────────
#  VERDICT LOGIC
# ─────────────────────────────────────────────────────────────

def get_verdict(score: float) -> tuple:
    """
    Map an overall similarity percentage to a (label, css_class) tuple.

    Thresholds
    ----------
    < 20%   → Likely Original   (green)
    20–50%  → Moderate Overlap  (amber)
    ≥ 50%   → High Similarity   (red)
    """
    if score < 20:
        return "✅ Likely Original", "verdict-low"
    if score < 50:
        return "⚠️ Moderate Overlap", "verdict-medium"
    return "🚨 High Similarity — Potential Plagiarism", "verdict-high"


def get_explanation(scores: dict) -> str:
    """
    Build a one-paragraph natural-language explanation of the results
    by combining observations from all four axis scores.
    """
    o    = scores["overall"]
    lm   = scores["lexical_match"]
    so   = scores["sentence_overlap"]
    ss   = scores["structure_similarity"]
    uniq = scores["uniqueness"]

    parts = []
    if o < 20:
        parts.append(
            "These documents appear to be independently written "
            "with minimal shared content."
        )
    elif o < 50:
        parts.append(
            "There is a moderate level of similarity between these documents."
        )
    else:
        parts.append(
            "These documents are highly similar and likely share "
            "substantial copied or closely paraphrased content."
        )

    if so >= 30:
        parts.append(
            f"Sentence-level overlap is notable ({so}%), "
            "pointing to direct passage copying."
        )
    elif ss >= 30:
        parts.append(
            f"Structural phrase patterns match closely ({ss}%), "
            "which can indicate paraphrasing."
        )
    elif lm >= 35:
        parts.append(
            f"Vocabulary overlap is high ({lm}%), "
            "though this may be explained by shared subject matter."
        )

    if uniq > 70:
        parts.append("Most of Document 1's content is still unique to itself.")
    elif uniq < 40:
        parts.append(
            "Very little of Document 1 reads as original "
            "when compared to Document 2."
        )

    return " ".join(parts)


def get_axis_explanation(key: str, score: float) -> str:
    """Return a one-sentence plain-English explanation for a single axis."""
    explanations = {
        "lexical_match": [
            (15,  "The two documents use very different vocabulary with few words in common."),
            (35,  "Some vocabulary is shared, likely due to common topic terminology."),
            (101, "A large portion of the vocabulary is identical — a strong signal."),
        ],
        "sentence_overlap": [
            (10,  "Virtually no sentences match — documents appear independently written."),
            (30,  "A small number of near-duplicate sentences exist. May be coincidental."),
            (101, "Many sentences appear in both documents — strong indicator of direct copying."),
        ],
        "structure_similarity": [
            (10,  "Phrase patterns are distinct, indicating different writing styles."),
            (30,  "Moderate structural overlap — some shared phrases or sentence patterns."),
            (101, "The documents share many 3-word phrases, suggesting paraphrasing or copying."),
        ],
        "uniqueness": [
            # Note: uniqueness is inverted — HIGH score = LOW risk
            (40,  "Very little of Document 1 is unique — most content also appears in Document 2."),
            (80,  "Document 1 has moderate uniqueness; a notable portion echoes Document 2."),
            (101, "Document 1 is largely unique — most content is not found in Document 2."),
        ],
    }
    for threshold, text in explanations.get(key, []):
        if score < threshold:
            return text
    return ""
