"""
app.py — Mini Plagiarism Checker
=================================
Entry point.  All analysis logic lives in src/analyzer.py,
all chart logic lives in src/visuals.py, and all utilities
live in src/utils.py.  This file is intentionally thin.
"""

import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils     import extract_text, split_sentences, preprocess
from src.analyzer  import (
    compute_all_scores,
    find_matching_sentences,
    get_top_shared_keywords,
    get_verdict,
    get_explanation,
    get_axis_explanation,
)
from src.visuals   import (
    build_radar_chart,
    build_keywords_bar_chart,
    build_word_length_chart,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mini Plagiarism Checker",
    page_icon="📄",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Academic IBM Plex theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600&family=IBM+Plex+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Serif', Georgia, serif; }
  .main { background-color: #fafaf8; }

  /* ── Title ── */
  .title-block {
    text-align: center; padding: 1.6rem 0 1rem;
    border-bottom: 2px solid #1a1a1a; margin-bottom: 1.5rem;
  }
  .title-block h1 { font-size: 2.1rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.2rem; }
  .title-block p  { color: #666; font-size: 0.9rem; }

  /* ── Score card ── */
  .score-card {
    background: #1a1a1a; color: #fff;
    border-radius: 8px; padding: 1.8rem;
    text-align: center; margin: 0.5rem 0 1rem;
  }
  .score-number {
    font-size: 3.6rem; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
  }
  .score-label  { font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; color: #aaa; margin-top: 0.3rem; }
  .score-verdict { font-size: 0.95rem; margin-top: 0.9rem; padding: 0.35rem 1rem; border-radius: 20px; display: inline-block; }
  .verdict-low    { background: #d4edda; color: #155724; }
  .verdict-medium { background: #fff3cd; color: #856404; }
  .verdict-high   { background: #f8d7da; color: #721c24; }

  /* ── Axis pills 2×2 ── */
  .axis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin: 1rem 0; }
  .axis-pill { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.7rem 1rem; }
  .axis-name  { font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; color: #888; }
  .axis-score { font-size: 1.6rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; color: #1a1a1a; }
  .axis-bar-bg   { background: #eee; border-radius: 4px; height: 5px; margin-top: 0.4rem; }
  .axis-bar-fill { height: 5px; border-radius: 4px; }

  /* ── Explanation box ── */
  .explain-box {
    background: #f5f5f2; border-left: 3px solid #1a1a1a;
    border-radius: 0 4px 4px 0; padding: 0.9rem 1.1rem;
    font-size: 0.88rem; color: #333; line-height: 1.6; margin: 0.6rem 0;
  }
  .explain-title { font-weight: 600; font-size: 0.78rem; letter-spacing: 1px;
    text-transform: uppercase; color: #555; margin-bottom: 0.4rem; }

  /* ── Match cards ── */
  .match-card {
    background: #fff; border: 1px solid #e0e0e0;
    border-left: 4px solid #1a1a1a; border-radius: 4px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
    font-size: 0.9rem; line-height: 1.6; color: #333;
  }
  .match-score-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: #888; margin-bottom: 0.3rem; }

  /* ── Section header ── */
  .section-header {
    font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase;
    color: #888; border-bottom: 1px solid #ddd;
    padding-bottom: 0.35rem; margin: 1.4rem 0 0.9rem;
  }

  /* ── Uploads ── */
  [data-testid="stFileUploader"] { border: 2px dashed #ccc !important; border-radius: 6px; padding: 0.4rem; }

  /* ── Button ── */
  .stButton > button {
    background: #1a1a1a !important; color: white !important;
    border: none !important; border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'IBM Plex Serif', serif !important;
    font-size: 1rem !important; width: 100%; letter-spacing: 0.5px;
  }
  .stButton > button:hover { background: #444 !important; }

  /* ── Info box ── */
  .info-box { background: #f0f0f0; border-radius: 4px; padding: 0.8rem 1rem;
    font-size: 0.88rem; color: #444; margin-top: 1rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] { background: #f5f5f2; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — About / How It Works
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 About")
    st.markdown(
        "A **fully offline** plagiarism detector using classical NLP.\n\n"
        "No internet · No API · No deep learning."
    )
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. **Extract** text from PDF or TXT  \n"
        "2. **Preprocess** (lowercase, strip special chars)  \n"
        "3. **TF-IDF (1–3 gram)** vectorization  \n"
        "4. **Cosine similarity** → overall score  \n"
        "5. **Jaccard / n-gram** axes → radar chart  \n"
        "6. **Sentence-level** pairwise matching"
    )
    st.divider()
    st.markdown("### Thresholds")
    st.markdown("🟢 **< 20%** — Likely Original  \n"
                "🟡 **20–50%** — Moderate Overlap  \n"
                "🔴 **≥ 50%** — High Similarity")
    st.divider()


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>📄 Mini Plagiarism Checker</h1>
  <p>Offline · Classical NLP · TF-IDF (1–3 gram) + Cosine Similarity · 4-Axis Radar</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-header">Document 1</div>', unsafe_allow_html=True)
    file1 = st.file_uploader("Upload File 1", type=["pdf", "txt"],
                              label_visibility="collapsed", key="file1")
with c2:
    st.markdown('<div class="section-header">Document 2</div>', unsafe_allow_html=True)
    file2 = st.file_uploader("Upload File 2", type=["pdf", "txt"],
                              label_visibility="collapsed", key="file2")

st.markdown("<br>", unsafe_allow_html=True)
check_clicked = st.button("🔍 Check Similarity")


# ─────────────────────────────────────────────
# HELPER — progress bar colour
# ─────────────────────────────────────────────
def _bar_color(score: float, invert: bool = False) -> str:
    risk = score if not invert else (100 - score)
    if risk < 25:   return "#2ecc71"
    if risk < 55:   return "#f39c12"
    return "#e74c3c"


# ═════════════════════════════════════════════
# ANALYSIS
# ═════════════════════════════════════════════
if check_clicked:

    if not file1 or not file2:
        st.error("⚠️ Please upload both documents before checking.")
        st.stop()

    with st.spinner("Extracting text…"):
        text1 = extract_text(file1)
        text2 = extract_text(file2)

    if not text1:
        st.error("Document 1 is empty or unreadable."); st.stop()
    if not text2:
        st.error("Document 2 is empty or unreadable."); st.stop()

    with st.spinner("Running NLP analysis…"):
        scores   = compute_all_scores(text1, text2)
        matches  = find_matching_sentences(text1, text2, top_n=5)
        keywords = get_top_shared_keywords(text1, text2, top_n=10)

    verdict_text, verdict_class = get_verdict(scores["overall"])

    # ── Three tabs ────────────────────────────
    tab_dash, tab_sent, tab_stats = st.tabs([
        "📊  Dashboard",
        "🔍  Sentence Analysis",
        "📈  Document Statistics",
    ])

    # ══════════════════════════════════════════
    #  TAB 1 — DASHBOARD
    # ══════════════════════════════════════════
    with tab_dash:

        # Row A — Score card | Radar
        left, right = st.columns([1, 1.55], gap="large")

        with left:
            st.markdown(f"""
            <div class="score-card">
              <div class="score-number">{scores['overall']}%</div>
              <div class="score-label">Overall Similarity</div>
              <div class="score-verdict {verdict_class}">{verdict_text}</div>
            </div>""", unsafe_allow_html=True)

            axis_info = [
                ("lexical_match",        "Lexical Match",        scores["lexical_match"],        False),
                ("sentence_overlap",     "Sentence Overlap",     scores["sentence_overlap"],     False),
                ("structure_similarity", "Structure Similarity", scores["structure_similarity"], False),
                ("uniqueness",           "Uniqueness",           scores["uniqueness"],           True),
            ]
            pills = '<div class="axis-grid">'
            for key, label, val, inv in axis_info:
                color = _bar_color(val, invert=inv)
                pills += f"""
                <div class="axis-pill">
                  <div class="axis-name">{label}</div>
                  <div class="axis-score">{val}%</div>
                  <div class="axis-bar-bg">
                    <div class="axis-bar-fill" style="width:{val}%;background:{color}"></div>
                  </div>
                </div>"""
            pills += "</div>"
            st.markdown(pills, unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-header">Similarity Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(
                build_radar_chart(scores),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # Row B — st.metric strip
        st.markdown('<div class="section-header">Quick Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Overall Similarity",   f"{scores['overall']}%")
        m2.metric("Lexical Match",         f"{scores['lexical_match']}%")
        m3.metric("Sentence Overlap",      f"{scores['sentence_overlap']}%")
        m4.metric("Structure Similarity",  f"{scores['structure_similarity']}%")
        m5.metric("Uniqueness",            f"{scores['uniqueness']}%")

        # Row C — Explanations
        st.markdown('<div class="section-header">What the Scores Mean</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explain-box">
          <div class="explain-title">📋 Overall Summary</div>
          {get_explanation(scores)}
        </div>""", unsafe_allow_html=True)

        el, er = st.columns(2)
        axis_labels = [
            ("lexical_match",        "🔤 Lexical Match"),
            ("sentence_overlap",     "📝 Sentence Overlap"),
            ("structure_similarity", "🧱 Structure Similarity"),
            ("uniqueness",           "✨ Uniqueness"),
        ]
        for idx, (key, label) in enumerate(axis_labels):
            col = el if idx % 2 == 0 else er
            with col:
                st.markdown(f"""
                <div class="explain-box">
                  <div class="explain-title">{label} — {scores[key]}%</div>
                  {get_axis_explanation(key, scores[key])}
                </div>""", unsafe_allow_html=True)

        # Row D — Top 10 Keywords bar chart
        st.markdown('<div class="section-header">Top 10 Shared Keywords</div>', unsafe_allow_html=True)
        fig_bar = build_keywords_bar_chart(keywords)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "TF-IDF weight of each shared keyword in each document. "
                "Higher weight = more distinctive in that document. Stop words excluded."
            )
        else:
            st.info("No significant shared keywords found.")

    # ══════════════════════════════════════════
    #  TAB 2 — SENTENCE ANALYSIS
    # ══════════════════════════════════════════
    with tab_sent:
        st.markdown('<div class="section-header">Top Matching Sentence Pairs</div>', unsafe_allow_html=True)
        st.caption(
            "Sentences (≥ 5 words) are extracted from both documents and compared "
            "with TF-IDF cosine similarity.  Top 5 unique pairs shown."
        )
        if matches:
            for i, m in enumerate(matches, 1):
                pct = round(m["score"] * 100, 1)
                st.markdown(f"""
                <div class="match-card">
                  <div class="match-score-tag">Match #{i} — {pct}% similar</div>
                  <strong>Doc 1:</strong> {m['sent1']}<br><br>
                  <strong>Doc 2:</strong> {m['sent2']}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No significantly matching sentence pairs were found.")

    # ══════════════════════════════════════════
    #  TAB 3 — DOCUMENT STATISTICS
    # ══════════════════════════════════════════
    with tab_stats:
        sents1 = split_sentences(text1)
        sents2 = split_sentences(text2)
        words1 = text1.split()
        words2 = text2.split()

        st.markdown('<div class="section-header">Document Statistics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Doc 1 — Words",     f"{len(words1):,}")
        c2.metric("Doc 1 — Sentences", f"{len(sents1):,}")
        c3.metric("Doc 1 — Chars",     f"{len(text1):,}")
        c4.metric("Doc 2 — Words",     f"{len(words2):,}")
        c5.metric("Doc 2 — Sentences", f"{len(sents2):,}")
        c6.metric("Doc 2 — Chars",     f"{len(text2):,}")

        st.markdown('<div class="section-header">Vocabulary Comparison</div>', unsafe_allow_html=True)
        vocab1       = set(preprocess(text1).split()) - set(ENGLISH_STOP_WORDS)
        vocab2       = set(preprocess(text2).split()) - set(ENGLISH_STOP_WORDS)
        shared_vocab = vocab1 & vocab2
        only_in_1    = vocab1 - vocab2

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Unique Words — Doc 1", f"{len(vocab1):,}")
        v2.metric("Unique Words — Doc 2", f"{len(vocab2):,}")
        v3.metric("Shared Vocabulary",     f"{len(shared_vocab):,}")
        v4.metric("Exclusive to Doc 1",    f"{len(only_in_1):,}")

        st.markdown('<div class="section-header">Word Length Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(
            build_word_length_chart(words1, words2),
            use_container_width=True,
            config={"displayModeBar": False},
        )

else:
    st.markdown("""
    <div class="info-box">
      👆 Upload two documents (PDF or TXT) and click <strong>Check Similarity</strong>.<br>
      Results are organised across three tabs:
      <strong>Dashboard</strong>, <strong>Sentence Analysis</strong>, and <strong>Document Statistics</strong>.
    </div>""", unsafe_allow_html=True)
