"""
visuals.py
----------
All Plotly figure builders.  No Streamlit or analysis logic lives here.
Each function returns a go.Figure ready to be passed to st.plotly_chart().

Exported functions
------------------
build_radar_chart(scores)           -> go.Figure
build_keywords_bar_chart(keywords)  -> go.Figure | None
build_word_length_chart(words1, words2) -> go.Figure
"""

from collections import Counter

import plotly.graph_objects as go

# Shared typography constants
_SERIF  = "IBM Plex Serif, Georgia, serif"
_MONO   = "IBM Plex Mono, Courier New, monospace"
_BG     = "#fafaf8"
_DARK   = "#1a1a1a"
_TEAL   = "#7eb8a4"
_MUTED  = "#888888"
_GRID   = "#e0e0e0"


def build_radar_chart(scores: dict) -> go.Figure:
    """
    4-axis radar chart for the plagiarism analysis.

    All spokes point outward = more risk.
    The Uniqueness axis is inverted (plotted as 100 − value) so that a
    large filled polygon consistently signals high overall risk.

    Parameters
    ----------
    scores : dict
        Must contain keys: lexical_match, sentence_overlap,
        structure_similarity, uniqueness.

    Returns
    -------
    go.Figure
    """
    labels = [
        "Lexical Match",
        "Sentence Overlap",
        "Structure Similarity",
        "Lack of Uniqueness",
    ]
    values = [
        scores["lexical_match"],
        scores["sentence_overlap"],
        scores["structure_similarity"],
        100 - scores["uniqueness"],      # invert: high spoke = high risk
    ]

    # Close the polygon
    labels_loop = labels + [labels[0]]
    values_loop = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_loop,
        theta=labels_loop,
        fill="toself",
        fillcolor="rgba(26,26,26,0.10)",
        line=dict(color=_DARK, width=2.5),
        name="Risk Profile",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, family=_MONO, color=_MUTED),
                gridcolor=_GRID,
                linecolor=_GRID,
                ticksuffix="%",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family=_SERIF, color="#333"),
                linecolor=_GRID,
                gridcolor=_GRID,
            ),
        ),
        showlegend=False,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(t=30, b=30, l=55, r=55),
        height=390,
    )
    return fig


def build_keywords_bar_chart(keywords: list) -> go.Figure | None:
    """
    Horizontal grouped bar chart comparing the TF-IDF weight of the top
    shared keywords in each document.

    Parameters
    ----------
    keywords : list of (word, score_doc1, score_doc2) tuples

    Returns
    -------
    go.Figure or None if keywords is empty.
    """
    if not keywords:
        return None

    # Reverse so the highest-ranked word appears at the top
    words   = [k[0] for k in keywords][::-1]
    scores1 = [k[1] for k in keywords][::-1]
    scores2 = [k[2] for k in keywords][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Document 1",
        y=words,
        x=scores1,
        orientation="h",
        marker_color=_DARK,
        hovertemplate="%{y}: %{x:.4f}<extra>Doc 1</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Document 2",
        y=words,
        x=scores2,
        orientation="h",
        marker_color=_TEAL,
        hovertemplate="%{y}: %{x:.4f}<extra>Doc 2</extra>",
    ))
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Top 10 Shared Keywords — TF-IDF Weight Comparison",
            font=dict(family=_SERIF, size=13, color=_DARK),
            x=0.01,
        ),
        xaxis=dict(
            title="TF-IDF Weight",
            tickfont=dict(family=_MONO, size=10, color="#555"),
            gridcolor=_GRID,
            linecolor=_GRID,
        ),
        yaxis=dict(
            tickfont=dict(family=_SERIF, size=11, color="#333"),
            gridcolor=_GRID,
        ),
        legend=dict(
            font=dict(family=_SERIF, size=11),
            orientation="h",
            y=1.08,
            x=0,
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(t=60, b=40, l=20, r=20),
        height=380,
    )
    return fig


def build_word_length_chart(words1: list, words2: list) -> go.Figure:
    """
    Grouped bar chart showing the word-length distribution
    (number of characters per word) for each document.

    Parameters
    ----------
    words1 : list of str  — tokenised words from Document 1
    words2 : list of str  — tokenised words from Document 2

    Returns
    -------
    go.Figure
    """
    def _length_counts(words):
        lengths = [len(w) for w in words if w.isalpha()]
        return Counter(lengths)

    wlc1 = _length_counts(words1)
    wlc2 = _length_counts(words2)
    all_lengths = sorted(set(wlc1) | set(wlc2))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Document 1",
        x=all_lengths,
        y=[wlc1.get(l, 0) for l in all_lengths],
        marker_color=_DARK,
    ))
    fig.add_trace(go.Bar(
        name="Document 2",
        x=all_lengths,
        y=[wlc2.get(l, 0) for l in all_lengths],
        marker_color=_TEAL,
        opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Word Length Distribution",
            font=dict(family=_SERIF, size=13, color=_DARK),
        ),
        xaxis=dict(
            title="Word length (characters)",
            tickfont=dict(family=_MONO, size=10, color="#555"),
        ),
        yaxis=dict(
            title="Count",
            tickfont=dict(family=_MONO, size=10, color="#555"),
            gridcolor=_GRID,
        ),
        legend=dict(font=dict(size=11), orientation="h", y=1.08),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(t=50, b=40, l=10, r=10),
        height=300,
    )
    return fig
