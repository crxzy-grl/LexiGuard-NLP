# 📄 Mini Plagiarism Detection System using NLP


A fully **offline**, lightweight plagiarism checker built with Python and Streamlit. No internet connection, no API keys, no deep learning models — just interpretable classical NLP that runs on any standard laptop.

---

## ✨ Features

| Feature | Details |
|---|---|
| **4-Axis Radar Analysis** | Lexical Match · Sentence Overlap · Structure Similarity · Uniqueness |
| **TF-IDF (1–3 gram)** | Captures word, bigram, and trigram patterns for deeper structural comparison |
| **Cosine Similarity** | Length-normalised overall document similarity score |
| **Jaccard Similarity** | Set-based vocabulary overlap (Lexical Match axis) |
| **Top 10 Keywords** | Grouped bar chart comparing TF-IDF weights of shared terms |
| **Sentence Matching** | Top 5 near-duplicate sentence pairs with similarity percentages |
| **Natural Language Verdicts** | Per-axis plain-English explanations of what each score means |
| **3-Tab Streamlit UI** | Dashboard · Sentence Analysis · Document Statistics |
| **PDF + TXT support** | pdfplumber handles standard (non-scanned) PDFs |
| **100% Offline** | No network calls at runtime |

---

## 🏗️ Architecture

```
plagiarism_checker/
│
├── app.py                  ← Streamlit entry point (UI only)
│
├── src/
│   ├── __init__.py
│   ├── analyzer.py         ← TF-IDF, n-grams, cosine/Jaccard scoring, verdicts
│   ├── visuals.py          ← Plotly radar chart, bar charts
│   └── utils.py            ← PDF/TXT extraction, preprocessing, sentence splitting
│
├── demo/
│   ├── original_document.txt
│   └── paraphrased_document.txt
│
├── requirements.txt
├── LICENSE
└── README.md
```

**Design principle:** `app.py` is intentionally thin — it only handles layout and wires together the three modules. All analysis logic is in `src/analyzer.py`, all chart logic in `src/visuals.py`, and all I/O and preprocessing in `src/utils.py`.

---

## 🔬 Technical Stack

| Layer | Technology |
|---|---|
| **UI Framework** | Streamlit 1.32+ |
| **Vectorization** | scikit-learn `TfidfVectorizer` (ngram_range 1–3) |
| **Similarity** | scikit-learn `cosine_similarity`, custom Jaccard |
| **N-gram Analysis** | Python `collections.Counter` + NumPy vector ops |
| **Visualisation** | Plotly `graph_objects` (radar + bar charts) |
| **PDF Extraction** | pdfplumber |
| **Preprocessing** | Python `re` (regex) |

---

## 🧮 How the Four Axes Work

### 1. Lexical Match (Jaccard Similarity)
Computes `|A ∩ B| / |A ∪ B|` on the unique word sets of both documents.
Measures raw vocabulary overlap without considering word importance.

### 2. Sentence Overlap
Splits both documents into sentences and applies TF-IDF cosine similarity to every pair. Reports the average fraction of sentences that have a near-duplicate (cosine ≥ 0.60) in the other document. Symmetric across both documents.

### 3. Structure Similarity
Builds **trigram** (3-word phrase) frequency vectors for both documents and computes their cosine similarity. Trigrams are more sensitive to paraphrased or reordered content than individual words, making this the primary detector of structural copying.

### 4. Uniqueness
Computes what percentage of Document 1's **bigrams** are absent from Document 2. Higher = more original. Plotted as *Lack of Uniqueness* (inverted) on the radar chart so that all spokes consistently represent increasing risk.

---

## 🚦 Verdict Thresholds

| Score Range | Verdict |
|---|---|
| < 20% | ✅ Likely Original |
| 20–50% | ⚠️ Moderate Overlap |
| ≥ 50% | 🚨 High Similarity — Potential Plagiarism |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/plagiarism-checker.git
cd plagiarism-checker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### 4. Try the demo files

Upload `demo/original_document.txt` as Document 1 and `demo/paraphrased_document.txt` as Document 2 to see all features in action.

---

## 📋 Use Cases

- **Students** — verify originality before submitting assignments or reports
- **Educators** — compare submitted work against reference material
- **Researchers** — check drafts against prior publications or notes
- **Content creators** — audit articles for accidental duplication

---

## ⚠️ Limitations

- **No database comparison** — pairwise only (two specific documents)
- **Scanned PDFs not supported** — requires embedded text layer
- **Semantic paraphrasing** — deep synonym substitution is not detected (requires sentence embeddings)
- **Language** — optimised for English

---

## 🔮 Future Enhancements

- [ ] Lightweight sentence embeddings (e.g., sentence-transformers `all-MiniLM`) for semantic similarity
- [ ] OCR support via pytesseract for scanned PDFs
- [ ] DOCX file support via python-docx
- [ ] Highlighted diff view with colour-coded matching passages
- [ ] Downloadable PDF analysis report

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

