# UNLP Baselines: Rationale and Implementation

## What was built

A 3-phase baseline system for the UNLP 2026 shared task: given a Ukrainian multiple-choice question (6 options A-F), predict the correct answer and the supporting document page from a corpus of 41 PDFs.

## Step 0: Corpus Index (`converters/build_unlp_corpus.py`)

### Problem
The evaluation harness (`eval.py`) expects a corpus JSONL file, but the UNLP config had `"corpus": None`. The raw data lives as extracted text files in `unlp-2026-shared-task/data/domain_*/dev-text/*.txt`, with page boundaries marked by form-feed characters (`\f`).

### What was done
- Created `converters/build_unlp_corpus.py` that reads all 41 text files across domain_1 (11 docs) and domain_2 (30 docs)
- Splits each file by `\f` to recover page boundaries
- Writes `data/unlp_corpus.jsonl` — one JSON line per document with fields:
  - `doc_id`: filename with `.pdf` extension (matches gold evidence references)
  - `domain`: `domain_1` or `domain_2`
  - `pages`: dict mapping 1-indexed page numbers (as strings) to page text
- Updated `eval.py` line 59: changed `"corpus": None` to `"corpus": "unlp_corpus.jsonl"`

### Result
41 documents, 1,118 pages indexed.

---

## Phase 1: TF-IDF Baseline (`baselines/unlp_tfidf_baseline.py`)

### Rationale
TF-IDF is the simplest retrieval method — no model downloads, no GPU, runs on any machine with sklearn. It provides a lexical matching floor that more sophisticated methods should beat.

### Implementation
1. **Index**: TfidfVectorizer over all 1,118 pages (50K features, unigrams+bigrams, sublinear TF). No stop words removed — Ukrainian doesn't have a standard stop word list in sklearn.
2. **Retrieval**: For each question, retrieve top-10 pages by cosine similarity of the question text against the page index.
3. **Answer selection**: For each of the 6 options, compute TF-IDF cosine similarity of `(question + option)` against each of the 10 retrieved pages. The option with the highest similarity to any retrieved page wins.
4. **Evidence**: The page that produced the best similarity score for the winning option.

### Design decisions
- **No stop words**: Ukrainian stop words aren't built into sklearn. Adding a custom list would help but was deferred to keep Phase 1 minimal.
- **Page-level granularity**: The gold evidence uses page numbers, not sentences. So we index whole pages rather than splitting further.
- **Top-10 retrieval**: Wider than the 5 used in SciFact TF-IDF because pages are coarser units than sentences.

### Results
```
Answer accuracy:   40.78%
Document accuracy: 77.87%
Page proximity:    0.4788
Evidence score:    0.6288
UNLP score:        51.83%
```

Document retrieval is strong (78%) because the corpus is only 41 documents. The bottleneck is answer selection — lexical overlap between `(question + option)` and page text is a weak signal for choosing among 6 options.

---

## Phase 2: Embedding Baseline (`baselines/unlp_embedding_baseline.py`)

### Rationale
Semantic embeddings capture paraphrases and meaning that TF-IDF misses. For Ukrainian, `lang-uk/ukr-paraphrase-multilingual-mpnet-base` is the best available model — it's specifically fine-tuned for Ukrainian paraphrase detection on top of the multilingual MPNet base.

### Implementation
1. **Model**: `lang-uk/ukr-paraphrase-multilingual-mpnet-base` (~400MB, 768-dim embeddings)
2. **Index**: Encode all 1,118 pages, normalize embeddings for fast dot-product cosine similarity
3. **Caching**: Page embeddings saved to `data/unlp_page_embeddings.npy` so subsequent runs skip encoding
4. **Retrieval**: Top-10 pages by cosine similarity of question embedding vs page embeddings
5. **Answer selection**: Same strategy as Phase 1 but using semantic similarity — encode `(question + option)` and each retrieved page, pick the highest cosine pair

### Design decisions
- **Cached embeddings**: Encoding 1,118 pages takes a while; caching to `.npy` makes re-runs fast
- **Same model for retrieval and scoring**: Using one model keeps memory under 1GB total
- **Fits on 8GB MacBook**: model (~400MB) + embeddings (~7MB) + overhead (~300MB) ≈ ~1GB

### Expected results
~35-50% answer accuracy, ~60-80% document accuracy. Semantic matching should improve answer selection over TF-IDF's lexical matching.

### Requirements
```
pip install sentence-transformers
```

---

## Phase 3: API-Based Baseline (`baselines/unlp_api_baseline.py`)

### Rationale
LLMs can reason about questions and options in context, rather than relying on similarity scores. This tests the ceiling of what's achievable with good retrieval + strong reasoning.

### Implementation
1. **Retrieval**: Same TF-IDF retriever as Phase 1 (to keep it dependency-light), top-5 pages
2. **Prompt**: System prompt instructs the model to read context pages, pick the correct answer (A-F), and identify the evidence page number
3. **Parsing**: Regex extraction of `Answer: <letter>` and `Page: <number>` from response
4. **Providers**: Supports both Claude (Anthropic) and OpenAI APIs via `--provider` flag
5. **Rate limiting**: Configurable `--delay` between API calls (default 0.5s)

### Design decisions
- **TF-IDF for retrieval, not embeddings**: Avoids requiring sentence-transformers as a dependency for Phase 3. The retrieval quality is already good (78% doc accuracy).
- **Page text truncated to 2,000 chars**: Keeps context window manageable; most relevant info is in the first portion of a page.
- **Graceful error handling**: API failures default to answer "A" with top retrieved page, so the run completes even with intermittent errors.

### Expected results
~70-85% answer accuracy if retrieval surfaces the right document. The LLM can reason about Ukrainian medical/sports questions far better than similarity scores.

### Requirements
```
# For Claude
pip install anthropic
export ANTHROPIC_API_KEY=...

# For OpenAI
pip install openai
export OPENAI_API_KEY=...
```

---

## Comparison with English (SciFact) TF-IDF Baseline

| Metric | SciFact (EN) | UNLP (UK) |
|--------|-------------|-----------|
| Task type | 3-class label (SUPPORTS/REFUTES/NEI) | 6-choice multiple choice |
| Corpus size | 5,183 docs / 45,952 sentences | 41 docs / 1,118 pages |
| Questions | 450 | 461 |
| Answer/Label accuracy | 40.89% | 40.78% |
| Evidence accuracy | 8.44% | 77.87% (doc-level) |
| Overall score | 20.89% (FEVER) | 51.83% (UNLP) |

Key insight: both TF-IDF baselines hit ~41% label accuracy, but through different mechanisms. SciFact's evidence accuracy is low because finding the right sentence among 46K is hard. UNLP's document accuracy is high because the corpus is tiny (41 docs), but answer selection among 6 options using lexical similarity is the limiting factor.

---

## Files created/modified

| File | Action | Description |
|------|--------|-------------|
| `converters/build_unlp_corpus.py` | Created | Corpus builder: text files → JSONL |
| `data/unlp_corpus.jsonl` | Generated | 41 docs, 1,118 pages |
| `eval.py` | Modified (line 59) | Set corpus path for UNLP |
| `baselines/unlp_tfidf_baseline.py` | Created | Phase 1: TF-IDF retrieval + scoring |
| `baselines/unlp_embedding_baseline.py` | Created | Phase 2: Semantic embedding retrieval |
| `baselines/unlp_api_baseline.py` | Created | Phase 3: LLM API answer selection |
