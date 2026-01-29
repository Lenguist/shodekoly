#!/usr/bin/env python3
"""
Embedding retrieval baseline for UNLP multi-choice QA.

Uses lang-uk/ukr-paraphrase-multilingual-mpnet-base (~400MB) for semantic
retrieval over corpus pages. Page embeddings are cached to .npy for fast re-runs.

Pipeline:
1. Encode all corpus pages with sentence-transformers model
2. For each question, retrieve top-10 pages by cosine similarity
3. For each option, score (question + option) against retrieved pages
4. Predict = highest-scoring option; evidence = top page for that option

Usage:
    python baselines/unlp_embedding_baseline.py \
        --questions data/unlp_dev.jsonl \
        --corpus data/unlp_corpus.jsonl \
        --output predictions.jsonl

    # Via eval harness
    python eval.py --corpus unlp --system baselines/unlp_embedding_baseline.py

Requirements:
    pip install sentence-transformers
"""

import json
import argparse
import numpy as np
from pathlib import Path


OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]
MODEL_NAME = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


class EmbeddingRetriever:
    """Semantic retriever using sentence-transformers embeddings."""

    def __init__(self, corpus: list[dict], cache_dir: Path = None):
        from sentence_transformers import SentenceTransformer

        self.pages = []      # (doc_id, page_num, text)
        self.page_texts = []

        for doc in corpus:
            doc_id = doc["doc_id"]
            for page_num, text in doc["pages"].items():
                self.pages.append((doc_id, int(page_num), text))
                self.page_texts.append(text)

        print(f"Loading model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        # Try to load cached embeddings
        cache_path = None
        if cache_dir:
            cache_path = cache_dir / "unlp_page_embeddings.npy"

        if cache_path and cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            self.page_embeddings = np.load(cache_path)
            if self.page_embeddings.shape[0] != len(self.page_texts):
                print("Cache size mismatch, re-encoding...")
                self.page_embeddings = self._encode_pages()
                np.save(cache_path, self.page_embeddings)
        else:
            self.page_embeddings = self._encode_pages()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, self.page_embeddings)
                print(f"Saved embeddings cache to {cache_path}")

        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.page_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.page_embeddings_normed = self.page_embeddings / norms

    def _encode_pages(self) -> np.ndarray:
        print(f"Encoding {len(self.page_texts)} pages...")
        embeddings = self.model.encode(
            self.page_texts,
            show_progress_bar=True,
            batch_size=32,
        )
        print("Encoding complete.")
        return embeddings

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k pages by cosine similarity to query."""
        query_emb = self.model.encode([query])
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        scores = (query_norm @ self.page_embeddings_normed.T)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id, page_num, text = self.pages[idx]
            results.append({
                "doc_id": doc_id,
                "page_num": page_num,
                "text": text,
                "score": float(scores[idx]),
            })
        return results

    def score_text(self, text: str, page_text: str) -> float:
        """Compute cosine similarity between two texts."""
        embs = self.model.encode([text, page_text])
        a, b = embs[0], embs[1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


def predict_question(question: dict, retriever: EmbeddingRetriever) -> dict:
    """Predict answer and evidence for a single question."""
    q_text = question["input"]
    options = question.get("options", [])

    # Retrieve pages using the question text
    retrieved = retriever.retrieve(q_text, top_k=10)

    # Score each option: similarity of (question + option) against retrieved pages
    best_label = OPTION_LABELS[0]
    best_score = -1.0
    best_page = retrieved[0] if retrieved else None

    for i, option_text in enumerate(options):
        if i >= len(OPTION_LABELS):
            break
        label = OPTION_LABELS[i]
        combined = q_text + " " + option_text

        for page in retrieved:
            sim = retriever.score_text(combined, page["text"])
            if sim > best_score:
                best_score = sim
                best_label = label
                best_page = page

    pred_evidence = None
    if best_page:
        pred_evidence = [{"doc_id": best_page["doc_id"], "location": best_page["page_num"]}]

    return {
        "id": question["id"],
        "pred_label": best_label,
        "pred_evidence": pred_evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Embedding baseline for UNLP multi-choice QA")
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSONL")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--train", type=Path, help="Training data (unused, for CLI compatibility)")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory to cache page embeddings",
    )
    args = parser.parse_args()

    # Load data
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    print(f"Loaded {len(corpus)} documents, {len(questions)} questions")

    # Build retriever
    retriever = EmbeddingRetriever(corpus, cache_dir=args.cache_dir)

    # Generate predictions
    predictions = []
    for i, q in enumerate(questions):
        pred = predict_question(q, retriever)
        predictions.append(pred)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(questions)}")

    # Save predictions
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
