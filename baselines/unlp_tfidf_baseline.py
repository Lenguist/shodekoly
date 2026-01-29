#!/usr/bin/env python3
"""
TF-IDF baseline for UNLP multi-choice QA over domain PDFs.

Pipeline:
1. Build TF-IDF index over all corpus pages (50K features, bigrams, sublinear TF)
2. For each question, retrieve top-10 pages by cosine similarity
3. For each of the 6 options, score by TF-IDF similarity of (question + option)
   against retrieved pages
4. Predict = highest-scoring option; evidence = top retrieved page for that option

Usage:
    python baselines/unlp_tfidf_baseline.py \
        --questions data/unlp_dev.jsonl \
        --corpus data/unlp_corpus.jsonl \
        --output predictions.jsonl

    # Via eval harness
    python eval.py --corpus unlp --system baselines/unlp_tfidf_baseline.py
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


class PageTfidfRetriever:
    """TF-IDF retriever over corpus pages."""

    def __init__(self, corpus: list[dict]):
        self.pages = []      # (doc_id, page_num, text)
        self.page_texts = []

        for doc in corpus:
            doc_id = doc["doc_id"]
            for page_num, text in doc["pages"].items():
                self.pages.append((doc_id, int(page_num), text))
                self.page_texts.append(text)

        print(f"Building TF-IDF index over {len(self.page_texts)} pages...")
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.page_matrix = self.vectorizer.fit_transform(self.page_texts)
        print("Index built.")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k pages by cosine similarity to query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.page_matrix)[0]
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


def predict_question(question: dict, retriever: PageTfidfRetriever) -> dict:
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

        # Score against each retrieved page
        option_vec = retriever.vectorizer.transform([combined])
        option_best_score = -1.0
        option_best_page = None

        for page in retrieved:
            page_vec = retriever.vectorizer.transform([page["text"]])
            sim = cosine_similarity(option_vec, page_vec)[0, 0]
            if sim > option_best_score:
                option_best_score = sim
                option_best_page = page

        if option_best_score > best_score:
            best_score = option_best_score
            best_label = label
            best_page = option_best_page

    pred_evidence = None
    if best_page:
        pred_evidence = [{"doc_id": best_page["doc_id"], "location": best_page["page_num"]}]

    return {
        "id": question["id"],
        "pred_label": best_label,
        "pred_evidence": pred_evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline for UNLP multi-choice QA")
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSONL")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--train", type=Path, help="Training data (unused, for CLI compatibility)")
    args = parser.parse_args()

    # Load data
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    print(f"Loaded {len(corpus)} documents, {len(questions)} questions")

    # Build retriever
    retriever = PageTfidfRetriever(corpus)

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
