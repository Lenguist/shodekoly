#!/usr/bin/env python3
"""
TF-IDF baseline for fact verification (SciFact / FEVER).

Pipeline:
1. Document retrieval: TF-IDF cosine similarity over corpus sentences
2. Sentence selection: Top-k most similar sentences to claim
3. Classification: Logistic regression trained on (claim, evidence) features

Usage:
    # Run standalone
    python baselines/tfidf_baseline.py \
        --questions data/scifact_validation.jsonl \
        --corpus data/scifact_corpus.jsonl \
        --output predictions.jsonl

    # Via eval harness
    python eval.py --corpus scifact --system baselines/tfidf_baseline.py

    # Train on training data first (uses scifact_train.jsonl for classifier)
    python baselines/tfidf_baseline.py \
        --questions data/scifact_validation.jsonl \
        --corpus data/scifact_corpus.jsonl \
        --train data/scifact_train.jsonl \
        --output predictions.jsonl
"""

import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


class TfidfRetriever:
    """TF-IDF-based document and sentence retrieval."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_index = {doc["doc_id"]: doc for doc in corpus}

        # Build sentence-level index
        self.sentences = []      # (doc_id, sent_idx, text)
        self.sent_texts = []

        for doc in corpus:
            for i, sent in enumerate(doc["sentences"]):
                self.sentences.append((doc["doc_id"], i, sent))
                self.sent_texts.append(sent)

        print(f"Building TF-IDF index over {len(self.sent_texts)} sentences...")
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.sent_matrix = self.vectorizer.fit_transform(self.sent_texts)
        print("Index built.")

    def retrieve(self, claim: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k most similar sentences to the claim.

        Returns list of {"doc_id", "location", "sentence", "score"}.
        """
        claim_vec = self.vectorizer.transform([claim])
        scores = cosine_similarity(claim_vec, self.sent_matrix)[0]

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_id, sent_idx, text = self.sentences[idx]
            results.append({
                "doc_id": doc_id,
                "location": sent_idx,
                "sentence": text,
                "score": float(scores[idx]),
            })
        return results


def extract_features(claim: str, retrieved: list[dict]) -> np.ndarray:
    """Extract features from claim + retrieved evidence for classification.

    Features:
    - Top-1 retrieval score
    - Top-3 avg retrieval score
    - Top-5 avg retrieval score
    - Max score
    - Score gap (top1 - top2)
    - Number of unique docs in top-5
    - Word overlap ratio between claim and top sentence
    """
    scores = [r["score"] for r in retrieved]

    top1_score = scores[0] if scores else 0.0
    top3_avg = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores) if scores else 0.0
    top5_avg = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    score_gap = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0

    unique_docs = len(set(r["doc_id"] for r in retrieved[:5]))

    # Word overlap
    claim_words = set(claim.lower().split())
    if retrieved:
        top_sent_words = set(retrieved[0]["sentence"].lower().split())
        overlap = len(claim_words & top_sent_words) / max(len(claim_words), 1)
    else:
        overlap = 0.0

    return np.array([
        top1_score, top3_avg, top5_avg, max_score,
        score_gap, unique_docs, overlap
    ])


def train_classifier(train_examples: list[dict], retriever: TfidfRetriever) -> LogisticRegression:
    """Train a logistic regression classifier on training data."""
    print(f"Training classifier on {len(train_examples)} examples...")

    X = []
    y = []
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    for ex in train_examples:
        retrieved = retriever.retrieve(ex["input"], top_k=5)
        features = extract_features(ex["input"], retrieved)
        X.append(features)
        y.append(label_map[ex["gold_label"]])

    X = np.array(X)
    y = np.array(y)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X, y)

    train_acc = clf.score(X, y)
    print(f"Training accuracy: {train_acc:.4f}")

    return clf


def predict_heuristic(claim: str, retrieved: list[dict]) -> tuple[str, list[dict]]:
    """Simple heuristic baseline (no training data needed).

    - If top retrieval score > threshold → SUPPORTS
    - If very low → NOT ENOUGH INFO
    - Otherwise → REFUTES (least common in practice)
    """
    if not retrieved or retrieved[0]["score"] < 0.05:
        return "NOT ENOUGH INFO", None

    top = retrieved[0]
    evidence = [{"doc_id": top["doc_id"], "location": top["location"]}]

    if top["score"] > 0.3:
        return "SUPPORTS", evidence
    elif top["score"] > 0.15:
        return "REFUTES", evidence
    else:
        return "NOT ENOUGH INFO", None


def predict_with_classifier(
    claim: str,
    retrieved: list[dict],
    clf: LogisticRegression,
) -> tuple[str, list[dict]]:
    """Predict using trained logistic regression."""
    inv_label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

    features = extract_features(claim, retrieved).reshape(1, -1)
    pred_idx = clf.predict(features)[0]
    pred_label = inv_label_map[pred_idx]

    if pred_label == "NOT ENOUGH INFO":
        return pred_label, None

    top = retrieved[0]
    evidence = [{"doc_id": top["doc_id"], "location": top["location"]}]
    return pred_label, evidence


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline for fact verification")
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSONL")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--train", type=Path, help="Training data JSONL (for classifier)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sentences to retrieve")
    args = parser.parse_args()

    # Load data
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    print(f"Loaded {len(corpus)} documents, {len(questions)} questions")

    # Build retriever
    retriever = TfidfRetriever(corpus)

    # Train classifier if training data provided
    clf = None
    if args.train and args.train.exists():
        train_examples = load_jsonl(args.train)
        clf = train_classifier(train_examples, retriever)

    # Generate predictions
    predictions = []
    for i, q in enumerate(questions):
        retrieved = retriever.retrieve(q["input"], top_k=args.top_k)

        if clf is not None:
            pred_label, pred_evidence = predict_with_classifier(
                q["input"], retrieved, clf
            )
        else:
            pred_label, pred_evidence = predict_heuristic(q["input"], retrieved)

        predictions.append({
            "id": q["id"],
            "pred_label": pred_label,
            "pred_evidence": pred_evidence,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(questions)}")

    # Save predictions
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
