#!/usr/bin/env python3
"""
Random baseline for fact verification.

This baseline randomly predicts labels and evidence.
Used to verify the evaluation pipeline works.

Expected performance:
- Label accuracy: ~33% (random choice among 3 labels)
- Evidence accuracy: ~0% (random document selection)
"""

import json
import random
import argparse
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def load_corpus(path: Path) -> dict:
    """Load corpus and return dict of doc_id -> document."""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[doc["doc_id"]] = doc
    return corpus


def predict_random(question: dict, corpus: dict, labels: list[str]) -> dict:
    """Generate random prediction for a question."""
    # Random label
    pred_label = random.choice(labels)

    # Random evidence (if corpus available)
    pred_evidence = None
    if corpus and pred_label != "NOT ENOUGH INFO":
        doc_id = random.choice(list(corpus.keys()))
        doc = corpus[doc_id]
        num_sentences = len(doc.get("sentences", []))
        if num_sentences > 0:
            location = random.randint(0, num_sentences - 1)
            pred_evidence = [{"doc_id": doc_id, "location": location}]

    return {
        "id": question["id"],
        "pred_label": pred_label,
        "pred_evidence": pred_evidence
    }


def main():
    parser = argparse.ArgumentParser(description="Random baseline for fact verification")
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSONL file")
    parser.add_argument("--corpus", type=Path, help="Corpus JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load questions
    questions = load_jsonl(args.questions)
    print(f"Loaded {len(questions)} questions")

    # Load corpus if available
    corpus = {}
    if args.corpus and args.corpus.exists():
        corpus = load_corpus(args.corpus)
        print(f"Loaded {len(corpus)} documents")

    # Determine labels from first question
    first_q = questions[0] if questions else {}
    options = first_q.get("options", ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])

    # Generate predictions
    predictions = []
    for q in questions:
        pred = predict_random(q, corpus, options)
        predictions.append(pred)

    # Save predictions
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
