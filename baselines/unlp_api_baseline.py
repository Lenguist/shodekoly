#!/usr/bin/env python3
"""
API-based baseline for UNLP multi-choice QA.

Uses TF-IDF retrieval to get top-5 pages as context, then sends the context,
question, and options to an LLM API (Claude or OpenAI) for answer selection.

Pipeline:
1. Build TF-IDF index (same as Phase 1)
2. Retrieve top-5 pages per question
3. Send context + question + options to LLM API
4. Parse answer letter and evidence page from response

Usage:
    # With Claude (default)
    ANTHROPIC_API_KEY=... python baselines/unlp_api_baseline.py \
        --questions data/unlp_dev.jsonl \
        --corpus data/unlp_corpus.jsonl \
        --output predictions.jsonl

    # With OpenAI
    OPENAI_API_KEY=... python baselines/unlp_api_baseline.py \
        --questions data/unlp_dev.jsonl \
        --corpus data/unlp_corpus.jsonl \
        --output predictions.jsonl \
        --provider openai

    # Via eval harness
    python eval.py --corpus unlp --system baselines/unlp_api_baseline.py

Requirements:
    pip install anthropic   # for Claude
    pip install openai      # for OpenAI
    pip install scikit-learn
"""

import json
import re
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]

SYSTEM_PROMPT = """You are a Ukrainian-language question answering assistant. You are given a multiple-choice question with 6 options (A-F) and supporting document pages as context. Your task is to:
1. Read the context carefully
2. Choose the correct answer
3. Identify which page contains the supporting evidence

Respond in exactly this format:
Answer: <letter>
Page: <page_number>

Do not include any other text."""

USER_PROMPT_TEMPLATE = """Context pages:
{context}

Question: {question}

Options:
A) {opt_a}
B) {opt_b}
C) {opt_c}
D) {opt_d}
E) {opt_e}
F) {opt_f}

Choose the correct answer (A-F) and identify the evidence page number."""


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


class PageTfidfRetriever:
    """TF-IDF retriever over corpus pages (same as Phase 1)."""

    def __init__(self, corpus: list[dict]):
        self.pages = []
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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
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


def call_claude(prompt: str, system: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=100,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_openai(prompt: str, system: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=100,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def parse_response(response: str) -> tuple[str, int | None]:
    """Parse answer letter and page number from LLM response."""
    answer = None
    page = None

    # Match answer
    answer_match = re.search(r"Answer:\s*([A-Fa-f])", response)
    if answer_match:
        answer = answer_match.group(1).upper()

    # Match page
    page_match = re.search(r"Page:\s*(\d+)", response)
    if page_match:
        page = int(page_match.group(1))

    # Fallback: look for a standalone letter
    if answer is None:
        letter_match = re.search(r"\b([A-F])\b", response)
        if letter_match:
            answer = letter_match.group(1)

    return answer or "A", page


def format_context(retrieved: list[dict]) -> str:
    """Format retrieved pages as context string."""
    parts = []
    for page in retrieved:
        # Truncate long pages to fit context window
        text = page["text"][:2000]
        parts.append(
            f"[Document: {page['doc_id']}, Page {page['page_num']}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def predict_question(
    question: dict,
    retriever: PageTfidfRetriever,
    call_fn,
    top_k: int = 5,
) -> dict:
    """Predict answer and evidence using LLM API."""
    q_text = question["input"]
    options = question.get("options", [])

    # Pad options to 6 if needed
    while len(options) < 6:
        options.append("")

    # Retrieve context
    retrieved = retriever.retrieve(q_text, top_k=top_k)
    context = format_context(retrieved)

    # Build prompt
    prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=q_text,
        opt_a=options[0],
        opt_b=options[1],
        opt_c=options[2],
        opt_d=options[3],
        opt_e=options[4],
        opt_f=options[5],
    )

    # Call LLM
    try:
        response = call_fn(prompt, SYSTEM_PROMPT)
        answer, page = parse_response(response)
    except Exception as e:
        print(f"  API error for question {question['id']}: {e}", file=sys.stderr)
        answer = "A"
        page = None

    # Build evidence from parsed page or fallback to top retrieved
    pred_evidence = None
    if page is not None:
        # Find matching page in retrieved results
        for r in retrieved:
            if r["page_num"] == page:
                pred_evidence = [{"doc_id": r["doc_id"], "location": r["page_num"]}]
                break
        # If page not in retrieved, use top result's doc with parsed page
        if pred_evidence is None and retrieved:
            pred_evidence = [{"doc_id": retrieved[0]["doc_id"], "location": page}]
    elif retrieved:
        pred_evidence = [{"doc_id": retrieved[0]["doc_id"], "location": retrieved[0]["page_num"]}]

    return {
        "id": question["id"],
        "pred_label": answer,
        "pred_evidence": pred_evidence,
    }


def main():
    parser = argparse.ArgumentParser(description="API-based baseline for UNLP multi-choice QA")
    parser.add_argument("--questions", type=Path, required=True, help="Questions JSONL")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output predictions JSONL")
    parser.add_argument("--train", type=Path, help="Training data (unused, for CLI compatibility)")
    parser.add_argument(
        "--provider", choices=["claude", "openai"], default="claude",
        help="LLM provider (default: claude)",
    )
    parser.add_argument("--model", type=str, help="Model name override")
    parser.add_argument("--top-k", type=int, default=5, help="Pages to retrieve per question")
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (rate limiting)",
    )
    args = parser.parse_args()

    # Validate API keys
    if args.provider == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        model = args.model or "claude-sonnet-4-20250514"
        call_fn = lambda prompt, system: call_claude(prompt, system, model)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        model = args.model or "gpt-4o-mini"
        call_fn = lambda prompt, system: call_openai(prompt, system, model)

    print(f"Provider: {args.provider}, Model: {model}")

    # Load data
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    print(f"Loaded {len(corpus)} documents, {len(questions)} questions")

    # Build retriever
    retriever = PageTfidfRetriever(corpus)

    # Generate predictions
    predictions = []
    for i, q in enumerate(questions):
        pred = predict_question(q, retriever, call_fn, top_k=args.top_k)
        predictions.append(pred)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(questions)}")

        if args.delay > 0:
            time.sleep(args.delay)

    # Save predictions
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
