#!/usr/bin/env python3
"""
Unified evaluation CLI for fact verification benchmarks.

Usage examples:
    # Evaluate on SciFact validation
    python eval.py --corpus scifact --split validation --predictions predictions.jsonl

    # Evaluate on FEVER dev
    python eval.py --corpus fever --predictions predictions.jsonl

    # Evaluate on UNLP dev
    python eval.py --corpus unlp --lang uk --predictions predictions.jsonl

    # Evaluate on custom questions file
    python eval.py --questions path/to/questions.jsonl --predictions predictions.jsonl

    # Run a baseline system and evaluate
    python eval.py --corpus scifact --split validation --system baselines/random_baseline.py
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional


# ============================================================================
# Data loading
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"

CORPUS_CONFIG = {
    "scifact": {
        "corpus": "scifact_corpus.jsonl",
        "splits": {
            "train": "scifact_train.jsonl",
            "validation": "scifact_validation.jsonl",
            "test": "scifact_test.jsonl",
        },
        "default_split": "validation",
        "lang": "en",
        "task": "fever",  # Same scoring as FEVER
    },
    "fever": {
        "corpus": None,  # Wikipedia - not included
        "splits": {
            "dev": "fever_dev.jsonl",
        },
        "default_split": "dev",
        "lang": "en",
        "task": "fever",
    },
    "unlp": {
        "corpus": "unlp_corpus.jsonl",
        "splits": {
            "dev": "unlp_dev.jsonl",
        },
        "default_split": "dev",
        "lang": "uk",
        "task": "unlp",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def get_questions_path(corpus: str, split: Optional[str] = None) -> Path:
    """Get the path to questions file for a corpus/split."""
    if corpus not in CORPUS_CONFIG:
        raise ValueError(f"Unknown corpus: {corpus}. Available: {list(CORPUS_CONFIG.keys())}")

    config = CORPUS_CONFIG[corpus]
    split = split or config["default_split"]

    if split not in config["splits"]:
        raise ValueError(f"Unknown split '{split}' for {corpus}. Available: {list(config['splits'].keys())}")

    return DATA_DIR / config["splits"][split]


def get_corpus_path(corpus: str) -> Optional[Path]:
    """Get the path to corpus file."""
    config = CORPUS_CONFIG[corpus]
    if config["corpus"]:
        return DATA_DIR / config["corpus"]
    return None


# ============================================================================
# Scoring functions
# ============================================================================

def compute_fever_score(gold: dict, pred: dict) -> dict:
    """FEVER-style scoring: label accuracy + evidence F1."""
    label_correct = gold["gold_label"] == pred.get("pred_label")

    evidence_correct = False
    if gold["gold_evidence"] and pred.get("pred_evidence"):
        gold_set = {(e["doc_id"], e["location"]) for e in gold["gold_evidence"]}
        pred_set = {(e["doc_id"], e["location"]) for e in pred.get("pred_evidence", [])}
        evidence_correct = bool(gold_set & pred_set)

    # For NEI, evidence not required
    if gold["gold_label"] == "NOT ENOUGH INFO":
        fever_score = 1.0 if label_correct else 0.0
    else:
        fever_score = 1.0 if (label_correct and evidence_correct) else 0.0

    return {
        "label_correct": label_correct,
        "evidence_correct": evidence_correct,
        "fever_score": fever_score
    }


def compute_unlp_score(gold: dict, pred: dict) -> dict:
    """UNLP-style scoring: answer + document + page proximity."""
    answer_correct = gold["gold_label"] == pred.get("pred_label")

    doc_correct = False
    page_proximity = 0.0

    if gold["gold_evidence"] and pred.get("pred_evidence"):
        gold_ev = gold["gold_evidence"][0]
        pred_ev = pred["pred_evidence"][0] if pred["pred_evidence"] else {}

        doc_correct = gold_ev["doc_id"] == pred_ev.get("doc_id")

        if doc_correct and gold_ev.get("location") is not None and pred_ev.get("location") is not None:
            page_diff = abs(gold_ev["location"] - pred_ev["location"])
            page_proximity = 1.0 / (1.0 + page_diff)

    evidence_score = 0.5 * (1.0 if doc_correct else 0.0) + 0.5 * page_proximity
    unlp_score = 0.5 * (1.0 if answer_correct else 0.0) + 0.5 * evidence_score

    return {
        "answer_correct": answer_correct,
        "doc_correct": doc_correct,
        "page_proximity": page_proximity,
        "evidence_score": evidence_score,
        "unlp_score": unlp_score
    }


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate(gold_examples: list[dict], pred_examples: list[dict], task: str = "fever") -> dict:
    """Run evaluation and return metrics."""
    pred_by_id = {str(p["id"]): p for p in pred_examples}

    scores = []
    missing = 0

    for gold in gold_examples:
        example_id = str(gold["id"])
        pred = pred_by_id.get(example_id)

        if pred is None:
            missing += 1
            if task == "fever":
                scores.append({"label_correct": False, "evidence_correct": False, "fever_score": 0.0})
            else:
                scores.append({"answer_correct": False, "doc_correct": False,
                              "page_proximity": 0.0, "evidence_score": 0.0, "unlp_score": 0.0})
            continue

        if task == "fever":
            scores.append(compute_fever_score(gold, pred))
        else:
            scores.append(compute_unlp_score(gold, pred))

    n = len(scores)
    results = {"task": task, "total": n, "missing_predictions": missing}

    if task == "fever":
        results["label_accuracy"] = sum(s["label_correct"] for s in scores) / n if n else 0
        results["evidence_accuracy"] = sum(s["evidence_correct"] for s in scores) / n if n else 0
        results["fever_score"] = sum(s["fever_score"] for s in scores) / n if n else 0
    else:
        results["answer_accuracy"] = sum(s["answer_correct"] for s in scores) / n if n else 0
        results["doc_accuracy"] = sum(s["doc_correct"] for s in scores) / n if n else 0
        results["page_proximity_avg"] = sum(s["page_proximity"] for s in scores) / n if n else 0
        results["evidence_score"] = sum(s["evidence_score"] for s in scores) / n if n else 0
        results["unlp_score"] = sum(s["unlp_score"] for s in scores) / n if n else 0

    return results


def run_system(system_path: Path, questions_path: Path, corpus_path: Optional[Path],
               output_path: Path, train_path: Optional[Path] = None) -> bool:
    """Run a baseline system script and generate predictions."""
    cmd = [
        sys.executable, str(system_path),
        "--questions", str(questions_path),
        "--output", str(output_path),
    ]
    if corpus_path and corpus_path.exists():
        cmd.extend(["--corpus", str(corpus_path)])
    if train_path and train_path.exists():
        cmd.extend(["--train", str(train_path)])

    print(f"Running system: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"System failed with error:\n{result.stderr}")
        return False

    if result.stdout:
        print(result.stdout)

    return True


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if results["task"] == "fever":
        print(f"{'Total examples:':<25} {results['total']}")
        print(f"{'Missing predictions:':<25} {results['missing_predictions']}")
        print()
        print(f"{'Label accuracy:':<25} {results['label_accuracy']:.4f} ({results['label_accuracy']*100:.2f}%)")
        print(f"{'Evidence accuracy:':<25} {results['evidence_accuracy']:.4f} ({results['evidence_accuracy']*100:.2f}%)")
        print(f"{'FEVER score:':<25} {results['fever_score']:.4f} ({results['fever_score']*100:.2f}%)")
    else:
        print(f"{'Total examples:':<25} {results['total']}")
        print(f"{'Missing predictions:':<25} {results['missing_predictions']}")
        print()
        print(f"{'Answer accuracy:':<25} {results['answer_accuracy']:.4f} ({results['answer_accuracy']*100:.2f}%)")
        print(f"{'Document accuracy:':<25} {results['doc_accuracy']:.4f} ({results['doc_accuracy']*100:.2f}%)")
        print(f"{'Page proximity:':<25} {results['page_proximity_avg']:.4f}")
        print(f"{'Evidence score:':<25} {results['evidence_score']:.4f}")
        print(f"{'UNLP score:':<25} {results['unlp_score']:.4f} ({results['unlp_score']*100:.2f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fact verification systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py --corpus scifact --predictions preds.jsonl
  python eval.py --corpus fever --predictions preds.jsonl
  python eval.py --corpus unlp --predictions preds.jsonl
  python eval.py --questions custom.jsonl --predictions preds.jsonl
  python eval.py --corpus scifact --system baselines/random.py
        """
    )

    # Data selection
    parser.add_argument("--corpus", choices=list(CORPUS_CONFIG.keys()),
                        help="Corpus to evaluate on (scifact, fever, unlp)")
    parser.add_argument("--split", help="Data split (default depends on corpus)")
    parser.add_argument("--lang", help="Language filter (not yet implemented)")
    parser.add_argument("--questions", type=Path,
                        help="Custom questions file (overrides --corpus)")

    # Predictions
    parser.add_argument("--predictions", type=Path,
                        help="Predictions JSONL file")
    parser.add_argument("--system", type=Path,
                        help="System script to run (generates predictions)")
    parser.add_argument("--train", type=Path,
                        help="Training data JSONL (passed to --system if set)")

    # Output
    parser.add_argument("--output", type=Path,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Validate arguments
    if not args.corpus and not args.questions:
        parser.error("Either --corpus or --questions is required")

    if not args.predictions and not args.system:
        parser.error("Either --predictions or --system is required")

    # Determine questions path
    if args.questions:
        questions_path = args.questions
        task = "fever"  # Default task type for custom questions
    else:
        questions_path = get_questions_path(args.corpus, args.split)
        task = CORPUS_CONFIG[args.corpus]["task"]

    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_path}")
        sys.exit(1)

    print(f"Questions: {questions_path}")

    # Get corpus path if available
    corpus_path = None
    if args.corpus:
        corpus_path = get_corpus_path(args.corpus)
        if corpus_path:
            print(f"Corpus: {corpus_path}")

    # Determine train path
    train_path = args.train
    if train_path is None and args.corpus and args.corpus in CORPUS_CONFIG:
        config = CORPUS_CONFIG[args.corpus]
        if "train" in config["splits"]:
            candidate = DATA_DIR / config["splits"]["train"]
            if candidate.exists():
                train_path = candidate

    # Run system if specified
    if args.system:
        pred_path = Path("/tmp/predictions.jsonl")
        if not run_system(args.system, questions_path, corpus_path, pred_path, train_path):
            sys.exit(1)
        args.predictions = pred_path

    # Load data
    print(f"Predictions: {args.predictions}")
    gold_examples = load_jsonl(questions_path)
    pred_examples = load_jsonl(args.predictions)

    # Auto-detect task from data if custom questions
    if args.questions:
        first = gold_examples[0] if gold_examples else {}
        meta_task = first.get("metadata", {}).get("task")
        if meta_task:
            task = meta_task

    print(f"Task type: {task}")

    # Evaluate
    results = evaluate(gold_examples, pred_examples, task)
    print_results(results)

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
