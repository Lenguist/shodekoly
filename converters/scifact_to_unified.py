#!/usr/bin/env python3
"""Convert SciFact data to unified format and save corpus."""

import json
import argparse
from pathlib import Path
from datasets import load_dataset


def convert_scifact_claim(claim: dict) -> dict:
    """Convert a SciFact claim to unified format.

    SciFact format:
        {
            "id": 3,
            "claim": "1,000 genomes project enables...",
            "evidence_doc_id": "14717500",
            "evidence_label": "SUPPORT",
            "evidence_sentences": [2, 5],
            "cited_doc_ids": [14717500]
        }

    Unified format:
        {
            "id": "3",
            "input": "1,000 genomes project enables...",
            "options": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            "gold_label": "SUPPORTS",
            "gold_evidence": [{"doc_id": "14717500", "location": 2}, ...],
            "metadata": {"task": "scifact"}
        }
    """
    # Map SciFact labels to FEVER-style labels
    label_map = {
        "SUPPORT": "SUPPORTS",
        "CONTRADICT": "REFUTES",
        "": "NOT ENOUGH INFO"
    }

    raw_label = claim.get("evidence_label", "") or ""
    gold_label = label_map.get(raw_label, "NOT ENOUGH INFO")

    # Build evidence list
    gold_evidence = None
    if claim.get("evidence_doc_id") and claim.get("evidence_sentences"):
        doc_id = str(claim["evidence_doc_id"])
        gold_evidence = [
            {"doc_id": doc_id, "location": sent_idx}
            for sent_idx in claim["evidence_sentences"]
        ]

    return {
        "id": str(claim["id"]),
        "input": claim["claim"],
        "options": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        "gold_label": gold_label,
        "gold_evidence": gold_evidence,
        "metadata": {
            "task": "scifact",
            "cited_doc_ids": [str(d) for d in claim.get("cited_doc_ids", [])]
        }
    }


def convert_scifact_doc(doc: dict) -> dict:
    """Convert a SciFact document to corpus format.

    Output format:
        {
            "doc_id": "4983",
            "title": "Microstructural development...",
            "sentences": ["sentence 0", "sentence 1", ...]
        }
    """
    return {
        "doc_id": str(doc["doc_id"]),
        "title": doc["title"],
        "sentences": doc["abstract"]  # Already a list of sentences
    }


def setup_scifact(output_dir: Path):
    """Download SciFact and convert to unified format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SciFact from Hugging Face...")

    # Load corpus
    corpus_ds = load_dataset("allenai/scifact", "corpus")
    claims_ds = load_dataset("allenai/scifact", "claims")

    # Save corpus
    corpus_path = output_dir / "scifact_corpus.jsonl"
    print(f"Saving corpus to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus_ds["train"]:
            converted = convert_scifact_doc(doc)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
    print(f"  Saved {len(corpus_ds['train'])} documents")

    # Save claims for each split
    stats = {}
    for split in ["train", "validation", "test"]:
        split_path = output_dir / f"scifact_{split}.jsonl"
        print(f"Saving {split} claims to {split_path}...")

        split_stats = {"total": 0, "supports": 0, "refutes": 0, "nei": 0, "with_evidence": 0}

        with open(split_path, "w", encoding="utf-8") as f:
            for claim in claims_ds[split]:
                converted = convert_scifact_claim(claim)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")

                split_stats["total"] += 1
                if converted["gold_label"] == "SUPPORTS":
                    split_stats["supports"] += 1
                elif converted["gold_label"] == "REFUTES":
                    split_stats["refutes"] += 1
                else:
                    split_stats["nei"] += 1
                if converted["gold_evidence"]:
                    split_stats["with_evidence"] += 1

        stats[split] = split_stats
        print(f"  Saved {split_stats['total']} claims (SUPPORTS: {split_stats['supports']}, "
              f"REFUTES: {split_stats['refutes']}, NEI: {split_stats['nei']})")

    print("\nSciFact setup complete!")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and convert SciFact to unified format")
    parser.add_argument("--output-dir", type=Path, default=Path("data"),
                        help="Output directory (default: data)")
    args = parser.parse_args()

    setup_scifact(args.output_dir)


if __name__ == "__main__":
    main()
