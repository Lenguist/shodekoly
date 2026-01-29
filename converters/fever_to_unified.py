#!/usr/bin/env python3
"""Convert FEVER JSONL data to unified format."""

import json
import argparse
from pathlib import Path


def convert_fever_example(example: dict) -> dict:
    """Convert a single FEVER example to unified format.

    FEVER format:
        {
            "id": 137334,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Fox 2000 Pictures released the film Soul Food.",
            "evidence": [[[289914, 283015, "Soul_Food_-LRB-film-RRB-", 0]], ...]
        }

    Unified format:
        {
            "id": "137334",
            "input": "Fox 2000 Pictures released the film Soul Food.",
            "options": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            "gold_label": "SUPPORTS",
            "gold_evidence": [{"doc_id": "Soul_Food_-LRB-film-RRB-", "location": 0}],
            "metadata": {"task": "fever", "verifiable": "VERIFIABLE"}
        }
    """
    # Extract evidence - FEVER can have multiple evidence sets
    # Each evidence set is a list of [annotation_id, evidence_id, wiki_page, sentence_id]
    gold_evidence = []

    for evidence_set in example.get("evidence", []):
        for evidence_item in evidence_set:
            if len(evidence_item) >= 4:
                _, _, doc_id, location = evidence_item[:4]
                if doc_id is not None:  # Skip null evidence (NOT ENOUGH INFO)
                    gold_evidence.append({
                        "doc_id": doc_id,
                        "location": location
                    })

    # Deduplicate evidence while preserving order
    seen = set()
    unique_evidence = []
    for ev in gold_evidence:
        key = (ev["doc_id"], ev["location"])
        if key not in seen:
            seen.add(key)
            unique_evidence.append(ev)

    return {
        "id": str(example["id"]),
        "input": example["claim"],
        "options": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
        "gold_label": example["label"],
        "gold_evidence": unique_evidence if unique_evidence else None,
        "metadata": {
            "task": "fever",
            "verifiable": example.get("verifiable")
        }
    }


def convert_file(input_path: Path, output_path: Path) -> dict:
    """Convert a FEVER JSONL file to unified format.

    Returns stats about the conversion.
    """
    stats = {"total": 0, "supports": 0, "refutes": 0, "nei": 0, "with_evidence": 0}

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            example = json.loads(line)
            unified = convert_fever_example(example)

            f_out.write(json.dumps(unified, ensure_ascii=False) + "\n")

            # Update stats
            stats["total"] += 1
            label = unified["gold_label"]
            if label == "SUPPORTS":
                stats["supports"] += 1
            elif label == "REFUTES":
                stats["refutes"] += 1
            else:
                stats["nei"] += 1

            if unified["gold_evidence"]:
                stats["with_evidence"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert FEVER data to unified format")
    parser.add_argument("input", type=Path, help="Input FEVER JSONL file")
    parser.add_argument("output", type=Path, help="Output unified JSONL file")
    args = parser.parse_args()

    print(f"Converting {args.input} -> {args.output}")
    stats = convert_file(args.input, args.output)

    print(f"\nConversion complete:")
    print(f"  Total examples: {stats['total']}")
    print(f"  SUPPORTS: {stats['supports']}")
    print(f"  REFUTES: {stats['refutes']}")
    print(f"  NOT ENOUGH INFO: {stats['nei']}")
    print(f"  With evidence: {stats['with_evidence']}")


if __name__ == "__main__":
    main()
