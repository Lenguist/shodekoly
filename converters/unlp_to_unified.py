#!/usr/bin/env python3
"""Convert UNLP CSV data to unified format."""

import csv
import json
import argparse
from pathlib import Path


def convert_unlp_row(row: dict) -> dict:
    """Convert a single UNLP CSV row to unified format.

    UNLP CSV format:
        Question_ID, Domain, N_Pages, Question, A, B, C, D, E, F,
        Correct_Answer, Doc_ID, Page_Num

    Unified format:
        {
            "id": "0",
            "input": "Як рекомендовано приймати ретаболіл дорослим?",
            "options": ["внутрішньо", "підшкірно", "орально", ...],
            "gold_label": "E",
            "gold_evidence": [{"doc_id": "xxx.pdf", "location": 1}],
            "metadata": {"task": "unlp", "domain": "domain_2", "n_pages": 5}
        }
    """
    # Build options list
    options = [
        row.get("A", ""),
        row.get("B", ""),
        row.get("C", ""),
        row.get("D", ""),
        row.get("E", ""),
        row.get("F", ""),
    ]

    # Build evidence
    gold_evidence = None
    doc_id = row.get("Doc_ID", "").strip()
    page_num = row.get("Page_Num", "").strip()

    if doc_id:
        gold_evidence = [{
            "doc_id": doc_id,
            "location": int(page_num) if page_num else None
        }]

    # Build metadata
    metadata = {"task": "unlp"}
    if row.get("Domain"):
        metadata["domain"] = row["Domain"]
    if row.get("N_Pages"):
        try:
            metadata["n_pages"] = int(row["N_Pages"])
        except ValueError:
            pass

    return {
        "id": str(row.get("Question_ID", "")),
        "input": row.get("Question", ""),
        "options": options,
        "gold_label": row.get("Correct_Answer", ""),
        "gold_evidence": gold_evidence,
        "metadata": metadata
    }


def convert_file(input_path: Path, output_path: Path) -> dict:
    """Convert a UNLP CSV file to unified format.

    Returns stats about the conversion.
    """
    stats = {
        "total": 0,
        "by_answer": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0},
        "by_domain": {},
        "with_evidence": 0
    }

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)

        for row in reader:
            unified = convert_unlp_row(row)
            f_out.write(json.dumps(unified, ensure_ascii=False) + "\n")

            # Update stats
            stats["total"] += 1

            label = unified["gold_label"]
            if label in stats["by_answer"]:
                stats["by_answer"][label] += 1

            domain = unified["metadata"].get("domain", "unknown")
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1

            if unified["gold_evidence"]:
                stats["with_evidence"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert UNLP data to unified format")
    parser.add_argument("input", type=Path, help="Input UNLP CSV file")
    parser.add_argument("output", type=Path, help="Output unified JSONL file")
    args = parser.parse_args()

    print(f"Converting {args.input} -> {args.output}")
    stats = convert_file(args.input, args.output)

    print(f"\nConversion complete:")
    print(f"  Total examples: {stats['total']}")
    print(f"  With evidence: {stats['with_evidence']}")
    print(f"\n  By answer:")
    for ans, count in sorted(stats["by_answer"].items()):
        if count > 0:
            print(f"    {ans}: {count}")
    print(f"\n  By domain:")
    for domain, count in sorted(stats["by_domain"].items()):
        print(f"    {domain}: {count}")


if __name__ == "__main__":
    main()
