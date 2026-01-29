#!/usr/bin/env python3
"""
Build UNLP corpus index from extracted PDF text files.

Reads text files from unlp-2026-shared-task/data/domain_*/dev-text/*.txt,
splits each file by form-feed to get page boundaries, and writes
data/unlp_corpus.jsonl with one JSON line per document.

Usage:
    python converters/build_unlp_corpus.py
    python converters/build_unlp_corpus.py --data-dir path/to/unlp-2026-shared-task/data
"""

import json
import argparse
from pathlib import Path


def build_corpus(data_dir: Path, output_path: Path):
    """Read text files, split by form-feed, write corpus JSONL."""
    docs = []

    for domain_dir in sorted(data_dir.glob("domain_*")):
        text_dir = domain_dir / "dev-text"
        if not text_dir.exists():
            print(f"  Skipping {domain_dir.name}: no dev-text directory")
            continue

        domain = domain_dir.name

        for txt_path in sorted(text_dir.glob("*.txt")):
            doc_id = txt_path.stem + ".pdf"
            text = txt_path.read_text(encoding="utf-8")
            raw_pages = text.split("\f")

            # Build 1-indexed pages dict, skip empty trailing pages
            pages = {}
            for i, page_text in enumerate(raw_pages, 1):
                stripped = page_text.strip()
                if stripped:
                    pages[str(i)] = stripped

            docs.append({
                "doc_id": doc_id,
                "domain": domain,
                "pages": pages,
            })

        print(f"  {domain}: {len([d for d in docs if d['domain'] == domain])} documents")

    # Write corpus
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total_pages = sum(len(d["pages"]) for d in docs)
    print(f"\nWrote {len(docs)} documents ({total_pages} pages) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build UNLP corpus index from text files")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent.parent / "unlp-2026-shared-task" / "data",
        help="Path to unlp-2026-shared-task/data directory",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent.parent / "data" / "unlp_corpus.jsonl",
        help="Output corpus JSONL file",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}")
        return

    print(f"Building corpus from {args.data_dir}")
    build_corpus(args.data_dir, args.output)


if __name__ == "__main__":
    main()
