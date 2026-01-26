from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class DocStats:
    filename: str
    chars: int
    words: int


def repo_root_from_script() -> Path:
    # script lives at: <repo>/pdf-to-text extractoin/extract_dev_pdfs_to_text.py
    return Path(__file__).resolve().parents[1]


def run_pdftotext(input_pdf: Path, output_txt: Path, layout: bool) -> None:
    args = ["pdftotext"]
    if layout:
        args.append("-layout")
    args += [str(input_pdf), str(output_txt)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_pymupdf(input_pdf: Path, output_txt: Path) -> None:
    # Fallback if pdftotext isn't available.
    import fitz  # type: ignore

    doc = fitz.open(str(input_pdf))
    text_parts: list[str] = []
    for page in doc:
        text_parts.append(page.get_text())
    output_txt.write_text("".join(text_parts), encoding="utf-8")


def extract_domain(domain_dir: Path, layout: bool, overwrite: bool) -> list[Path]:
    pdf_dir = domain_dir / "dev"
    out_dir = domain_dir / "dev-text"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"Missing PDF directory: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")

    has_pdftotext = shutil.which("pdftotext") is not None

    for pdf in pdfs:
        out_txt = out_dir / (pdf.stem + ".txt")
        if out_txt.exists() and not overwrite:
            continue

        if has_pdftotext:
            run_pdftotext(pdf, out_txt, layout=layout)
            # pdftotext may emit Latin-1 in rare cases; normalize to UTF-8 if needed.
            try:
                out_txt.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = out_txt.read_bytes()
                out_txt.write_text(raw.decode("utf-8", errors="replace"), encoding="utf-8")
        else:
            try:
                run_pymupdf(pdf, out_txt)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "Neither `pdftotext` is available nor PyMuPDF is usable. "
                    "Install poppler (`brew install poppler`) or `pip install pymupdf`."
                ) from e

    return pdfs


def compute_stats(domain_dir: Path) -> dict[str, Any]:
    out_dir = domain_dir / "dev-text"
    txts = sorted(out_dir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No extracted text files found in: {out_dir}")

    docs: list[DocStats] = []
    total_chars = 0
    total_words = 0

    for txt in txts:
        text = txt.read_text(encoding="utf-8", errors="replace")
        chars = len(text)
        words = len(WORD_RE.findall(text))
        docs.append(DocStats(filename=txt.name, chars=chars, words=words))
        total_chars += chars
        total_words += words

    return {
        "domain": domain_dir.name,
        "n_documents": len(docs),
        "total_chars": total_chars,
        "total_words": total_words,
        "documents": [d.__dict__ for d in docs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dev PDFs to text for UNLP 2026 shared task domains + write corpus stats."
    )
    parser.add_argument("--layout", action="store_true", help="Use pdftotext -layout for extraction.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt outputs.")
    args = parser.parse_args()

    root = repo_root_from_script()
    data_dir = root / "unlp-2026-shared-task" / "data"

    domains = ["domain_1", "domain_2"]
    per_domain: dict[str, dict[str, Any]] = {}

    for d in domains:
        domain_dir = data_dir / d
        extract_domain(domain_dir, layout=args.layout, overwrite=args.overwrite)
        stats = compute_stats(domain_dir)
        per_domain[d] = stats

        (domain_dir / "dev-text-stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    combined_docs = sum(per_domain[d]["n_documents"] for d in domains)
    combined_chars = sum(per_domain[d]["total_chars"] for d in domains)
    combined_words = sum(per_domain[d]["total_words"] for d in domains)

    combined = {
        "domains": domains,
        "n_documents_total": combined_docs,
        "total_chars": combined_chars,
        "total_words": combined_words,
        "per_domain": per_domain,
    }

    (data_dir / "dev-text-stats.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

