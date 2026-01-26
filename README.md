## shodekoly

Local workspace for the **UNLP 2026 Shared Task on Multi-Domain Document Understanding** (see `unlp-2026-shared-task/README.md`).

### What this task is

Given a question with multiple-choice options, the goal is to predict:
- **the correct option (A–F)**
- **the supporting document (`Doc_ID`) and page (`Page_Num`)** from a domain’s PDFs

### Getting started (local dev)

Clone the repo, then extract the provided dev PDFs and run the PDF→text extraction script.

#### 1) Unzip dev PDFs (domain_1 and domain_2)

```bash
unzip -o "unlp-2026-shared-task/data/domain_1/dev.zip" -d "unlp-2026-shared-task/data/domain_1"
unzip -o "unlp-2026-shared-task/data/domain_2/dev.zip" -d "unlp-2026-shared-task/data/domain_2"
```

This creates:
- `unlp-2026-shared-task/data/domain_1/dev/*.pdf`
- `unlp-2026-shared-task/data/domain_2/dev/*.pdf`

#### 2) Extract PDFs to text + write corpus stats

Preferred: install Poppler so the script can use `pdftotext`:

```bash
brew install poppler
```

Run the script:

```bash
python3 "pdf-to-text extractoin/extract_dev_pdfs_to_text.py"
```

Outputs:
- `unlp-2026-shared-task/data/domain_1/dev-text/*.txt`
- `unlp-2026-shared-task/data/domain_2/dev-text/*.txt`
- `unlp-2026-shared-task/data/domain_1/dev-text-stats.json`
- `unlp-2026-shared-task/data/domain_2/dev-text-stats.json`
- `unlp-2026-shared-task/data/dev-text-stats.json`

Notes:
- Extraction details/journal: `pdf-to-text extractoin/pdf-to-text-journal.md`
- Generated PDFs/text/stats are ignored by git (see root `.gitignore`).
