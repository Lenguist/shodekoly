✅ BEST OPTIONS (NO OCR)

✅ Option 1 — Command line (best, cleanest)
pdftotext (from Poppler)

This is the gold standard.

Install (Mac):
brew install poppler
Extract text:
pdftotext input.pdf output.txt

That’s it.

✔ Preserves Ukrainian
✔ Preserves paragraphs
✔ Extremely accurate
✔ Fast
✔ No encoding issues

If you want layout preserved:

pdftotext -layout input.pdf output.txt

✅ Option 2 — Python (very good)
Install:
pip install pymupdf
Full working code:
import fitz  # PyMuPDF


doc = fitz.open("input.pdf")


text = ""
for page in doc:
    text += page.get_text()


with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

✔ Handles Ukrainian perfectly
✔ UTF-8 output
✔ Works if copy–paste works in Preview

✅ Option 3 — pdfplumber (best formatting)
pip install pdfplumber
import pdfplumber


text = ""


with pdfplumber.open("input.pdf") as pdf:
    for page in pdf.pages:
        text += page.extract_text() + "\n"


with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

Often best for academic PDFs.

✅ Option 4 — macOS Preview (manual)

If you can:

open PDF in Preview

select text

copy & paste

then all tools above will work perfectly.

We will proceed to try to use the pdf to text utility.
