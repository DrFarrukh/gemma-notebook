"""Compare legacy pypdf text extraction with PyMuPDF4LLM Markdown conversion.

Run from the repository root:
    python scripts/benchmark_pdf_extraction.py [PDF ...]

With no paths, the two local EMG benchmark PDFs are used when present. A
Markdown report including samples from the first several pages is written to
benchmark/pdf_extraction.md.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader
import pymupdf4llm

from app.documents import clean_text, normalize_pdf_headings, split_pages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDFS = (
    "E2CNN_An_Efficient_Concatenated_CNN_for_Classification_of_Surface_EMG_Extracted_From_Upper_Limb.pdf",
    "Spectral_Image-Based_Multiday_Surface_Electromyography_Classification_of_Hand_Motions_Using_CNN_for_HumanComputer_Interaction.pdf",
)
SAMPLE_PAGES = 4


def old_extract(path: Path):
    started = time.perf_counter()
    reader = PdfReader(str(path))
    pages = [(index, clean_text(page.extract_text() or "")) for index, page in enumerate(reader.pages, 1)]
    return pages, time.perf_counter() - started


def new_extract(path: Path):
    started = time.perf_counter()
    result = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    pages = [(chunk["metadata"].get("page_number", chunk["metadata"].get("page")),
              normalize_pdf_headings(clean_text(chunk["text"]))) for chunk in result]
    return pages, time.perf_counter() - started


def chunk_count(pages, markdown=False, size=1800, overlap=250):
    return len(split_pages([{"page": number, "text": text} for number, text in pages],
                           preserve_sections=markdown, chunk_size=size,
                           overlap=overlap, hard_cap=4000))


def heading_count(pages):
    return sum(len(re.findall(r"(?m)^#{1,6}\s+.{3,}$", text)) for _, text in pages)


def resolve_pdfs(args):
    if args:
        return [Path(arg).resolve() for arg in args]
    candidates = list((ROOT / "data/files").glob("*/*.pdf"))
    selected = []
    for filename in DEFAULT_PDFS:
        selected.extend(path for path in candidates if path.name.endswith(filename))
    return selected


def main():
    pdfs = resolve_pdfs(sys.argv[1:])
    if not pdfs:
        raise SystemExit("No PDFs found. Pass one or more PDF paths.")

    report = ["# PDF extraction benchmark", "", "Legacy: `pypdf.Page.extract_text()`. New: `pymupdf4llm.to_markdown(page_chunks=True)`. Structured output chunk counts use paragraph and heading boundaries, with a 4000-character hard cap.", "", "OCR note: the core PyMuPDF4LLM package is used without its optional Layout/OCR modules. Scanned pages continue through the app's existing image-only PDF detection.", "", "## Chunk target comparison", "", "The new extractor is run once per PDF. Chunk counts below compare 1800/250, 2800/350, and 3500/350 targets; the pypdf baseline uses 1800/250.", ""]
    for path in pdfs:
        old_pages, old_seconds = old_extract(path)
        md_pages, md_seconds = new_extract(path)
        report.extend((f"## {path.name}", "", "| Extractor / target | Pages | Characters | Chunks | Headings | Wall time (s) |", "|---|---:|---:|---:|---:|---:|"))
        variants = [("pypdf 1800/250", old_pages, old_seconds, False, 1800, 250)]
        variants.extend((f"PyMuPDF4LLM {size}/{overlap}", md_pages, md_seconds, True, size, overlap)
                        for size, overlap in ((1800, 250), (2800, 350), (3500, 350)))
        for name, pages, seconds, markdown, size, overlap in variants:
            chars = sum(len(text) for _, text in pages)
            report.append(f"| {name} | {len(pages)} | {chars} | {chunk_count(pages, markdown, size, overlap)} | {heading_count(pages)} | {seconds:.3f} |")
        report.extend(("", "### PyMuPDF4LLM sample", ""))
        for number, text in md_pages[:SAMPLE_PAGES]:
            report.extend((f"#### Source page {number}", "", text.strip() or "_(no text)_", ""))

    output = ROOT / "benchmark/pdf_extraction.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report).rstrip() + "\n", encoding="utf-8")
    print(output)
    print("\n".join(report[:4]))
    for line in report:
        if line.startswith("| pypdf") or line.startswith("| PyMuPDF4LLM"):
            print(line)


if __name__ == "__main__":
    main()
