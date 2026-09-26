"""Local document extraction, chunking and atomic ingestion."""
import csv
import io
import re
import struct
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from odf import teletype
from odf.opendocument import load as load_odf

from . import db, models

MAX_CHARS = 2_000_000
CHUNK_SIZE = 2800
CHUNK_OVERLAP = 350
CHUNK_HARD_CAP = 4000
LEGACY_CHUNK_SIZE = 1800
LEGACY_CHUNK_OVERLAP = 250


def clean_text(text):
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_pdf_headings(text):
    """Mark conventional, visually formatted document headings as Markdown."""
    output = []
    roman_heading = re.compile(r"^([IVXLCDM]+)\.\s+([A-Z][A-Z0-9 /,&:;()'’\-]{2,})$")
    previous_nonempty = ""
    for line in text.splitlines():
        plain = re.sub(r"[*_`]+", "", line).strip()
        existing_heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if existing_heading:
            title = re.sub(r"[*_`]+", "", existing_heading.group(1)).strip()
            output.append(line if len(title) >= 3 else title)
            continue
        roman = roman_heading.match(plain)
        emphasized = line.lstrip().startswith(("_", "*")) or line.rstrip().endswith(("_", "*"))
        alpha = re.match(r"^([A-Z])\.\s+(.{4,})$", plain)
        alpha_title = alpha.group(2).strip() if alpha else ""
        alpha_heading = alpha and (emphasized or (
            len(alpha_title.split()) <= 12 and not alpha_title.endswith((".", ",", ";"))
        ))
        uppercase_heading = re.match(r"^[A-Z][A-Z &()'’\-]{8,}$", plain)
        if uppercase_heading and (not 2 <= len(plain.split()) <= 12
                                  or re.match(r"^TABLE\s+\w+", previous_nonempty, re.I)):
            uppercase_heading = None
        numbered = re.match(r"^(\d+)[.)]\s+(.{4,}?[:.])(?:\s+(.+))?$", plain) if emphasized else None
        if roman:
            output.append(f"## {roman.group(1)}. {roman.group(2).strip()}")
        elif alpha_heading:
            output.append(f"### {alpha.group(1)}. {alpha.group(2).strip()}")
        elif numbered:
            output.append(f"#### {numbered.group(1)}. {numbered.group(2).strip()}")
            if numbered.group(3):
                output.append(numbered.group(3).strip())
        elif uppercase_heading:
            output.append(f"### {plain}")
        else:
            output.append(line)
        if plain:
            previous_nonempty = plain
    return "\n".join(output)


def extract(path: Path, kind: str):
    suffix = path.suffix.lower()
    pages = []
    scanned = False
    if suffix == ".pdf":
        # Keep the import local so non-PDF extraction does not depend on the
        # PDF conversion stack at module import time. The core package is used
        # without its optional Layout/OCR modules to retain scanned-PDF behavior.
        try:
            import pymupdf4llm
            converted = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        except Exception as exc:
            raise ValueError(f"PDF Markdown extraction failed ({type(exc).__name__})") from exc
        if not isinstance(converted, list):
            raise ValueError("PDF Markdown extraction returned an invalid result")
        for item in converted:
            try:
                metadata = item["metadata"]
                # PyMuPDF4LLM 0.x used metadata.page; 1.x names the same
                # 1-based value page_number.
                page_number = int(metadata.get("page_number", metadata.get("page")))
                markdown = item["text"]
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError("PDF Markdown extraction returned invalid page data") from exc
            if page_number < 1 or not isinstance(markdown, str):
                raise ValueError("PDF Markdown extraction returned invalid page data")
            pages.append({"page": page_number, "text": normalize_pdf_headings(clean_text(markdown))})
        total = sum(len(p["text"]) for p in pages)
        scanned = bool(pages and total < max(80, len(pages) * 30))
    elif suffix == ".docx":
        document = Document(str(path))
        pages = [{"page": None, "text": clean_text("\n\n".join(p.text for p in document.paragraphs if p.text.strip()))}]
    elif suffix == ".odt":
        document = load_odf(str(path))
        pages = [{"page": None, "text": clean_text(teletype.extractText(document.text))}]
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix in {".html", ".htm"}:
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            raw = soup.get_text("\n")
        elif suffix == ".csv":
            raw = "\n".join(" | ".join(cell.strip() for cell in row) for row in csv.reader(io.StringIO(raw)))
        pages = [{"page": None, "text": clean_text(raw)}]
    combined = "\n\n".join(p["text"] for p in pages)
    if len(combined) > MAX_CHARS:
        raise ValueError(f"Extracted text exceeds the {MAX_CHARS:,}-character source limit")
    if not combined and not scanned:
        raise ValueError("No readable text was found in this source")
    return pages, scanned


def split_pages(pages, preserve_sections=False, chunk_size=None, overlap=None, hard_cap=None):
    chunks = []
    ordinal = 0
    if not preserve_sections:
        chunk_size = LEGACY_CHUNK_SIZE
        overlap = LEGACY_CHUNK_OVERLAP
        for page in pages:
            text = page["text"]
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text):
                    boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
                    if boundary > start + chunk_size // 2:
                        end = boundary + 1
                piece = text[start:end].strip()
                if piece:
                    chunks.append({"ordinal": ordinal, "page": page["page"], "text": piece})
                    ordinal += 1
                if end >= len(text):
                    break
                start = max(start + 1, end - overlap)
        return chunks

    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap if overlap is not None else CHUNK_OVERLAP
    hard_cap = hard_cap or CHUNK_HARD_CAP
    chunk_size = min(chunk_size, hard_cap)

    heading_stack = []
    for page in pages:
        text = page["text"]
        # Treat Markdown headings as deterministic section boundaries. Build
        # blocks first so ordinary paragraph boundaries remain preferred.
        blocks = []
        current = []
        for line in text.splitlines():
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading and len(re.sub(r"[*_`]+", "", heading.group(2)).strip()) < 3:
                heading = None
            if heading:
                if current:
                    blocks.append(("\n".join(current).strip(), None))
                    current = []
                level = len(heading.group(1))
                title = re.sub(r"[*_`]+", "", heading.group(2)).strip()
                blocks.append((line.strip(), (level, title)))
            elif not line.strip():
                if current:
                    blocks.append(("\n".join(current).strip(), None))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append(("\n".join(current).strip(), None))

        # Paragraphs inherit the most recent Markdown heading, including
        # headings established on prior pages.
        page_chunks = []
        for block, section in blocks:
            if section:
                level, title = section
                heading_stack = heading_stack[:level - 1]
                while len(heading_stack) < level - 1:
                    heading_stack.append(None)
                heading_stack.append(title)
                page_chunks.append((block, " > ".join(item for item in heading_stack if item)))
            else:
                page_chunks.append((block, " > ".join(item for item in heading_stack if item) or None))

        pending_text, pending_section = "", None
        for block, section in page_chunks:
            if not block:
                continue
            if pending_text and (section != pending_section or len(pending_text) + 2 + len(block) > chunk_size):
                chunks.append({"ordinal": ordinal, "page": page["page"],
                               "section": pending_section, "text": pending_text})
                ordinal += 1
                # Carry a small same-section overlap across chunks. A heading
                # change starts a clean chunk with the new section.
                if section == pending_section:
                    pending_text = pending_text[-overlap:]
                else:
                    pending_text = ""
            if not pending_text:
                pending_section = section
            pending_text = f"{pending_text}\n\n{block}".strip() if pending_text else block

            while len(pending_text) > chunk_size:
                end = min(chunk_size, len(pending_text))
                boundary = max(pending_text.rfind("\n\n", 0, end), pending_text.rfind(". ", 0, end))
                if boundary > chunk_size // 2:
                    end = boundary + 1
                else:
                    # If this is one unusually long paragraph with no useful
                    # boundary near the target, split only at the hard cap.
                    end = min(hard_cap, len(pending_text))
                piece = pending_text[:end].strip()
                chunks.append({"ordinal": ordinal, "page": page["page"],
                               "section": pending_section, "text": piece})
                ordinal += 1
                pending_text = pending_text[max(end - overlap, end if end == len(pending_text) else 0):].strip()
        if pending_text:
            chunks.append({"ordinal": ordinal, "page": page["page"],
                           "section": pending_section, "text": pending_text})
            ordinal += 1
    return chunks


def source_kind(filename):
    kinds = {".pdf": "pdf", ".docx": "docx", ".odt": "odt", ".txt": "text",
             ".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html", ".csv": "csv"}
    suffix = Path(filename).suffix.lower()
    if suffix not in kinds:
        raise ValueError("Unsupported file type. Use PDF, DOCX, ODT, TXT, Markdown, HTML, or CSV.")
    return kinds[suffix]


def safe_filename(name):
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name).strip(" .")
    return cleaned[:180] or "source.txt"


def process_source(source_id):
    source = db.row("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source:
        return
    with db.connection() as conn:
        conn.execute("UPDATE sources SET status='processing', error=NULL, updated_at=? WHERE id=?", (db.now(), source_id))
        db.touch_notebook(conn, source["notebook_id"])
    try:
        pages, scanned = extract(Path(source["path"]), source["kind"])
        if scanned:
            with db.connection() as conn:
                conn.execute("UPDATE sources SET status='error', scanned=1, error=?, page_count=?, updated_at=? WHERE id=?",
                             ("This appears to be an image-only PDF. Local OCR is not enabled in v1.", len(pages), db.now(), source_id))
                db.touch_notebook(conn, source["notebook_id"])
            return
        chunks = split_pages(pages, preserve_sections=source["kind"] == "pdf")
        vectors = []
        dimension = None
        embedding_digest = models.embedding_model_digest()
        for offset in range(0, len(chunks), 24):
            batch = models.provider.embed([c["text"] for c in chunks[offset:offset + 24]])
            dimension = models.validate_vectors(batch, len(chunks[offset:offset + 24]), dimension)
            vectors.extend(batch)
        models.validate_vectors(vectors, len(chunks), dimension)
        if embedding_digest != models.embedding_model_digest():
            embedding_digest = None
        with db.connection() as conn:
            conn.execute("DELETE FROM chunk_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id=?)", (source_id,))
            conn.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            for chunk, vector in zip(chunks, vectors):
                chunk_id = db.uid()
                conn.execute("INSERT INTO chunks(id,source_id,notebook_id,ordinal,page,section,text,embedding) VALUES(?,?,?,?,?,?,?,?)",
                             (chunk_id, source_id, source["notebook_id"], chunk["ordinal"], chunk["page"], chunk.get("section"), chunk["text"], struct.pack(f"<{len(vector)}f", *vector)))
                if embedding_digest:
                    conn.execute("INSERT INTO chunk_embeddings VALUES(?,?,?,?,?)",
                                 (chunk_id, models.EMBEDDING_MODEL, embedding_digest,
                                  len(vector), struct.pack(f"<{len(vector)}f", *vector)))
                conn.execute("INSERT INTO chunk_fts(chunk_id,text) VALUES(?,?)", (chunk_id, chunk["text"]))
            count = sum(len(p["text"]) for p in pages)
            conn.execute("UPDATE sources SET status='ready', scanned=0, error=NULL, char_count=?, page_count=?, updated_at=? WHERE id=?",
                         (count, len(pages) if source["kind"] == "pdf" else None, db.now(), source_id))
            db.touch_notebook(conn, source["notebook_id"])
    except Exception as exc:
        with db.connection() as conn:
            conn.execute("UPDATE sources SET status='error', error=?, updated_at=? WHERE id=?",
                         (models.safe_error(exc) if not isinstance(exc, ValueError) else str(exc)[:1000], db.now(), source_id))
            db.touch_notebook(conn, source["notebook_id"])
