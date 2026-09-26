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
from pypdf import PdfReader

from . import db, models

MAX_CHARS = 2_000_000
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250


def clean_text(text):
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract(path: Path, kind: str):
    suffix = path.suffix.lower()
    pages = []
    scanned = False
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        for index, page in enumerate(reader.pages, 1):
            pages.append({"page": index, "text": clean_text(page.extract_text() or "")})
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


def split_pages(pages):
    chunks = []
    ordinal = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append({"ordinal": ordinal, "page": page["page"], "text": piece})
                ordinal += 1
            if end >= len(text):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
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
        chunks = split_pages(pages)
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
                             (chunk_id, source_id, source["notebook_id"], chunk["ordinal"], chunk["page"], None, chunk["text"], struct.pack(f"<{len(vector)}f", *vector)))
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
