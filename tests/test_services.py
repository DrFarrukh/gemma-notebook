import asyncio
import json
import os
import sys
import types
from pathlib import Path

import httpx

os.environ.setdefault("DATA_DIR", "/tmp/gemma-notebook-tests")

from app import documents, services


def test_clean_text_normalizes_whitespace():
    assert services.clean_text("one   two\n\n\n\nthree") == "one two\n\nthree"


def test_chunking_preserves_page_metadata_and_overlap():
    text = "Sentence one. " * 220
    chunks = services.split_pages([{"page": 7, "text": text}])
    assert len(chunks) >= 2
    assert all(chunk["page"] == 7 for chunk in chunks)
    assert [chunk["ordinal"] for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0]["text"][-80:] in chunks[1]["text"]


def test_vector_roundtrip_and_cosine():
    values = [0.1, 0.2, 0.3]
    restored = services.unpack_vector(services.pack_vector(values))
    assert list(restored) == pytest.approx(values)
    assert services.cosine(values, values) == pytest.approx(1.0)
    assert services.cosine(values, [-x for x in values]) == pytest.approx(-1.0)


def test_text_and_html_extraction(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Alpha\n\nBeta", encoding="utf-8")
    pages, scanned = services.extract(text_file, "text")
    assert pages[0]["text"] == "Alpha\n\nBeta"
    assert not scanned

    html_file = tmp_path / "sample.html"
    html_file.write_text("<h1>Heading</h1><script>ignore()</script><p>Body</p>", encoding="utf-8")
    pages, _ = services.extract(html_file, "html")
    assert "Heading" in pages[0]["text"] and "Body" in pages[0]["text"]
    assert "ignore()" not in pages[0]["text"]


def test_pdf_extraction_returns_page_associated_markdown(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-placeholder")
    calls = []
    markdown_pages = [
        {"metadata": {"page": 2}, "text": "## II. Dataset\n\n" + "Body paragraph. " * 12},
        {"metadata": {"page": 3}, "text": "Continuation. " * 12},
    ]
    monkeypatch.setitem(sys.modules, "pymupdf4llm", types.SimpleNamespace(
        to_markdown=lambda *args, **kwargs: calls.append((args, kwargs)) or markdown_pages
    ))

    pages, scanned = services.extract(pdf, "pdf")

    assert [page["page"] for page in pages] == [2, 3]
    assert pages[0]["text"].startswith("## II. Dataset")
    assert not scanned
    assert calls[0][1] == {"page_chunks": True}
    assert [chunk["page"] for chunk in services.split_pages(pages, preserve_sections=True)] == [2, 3]


def test_pdf_markdown_heading_sections_change_deterministically():
    chunks = services.split_pages([
        {"page": 4, "text": "## II. Dataset\n\nDataset body.\n\n### A. Longitudinal Dataset\n\nStudy body."},
        {"page": 5, "text": "Continuation on next page."},
        {"page": 6, "text": "## III. Architecture\n\nModel body."},
    ], preserve_sections=True)

    assert chunks[0]["page"] == 4
    assert chunks[0]["section"] == "II. Dataset"
    assert "## II. Dataset" in chunks[0]["text"]
    assert any(chunk["section"] == "II. Dataset > A. Longitudinal Dataset" and chunk["page"] == 4
               for chunk in chunks)
    assert any(chunk["section"] == "II. Dataset > A. Longitudinal Dataset" and chunk["page"] == 5
               for chunk in chunks)
    assert any(chunk["section"] == "III. Architecture" and chunk["page"] == 6
               for chunk in chunks)


def test_pdf_heading_normalization_marks_generic_academic_sections():
    normalized = documents.normalize_pdf_headings(
        "I. INTRODUCTION\n\n_A._ _Experimental Dataset and Setup_\n\nB. Mel-Spectrogram-Based Feature Extraction\n\nANALYSIS OF EXISTING DATASETS\n\n_1)_ _First Dataset:_ Dataset details follow.\n# **E**"
    )
    assert normalized.splitlines() == [
        "## I. INTRODUCTION", "", "### A. Experimental Dataset and Setup", "",
        "### B. Mel-Spectrogram-Based Feature Extraction", "", "### ANALYSIS OF EXISTING DATASETS", "",
        "#### 1. First Dataset:", "Dataset details follow.", "E"
    ]


def test_pdf_plain_text_chunks_without_a_section():
    chunks = services.split_pages([{"page": 9, "text": "Plain paragraph. " * 220}], preserve_sections=True)
    assert len(chunks) > 1
    assert all(chunk["page"] == 9 and chunk["section"] is None for chunk in chunks)


def test_pdf_oversized_paragraph_respects_hard_cap_with_overlap():
    text = "x" * 8500
    chunks = services.split_pages([{"page": 10, "text": text}], preserve_sections=True)
    assert all(len(chunk["text"]) <= 4000 for chunk in chunks)
    assert chunks[0]["text"][-350:] in chunks[1]["text"]


def test_pdf_extraction_empty_and_broken_results_fail_safely(tmp_path, monkeypatch):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-placeholder")
    monkeypatch.setitem(sys.modules, "pymupdf4llm", types.SimpleNamespace(to_markdown=lambda *a, **k: []))
    with pytest.raises(ValueError, match="No readable text"):
        services.extract(pdf, "pdf")
    monkeypatch.setitem(sys.modules, "pymupdf4llm", types.SimpleNamespace(
        to_markdown=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken PDF"))
    ))
    with pytest.raises(ValueError, match=r"PDF Markdown extraction failed \(RuntimeError\)"):
        services.extract(pdf, "pdf")


def test_non_pdf_extraction_and_chunk_sections_remain_unchanged(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("## Heading\n\nBody", encoding="utf-8")
    pages, scanned = services.extract(path, "markdown")
    chunks = services.split_pages(pages)
    assert pages == [{"page": None, "text": "## Heading\n\nBody"}]
    assert not scanned
    assert chunks == [{"ordinal": 0, "page": None, "text": "## Heading\n\nBody"}]


def test_safe_filename_and_type_validation():
    assert services.safe_filename("../../odd:name.txt") == "odd_name.txt"
    assert services.source_kind("notes.MD") == "markdown"
    with pytest.raises(ValueError):
        services.source_kind("malware.exe")


def test_citations_include_page_and_excerpt():
    chunks = [{"id":"c1", "source_id":"s1", "source_name":"Paper", "page":3, "text":"Evidence" * 200}]
    citation = services.citations_for(chunks)[0]
    assert citation["number"] == 1
    assert citation["namespace"] == "C"
    assert citation["page"] == 3
    assert len(citation["excerpt"]) == 700


def test_ollama_stream_uses_configured_model_and_context(monkeypatch):
    assert services.GENERATION_MODEL == os.getenv("GENERATION_MODEL", "gemma4:e4b")
    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, text='{"message":{"content":"ok"}}\n{"done":true,"eval_count":1}\n')

    async_client = httpx.AsyncClient
    monkeypatch.setattr(services.httpx, "AsyncClient", lambda **kwargs: async_client(
        **kwargs, transport=httpx.MockTransport(respond)
    ))

    async def collect():
        return [piece async for piece in services.ollama_stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(collect()) == ["ok"]
    assert requests[0][0] == "/api/chat"
    assert requests[0][1]["model"] == services.GENERATION_MODEL
    assert requests[0][1]["options"]["num_ctx"] == int(os.getenv("NUM_CTX", "16384"))
    assert requests[0][1]["options"]["temperature"] == float(os.getenv("TEMPERATURE", "0.2"))


import pytest
