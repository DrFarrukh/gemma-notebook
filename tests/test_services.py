import asyncio
import json
import os
from pathlib import Path

import httpx

os.environ.setdefault("DATA_DIR", "/tmp/gemma-notebook-tests")

from app import services


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


def test_safe_filename_and_type_validation():
    assert services.safe_filename("../../odd:name.txt") == "odd_name.txt"
    assert services.source_kind("notes.MD") == "markdown"
    with pytest.raises(ValueError):
        services.source_kind("malware.exe")


def test_citations_include_page_and_excerpt():
    chunks = [{"id":"c1", "source_id":"s1", "source_name":"Paper", "page":3, "text":"Evidence" * 200}]
    citation = services.citations_for(chunks)[0]
    assert citation["number"] == 1
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
    assert requests[0][1]["options"]["num_ctx"] == int(os.getenv("NUM_CTX", "4096"))


import pytest
