"""Text-free classification of completed chat generation failures."""
import asyncio
import json

import pytest

from app import chat, db, models, retrieval


class ScriptedProvider:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def stream(self, messages):
        self.calls.append(messages)
        yield models.ModelEvent("text", text=self.outputs[len(self.calls) - 1])
        yield models.ModelEvent("metrics", metrics={"eval_count": 1})

    async def vram_bytes(self):
        return None


def validation_state():
    return {"citation_retry_used": False, "citation_retry_succeeded": False,
            "generation_attempts": 0, "validation_failures": []}


@pytest.mark.parametrize(("draft", "source_refs", "code", "attempts"), [
    ("Uncited claim", None, "no_valid_citations", 2),
    ("Claim [9]", None, "invalid_citation_ids", 2),
    ("Claim [1,]", None, "malformed_citation", 2),
    ("Claim [1].\n| Source | Evidence |\n|---|---|\n| S1 Alpha | none |",
     {"S1": {1}}, "uncited_source_row", 2),
    ("Claim [1].\n| Source | Evidence |\n|---|---|\n"
     "| S1 Alpha | Not reported in supplied evidence |",
     {"S1": {1}}, "uncited_source_row", 2),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [2] |\n| S2 Beta | [1] |",
     {"S1": {1}, "S2": {2}}, "invalid_citation_ids", 2),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [1] |",
     {"S1": {1}, "S2": {2}}, "missing_source_row", 1),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [1] |\n| S1 Alpha | [1] |",
     {"S1": {1}, "S2": {2}}, "duplicate_source_row", 1),
    ("Claim [1].", {"S1": {1}}, "source_completeness_failure", 1),
])
def test_final_attempt_reason_codes(monkeypatch, draft, source_refs, code, attempts):
    provider = ScriptedProvider([draft, draft])
    monkeypatch.setattr(models, "provider", provider)
    state = validation_state()

    async def consume():
        return [event async for event in chat._validated_stream(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
            {1, 2}, chat.RunMetrics(), "exhaustive_synthesis", state, source_refs)]

    expected = chat.SourceCompletenessError if attempts == 1 else chat.CitationValidationError
    with pytest.raises(expected):
        asyncio.run(consume())
    assert state["validation_failures"] == [
        {"stage": "exhaustive_synthesis", "attempt": index, "code": code}
        for index in range(1, attempts + 1)]
    assert len(provider.calls) == attempts
    assert draft not in json.dumps(state)


@pytest.mark.parametrize(("draft", "code"), [
    ("No supported fact", "no_valid_citations"),
    ("Fact [9]", "invalid_citation_ids"),
    ("X" * 801, "malformed_exhaustive_record"),
])
def test_map_attempt_reason_codes(monkeypatch, draft, code):
    provider = ScriptedProvider([draft, "Fact [1]"])
    monkeypatch.setattr(models, "provider", provider)
    state = validation_state()
    source = {"id": "source-id", "name": "Alpha"}
    chunk = {"id": "chunk-id", "source_id": "source-id", "source_name": "Alpha",
             "page": None, "text": "Fact"}
    asyncio.run(chat._exhaustive_context("One row per source", [(source, [chunk])],
                                         chat.RunMetrics(), [], state))
    assert state["validation_failures"] == [
        {"stage": "exhaustive_map", "attempt": 1, "code": code}]
    assert len(provider.calls) == 2


def test_failed_attempt_codes_are_persisted_without_drafts(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "notebook.db")
    monkeypatch.setattr(db, "FILES_DIR", tmp_path / "files")
    db.init_db()
    async def inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(chat, "run_in_threadpool", inline)
    provider = ScriptedProvider(["Rejected claim [9]", "Rejected claim [9]"])
    monkeypatch.setattr(models, "provider", provider)
    notebook_id, conversation_id, source_id, chunk_id = [db.uid() for _ in range(4)]
    stamp = db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO notebooks VALUES(?,?,?,?)", (notebook_id, "Test", stamp, stamp))
        conn.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conversation_id, notebook_id, stamp, stamp))
        conn.execute("INSERT INTO sources(id,notebook_id,name,kind,status,enabled,created_at,updated_at) "
                     "VALUES(?,?,?,'text','ready',1,?,?)", (source_id, notebook_id, "Alpha", stamp, stamp))
        conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                     (chunk_id, source_id, notebook_id, 0, None, None, "Fact", retrieval.pack_vector([1, 0])))
        conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (chunk_id, "Fact"))

    async def consume():
        return [json.loads(item) async for item in chat.stream_chat(
            notebook_id, conversation_id, "Fact?", [source_id], [])]

    events = asyncio.run(consume())
    assert events[-1]["reason"] == "citation_validation"
    assert "code" not in events[-1]
    coverage = db.row("SELECT coverage FROM generation_runs WHERE notebook_id=?", (notebook_id,))["coverage"]
    assert json.loads(coverage)["validation_failures"] == [
        {"stage": "chat", "attempt": 1, "code": "invalid_citation_ids"},
        {"stage": "chat", "attempt": 2, "code": "invalid_citation_ids"}]
    assert "Rejected claim" not in coverage
    assert "Fact?" not in coverage
    assert "Fact" not in coverage
