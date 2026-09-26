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

    async def stream(self, messages, think=None):
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
    ("Claim [C9]", None, "invalid_citation_ids", 2),
    ("Claim [C1,]", None, "malformed_citation", 2),
    ("Claim [C1].\n| Source | Evidence |\n|---|---|\n| S1 Alpha | none |",
     {"S1": {1}}, "uncited_source_row", 2),
    ("Claim [C1].\n| Source | Evidence |\n|---|---|\n"
     "| S1 Alpha | Not reported in supplied evidence |",
     {"S1": {1}}, "uncited_source_row", 2),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C2] |\n| S2 Beta | [C1] |",
     {"S1": {1}, "S2": {2}}, "invalid_citation_ids", 2),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C1] |",
     {"S1": {1}, "S2": {2}}, "missing_source_row", 1),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C1] |\n| S1 Alpha | [C1] |",
     {"S1": {1}, "S2": {2}}, "duplicate_source_row", 1),
    ("Claim [C1].", {"S1": {1}}, "unparseable_source_rows", 1),
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
    assert [(item["stage"], item["attempt"], item["code"])
            for item in state["validation_failures"]] == [
        ("exhaustive_synthesis", index, code) for index in range(1, attempts + 1)]
    assert len(provider.calls) == attempts
    assert draft not in json.dumps(state)


@pytest.mark.parametrize(("draft", "code"), [
    ("No supported fact", "map_no_valid_citations"),
    ("Fact [C9]", "map_invalid_citation_ids"),
    ("", "map_empty_record"),
    ("Fact [C1,]", "map_malformed_citation"),
    ("X" * 2398 + "[C1]", "map_record_too_large"),
])
def test_map_attempt_reason_codes(monkeypatch, draft, code):
    provider = ScriptedProvider([draft, "Fact [C1]"])
    monkeypatch.setattr(models, "provider", provider)
    state = validation_state()
    source = {"id": "source-id", "name": "Alpha"}
    chunk = {"id": "chunk-id", "source_id": "source-id", "source_name": "Alpha",
             "page": None, "text": "Fact"}
    asyncio.run(chat._exhaustive_context("One row per source", [(source, [chunk])],
                                         chat.RunMetrics(), [], state))
    assert [(item["stage"], item["attempt"], item["code"])
            for item in state["validation_failures"]] == [("exhaustive_map", 1, code)]
    failure = state["validation_failures"][0]
    assert failure["output_char_count"] == len(draft)
    assert failure["output_char_limit"] == 2400
    assert failure["allowed_citation_count"] == 1
    assert len(provider.calls) == 2


def test_map_budget_accepts_valid_records_above_old_limit(monkeypatch):
    source = {"id": "source-id", "name": "Alpha"}
    chunk = {"id": "chunk-id", "source_id": "source-id", "source_name": "Alpha",
             "page": None, "text": "Fact"}
    for size in (801, 2400):
        record = "X" * (size - 4) + "[C1]"
        provider = ScriptedProvider([record])
        monkeypatch.setattr(models, "provider", provider)
        state = validation_state()
        _, messages, _ = asyncio.run(chat._exhaustive_context(
            "One row per source", [(source, [chunk])], chat.RunMetrics(), [], state))
        assert len(record) == size
        assert record in messages[-1]["content"]
        assert state["validation_failures"] == []
        assert len(provider.calls) == 1
        assert "COMPACT" in provider.calls[0][0]["content"]


def test_map_stream_safety_ceiling_remains_8192(monkeypatch):
    provider = ScriptedProvider(["X" * 8193, "Fact [C1]"])
    monkeypatch.setattr(models, "provider", provider)
    source = {"id": "source-id", "name": "Alpha"}
    chunk = {"id": "chunk-id", "source_id": "source-id", "source_name": "Alpha",
             "page": None, "text": "Fact"}
    state = validation_state()
    asyncio.run(chat._exhaustive_context("One row per source", [(source, [chunk])],
                                         chat.RunMetrics(), [], state))
    failure = state["validation_failures"][0]
    assert failure["code"] == "map_record_too_large"
    assert failure["output_char_count"] == 8193
    assert failure["output_char_limit"] == 8192


@pytest.mark.parametrize(("answer", "code", "counts"), [
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C1] |\n| S2 Beta | [C2] |",
     None, (2, 2, 0, 0, 0)),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C1] |",
     "missing_source_row", (1, 1, 1, 0, 0)),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C1] |\n| S1 Alpha | [C1] |",
     "duplicate_source_row", (2, 1, 1, 1, 0)),
    ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [C1] |\n| Beta | [C2] |",
     "unmatched_source_row", (2, 1, 1, 0, 1)),
    ("Source | Evidence\n--- | ---\nS1 Alpha | [C1]\nS2 Beta | [C2]",
     "unparseable_source_rows", (0, 0, 2, 0, 0)),
    ("  | Source | Evidence |  \n  | :--- | ---: |  \n  | S1 Alpha | [C1] |  \n  | S2 Beta | [C2] |",
     None, (2, 2, 0, 0, 0)),
])
def test_source_row_failure_counts(answer, code, counts):
    refs = {"S1": {1}, "S2": {2}}
    if code is None:
        chat._validate_source_rows(answer, refs)
        return
    with pytest.raises(chat.SourceCompletenessError) as caught:
        chat._validate_source_rows(answer, refs)
    assert caught.value.code == code
    details = caught.value.details
    assert (details["parsed_row_count"], details["matched_source_count"],
            details["missing_source_count"], details["duplicate_source_count"],
            details["unmatched_row_count"]) == counts
    assert details["expected_source_labels"] == ["S1", "S2"]


def test_failed_attempt_codes_are_persisted_without_drafts(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "notebook.db")
    monkeypatch.setattr(db, "FILES_DIR", tmp_path / "files")
    db.init_db()
    async def inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(chat, "run_in_threadpool", inline)
    provider = ScriptedProvider(["Rejected claim [C9]", "Rejected claim [C9]"])
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
