import asyncio
import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from app import chat, db, documents, main, models, retrieval


class FakeProvider:
    def __init__(self, answers=None, fail_at=None):
        self.answers = answers or ["Evidence [1]"]
        self.fail_at = fail_at
        self.calls = []

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    async def stream(self, messages):
        self.calls.append(messages)
        index = len(self.calls)
        if index == self.fail_at:
            yield models.ModelEvent("text", text="partial ")
            raise RuntimeError("secret upstream exception")
        answer = self.answers[min(index - 1, len(self.answers) - 1)]
        yield models.ModelEvent("text", text=answer)
        yield models.ModelEvent("metrics", metrics={"eval_count": 8, "total_duration": 42})

    async def health(self):
        return {"app": "ok", "ollama": True}

    async def vram_bytes(self):
        return 1234


class ScriptedProvider(FakeProvider):
    def __init__(self, answers):
        super().__init__()
        self.script = answers

    async def stream(self, messages):
        self.calls.append(messages)
        answer = self.script[len(self.calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        yield models.ModelEvent("text", text=answer)
        yield models.ModelEvent("metrics", metrics={"eval_count": 8})


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "notebook.db")
    monkeypatch.setattr(db, "FILES_DIR", tmp_path / "files")
    db.init_db()
    fake = FakeProvider()
    monkeypatch.setattr(models, "provider", fake)
    with TestClient(main.app) as client:
        notebook = client.post("/api/notebooks", json={"title": "Research"}).json()
        yield client, notebook["id"], fake, tmp_path


def events(response):
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


def add_source(nid, text, name="Paper", vector=(1, 0), status="ready", enabled=1):
    sid, cid, stamp = db.uid(), db.uid(), db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO sources(id,notebook_id,name,kind,status,enabled,created_at,updated_at) VALUES(?,?,?,'text',?,?,?,?)",
                     (sid, nid, name, status, enabled, stamp, stamp))
        conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", (cid, sid, nid, 0, None, None, text, retrieval.pack_vector(vector)))
        conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (cid, text))
    return sid, cid


def test_file_headers_and_activity(env):
    client, nid, fake, root = env
    stamp = db.row("SELECT updated_at FROM notebooks WHERE id=?", (nid,))["updated_at"]
    note = client.post(f"/api/notebooks/{nid}/notes", json={"title": "n"}).json()
    assert db.row("SELECT updated_at FROM notebooks WHERE id=?", (nid,))["updated_at"] >= stamp
    client.put(f"/api/notes/{note['id']}", json={"title": "updated", "content": "text"})
    client.delete(f"/api/notes/{note['id']}")
    for name, typ, disposition in [("attack.html", "application/octet-stream", "attachment"),
                                   ("attack.txt", "text/plain", "attachment"),
                                   ("scan.pdf", "application/pdf", "inline")]:
        path = root / name
        path.write_bytes(b"<script>alert('x')</script>")
        sid = db.uid()
        db.execute("INSERT INTO sources(id,notebook_id,name,kind,path,status,created_at,updated_at) VALUES(?,?,?,?,?,'ready',?,?)",
                   (sid, nid, name, "html", str(path), db.now(), db.now()))
        response = client.get(f"/api/files/{sid}")
        assert response.headers["content-type"].startswith(typ)
        assert disposition in response.headers["content-disposition"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "sandbox" in response.headers["content-security-policy"]
    assert client.get("/api/files/missing").status_code == 404


def test_retrieval_scope_rrf_rescue_and_budget(env, monkeypatch):
    client, nid, fake, root = env
    sid, cid = add_source(nid, "rarequartz mechanism", vector=(0, 1))
    sid2, cid2 = add_source(nid, "unrelated", vector=(1, 0))
    add_source(nid, "rarequartz disabled", status="queued")
    assert retrieval.retrieve(nid, "rarequartz", []) == []
    assert [c["id"] for c in retrieval.retrieve(nid, "rarequartz", [sid])] == [cid]
    assert cid in [c["id"] for c in retrieval.retrieve(nid, "rarequartz")]
    assert retrieval.retrieve(nid, "a the to", [sid]) == []
    assert retrieval.retrieve(nid, "meaningless unrelated", [sid]) == []
    assert cid2 in [c["id"] for c in retrieval.retrieve(nid, "semantic only", [sid2])]
    assert retrieval.retrieve(nid, "rarequartz", ["other-notebook"]) == []
    assert retrieval.budgeted([{"id": "x", "source_id": sid, "source_name": "a", "page": None, "text": "z" * 80}], 20) == []
    assert all(c["score"] > 0 for c in retrieval.retrieve(nid, "rarequartz"))


def test_retrieval_intent_and_bounded_followup_query(env):
    _, nid, fake, _ = env
    add_source(nid, "Asim Waris reports a result")
    embedded = []
    fake.embed = lambda texts: embedded.extend(texts) or [[1.0, 0.0]]
    history = ["Old unrelated question", "What does Asim Waris report?"]
    assert retrieval.retrieval_mode("What classifier was used in paper X?") == "focused"
    assert retrieval.retrieval_mode("What accuracy did the SSAE achieve between days?") == "focused"
    assert retrieval.retrieval_mode("What are the main themes across these sources?") == "coverage"
    assert retrieval.retrieval_mode("How is each author connected with the others across these papers?") == "coverage"
    assert retrieval.retrieval_mode("Where do all of these sources agree and disagree?") == "coverage"
    assert retrieval.contextualize_retrieval_query("What classifier was used in paper X?", history) == "What classifier was used in paper X?"
    retrieval.retrieve(nid, "What classifier was used in paper X?", prior_user_questions=history)
    assert embedded[-1] == "What classifier was used in paper X?"
    retrieval.retrieve(nid, "And what about him?", prior_user_questions=["Too old", *history])
    assert embedded[-1].startswith("And what about him?")
    assert "What does Asim Waris report?" in embedded[-1]
    assert "Old unrelated question" in embedded[-1]
    assert "Too old" not in embedded[-1]
    assert "Asim Waris" in retrieval.contextualize_retrieval_query("About Zia?", history)


def test_explicit_new_entity_keeps_lexical_priority(env):
    _, nid, _, _ = env
    add_source(nid, "Asim Waris and I.K. Niazi collaborated")
    _, farrukh = add_source(nid, "Farrukh collaborated on the project")
    chosen = retrieval.retrieve(nid, "And Farrukh?", prior_user_questions=["How are Asim Waris and I.K. Niazi connected?"])
    assert chosen[0]["id"] == farrukh


def test_coverage_rounds_budget_and_source_scope(env, monkeypatch):
    _, nid, _, _ = env
    ids = []
    for name in ("Alpha", "Beta", "Gamma"):
        sid, first = add_source(nid, f"{name} methods and dataset first finding")
        second = db.uid()
        text = f"{name} methods and dataset second distinct finding"
        with db.connection() as conn:
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (second, sid, nid, 1, None, None, text, retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (second, text))
        ids.append((sid, first, second))
    disabled, _ = add_source(nid, "Disabled methods and dataset", enabled=0)
    weak, _ = add_source(nid, "Unrelated content", vector=(-1, 0))
    query = "What methods and datasets were used across the papers?"
    chosen = retrieval.retrieve(nid, query)
    assert len(chosen) == 6
    assert {c["source_id"] for c in chosen[:3]} == {sid for sid, _, _ in ids}
    assert {c["source_id"] for c in chosen[3:]} == {sid for sid, _, _ in ids}
    assert all(sum(c["source_id"] == sid for c in chosen) == 2 for sid, _, _ in ids)
    assert disabled not in {c["source_id"] for c in chosen}
    assert weak not in {c["source_id"] for c in chosen}
    selected = retrieval.retrieve(nid, query, source_ids=[ids[0][0], disabled, weak])
    assert {c["source_id"] for c in selected} == {ids[0][0]}
    focused = retrieval.retrieve(nid, "methods", source_ids=[ids[0][0], disabled])
    assert {c["source_id"] for c in focused} == {ids[0][0]}
    assert retrieval.retrieve(nid, query, source_ids=[]) == []
    monkeypatch.setattr(retrieval, "EVIDENCE_BUDGET", len(retrieval.evidence_text(chosen[:3])))
    budgeted = retrieval.retrieve(nid, query)
    assert len(retrieval.evidence_text(budgeted)) <= retrieval.EVIDENCE_BUDGET
    assert {c["source_id"] for c in budgeted} == {sid for sid, _, _ in ids}


def test_coverage_can_exceed_focused_chunk_limit(env):
    _, nid, _, _ = env
    sources = []
    for index in range(7):
        sid, _ = add_source(nid, f"Study {index} methods first finding")
        second = db.uid()
        text = f"Study {index} methods second finding"
        with db.connection() as conn:
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (second, sid, nid, 1, None, None, text, retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (second, text))
        sources.append(sid)
    chosen = retrieval.retrieve(nid, "Compare methods across the studies")
    assert len(chosen) == 14
    assert set(sources) == {c["source_id"] for c in chosen[:7]}
    assert set(sources) == {c["source_id"] for c in chosen[7:]}


def test_chat_passes_bounded_user_history_to_retrieval(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Asim Waris and Farrukh collaborated")
    conversation = db.row("SELECT id FROM conversations WHERE notebook_id=?", (nid,))["id"]
    questions = ["First old question", "Second old question", "What does Asim Waris report?"]
    with db.connection() as conn:
        for question in questions:
            conn.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)", (db.uid(), conversation, "user", question, "[]", db.now()))
            conn.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)", (db.uid(), conversation, "assistant", "Prior assistant answer is not evidence", "[]", db.now()))
    original = retrieval.retrieve
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs["prior_user_questions"])
        return original(*args, **kwargs)
    monkeypatch.setattr(retrieval, "retrieve", capture)
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "And what about him?"}))[-1]["type"] == "done"
    assert seen == [questions[-2:]]
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["coverage"]["retrieval_mode"] == "focused"
    assert run["coverage"]["contextualized_retrieval"] is True
    assert run["coverage"]["represented_sources"] == 1
    assert sum(run["coverage"]["chunks_per_source"].values()) == 1
    assert not any(text in json.dumps(run) for text in questions + ["And what about him?", "Prior assistant answer is not evidence", "Asim Waris and Farrukh collaborated"])


@pytest.mark.parametrize("answer", ["A fact [1].", "Linked facts [1] and [2, 3]."])
def test_valid_citations_complete_without_retry(env, monkeypatch, answer):
    client, nid, _, _ = env
    for name in ("Alpha", "Beta", "Gamma"):
        add_source(nid, f"{name} evidence")
    provider = ScriptedProvider([answer])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "What evidence?"}))
    assert [event["type"] for event in received] == ["citations", "delta", "done"]
    assert len(provider.calls) == 1
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["coverage"]["citation_retry_used"] is False
    assert run["coverage"]["generation_attempts"] == 1


@pytest.mark.parametrize("first", ["Only prose (citation 2).", "```mermaid\nA --> B [1]\n```", "Wrong current source [99]."])
def test_citation_retry_reuses_evidence_and_persists_only_valid_answer(env, monkeypatch, first):
    client, nid, _, _ = env
    add_source(nid, "Alpha supported claim")
    add_source(nid, "Beta supported claim")
    provider = ScriptedProvider([first, "Supported answer [1, 2]."])
    monkeypatch.setattr(models, "provider", provider)
    original = retrieval.retrieve
    retrieval_calls = []
    def capture(*args, **kwargs):
        retrieval_calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(retrieval, "retrieve", capture)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "What is supported?"}))
    assert [event["type"] for event in received] == ["citations", "delta", "retry", "delta", "done"]
    assert received[2]["reason"] == "citation_validation"
    assert len(retrieval_calls) == 1
    assert len(provider.calls) == 2
    assert provider.calls[0][-1]["content"] == provider.calls[1][-1]["content"]
    assert "previous response was rejected" in provider.calls[1][0]["content"]
    assert first not in json.dumps(provider.calls[1])
    saved = client.get(f"/api/notebooks/{nid}/messages").json()
    assert [item["role"] for item in saved] == ["user", "assistant"]
    assert saved[1]["content"] == "Supported answer [1, 2]."
    assert first not in json.dumps(saved)
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["coverage"]["citation_retry_used"] is True
    assert run["coverage"]["citation_retry_succeeded"] is True
    assert run["coverage"]["generation_attempts"] == 2
    assert run["metrics"]["stage_calls"] == {"chat": 1, "chat_citation_retry": 1}
    assert first not in json.dumps(run)


def test_citation_retry_stops_after_two_completed_invalid_answers(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Supported claim")
    provider = ScriptedProvider(["Draft (citation 1)", "Still only source 1 prose"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "Claim?"}))
    assert [event["type"] for event in received] == ["citations", "delta", "retry", "delta", "error"]
    assert received[-1]["message"] == chat.CITATION_REJECTION
    assert received[-1]["reason"] == "citation_validation"
    assert len(provider.calls) == 2
    assert client.get(f"/api/notebooks/{nid}/messages").json() == []
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["coverage"]["citation_retry_used"] is True
    assert run["coverage"]["citation_retry_succeeded"] is False
    assert run["coverage"]["validation_failures"] == [
        {"stage": "chat", "attempt": 1, "code": "no_valid_citations"},
        {"stage": "chat", "attempt": 2, "code": "no_valid_citations"}]


def test_transport_error_does_not_retry_citation_generation(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Supported claim")
    provider = ScriptedProvider([RuntimeError("transport interrupted")])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "Claim?"}))
    assert [event["type"] for event in received] == ["citations", "error"]
    assert len(provider.calls) == 1
    assert client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]["coverage"]["citation_retry_used"] is False


def test_exhaustive_router_is_narrow():
    for question in ("Create a complete table for all papers",
                     "Create a comprehensive table across all 11 papers",
                     "One row per source", "Compare every paper",
                     "List each study and its accuracy",
                     "For each source, give model, classes, dataset and accuracy",
                     "Make a table covering all selected sources",
                     "Summarize each of the 11 studies"):
        assert retrieval.retrieval_mode(question) == "exhaustive"
    for question in ("What does paper X report?", "What accuracy did SSAE achieve?",
                     "Compare LDA and SSAE"):
        assert retrieval.retrieval_mode(question) == "focused"
    assert retrieval.retrieval_mode("What are the main themes across these sources?") == "coverage"


def test_diagnosis_one_source_cited_sentence(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Alpha reports a measured result", name="Alpha")
    provider = ScriptedProvider(["Alpha reports a measured result [1]."])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "What does Alpha report?"}))
    assert [event["type"] for event in received] == ["citations", "delta", "done"]
    assert len(provider.calls) == 1
    assert client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]["coverage"]["retrieval_mode"] == "focused"


def test_diagnosis_one_source_one_cited_table_row(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Alpha reports a measured result", name="Alpha")
    provider = ScriptedProvider(["Alpha result [1].",
                                 "| Source | Result | Evidence |\n|---|---|---|\n| S1 Alpha | Measured result | [1] |"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert [event["type"] for event in received] == ["citations", "status", "delta", "done"]
    assert len(provider.calls) == 2
    assert client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]["coverage"]["retrieval_mode"] == "exhaustive"


def test_diagnosis_two_sources_two_cited_table_rows(env, monkeypatch):
    client, nid, _, _ = env
    first, _ = add_source(nid, "Alpha reports result A", name="Alpha")
    second, _ = add_source(nid, "Beta reports result B", name="Beta")
    for index, sid in enumerate((first, second)):
        db.execute("UPDATE sources SET created_at=? WHERE id=?", (f"2026-01-01T00:00:{index:02d}Z", sid))
    provider = ScriptedProvider(["Alpha result [1].", "Beta result [2].",
                                 "| Source | Result | Evidence |\n|---|---|---|\n"
                                 "| S1 Alpha | Result A | [1] |\n| S2 Beta | Result B | [2] |"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert [event["type"] for event in received] == ["citations", "status", "delta", "done"]
    assert [item["number"] for item in received[0]["citations"]] == [1, 2]
    assert len(provider.calls) == 3
    final_input = provider.calls[-1][-1]["content"]
    assert '"citation": 1' in final_input and '"citation": 2' in final_input


def test_diagnosis_parser_and_row_failure_categories():
    assert chat.references("(citation 1) and source 1", {1}) == set()
    assert chat.references("**Supported [1, 2]** and [1-2]", {1, 2}) == {1, 2}
    assert chat.references("| S1 Alpha | claim | [1] |", {1}) == {1}
    with pytest.raises(ValueError, match="Citation outside current evidence"):
        chat.references("Unsupported [3]", {1, 2})
    with pytest.raises(ValueError, match="Malformed numeric citation"):
        chat.references("Malformed [1,]", {1})
    row = "| Source | Claim | Evidence |\n|---|---|---|\n| S1 Alpha | claim | [1] |"
    chat._validate_source_rows(row, {"S1": {1}})
    with pytest.raises(chat.CitationValidationError):
        chat._validate_source_rows("Intro [1]\n" + row.replace("| [1] |", "| empty |"), {"S1": {1}})
    with pytest.raises(chat.SourceCompletenessError):
        chat._validate_source_rows(row.replace("S1 Alpha", "Alpha"), {"S1": {1}})


def test_diagnosis_valid_global_citation_but_uncited_table_row_is_rejected(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Alpha reports a measured result", name="Alpha")
    uncited_row = ("Intro grounded in current evidence [1].\n"
                   "| Source | Result | Evidence |\n|---|---|---|\n"
                   "| S1 Alpha | Measured result | empty |")
    provider = ScriptedProvider(["Alpha result [1].", uncited_row, uncited_row])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert [event["type"] for event in received] == ["citations", "status", "delta", "retry", "delta", "error"]
    assert received[-1]["message"] == chat.CITATION_REJECTION
    assert chat.references(uncited_row, {1}) == {1}
    assert len(provider.calls) == 3


def test_diagnosis_map_to_final_citation_numbers_remain_stable(env, monkeypatch):
    client, nid, _, _ = env
    sources = []
    for index, name in enumerate(("Alpha", "Beta")):
        sid, _ = add_source(nid, f"{name} first result", name=name)
        db.execute("UPDATE sources SET created_at=? WHERE id=?", (f"2026-01-01T00:00:{index:02d}Z", sid))
        second = db.uid()
        text = f"{name} second distinct result"
        with db.connection() as conn:
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (second, sid, nid, 1, None, None, text, retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (second, text))
        sources.append(sid)
    provider = ScriptedProvider(["Alpha second result [2].", "Beta second result [4].",
                                 "| Source | Result | Evidence |\n|---|---|---|\n"
                                 "| S1 Alpha | Second result | [2] |\n| S2 Beta | Second result | [4] |"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert received[-1]["type"] == "done"
    assert [c["number"] for c in received[0]["citations"]] == [1, 2, 3, 4]
    payload = json.loads(provider.calls[-1][-1]["content"].removeprefix("<synthesis_input_json>\n").removesuffix("\n</synthesis_input_json>"))
    assert [[e["citation"] for e in record["original_evidence"]] for record in payload["source_records"]] == [[1, 2], [3, 4]]


def test_exhaustive_source_records_scope_and_missing_fields(env, monkeypatch):
    client, nid, _, _ = env
    selected = [add_source(nid, f"{name} reports a method", name=name)[0]
                for name in ("Alpha", "Beta", "Gamma")]
    for index, sid in enumerate(selected):
        db.execute("UPDATE sources SET created_at=? WHERE id=?", (f"2026-01-01T00:00:{index:02d}Z", sid))
    disabled, _ = add_source(nid, "Disabled reports a method", enabled=0)
    unselected, _ = add_source(nid, "Other reports a method")
    final = ("| Source | Accuracy | Evidence |\n| --- | --- | --- |\n" +
             "\n".join(f"| S{i} {name} | Not reported in supplied evidence | [{i}] |"
                       for i, name in enumerate(("Alpha", "Beta", "Gamma"), 1)))
    provider = ScriptedProvider([f"Method reported; accuracy Not reported in supplied evidence [{i}]."
                                 for i in range(1, 4)] + [final])
    monkeypatch.setattr(models, "provider", provider)
    query = "Create a complete table with accuracy for all papers"
    received = events(client.post(f"/api/notebooks/{nid}/chat",
                                  json={"question": query, "source_ids": selected + [disabled]}))
    assert [event["type"] for event in received] == ["citations", "status", "delta", "done"]
    assert {c["source_id"] for c in received[0]["citations"]} == set(selected)
    assert disabled not in {c["source_id"] for c in received[0]["citations"]}
    assert unselected not in {c["source_id"] for c in received[0]["citations"]}
    assert len(provider.calls) == 4
    synthesis = provider.calls[-1][-1]["content"]
    payload = json.loads(synthesis.removeprefix("<synthesis_input_json>\n").removesuffix("\n</synthesis_input_json>"))
    assert payload["selected_source_count"] == 3
    assert payload["required_source_labels"] == ["S1", "S2", "S3"]
    assert [r["source_id"] for r in payload["source_records"]] == selected
    assert all(len(r["original_evidence"]) == 1 for r in payload["source_records"])
    assert all("Not reported in supplied evidence" in r["summary_untrusted"] for r in payload["source_records"])
    saved = client.get(f"/api/notebooks/{nid}/messages").json()
    assert saved[-1]["content"] == final
    assert all("Not reported in supplied evidence" in row for row in final.splitlines()[2:])
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["coverage"]["retrieval_mode"] == "exhaustive"
    assert run["coverage"]["eligible_sources"] == 3
    assert query not in json.dumps(run)
    assert "reports a method" not in json.dumps(run)


def test_exhaustive_retrieval_is_bounded_and_keeps_empty_source_record(env):
    _, nid, _, _ = env
    sid, _ = add_source(nid, "method first finding")
    with db.connection() as conn:
        for index in range(1, 6):
            cid = db.uid()
            text = f"method distinct finding {index}"
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (cid, sid, nid, index, None, None, text, retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (cid, text))
    empty, _ = add_source(nid, "irrelevant", vector=(-1, 0))
    groups = retrieval.retrieve_by_source(nid, "Compare every paper", source_ids=[sid, empty])
    assert [source["id"] for source, _ in groups] == [sid, empty]
    assert len(groups[0][1]) == 3
    assert groups[1][1] == []
    assert len(retrieval.evidence_text(groups[0][1])) <= 6000


def test_exhaustive_final_citation_retry_and_completeness(env, monkeypatch):
    client, nid, _, _ = env
    alpha, _ = add_source(nid, "Alpha method", name="Alpha")
    beta, _ = add_source(nid, "Beta method", name="Beta")
    for index, sid in enumerate((alpha, beta)):
        db.execute("UPDATE sources SET created_at=? WHERE id=?", (f"2026-01-01T00:00:{index:02d}Z", sid))
    valid = ("| Source | Method | Evidence |\n| --- | --- | --- |\n"
             "| S1 Alpha | Method A | [1] |\n| S2 Beta | Method B | [2] |")
    provider = ScriptedProvider(["Alpha method [1]", "Beta method [2]",
                                 "| Source | Method | Evidence |\n|---|---|---|\n| S1 Alpha | A (citation 1) | none |\n| S2 Beta | B (citation 2) | none |",
                                 valid])
    monkeypatch.setattr(models, "provider", provider)
    original = retrieval.retrieve_by_source
    calls = []
    def capture(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(retrieval, "retrieve_by_source", capture)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "Compare every paper"}))
    assert [event["type"] for event in received] == ["citations", "status", "delta", "retry", "delta", "done"]
    assert len(calls) == 1
    assert len(provider.calls) == 4
    assert provider.calls[2][-1]["content"] == provider.calls[3][-1]["content"]
    assert client.get(f"/api/notebooks/{nid}/messages").json()[-1]["content"] == valid
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["coverage"]["generation_attempts"] == 2
    assert run["metrics"]["stage_calls"] == {"exhaustive_map": 2, "exhaustive_synthesis": 1,
                                                "exhaustive_synthesis_citation_retry": 1}


def test_exhaustive_rejects_omitted_row_and_oversized_input(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Alpha method")
    add_source(nid, "Beta method")
    provider = ScriptedProvider(["Alpha [1]", "Beta [2]",
                                 "| Source | Evidence |\n|---|---|\n| S1 Alpha | [1] |"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert received[-1]["message"] == chat.COMPLETENESS_REJECTION
    assert received[-1]["reason"] == "source_completeness"
    assert len(provider.calls) == 3
    assert client.get(f"/api/notebooks/{nid}/messages").json() == []
    assert client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]["coverage"]["validation_failures"] == [
        {"stage": "exhaustive_synthesis", "attempt": 1, "code": "missing_source_row"}]
    monkeypatch.setattr(chat, "MAX_SYNTHESIS_INPUT_CHARS", 100)
    provider = ScriptedProvider(["Alpha [1]", "Beta [2]"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert received[-1]["type"] == "error"
    assert "size budget" in received[-1]["message"]
    assert len(provider.calls) == 2  # No final synthesis request bypasses the size gate.
    monkeypatch.setattr(chat, "MAX_SYNTHESIS_INPUT_CHARS", 42_000)
    provider = ScriptedProvider([])
    monkeypatch.setattr(models, "provider", provider)
    huge = "Create a complete table for all papers " + "\x01" * 5000
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": huge}))
    assert received[-1]["type"] == "error"
    assert "size budget" in received[-1]["message"]
    assert provider.calls == []  # Escaped map input is checked before the first model call.


def test_exhaustive_row_check_rejects_extra_or_fenced_rows():
    two_rows = ("| Source | Evidence |\n|---|---|\n| S1 Alpha | [1] |\n| S2 Beta | [2] |")
    chat._validate_source_rows(two_rows, {"S1": {1}, "S2": {2}})
    with pytest.raises(chat.SourceCompletenessError):
        chat._validate_source_rows(two_rows + "\n| Alpha again | [1] |", {"S1": {1}, "S2": {2}})
    with pytest.raises(chat.SourceCompletenessError):
        chat._validate_source_rows("```\n" + two_rows + "\n```\nSummary [1]", {"S1": {1}, "S2": {2}})


def test_exhaustive_map_rejects_uncited_record_without_synthesis(env, monkeypatch):
    client, nid, _, _ = env
    add_source(nid, "Alpha method")
    provider = ScriptedProvider(["Uncited method", "Still uncited method"])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert received[-1]["type"] == "error"
    assert received[-1]["reason"] == "citation_validation"
    assert len(provider.calls) == 2
    assert client.get(f"/api/notebooks/{nid}/messages").json() == []


def test_exhaustive_eleven_sources_and_unavailable_record(env, monkeypatch):
    client, nid, _, _ = env
    for index in range(11):
        sid, _ = add_source(nid, f"Study {index} reports a method", name=f"Study {index}")
        db.execute("UPDATE sources SET created_at=? WHERE id=?", (f"2026-01-01T00:00:{index:02d}Z", sid))
    final = ("| Source | Finding | Evidence |\n|---|---|---|\n" +
             "\n".join(f"| S{i} Study {i - 1} | Method | [{i}] |" for i in range(1, 12)))
    provider = ScriptedProvider([f"Method [{i}]" for i in range(1, 12)] + [final])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "Summarize each of the 11 studies"}))
    assert received[-1]["type"] == "done"
    assert len(received[0]["citations"]) == 11
    assert len(provider.calls) == 12
    payload = json.loads(provider.calls[-1][-1]["content"].removeprefix("<synthesis_input_json>\n").removesuffix("\n</synthesis_input_json>"))
    assert payload["selected_source_count"] == 11
    assert len(payload["source_records"]) == 11

    client.delete(f"/api/notebooks/{nid}/messages")
    disabled = [row["id"] for row in retrieval.scoped_sources(nid)][1:]
    for sid in disabled:
        db.execute("UPDATE sources SET enabled=0 WHERE id=?", (sid,))
    missing, _ = add_source(nid, "irrelevant text", name="No relevant data", vector=(-1, 0))
    db.execute("UPDATE sources SET created_at=? WHERE id=?", ("2026-01-02T00:00:00Z", missing))
    no_data = ("| Source | Finding | Evidence |\n|---|---|---|\n"
               "| S1 Study 0 | Method | [1] |\n"
               "| S2 No relevant data | Not reported in supplied evidence | |")
    provider = ScriptedProvider(["Method [1]", no_data])
    monkeypatch.setattr(models, "provider", provider)
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "One row per source"}))
    assert received[-1]["type"] == "done"
    assert len(provider.calls) == 2
    payload = json.loads(provider.calls[-1][-1]["content"].removeprefix("<synthesis_input_json>\n").removesuffix("\n</synthesis_input_json>"))
    assert payload["source_records"][1]["source_id"] == missing
    assert payload["source_records"][1]["original_evidence"] == []


def test_chat_stream_history_citations_failure_and_diagnostics(env):
    client, nid, fake, root = env
    sid, cid = add_source(nid, "key scientific fact")
    first = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "key?"}))
    assert [e["type"] for e in first] == ["citations", "delta", "done"]
    assert first[0]["citations"][0]["chunk_id"] == cid
    fake.answers = ["Evidence [1]", "Wrong [2]"]
    second = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "key follow-up?"}))
    assert [e["type"] for e in second] == ["citations", "delta", "retry", "delta", "error"]
    assert len(client.get(f"/api/notebooks/{nid}/messages").json()) == 2
    assert all(m["role"] != "assistant" for m in fake.calls[1])
    assert "prior_user_questions_not_evidence" in fake.calls[1][-1]["content"]
    runs = client.get(f"/api/notebooks/{nid}/diagnostics?limit=2").json()
    assert {r["status"] for r in runs} == {"success", "error"}
    assert runs[0]["chunk_ids"] == [cid]
    assert runs[1]["vram_bytes"] == 1234
    assert runs[0]["coverage"]["scores"][cid]["rrf"] > 0
    assert runs[0]["coverage"]["min_similarity"] == retrieval.MIN_SIMILARITY
    assert "key scientific fact" not in json.dumps(runs)
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "key?", "source_ids": []}))[-1]["type"] == "done"
    assert client.get(f"/api/notebooks/{nid}/diagnostics?limit=101").status_code == 422


def test_studio_balanced_provenance_and_failure(env):
    client, nid, fake, root = env
    a, ca = add_source(nid, "orchid botanical findings")
    b, cb = add_source(nid, "neutron physics findings", vector=(0, 1))
    fake.answers = ["Orchids [1]", "Neutrons [2]", "Synthesis [1] [2]"]
    result = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))
    assert [r["type"] for r in result] == ["citations", "status", "delta", "done"]
    assert [r["chunk_id"] for r in result[0]["citations"]] == [ca, cb]
    saved = db.row("SELECT * FROM artifacts WHERE notebook_id=?", (nid,))
    assert "not exhaustive" in saved["content"]
    assert len(fake.calls) == 3
    assert "neutron" in fake.calls[1][-1]["content"]
    fake.calls.clear()
    fake.fail_at = 2
    failed = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))
    assert failed[-1]["type"] == "error"
    assert db.row("SELECT COUNT(*) AS n FROM artifacts WHERE notebook_id=?", (nid,))["n"] == 1
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary", "source_ids": []}))[-1]["type"] == "error"


def test_embedding_validation_and_atomic_ingestion(env):
    client, nid, fake, root = env
    path = root / "text.txt"
    path.write_text("Useful words in text")
    sid = db.uid()
    db.execute("INSERT INTO sources(id,notebook_id,name,kind,path,status,created_at,updated_at) VALUES(?,?,?,?,?,'queued',?,?)",
               (sid, nid, "text.txt", "text", str(path), db.now(), db.now()))
    fake.embed = lambda texts: []
    documents.process_source(sid)
    assert db.row("SELECT status FROM sources WHERE id=?", (sid,))["status"] == "error"
    assert db.row("SELECT COUNT(*) AS n FROM chunks WHERE source_id=?", (sid,))["n"] == 0
    for vectors in ([[float("nan"), 1]], [[1], [2]]):
        with pytest.raises(ValueError):
            models.validate_vectors(vectors, 1)
    fake.embed = lambda texts: [[1., 0.] for _ in texts]
    documents.process_source(sid)
    assert db.row("SELECT status FROM sources WHERE id=?", (sid,))["status"] == "ready"


def test_upload_lifecycle_additive_schema(env):
    client, nid, fake, root = env
    db.execute("CREATE TABLE IF NOT EXISTS private_legacy_data (x TEXT)")
    db.execute("INSERT INTO private_legacy_data VALUES ('preserved')")
    db.init_db()
    assert db.row("SELECT x FROM private_legacy_data")["x"] == "preserved"
    assert client.post(f"/api/notebooks/{nid}/sources", files={"file": ("bad.exe", b"x")}).status_code == 400
    uploaded = client.post(f"/api/notebooks/{nid}/sources", files={"file": ("safe.txt", b"Read this safe text")})
    assert uploaded.status_code == 202
    sid = uploaded.json()["id"]
    assert db.row("SELECT status FROM sources WHERE id=?", (sid,))["status"] == "ready"
    assert client.patch(f"/api/sources/{sid}/toggle").json() == {"enabled": False}
    assert client.post(f"/api/sources/{sid}/retry").status_code == 202
    assert client.delete(f"/api/sources/{sid}").status_code == 204
    assert client.get(f"/api/files/{sid}").status_code == 404


def test_retrieval_diversity_and_no_budget_phantom_citations(env, monkeypatch):
    client, nid, fake, root = env
    sid, first = add_source(nid, "photosynthesis chlorophyll " * 6)
    duplicate = db.uid()
    distinct = db.uid()
    other, outside = add_source(nid, "photosynthesis in forests")
    with db.connection() as conn:
        for cid, ordinal, text in [(duplicate, 1, "photosynthesis chlorophyll " * 6),
                                   (distinct, 2, "photosynthesis light energy")]:
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (cid, sid, nid, ordinal, None, None, text, retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (cid, text))
    chosen = retrieval.retrieve(nid, "photosynthesis chlorophyll", limit=4)
    assert first in {c["id"] for c in chosen} or duplicate in {c["id"] for c in chosen}
    assert not {first, duplicate} <= {c["id"] for c in chosen}
    assert {outside, distinct} <= {c["id"] for c in chosen}
    monkeypatch.setattr(retrieval, "EVIDENCE_BUDGET", 10)
    assert retrieval.retrieve(nid, "photosynthesis") == []
    assert retrieval.citations_for(retrieval.retrieve(nid, "photosynthesis")) == []


def test_partial_model_stream_is_not_persisted(env, monkeypatch):
    client, nid, fake, root = env
    add_source(nid, "model evidence")

    class Truncated(FakeProvider):
        async def stream(self, messages):
            yield models.ModelEvent("text", text="partial [1]")

    monkeypatch.setattr(models, "provider", Truncated())
    received = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "model?"}))
    assert [e["type"] for e in received] == ["citations", "delta", "error"]
    assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 0
    assert client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]["status"] == "error"


def test_selection_and_activity_with_source_chat_artifact(env):
    client, nid, fake, root = env
    active, cid = add_source(nid, "active evidence")
    disabled, _ = add_source(nid, "disabled evidence", enabled=0)
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "active", "source_ids": [disabled]}))[-1]["type"] == "done"
    assert fake.calls == []
    conversation_before = db.row("SELECT updated_at FROM conversations WHERE notebook_id=?", (nid,))["updated_at"]
    events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "active", "source_ids": [active]}))
    assert db.row("SELECT updated_at FROM conversations WHERE notebook_id=?", (nid,))["updated_at"] >= conversation_before
    client.delete(f"/api/notebooks/{nid}/messages")
    assert client.get(f"/api/notebooks/{nid}/messages").json() == []
    fake.calls.clear()
    fake.answers = ["Map [1]", "Final [1]"]
    done = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary", "source_ids": [active, disabled]}))
    assert done[0]["citations"][0]["source_id"] == active
    assert len(done[0]["citations"]) == 1
    aid = done[-1]["artifact_id"]
    assert client.put(f"/api/artifacts/{aid}", json={"title": "Edited", "content": "notes"}).status_code == 200
    assert client.delete(f"/api/artifacts/{aid}").status_code == 204


def test_diagnostics_bounded_retention(env):
    client, nid, fake, root = env
    for i in range(102):
        db.record_run(nid, "chat", "insufficient", chunk_ids=[str(i)])
    assert db.row("SELECT COUNT(*) AS n FROM generation_runs WHERE notebook_id=?", (nid,))["n"] == 100
    assert len(client.get(f"/api/notebooks/{nid}/diagnostics?limit=3").json()) == 3
    assert client.get("/api/notebooks/unknown/diagnostics").status_code == 404


def test_ollama_final_metrics_vram_and_truncation(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": models.GENERATION_MODEL, "size_vram": 4096}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, text='{"message":{"content":"hello"}}\n'
                                             '{"done":true,"total_duration":10,"load_duration":2,'
                                             '"prompt_eval_duration":3,"eval_duration":4,'
                                             '"prompt_eval_count":5,"eval_count":6}\n'
                                             '{"message":{"content":"must not be emitted"}}\n'
                                             '{"error":"after terminal"}\n')
        return httpx.Response(200, json={"models": [{"name": models.GENERATION_MODEL}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(respond)))

    async def check():
        provider = models.OllamaProvider()
        output = [event async for event in provider.stream([{"role": "user", "content": "hello"}])]
        assert output[0].text == "hello"
        assert output[1].metrics == {"total_duration": 10, "load_duration": 2,
                                     "prompt_eval_duration": 3, "eval_duration": 4,
                                     "prompt_eval_count": 5, "eval_count": 6}
        assert len(output) == 2
        assert await provider.vram_bytes() == 4096
        assert (await provider.health())["generation_ready"] is True
        assert "/api/chat" in requests

    import asyncio
    asyncio.run(check())

    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200, text='{"message":{"content":"partial"}}\n'))))

    async def truncated():
        with pytest.raises(RuntimeError, match="before completion"):
            _ = [event async for event in models.OllamaProvider().stream([])]

    asyncio.run(truncated())

    monkeypatch.setattr(models.httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(200, text='{"error":"secret model error"}\n'))))

    async def upstream_error():
        with pytest.raises(RuntimeError, match="Model generation failed"):
            _ = [event async for event in models.OllamaProvider().stream([])]

    asyncio.run(upstream_error())


def test_diagnostics_error_does_not_break_chat(env, monkeypatch):
    client, nid, fake, root = env
    add_source(nid, "observable evidence")
    monkeypatch.setattr(db, "record_run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk")))
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "observable?"}))[-1]["type"] == "done"
    assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 2


def test_full_numeric_citation_parsing():
    assert chat.references("supported [10] then [1]", range(1, 11)) == {1, 10}
    assert chat.references("[3,4] [1, 10] [1,2,5,7,10,12] [1][2] [3-5] [5–6]",
                           range(1, 13)) == {1, 2, 3, 4, 5, 6, 7, 10, 12}
    assert chat.references("[1,10] [10-12]", {1, 10, 11, 12}) == {1, 10, 11, 12}
    assert chat.references("[title](url) and [a] [10]", range(1, 11)) == {10}
    assert chat.references("`[2]`\n```python\n[3,4]\n```\n[2024 report](https://example.org) [2024](url) ![2025 cover](img.png) [10]",
                           range(1, 11)) == {10}
    assert chat.references("[1,2](url) ![1-2](img.png) [1]", range(1, 2)) == {1}
    links = ("[1,2](https://example.org/Foo_(bar)) ![1-2](https://example.org/plot_(final).png) "
             r"[1](https://example.org/a\(b\)/[2])")
    assert chat.references(links, {1, 2}) == set()
    assert chat.references(links + " [2]", {2}) == {2}
    for text in ("[1x]", "unfinished [1", "wrong [11]", "[1,11]", "[1-11]",
                 "[0]", "[-1]", "[1,-2]", "[2-1]", "[1–0]", "[1,,2]", "[1,]",
                 "[1-201]", "[1,2-201]", "[9007199254740992]", "[12345678901234567]"):
        with pytest.raises(ValueError):
            chat.references(text, range(1, 11))
    assert chat.references("[1-200]", range(1, 201)) == set(range(1, 201))
    assert chat.references("[1] [1]", {1}) == {1}
    intent = chat._messages("question", [], [{"role": "user", "content": "Earlier [1, 3-5] " + links + " `code [7]`"}])
    prior = json.loads(intent[-1]["content"].split('<request_json>\n')[1].split('\n</request_json>')[0])
    assert prior["prior_user_questions_not_evidence"] == ['Earlier  ' + links + ' `code [7]`']


def test_escaped_chat_question_and_history_never_start_oversized_generation(env):
    client, nid, fake, root = env
    sid, cid = add_source(nid, "searchable content")
    # 12k raw characters are permitted by the route, but JSON escaping expands
    # this control character to 72k characters before evidence is included.
    huge = "\x01" * 12_000
    result = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": huge}))
    assert [item["type"] for item in result] == ["error"]
    assert "too large" in result[-1]["message"]
    assert fake.calls == []
    assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 0
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["status"] == "error" and run["metrics"]["call_count"] == 0
    assert run["coverage"]["chat_input_budget_chars"] == chat.MAX_CHAT_INPUT_CHARS
    assert huge not in json.dumps(run)
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": huge, "source_ids": []}))[-1]["type"] == "error"
    assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 0

    conversation = db.row("SELECT id FROM conversations WHERE notebook_id=?", (nid,))["id"]
    with db.connection() as conn:
        for i in range(5):
            conn.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)",
                         (db.uid(), conversation, "user", "\x02" * 1000, "[]", db.now()))
        for i in range(12):
            source = db.uid()
            conn.execute("INSERT INTO sources(id,notebook_id,name,kind,status,created_at,updated_at) VALUES(?,?,?,'text','ready',?,?)",
                         (source, nid, f"escaped {i}", db.now(), db.now()))
            chunk = db.uid()
            text = '"\\' * 900
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (chunk, source, nid, 0, None, None, text, retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (chunk, text))
    response = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "ordinary question"}))
    assert response[-1]["type"] == "error"
    assert "too large" in response[-1]["message"]
    assert fake.calls == []
    assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 5


def test_numeric_leading_markdown_links_do_not_block_cited_routes(env):
    client, nid, fake, root = env
    add_source(nid, "supported claim")
    fake.answers = ["A [2024 report](https://example.org/report) supports this [1]"]
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "claim?"}))[-1]["type"] == "done"
    fake.calls.clear()
    fake.answers = ["Map [2024](https://example.org) [1]", "Final [2024 report](url) [1]"]
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))[-1]["type"] == "done"
    assert [c["number"] for c in json.loads(db.row("SELECT citations FROM artifacts")["citations"])] == [1]


def test_chat_tenth_citation_and_missing_citation(env):
    client, nid, fake, root = env
    for i in range(10):
        add_source(nid, f"Distinct evidence number {i}")
    fake.answers = ["Valid tenth [1, 10]", "no references"]
    success = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "evidence?"}))
    assert len(success[0]["citations"]) == 10
    assert success[-1]["type"] == "done"
    assert db.row("SELECT content FROM messages WHERE role='assistant'")["content"] == "Valid tenth [1, 10]"
    failed = events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "more evidence?"}))
    assert failed[-1]["type"] == "error"
    assert len(client.get(f"/api/notebooks/{nid}/messages").json()) == 2


def test_serialized_evidence_exact_budget_and_numbering(env, monkeypatch):
    client, nid, fake, root = env
    chunks = [{"id": str(i), "source_id": str(i), "source_name": '"\\' * 100,
               "page": None, "text": '"\\' * 1000} for i in range(12)]
    actual = retrieval.evidence_text(chunks)
    monkeypatch.setattr(retrieval, "EVIDENCE_BUDGET", len(actual) - 1)
    chosen = retrieval.budgeted(chunks)
    assert len(chosen) == 11
    assert len(retrieval.evidence_text(chosen)) <= retrieval.EVIDENCE_BUDGET
    monkeypatch.setattr(retrieval, "EVIDENCE_BUDGET", len(actual))
    assert len(retrieval.budgeted(chunks)) == 12
    assert '"citation": 10' in retrieval.evidence_text(chunks)


def test_no_zero_score_filler_after_rank_80(env):
    client, nid, fake, root = env
    sid, first = add_source(nid, "only high semantic evidence")
    with db.connection() as conn:
        for i in range(81):
            cid = db.uid()
            conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                         (cid, sid, nid, i + 1, None, None, f"unique high chunk {i}", retrieval.pack_vector([1, 0])))
            conn.execute("INSERT INTO chunk_fts VALUES(?,?)", (cid, f"unique high chunk {i}"))
    other, outsider = add_source(nid, "lexically unrelated", vector=(-1, 0))
    chosen = retrieval.retrieve(nid, "query absent vocabulary", limit=12)
    assert chosen
    assert outsider not in {c["id"] for c in chosen}
    assert all(c["score"] > 0 for c in chosen)


def test_studio_retry_no_truncation_and_missing_source(env):
    client, nid, fake, root = env
    one, c1 = add_source(nid, "First source")
    two, c2 = add_source(nid, "Second source")
    fake.answers = ["x" * 848 + "[1]", "Compact [1]", "Second [2]", "Both [1] [2]"]
    result = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))
    assert result[-1]["type"] == "done"
    assert len(fake.calls) == 4
    assert "Compact [1]" in fake.calls[-1][-1]["content"]
    assert "x" * 50 not in fake.calls[-1][-1]["content"]
    run = client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]
    assert run["metrics"]["stage_calls"] == {"map": 2, "map_retry": 1, "synthesis": 1}
    assert run["metrics"]["ollama_totals"]["eval_count"] == 32
    assert run["metrics"]["metric_samples"]["total_duration"] == 4
    assert run["coverage"]["config"] == {"generation_model": models.GENERATION_MODEL,
                                          "embedding_model": models.EMBEDDING_MODEL,
                                          "num_ctx": models.NUM_CTX, "num_predict": 4096}
    assert "First source" not in json.dumps(run)
    fake.calls.clear()
    fake.answers = ["Map [1]", "Map [2]", "Ignored second [1]"]
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))[-1]["type"] == "error"
    assert db.row("SELECT COUNT(*) AS n FROM artifacts")["n"] == 1
    fake.calls.clear()
    fake.answers = ["Map [1]", "Map [2]", "Citationless synthesis"]
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))[-1]["type"] == "error"
    fake.calls.clear()
    fake.answers = ["y" * 850 + "[1]", "z" * 850 + "[1]"]
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary", "source_ids": [one]}))[-1]["type"] == "error"
    assert len(fake.calls) == 2
    assert db.row("SELECT COUNT(*) AS n FROM artifacts")["n"] == 1


def test_fifty_sources_fit_serialized_synthesis_budget(env):
    client, nid, fake, root = env
    for i in range(50):
        add_source(nid, f"source-specific evidence {i}")
    fake.answers = [('"\\' * 160 + f" [{i}]") for i in range(1, 51)] + [" ".join(f"[{i}]" for i in range(1, 51))]
    result = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))
    assert result[-1]["type"] == "done"
    assert len(fake.calls) == 51
    assert len(fake.calls[-1][-1]["content"]) <= chat.MAX_SYNTHESIS_INPUT_CHARS
    saved = db.row("SELECT citations FROM artifacts WHERE notebook_id=?", (nid,))
    assert [c["number"] for c in json.loads(saved["citations"])] == list(range(1, 51))
    assert client.get(f"/api/notebooks/{nid}/diagnostics").json()[0]["metrics"]["call_count"] == 51


def test_studio_tenth_citation_map_and_final(env):
    client, nid, fake, root = env
    for i in range(10):
        add_source(nid, f"source evidence {i}")
    fake.answers = [f"Map [{i}]" for i in range(1, 11)] + ["All sources [1-10]"]
    result = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "faq"}))
    assert result[-1]["type"] == "done"
    saved = db.row("SELECT citations,content FROM artifacts")
    assert [c["number"] for c in json.loads(saved["citations"])] == list(range(1, 11))
    assert saved["content"].startswith("All sources [1-10]")


def test_studio_persists_only_finally_used_sparse_citations(env):
    client, nid, fake, root = env
    sid, first = add_source(nid, "first evidence")
    with db.connection() as conn:
        conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", (db.uid(), sid, nid, 1, None, None,
                     "second nonoverlapping evidence", retrieval.pack_vector([1, 0])))
    other, third = add_source(nid, "different source evidence")
    fake.answers = ["Map both [1-2]", "Second source [3]", "Only these facts [1,3]"]
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))[-1]["type"] == "done"
    saved = db.row("SELECT citations FROM artifacts WHERE notebook_id=?", (nid,))
    assert [c["number"] for c in json.loads(saved["citations"])] == [1, 3]
    assert db.row("SELECT content FROM artifacts WHERE notebook_id=?", (nid,))["content"].startswith("Only these facts [1,3]")


def test_group_with_unknown_member_does_not_save_chat_or_studio(env):
    client, nid, fake, root = env
    add_source(nid, "first evidence")
    add_source(nid, "second evidence")
    fake.answers = ["Partial [1,3]"]
    assert events(client.post(f"/api/notebooks/{nid}/chat", json={"question": "evidence?"}))[-1]["type"] == "error"
    assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 0
    fake.calls.clear()
    fake.answers = ["Map [1]", "Map [2]", "Synthesis [1,2,3]"]
    assert events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))[-1]["type"] == "error"
    assert db.row("SELECT COUNT(*) AS n FROM artifacts")["n"] == 0


def test_studio_does_not_silently_skip_ready_source_without_chunks(env):
    client, nid, fake, root = env
    add_source(nid, "indexed source")
    sid, stamp = db.uid(), db.now()
    db.execute("INSERT INTO sources(id,notebook_id,name,kind,status,created_at,updated_at) VALUES(?,?,?,'text','ready',?,?)",
               (sid, nid, "unindexed source", stamp, stamp))
    result = events(client.post(f"/api/notebooks/{nid}/artifacts", json={"kind": "summary"}))
    assert result[-1]["type"] == "error"
    assert fake.calls == []
    assert db.row("SELECT COUNT(*) AS n FROM artifacts")["n"] == 0


def test_cancelled_generators_record_without_vram_or_success(env, monkeypatch):
    client, nid, fake, root = env
    sid, cid = add_source(nid, "evidence for cancellation")
    conversation = db.row("SELECT id FROM conversations WHERE notebook_id=?", (nid,))["id"]

    class Blocking(FakeProvider):
        def __init__(self, at):
            super().__init__()
            self.at = at
            self.started = asyncio.Event()

        async def stream(self, messages):
            self.calls.append(messages)
            if len(self.calls) == self.at:
                self.started.set()
                await asyncio.Event().wait()
            yield models.ModelEvent("text", text="Map [1]" if len(self.calls) == 1 else "Final [1]")
            yield models.ModelEvent("metrics", metrics={"eval_count": 1})

        async def vram_bytes(self):
            await asyncio.Event().wait()  # Must never be probed during cancellation.

    async def cancel(gen, provider, prefix_count):
        for _ in range(prefix_count):
            await anext(gen)
        pending = asyncio.create_task(anext(gen))
        await asyncio.wait_for(provider.started.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)

    async def checks():
        for stage, first_events in [("chat", 1), ("map", 2), ("synthesis", 2)]:
            provider = Blocking(2 if stage == "synthesis" else 1)
            monkeypatch.setattr(models, "provider", provider)
            if stage == "chat":
                gen = chat.stream_chat(nid, conversation, "why?", None, [])
            else:
                gen = chat.stream_artifact(nid, "Notebook", "summary", None)
            await cancel(gen, provider, first_events)
            assert db.row("SELECT status FROM generation_runs ORDER BY created_at DESC,id DESC LIMIT 1")["status"] == "cancelled"
            if stage == "chat":
                assert len(provider.calls) == 1
                run = db.row("SELECT coverage FROM generation_runs ORDER BY created_at DESC,id DESC LIMIT 1")
                assert json.loads(run["coverage"])["citation_retry_used"] is False
            assert db.row("SELECT COUNT(*) AS n FROM messages")["n"] == 0
            assert db.row("SELECT COUNT(*) AS n FROM artifacts")["n"] == 0
        provider = Blocking(1)
        monkeypatch.setattr(models, "provider", provider)
        gen = chat.stream_chat(nid, conversation, "why?", None, [])
        assert json.loads(await anext(gen))["type"] == "citations"
        await gen.aclose()
        assert db.row("SELECT status FROM generation_runs ORDER BY created_at DESC,id DESC LIMIT 1")["status"] == "cancelled"

    asyncio.run(checks())


def test_additive_migration_preserves_real_baseline_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "legacy.db")
    monkeypatch.setattr(db, "FILES_DIR", tmp_path / "files")
    # The original database schema, before generation_runs existed.
    baseline = """
    CREATE TABLE notebooks (id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE sources (id TEXT PRIMARY KEY,notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
      name TEXT NOT NULL,kind TEXT NOT NULL,path TEXT,status TEXT NOT NULL DEFAULT 'queued',error TEXT,
      char_count INTEGER NOT NULL DEFAULT 0,page_count INTEGER,scanned INTEGER NOT NULL DEFAULT 0,
      enabled INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE chunks (id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
      notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,ordinal INTEGER NOT NULL,page INTEGER,
      section TEXT,text TEXT NOT NULL,embedding BLOB);
    CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED,text);
    CREATE TABLE conversations (id TEXT PRIMARY KEY,notebook_id TEXT NOT NULL UNIQUE REFERENCES notebooks(id) ON DELETE CASCADE,
      created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE messages (id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
      role TEXT NOT NULL,content TEXT NOT NULL,citations TEXT NOT NULL DEFAULT '[]',created_at TEXT NOT NULL);
    CREATE TABLE notes (id TEXT PRIMARY KEY,notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
      title TEXT NOT NULL,content TEXT NOT NULL DEFAULT '',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE artifacts (id TEXT PRIMARY KEY,notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
      kind TEXT NOT NULL,title TEXT NOT NULL,content TEXT NOT NULL DEFAULT '',citations TEXT NOT NULL DEFAULT '[]',
      created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    """
    with sqlite3.connect(db.DB_PATH) as conn:
        conn.executescript(baseline)
        conn.execute("INSERT INTO notebooks VALUES('nb','legacy','old','old')")
        conn.execute("INSERT INTO sources(id,notebook_id,name,kind,status,created_at,updated_at) VALUES('src','nb','original','text','ready','old','old')")
        conn.execute("INSERT INTO chunks VALUES('ch','src','nb',0,NULL,NULL,'source content',NULL)")
        conn.execute("INSERT INTO chunk_fts VALUES('ch','source content')")
        conn.execute("INSERT INTO conversations VALUES('conv','nb','old','old')")
        conn.execute("INSERT INTO messages VALUES('msg','conv','assistant','prior answer','[]','old')")
        conn.execute("INSERT INTO notes VALUES('note','nb','old note','retained','old','old')")
    db.init_db()
    db.init_db()
    assert db.row("SELECT text FROM chunks WHERE id='ch'")["text"] == "source content"
    assert db.row("SELECT content FROM messages WHERE id='msg'")["content"] == "prior answer"
    assert db.row("SELECT content FROM notes WHERE id='note'")["content"] == "retained"
    assert db.row("SELECT id FROM generation_runs") is None


def test_touched_timestamps_change_from_legacy_time(env):
    client, nid, fake, root = env
    old = "2001-01-01T00:00:00+00:00"
    note = client.post(f"/api/notebooks/{nid}/notes", json={"title": "A"}).json()
    sid, cid = add_source(nid, "timestamp evidence")
    conv = db.row("SELECT id FROM conversations WHERE notebook_id=?", (nid,))["id"]
    for method, route, body, conversation in [
        ("put", f"/api/notes/{note['id']}", {"title": "B"}, False),
        ("patch", f"/api/sources/{sid}/toggle", None, False),
        ("delete", f"/api/notes/{note['id']}", None, False),
        ("delete", f"/api/notebooks/{nid}/messages", None, True),
    ]:
        db.execute("UPDATE notebooks SET updated_at=? WHERE id=?", (old, nid))
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (old, conv))
        response = getattr(client, method)(route, json=body) if body is not None else getattr(client, method)(route)
        assert response.status_code < 300
        assert db.row("SELECT updated_at FROM notebooks WHERE id=?", (nid,))["updated_at"] > old
        if conversation:
            assert db.row("SELECT updated_at FROM conversations WHERE id=?", (conv,))["updated_at"] > old
