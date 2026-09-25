from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import chat as generation, db, documents, models

STATIC_DIR = Path(__file__).parent / "static"
MAX_FILE_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app):
    db.init_db()
    for source in db.rows("SELECT id FROM sources WHERE status='processing'"):
        with db.connection() as conn:
            conn.execute("UPDATE sources SET status='queued', error='Processing was interrupted; retry the source.' WHERE id=?", (source["id"],))
            row = conn.execute("SELECT notebook_id FROM sources WHERE id=?", (source["id"],)).fetchone()
            db.touch_notebook(conn, row["notebook_id"])
    yield


app = FastAPI(title="Gemma Notebook", version="1.0.0", lifespan=lifespan)


class NotebookIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class PasteIn(BaseModel):
    name: str = Field(default="Pasted text", min_length=1, max_length=180)
    text: str = Field(min_length=1, max_length=documents.MAX_CHARS)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=12_000)
    source_ids: list[str] | None = None


class NoteIn(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    content: str = Field(default="", max_length=500_000)


class ArtifactIn(BaseModel):
    kind: str
    source_ids: list[str] | None = None


class SettingsIn(BaseModel):
    generation_model: str | None = Field(default=None, min_length=1, max_length=200)
    num_ctx: int | None = None


def require_notebook(notebook_id):
    notebook = db.row("SELECT * FROM notebooks WHERE id=?", (notebook_id,))
    if not notebook:
        raise HTTPException(404, "Notebook not found")
    return notebook


def source_json(source):
    source["enabled"] = bool(source["enabled"])
    source["scanned"] = bool(source["scanned"])
    return source


@app.get("/api/health")
async def health():
    try:
        result = await models.provider.health()
    except Exception:
        result = {"app": "ok", "ollama": False, "error": "Local model unavailable",
                   "generation_model": models.GENERATION_MODEL, "embedding_model": models.EMBEDDING_MODEL,
                   "generation_ready": False, "embedding_ready": False,
                   "model_loaded": False, "gpu_percent": None, "cpu_percent": None}
    result["num_ctx"] = models.NUM_CTX
    return result


@app.get("/api/settings")
async def get_settings():
    try:
        available = await models.provider.list_models()
    except Exception:
        available = []
    return {"generation_model": models.GENERATION_MODEL, "embedding_model": models.EMBEDDING_MODEL,
            "num_ctx": models.NUM_CTX, "available_models": available, "context_options": list(models.CONTEXT_OPTIONS)}


@app.post("/api/settings")
async def update_settings(payload: SettingsIn):
    model, num_ctx = None, None
    if payload.generation_model is not None:
        try:
            available = await models.provider.list_models()
        except Exception:
            raise HTTPException(503, "Ollama unavailable; cannot verify model")
        if payload.generation_model not in available:
            raise HTTPException(400, "Unknown model; it must already be pulled in Ollama")
        model = payload.generation_model
    if payload.num_ctx is not None:
        if payload.num_ctx not in models.CONTEXT_OPTIONS:
            raise HTTPException(400, f"num_ctx must be one of {models.CONTEXT_OPTIONS}")
        num_ctx = payload.num_ctx
    models.set_generation_settings(model, num_ctx)
    return {"generation_model": models.GENERATION_MODEL, "num_ctx": models.NUM_CTX}


@app.get("/api/notebooks/{notebook_id}/diagnostics")
def diagnostics(notebook_id: str, limit: int = 20):
    require_notebook(notebook_id)
    if not 1 <= limit <= 100:
        raise HTTPException(422, "limit must be between 1 and 100")
    records = db.rows("SELECT * FROM generation_runs WHERE notebook_id=? ORDER BY created_at DESC,id DESC LIMIT ?", (notebook_id, limit))
    for record in records:
        for key in ("chunk_ids", "coverage", "metrics"):
            record[key] = db.json_value(record[key])
    return records


@app.get("/api/notebooks")
def list_notebooks():
    return db.rows("""
      SELECT n.*, COUNT(DISTINCT s.id) AS source_count
      FROM notebooks n LEFT JOIN sources s ON s.notebook_id=n.id
      GROUP BY n.id ORDER BY n.updated_at DESC
    """)


@app.post("/api/notebooks", status_code=201)
def create_notebook(body: NotebookIn):
    notebook_id, timestamp = db.uid(), db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO notebooks VALUES(?,?,?,?)", (notebook_id, body.title.strip(), timestamp, timestamp))
        conn.execute("INSERT INTO conversations VALUES(?,?,?,?)", (db.uid(), notebook_id, timestamp, timestamp))
    return db.row("SELECT * FROM notebooks WHERE id=?", (notebook_id,))


@app.patch("/api/notebooks/{notebook_id}")
def rename_notebook(notebook_id: str, body: NotebookIn):
    require_notebook(notebook_id)
    db.execute("UPDATE notebooks SET title=?, updated_at=? WHERE id=?", (body.title.strip(), db.now(), notebook_id))
    return db.row("SELECT * FROM notebooks WHERE id=?", (notebook_id,))


@app.delete("/api/notebooks/{notebook_id}", status_code=204)
def delete_notebook(notebook_id: str):
    require_notebook(notebook_id)
    paths = db.rows("SELECT path FROM sources WHERE notebook_id=? AND path IS NOT NULL", (notebook_id,))
    with db.connection() as conn:
        conn.execute("DELETE FROM chunk_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE notebook_id=?)", (notebook_id,))
        conn.execute("DELETE FROM notebooks WHERE id=?", (notebook_id,))
    for item in paths:
        try:
            Path(item["path"]).unlink(missing_ok=True)
        except OSError:
            pass
    try:
        (db.FILES_DIR / notebook_id).rmdir()
    except OSError:
        pass


@app.get("/api/notebooks/{notebook_id}")
def notebook_detail(notebook_id: str):
    notebook = require_notebook(notebook_id)
    notebook["sources"] = [source_json(s) for s in db.rows("SELECT * FROM sources WHERE notebook_id=? ORDER BY created_at DESC", (notebook_id,))]
    notebook["notes"] = db.rows("SELECT * FROM notes WHERE notebook_id=? ORDER BY updated_at DESC", (notebook_id,))
    notebook["artifacts"] = db.rows("SELECT * FROM artifacts WHERE notebook_id=? ORDER BY updated_at DESC", (notebook_id,))
    return notebook


@app.post("/api/notebooks/{notebook_id}/sources", status_code=202)
async def upload_source(notebook_id: str, background: BackgroundTasks, file: UploadFile = File(...)):
    require_notebook(notebook_id)
    if len(db.rows("SELECT id FROM sources WHERE notebook_id=?", (notebook_id,))) >= 50:
        raise HTTPException(400, "This notebook has reached the 50-source limit")
    filename = documents.safe_filename(file.filename or "source.txt")
    try:
        kind = documents.source_kind(filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    source_id = db.uid()
    folder = db.FILES_DIR / notebook_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source_id}-{filename}"
    size = 0
    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                output.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, "Files are limited to 50 MB")
            output.write(chunk)
    timestamp = db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO sources(id,notebook_id,name,kind,path,status,created_at,updated_at) VALUES(?,?,?,?,?,'queued',?,?)",
                     (source_id, notebook_id, filename, kind, str(path), timestamp, timestamp))
        db.touch_notebook(conn, notebook_id)
    background.add_task(documents.process_source, source_id)
    return source_json(db.row("SELECT * FROM sources WHERE id=?", (source_id,)))


@app.post("/api/notebooks/{notebook_id}/sources/paste", status_code=202)
def paste_source(notebook_id: str, body: PasteIn, background: BackgroundTasks):
    require_notebook(notebook_id)
    if len(db.rows("SELECT id FROM sources WHERE notebook_id=?", (notebook_id,))) >= 50:
        raise HTTPException(400, "This notebook has reached the 50-source limit")
    source_id = db.uid()
    folder = db.FILES_DIR / notebook_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source_id}-pasted.txt"
    path.write_text(body.text, encoding="utf-8")
    timestamp = db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO sources(id,notebook_id,name,kind,path,status,created_at,updated_at) VALUES(?,?,?,?,?,'queued',?,?)",
                     (source_id, notebook_id, body.name.strip(), "text", str(path), timestamp, timestamp))
        db.touch_notebook(conn, notebook_id)
    background.add_task(documents.process_source, source_id)
    return source_json(db.row("SELECT * FROM sources WHERE id=?", (source_id,)))


@app.patch("/api/sources/{source_id}/toggle")
def toggle_source(source_id: str):
    source = db.row("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source:
        raise HTTPException(404, "Source not found")
    value = 0 if source["enabled"] else 1
    with db.connection() as conn:
        conn.execute("UPDATE sources SET enabled=?, updated_at=? WHERE id=?", (value, db.now(), source_id))
        db.touch_notebook(conn, source["notebook_id"])
    return {"enabled": bool(value)}


@app.post("/api/sources/{source_id}/retry", status_code=202)
def retry_source(source_id: str, background: BackgroundTasks):
    source = db.row("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source:
        raise HTTPException(404, "Source not found")
    with db.connection() as conn:
        conn.execute("UPDATE sources SET status='queued', error=NULL, updated_at=? WHERE id=?", (db.now(), source_id))
        db.touch_notebook(conn, source["notebook_id"])
    background.add_task(documents.process_source, source_id)
    return {"status": "queued"}


@app.delete("/api/sources/{source_id}", status_code=204)
def delete_source(source_id: str):
    source = db.row("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source:
        raise HTTPException(404, "Source not found")
    with db.connection() as conn:
        conn.execute("DELETE FROM chunk_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id=?)", (source_id,))
        conn.execute("DELETE FROM sources WHERE id=?", (source_id,))
        db.touch_notebook(conn, source["notebook_id"])
    if source["path"]:
        Path(source["path"]).unlink(missing_ok=True)


@app.get("/api/notebooks/{notebook_id}/messages")
def messages(notebook_id: str):
    require_notebook(notebook_id)
    items = db.rows("""
      SELECT m.* FROM messages m JOIN conversations c ON c.id=m.conversation_id
      WHERE c.notebook_id=? ORDER BY m.created_at
    """, (notebook_id,))
    for item in items:
        item["citations"] = db.json_value(item["citations"], [])
    return items


@app.delete("/api/notebooks/{notebook_id}/messages", status_code=204)
def clear_messages(notebook_id: str):
    require_notebook(notebook_id)
    with db.connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=(SELECT id FROM conversations WHERE notebook_id=?)", (notebook_id,))
        db.touch_notebook(conn, notebook_id, conversation=True)


@app.post("/api/notebooks/{notebook_id}/chat")
async def chat(notebook_id: str, body: ChatIn):
    require_notebook(notebook_id)
    conversation = db.row("SELECT * FROM conversations WHERE notebook_id=?", (notebook_id,))
    history = db.rows("SELECT role,content FROM messages WHERE conversation_id=? AND role='user' ORDER BY created_at DESC LIMIT 5", (conversation["id"],))
    history.reverse()
    return StreamingResponse(generation.stream_chat(notebook_id, conversation["id"], body.question, body.source_ids, history), media_type="application/x-ndjson")


@app.get("/api/notebooks/{notebook_id}/notes")
def list_notes(notebook_id: str):
    require_notebook(notebook_id)
    return db.rows("SELECT * FROM notes WHERE notebook_id=? ORDER BY updated_at DESC", (notebook_id,))


@app.post("/api/notebooks/{notebook_id}/notes", status_code=201)
def create_note(notebook_id: str, body: NoteIn):
    require_notebook(notebook_id)
    note_id, timestamp = db.uid(), db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO notes VALUES(?,?,?,?,?,?)", (note_id, notebook_id, body.title.strip(), body.content, timestamp, timestamp))
        db.touch_notebook(conn, notebook_id)
    return db.row("SELECT * FROM notes WHERE id=?", (note_id,))


@app.put("/api/notes/{note_id}")
def update_note(note_id: str, body: NoteIn):
    note = db.row("SELECT notebook_id FROM notes WHERE id=?", (note_id,))
    if not note:
        raise HTTPException(404, "Note not found")
    with db.connection() as conn:
        conn.execute("UPDATE notes SET title=?,content=?,updated_at=? WHERE id=?", (body.title.strip(), body.content, db.now(), note_id))
        db.touch_notebook(conn, note["notebook_id"])
    return db.row("SELECT * FROM notes WHERE id=?", (note_id,))


@app.delete("/api/notes/{note_id}", status_code=204)
def delete_note(note_id: str):
    with db.connection() as conn:
        note = conn.execute("SELECT notebook_id FROM notes WHERE id=?", (note_id,)).fetchone()
        if not note:
            raise HTTPException(404, "Note not found")
        conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
        db.touch_notebook(conn, note["notebook_id"])


@app.post("/api/notebooks/{notebook_id}/artifacts")
async def create_artifact(notebook_id: str, body: ArtifactIn):
    notebook = require_notebook(notebook_id)
    if body.kind not in generation.ARTIFACT_PROMPTS:
        raise HTTPException(400, "Artifact kind must be summary, faq, or guide")
    return StreamingResponse(generation.stream_artifact(notebook_id, notebook["title"], body.kind, body.source_ids), media_type="application/x-ndjson")


@app.put("/api/artifacts/{artifact_id}")
def update_artifact(artifact_id: str, body: NoteIn):
    artifact = db.row("SELECT notebook_id FROM artifacts WHERE id=?", (artifact_id,))
    if not artifact:
        raise HTTPException(404, "Artifact not found")
    with db.connection() as conn:
        conn.execute("UPDATE artifacts SET title=?,content=?,updated_at=? WHERE id=?", (body.title.strip(), body.content, db.now(), artifact_id))
        db.touch_notebook(conn, artifact["notebook_id"])
    return db.row("SELECT * FROM artifacts WHERE id=?", (artifact_id,))


@app.delete("/api/artifacts/{artifact_id}", status_code=204)
def delete_artifact(artifact_id: str):
    with db.connection() as conn:
        artifact = conn.execute("SELECT notebook_id FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not artifact:
            raise HTTPException(404, "Artifact not found")
        conn.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
        db.touch_notebook(conn, artifact["notebook_id"])


@app.get("/api/files/{source_id}")
def original_file(source_id: str):
    source = db.row("SELECT * FROM sources WHERE id=?", (source_id,))
    if not source or not source["path"] or not Path(source["path"]).is_file():
        raise HTTPException(404, "Source file not found")
    suffix = Path(source["name"]).suffix.lower()
    # Only PDF is inline; even a disguised HTML payload gets a sandboxed, nosniff response.
    media_type = {".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/plain",
                  ".markdown": "text/plain", ".csv": "text/plain"}.get(suffix, "application/octet-stream")
    if suffix in {".html", ".htm"}:
        media_type = "application/octet-stream"
    return FileResponse(source["path"], filename=source["name"], media_type=media_type,
                        content_disposition_type="inline" if suffix == ".pdf" else "attachment",
                        headers={"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox; default-src 'none'"})


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str = ""):
    candidate = STATIC_DIR / path
    if path and candidate.is_file() and STATIC_DIR in candidate.resolve().parents:
        return FileResponse(candidate)
    return FileResponse(STATIC_DIR / "index.html")
