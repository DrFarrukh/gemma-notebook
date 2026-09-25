import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
DB_PATH = DATA_DIR / "notebook.db"
FILES_DIR = DATA_DIR / "files"
_lock = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid.uuid4())


@contextmanager
def connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db():
    with connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS notebooks (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
          id TEXT PRIMARY KEY, notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
          name TEXT NOT NULL, kind TEXT NOT NULL, path TEXT, status TEXT NOT NULL DEFAULT 'queued',
          error TEXT, char_count INTEGER NOT NULL DEFAULT 0, page_count INTEGER,
          scanned INTEGER NOT NULL DEFAULT 0, enabled INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
          notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL, page INTEGER, section TEXT, text TEXT NOT NULL, embedding BLOB
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(chunk_id UNINDEXED, text);
        CREATE TABLE IF NOT EXISTS conversations (
          id TEXT PRIMARY KEY, notebook_id TEXT NOT NULL UNIQUE REFERENCES notebooks(id) ON DELETE CASCADE,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
          role TEXT NOT NULL, content TEXT NOT NULL, citations TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS notes (
          id TEXT PRIMARY KEY, notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
          title TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifacts (
          id TEXT PRIMARY KEY, notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', citations TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS generation_runs (
          id TEXT PRIMARY KEY, notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
          retrieval_ms INTEGER, generation_ms INTEGER, chunk_ids TEXT NOT NULL DEFAULT '[]',
          coverage TEXT NOT NULL DEFAULT '{}', metrics TEXT NOT NULL DEFAULT '{}', vram_bytes INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_runs_notebook ON generation_runs(notebook_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_sources_notebook ON sources(notebook_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_notebook ON chunks(notebook_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
        """)


def rows(sql, params=()):
    with connection() as db:
        return [dict(r) for r in db.execute(sql, params).fetchall()]


def row(sql, params=()):
    with connection() as db:
        found = db.execute(sql, params).fetchone()
        return dict(found) if found else None


def execute(sql, params=()):
    with connection() as db:
        cur = db.execute(sql, params)
        return cur.rowcount


def json_value(value, fallback=None):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def touch_notebook(conn, notebook_id, conversation=False):
    timestamp = now()
    conn.execute("UPDATE notebooks SET updated_at=? WHERE id=?", (timestamp, notebook_id))
    if conversation:
        conn.execute("UPDATE conversations SET updated_at=? WHERE notebook_id=?", (timestamp, notebook_id))


def record_run(notebook_id, kind, status, retrieval_ms=None, generation_ms=None,
               chunk_ids=(), coverage=None, metrics=None, vram_bytes=None):
    with connection() as conn:
        conn.execute("INSERT INTO generation_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                     (uid(), notebook_id, kind, status, now(), retrieval_ms, generation_ms,
                      json.dumps(chunk_ids), json.dumps(coverage or {}), json.dumps(metrics or {}), vram_bytes))
        conn.execute("DELETE FROM generation_runs WHERE notebook_id=? AND id NOT IN "
                     "(SELECT id FROM generation_runs WHERE notebook_id=? ORDER BY created_at DESC,id DESC LIMIT 100)",
                     (notebook_id, notebook_id))
