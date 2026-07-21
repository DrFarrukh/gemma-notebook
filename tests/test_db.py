import importlib


def test_database_cascades_notebook_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app import db
    importlib.reload(db)
    db.init_db()
    notebook_id, timestamp = db.uid(), db.now()
    with db.connection() as conn:
        conn.execute("INSERT INTO notebooks VALUES(?,?,?,?)", (notebook_id, "Test", timestamp, timestamp))
        conn.execute("INSERT INTO conversations VALUES(?,?,?,?)", (db.uid(), notebook_id, timestamp, timestamp))
    assert db.row("SELECT title FROM notebooks WHERE id=?", (notebook_id,))["title"] == "Test"
    db.execute("DELETE FROM notebooks WHERE id=?", (notebook_id,))
    assert not db.row("SELECT id FROM conversations WHERE notebook_id=?", (notebook_id,))
