# Gemma Notebook

A private, local NotebookLM-style research workspace powered by `gemma4-e4b-64k:latest` and `nomic-embed-text:latest` through Ollama.

## Start

Prerequisites already present on this machine:

- Ollama listening at `127.0.0.1:11434`
- `gemma4-e4b-64k:latest`
- `nomic-embed-text:latest`
- rootless Podman with `podman-compose`

Run:

```bash
cd /home/farrukh/gemma-notebook
./start.sh
```

Open <http://127.0.0.1:8787>. Use `./logs.sh` for logs and `./stop.sh` to stop the application. The first build downloads the Python base image and dependencies; later starts reuse the local image.

## What it does

- Multiple persistent notebooks
- PDF, DOCX, ODT, text, Markdown, HTML, CSV, and pasted-text sources
- Hybrid semantic and keyword retrieval using local embeddings
- Streaming source-grounded chat with clickable page/excerpt citations
- Editable notes, summaries, FAQs, and study/briefing guides
- Scanned-PDF detection with a clear OCR warning
- Local-only storage and localhost-only network binding

The 50 MB upload, 50 sources-per-notebook, and 2 million extracted characters-per-source limits protect this laptop from accidental oversized jobs. The generation request retains the model's 65,536-token context configuration while retrieval keeps the evidence set focused for usable performance.

## Data and backup

Everything is under `data/`:

- `data/notebook.db` — notebooks, extracted chunks, embeddings, chats, notes, artifacts
- `data/files/` — original uploaded sources

For a consistent backup, stop the app and archive the entire directory:

```bash
./stop.sh
tar -czf gemma-notebook-backup.tar.gz data
./start.sh
```

To restore, stop the app, move the current `data/` somewhere safe, extract the archive in this project directory, and start again. Do not merge individual SQLite files from different backups.

## Configuration

Defaults are declared in `podman-compose.yml`:

| Variable | Default |
|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `GENERATION_MODEL` | `gemma4-e4b-64k:latest` |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` |
| `DATA_DIR` | `/app/data` |

Host networking is intentional: rootless Podman must reach the host-only Ollama listener. The app itself binds only to `127.0.0.1:8787`.

## Test

```bash
podman build -t localhost/gemma-notebook:test .
podman run --rm localhost/gemma-notebook:test python -m pytest -q
```

The OpenAPI schema is available locally at <http://127.0.0.1:8787/docs> while the app is running.
