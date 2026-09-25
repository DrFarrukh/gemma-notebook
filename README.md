# Gemma Notebook

A private, local NotebookLM-style research workspace powered by `gemma4:e4b` and `nomic-embed-text:latest` through Ollama.

## Start

On a Linux host, install Ollama and rootless Podman with Compose support (`podman compose`; for example, the `podman-compose` provider). Start Ollama with `ollama serve` in a separate terminal if it is not already running. Then, from a clean checkout:

```bash
git clone https://github.com/DrFarrukh/gemma-notebook.git
cd gemma-notebook
ollama list
ollama show gemma4:e4b || ollama pull gemma4:e4b
ollama show nomic-embed-text:latest || ollama pull nomic-embed-text:latest
./start.sh
```

The pull commands run only for missing models. Ollama must listen at `127.0.0.1:11434`; check with `ollama list` and `ollama show`. Use the Ollama **tag** (`gemma4:e4b`), not a blob filename/hash in Ollama's internal model store. `gemma4-e4b-64k:latest` is not the default tag.

Open <http://127.0.0.1:8787>. Use `./logs.sh` for logs and `./stop.sh` to stop the application. The first build downloads the Python base image and dependencies; later starts reuse the local image.

## What it does

- Multiple persistent notebooks
- PDF, DOCX, ODT, text, Markdown, HTML, CSV, and pasted-text sources
- Hybrid semantic and keyword retrieval using local embeddings
- Streaming source-grounded chat with clickable page/excerpt citations
- Editable notes, summaries, FAQs, and study/briefing guides
- Scanned-PDF detection with a clear OCR warning
- Local-only storage and localhost-only network binding

The 50 MB upload, 50 sources-per-notebook, and 2 million extracted characters-per-source limits protect against accidental oversized jobs. The generation request asks Ollama for a 65,536-token context via the runtime `num_ctx` option; `64k` is not part of the model tag. This is a request, not a guarantee that the model and context fit in your RAM/VRAM. On memory pressure or slow generation, close other workloads, use a smaller model or shorten source material; large contexts and two-stage Studio jobs may take a long time. Check `./logs.sh`, `ollama list`, and `ollama show gemma4:e4b` when troubleshooting. The health indicator checks API reachability and installed tags, not inference capacity.

Retrieval uses a heuristic relevance gate, not a guarantee of complete evidence or correctness. Follow-up retrieval uses the current question, **not** resolved conversational coreference; the prompt includes only recent prior user questions as context, not prior answers as evidence. Rephrase ambiguous follow-ups explicitly. Studio samples up to four chunks per source (the first 1,000 characters of each selected chunk), summarizes each source, then synthesizes the final artifact; sampling is not exhaustive. Treat source text and intermediate summaries as untrusted: prompt injection defenses cannot guarantee immunity. SQLite-backed retrieval is used today; a vector index is deferred. Verify claims against original files and citations. Studio and chat have Stop controls; interruption does not guarantee server-side rollback, so refresh to check saved output.

## Code layout

- `app/main.py` — HTTP routes and API validation; `app/chat.py` — chat and two-stage Studio streams.
- `app/documents.py` — source ingestion; `app/retrieval.py` — notebook-local hybrid retrieval; `app/models.py` — Ollama provider.
- `app/db.py` — SQLite schema and storage; `app/services.py` — compatibility facade for services.
- `app/static/` — offline notebook UI; `tests/` — local/mocked Python and dependency-free JavaScript tests.

## Data and backup

Everything is under `data/`:

- `data/notebook.db` — notebooks, extracted chunks, embeddings, chats, notes, artifacts
- `data/files/` — original uploaded sources

Before upgrades or other changes, make a consistent backup: stop the app and archive the entire directory:

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
| `GENERATION_MODEL` | `gemma4:e4b` |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` |
| `RETRIEVAL_MIN_SIMILARITY` | `0.35` |
| `DATA_DIR` | `/app/data` |

The first four values can be overridden as environment variables before `./start.sh` (for example, `GENERATION_MODEL=other:tag ./start.sh`); install any alternative tags in Ollama first. The `DATA_DIR` container path is fixed by the volume mount. No passwords are needed in the compose file.

Linux host networking is intentional: rootless Podman must reach the host-only Ollama listener. The app itself binds only to `127.0.0.1:8787`. Rootless Podman isolates the container from host root; the image currently runs its process as root **inside** the container, and `./data` is bind-mounted with `:Z`. Do not change data ownership casually on an existing installation. Compose declares a health check for API and tag readiness because Podman's default OCI image build may ignore the Containerfile `HEALTHCHECK`; it does not test actual inference.

The notebook UI uses local assets and system fonts and needs no internet after the image/dependencies and Ollama models are downloaded. FastAPI's default `/docs` Swagger UI may load assets from a CDN, so do not rely on that separate page offline. `GET /api/notebooks/{id}/diagnostics` returns recent SQLite run metadata, retaining up to 100 runs per notebook: status, durations, provider counters, chunk IDs, retrieval scores and sampled coverage. VRAM is a best-effort current reading, **not peak memory**. It does not return document or prompt text; the optional `limit` is 1–100 (default 20). Keep the endpoint local because even metadata can reveal usage patterns.

## Test

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
node --test tests/test_*.cjs  # optional: Node 22+; only for frontend tests
```

Tests use local/mocked backends; they do not require a running Ollama, downloaded models, or your notebook data. Node is not needed to run the app.

The OpenAPI schema is available locally at <http://127.0.0.1:8787/docs> while the app is running.

## License

Apache License 2.0; see [LICENSE](LICENSE) for the full text.
