"""Local model provider; no source content or prompts are logged or retained here."""
import json
import math
import os
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen3.5:9b") #gemma4:e4b
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
NUM_CTX = 16384 # 65536
NUM_PREDICT = 4096
METRIC_KEYS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
               "prompt_eval_count", "eval_count")


@dataclass
class ModelEvent:
    type: str  # text or metrics
    text: str = ""
    metrics: dict | None = None


class Provider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def stream(self, messages: list[dict]) -> AsyncIterator[ModelEvent]: ...
    async def health(self) -> dict: ...
    async def vram_bytes(self) -> int | None: ...


def validate_vectors(vectors, count, dimension=None):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedding provider returned an unexpected vector count")
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or not vector or (dimension is not None and len(vector) != dimension):
            raise ValueError("Embedding provider returned inconsistent dimensions")
        if dimension is None:
            dimension = len(vector)
        if any(not isinstance(n, (int, float)) or not math.isfinite(n) or abs(n) > 3.4e38 for n in vector):
            raise ValueError("Embedding provider returned invalid values")
    return dimension


class OllamaProvider:
    def embed(self, texts):
        if not texts:
            return []
        with httpx.Client(timeout=180) as client:
            response = client.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBEDDING_MODEL, "input": texts})
            response.raise_for_status()
            vectors = response.json()["embeddings"]
            validate_vectors(vectors, len(texts))
            return vectors

    async def stream(self, messages):
        payload = {"model": GENERATION_MODEL, "messages": messages, "stream": True,
                   "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}, "keep_alive": "15m"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("error"):
                        raise RuntimeError("Model generation failed")
                    content = item.get("message", {}).get("content", "")
                    if content:
                        yield ModelEvent("text", text=content)
                    if item.get("done") is True:
                        yield ModelEvent("metrics", metrics={key: item.get(key) for key in METRIC_KEYS})
                        return  # Do not accept trailing content or a second terminal event.
                raise RuntimeError("Model stream ended before completion")

    async def health(self):
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            names = [item["name"] for item in response.json().get("models", [])]
        return {"app": "ok", "ollama": True, "error": None,
                "generation_model": GENERATION_MODEL, "embedding_model": EMBEDDING_MODEL,
                "generation_ready": GENERATION_MODEL in names, "embedding_ready": EMBEDDING_MODEL in names}

    async def vram_bytes(self):
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(f"{OLLAMA_URL}/api/ps")
                response.raise_for_status()
                sizes = [m.get("size_vram") for m in response.json().get("models", [])
                         if m.get("name") == GENERATION_MODEL and isinstance(m.get("size_vram"), int)]
                return sizes[0] if sizes else None
        except Exception:
            return None


provider: Provider = OllamaProvider()


def safe_error(exc):
    if isinstance(exc, httpx.TimeoutException):
        return "Local model timed out; please retry."
    if isinstance(exc, (httpx.HTTPError, json.JSONDecodeError)):
        return "Local model unavailable or returned an invalid response; please retry."
    if isinstance(exc, ValueError) and str(exc).startswith("Embedding provider"):
        return str(exc)
    return "Generation failed; please retry."
