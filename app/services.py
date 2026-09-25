"""Compatibility imports for existing callers; implementations live in dedicated modules."""
import httpx
from .chat import ARTIFACT_PROMPTS, SYSTEM_PROMPT

from .documents import MAX_CHARS, CHUNK_SIZE, CHUNK_OVERLAP, clean_text, extract, split_pages, process_source, source_kind, safe_filename
from .retrieval import pack_vector, unpack_vector, cosine, retrieve, citations_for, evidence_text
from .models import OLLAMA_URL, GENERATION_MODEL, EMBEDDING_MODEL

def embed_texts(texts):
    from .models import provider
    return provider.embed(texts)


async def ollama_stream(messages):
    from .models import provider
    async for event in provider.stream(messages):
        if event.type == "text":
            yield event.text
