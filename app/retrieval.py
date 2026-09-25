"""Brute-force notebook-local retrieval; similarity is a heuristic, not entailment."""
import json
import math
import os
import re
import struct

from . import db, models

MIN_SIMILARITY = float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.35"))
EVIDENCE_BUDGET = 42_000
STOPWORDS = set("a an and are as at be by can do does for from how i in is it of on or our should the their this to was what when where which who why with you your about create concise comprehensive notebook overview key claims important evidence disagreements open questions useful faq guide structured study briefing source sources answer using only summarize".split())


def pack_vector(values):
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob):
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def scoped_chunks(notebook_id, source_ids=None):
    if source_ids == []:
        return []
    params = [notebook_id]
    clause = "c.notebook_id=? AND s.notebook_id=c.notebook_id AND s.status='ready' AND s.enabled=1"
    if source_ids is not None:
        clause += f" AND c.source_id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    return db.rows(f"SELECT c.*,s.name AS source_name FROM chunks c JOIN sources s ON s.id=c.source_id WHERE {clause} ORDER BY s.created_at,s.id,c.ordinal,c.id", params)


def _tokens(text):
    return set(re.findall(r"[\w]{3,}", text.lower(), re.UNICODE)) - STOPWORDS


def _fts_query(query):
    tokens = re.findall(r"[\w]{2,}", query.lower(), re.UNICODE)[:12]
    return " OR ".join(f'"{token}"' for token in tokens)


def _overlap(a, b):
    # Source-aware near-duplicate suppression; non-overlapping chunks of the same source remain eligible.
    if a["source_id"] != b["source_id"]:
        return False
    x, y = a["text"], b["text"]
    return x == y or (min(len(x), len(y)) > 80 and
                      (x[:100] in y or y[:100] in x or x[-100:] in y or y[-100:] in x))


def evidence_line(chunk, number):
    # Serialize once for both the prompt and its exact character budget.
    return json.dumps({"citation": number, "source": chunk["source_name"], "page": chunk["page"],
                       "text": chunk["text"]}, ensure_ascii=False)


def budgeted(chunks, budget=None):
    if budget is None:
        budget = EVIDENCE_BUDGET
    selected, used = [], 0
    for chunk in chunks:
        cost = len(evidence_line(chunk, len(selected) + 1)) + (1 if selected else 0)
        if used + cost > budget:
            continue
        if any(_overlap(chunk, old) for old in selected):
            continue
        selected.append(chunk)
        used += cost
    return selected


def retrieve(notebook_id, query, source_ids=None, limit=12):
    candidates = scoped_chunks(notebook_id, source_ids)
    if not candidates:
        return []
    vector = models.provider.embed([query])
    models.validate_vectors(vector, 1)
    similarities = {c["id"]: cosine(vector[0], unpack_vector(c["embedding"])) for c in candidates}
    semantic = sorted(candidates, key=lambda c: (-similarities[c["id"]], c["id"]))
    semantic_rank = {c["id"]: i for i, c in enumerate(semantic[:80], 1)}
    lexical_rank = {}
    fts = _fts_query(query)
    if fts:
        params = [fts, notebook_id]
        clause = ""
        if source_ids is not None:
            clause = f" AND c.source_id IN ({','.join('?' for _ in source_ids)})"
            params.extend(source_ids)
        matches = db.rows("SELECT c.id FROM chunk_fts f JOIN chunks c ON c.id=f.chunk_id "
                          "JOIN sources s ON s.id=c.source_id WHERE chunk_fts MATCH ? AND c.notebook_id=? "
                          f"AND s.notebook_id=c.notebook_id AND s.status='ready' AND s.enabled=1{clause} ORDER BY bm25(chunk_fts),c.id LIMIT 80", params)
        lexical_rank = {c["id"]: i for i, c in enumerate(matches, 1)}
    meaningful = _tokens(query)
    ranked = []
    for item in candidates:
        cid = item["id"]
        # Lexical rescue requires a non-instruction content word, not just a generic prompt term.
        rescue = cid in lexical_rank and bool(meaningful & _tokens(item["text"]))
        if similarities[cid] < MIN_SIMILARITY and not rescue:
            continue
        item["similarity"] = similarities[cid]
        item["score"] = (1 / (60 + semantic_rank[cid]) if cid in semantic_rank else 0) + (1 / (60 + lexical_rank[cid]) if cid in lexical_rank else 0)
        ranked.append(item)
    ranked = [c for c in ranked if c["score"] > 0]
    ranked.sort(key=lambda c: (-c["score"], -c["similarity"], c["id"]))
    ranked = ranked[:80]  # Only actual ranked candidates compete for diversity; no zero-score filler.
    # Round-robin diverse sources before filling remaining slots from the ranked list.
    chosen = []
    seen_sources = set()
    for item in ranked:
        if item["source_id"] not in seen_sources and not any(_overlap(item, c) for c in chosen):
            chosen.append(item)
            seen_sources.add(item["source_id"])
            if len(chosen) >= limit:
                return budgeted(chosen)
    for item in ranked:
        if item not in chosen and not any(_overlap(item, c) for c in chosen):
            chosen.append(item)
            if len(chosen) >= limit:
                break
    return budgeted(chosen)


def citations_for(chunks):
    return [{"number": i, "chunk_id": c["id"], "source_id": c["source_id"],
             "source_name": c["source_name"], "page": c["page"], "excerpt": c["text"][:700]}
            for i, c in enumerate(chunks, 1)]


def evidence_text(chunks):
    # JSON string escaping separates untrusted source metadata/content from prompt structure.
    return "\n".join(evidence_line(c, i) for i, c in enumerate(chunks, 1))
