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
FOLLOWUP_WORDS = set("he him his she her hers they them their theirs it its that those these this same former latter other one ones previous".split())


def retrieval_mode(question):
    """Route explicit source-complete requests before general corpus coverage."""
    q = question.lower()
    subject = r"(?:papers?|stud(?:y|ies)|sources?)"
    selected = rf"(?:\d+\s+|one\s+|two\s+|three\s+)?selected\s+{subject}"
    if (re.search(rf"\bfor\s+(?:each|every)\s+(?:of\s+the\s+)?{selected}\b", q) or
            re.search(rf"\b(?:one|1|exactly\s+one)\s+row\s+(?:per|for\s+each)\s+(?:selected\s+)?{subject}\b", q) or
            re.search(rf"\b(?:compare|list|summari[sz]e)\s+(?:each|every)\s+{selected}\b", q) or
            re.search(rf"\b(?:one|1)\s+row\s+per\s+{subject}\b", q) or
            re.search(rf"\b(?:compare|list|summari[sz]e)\s+(?:each|every)\s+(?:of\s+the\s+)?(?:\d+\s+)?{subject}\b", q) or
            re.search(rf"\bfor\s+each\s+{subject}\b", q) or
            re.search(rf"\b(?:complete|comprehensive|exhaustive)\s+table\b.*\b(?:all|every|each)\b.*\b{subject}\b", q) or
            re.search(rf"\btable\b.*\b(?:covering|for|across)\s+(?:all|every|each)\s+(?:selected\s+)?(?:\d+\s+)?{subject}\b", q)):
        return "exhaustive"
    if re.search(r"\b(?:across|among)\s+(?:all\s+|these\s+|the\s+)?(?:papers|studies|sources|authors)\b", q):
        return "coverage"
    if re.search(r"\b(?:each|all)\s+(?:of\s+)?(?:the\s+|these\s+)?(?:papers?|sources?|stud(?:y|ies)|authors?)\b", q):
        return "coverage"
    if re.search(r"\b(?:compare|comparison of)\s+(?:all\s+|the\s+|these\s+)?(?:papers|studies|sources)\b", q):
        return "coverage"
    if re.search(r"\b(?:common|main|shared)\s+themes\b", q):
        return "coverage"
    if re.search(r"\b(?:papers|studies|sources)\b.*\b(?:similarities|differences|agree|disagree)\b|\b(?:similarities|differences|agree|disagree)\b.*\b(?:papers|studies|sources)\b", q):
        return "coverage"
    if re.search(r"\b(?:connections|relationships)\s+(?:among|between)\s+authors\b", q):
        return "coverage"
    return "focused"


def is_contextual_followup(question):
    q = question.strip().lower()
    if len(q.split()) > 18:
        return False
    return bool(re.match(r"^(?:and\b|about\b|what about\b|how about\b)", q) or
                re.search(r"\b(?:he|him|his|she|her|hers|they|them|their|theirs|that|those|the former|the latter|the previous one|same)\b", q))


def contextualize_retrieval_query(question, prior_user_questions=()):
    """Bounded retrieval aid; never persisted or supplied as source evidence."""
    if not is_contextual_followup(question) or not prior_user_questions:
        return question
    previous = [q.strip()[:500] for q in prior_user_questions[-2:] if q.strip()]
    return " ".join([question] + previous) if previous else question


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


def scoped_chunks(notebook_id, source_ids=None, model_digest=None):
    if source_ids == []:
        return []
    params = [notebook_id]
    clause = "c.notebook_id=? AND s.notebook_id=c.notebook_id AND s.status='ready' AND s.enabled=1"
    if source_ids is not None:
        clause += f" AND c.source_id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    if model_digest:
        return db.rows(f"SELECT c.*,s.name AS source_name,e.embedding AS semantic_embedding,"
                       f"e.dimensions AS index_dimensions FROM chunks c JOIN sources s ON s.id=c.source_id "
                       f"LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id AND e.model_name=? AND e.model_digest=? "
                       f"WHERE {clause} ORDER BY s.created_at,s.id,c.ordinal,c.id",
                       [models.EMBEDDING_MODEL, model_digest, *params])
    return db.rows(f"SELECT c.*,s.name AS source_name,NULL AS semantic_embedding,NULL AS index_dimensions "
                   f"FROM chunks c JOIN sources s ON s.id=c.source_id WHERE {clause} "
                   f"ORDER BY s.created_at,s.id,c.ordinal,c.id", params)


def scoped_sources(notebook_id, source_ids=None):
    if source_ids == []:
        return []
    params = [notebook_id]
    clause = "notebook_id=? AND status='ready' AND enabled=1"
    if source_ids is not None:
        clause += f" AND id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    return db.rows(f"SELECT id,name FROM sources WHERE {clause} ORDER BY created_at,id", params)


def rebuild_embedding_index(notebook_id=None, batch_size=16):
    """Build the current model revision in the background; queries use lexical fallback meanwhile."""
    digest = models.embedding_model_digest()
    if not digest:
        return {"status": "model_digest_unavailable", "indexed_chunks": 0}
    params = [models.EMBEDDING_MODEL, digest]
    clause = ""
    if notebook_id is not None:
        clause = " AND c.notebook_id=?"
        params.append(notebook_id)
    missing = db.rows("SELECT c.id,c.text FROM chunks c JOIN sources s ON s.id=c.source_id "
                      "LEFT JOIN chunk_embeddings e ON e.chunk_id=c.id AND e.model_name=? AND e.model_digest=? "
                      "WHERE s.status='ready' AND e.chunk_id IS NULL" + clause + " ORDER BY c.id", params)
    indexed = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        vectors = models.provider.embed([item["text"] for item in batch])
        dimension = models.validate_vectors(vectors, len(batch))
        if models.embedding_model_digest() != digest:
            return {"status": "model_changed", "indexed_chunks": indexed}
        with db.connection() as conn:
            for item, vector in zip(batch, vectors):
                conn.execute("INSERT OR IGNORE INTO chunk_embeddings(chunk_id,model_name,model_digest,dimensions,embedding) "
                             "SELECT c.id,?,?,?,? FROM chunks c JOIN sources s ON s.id=c.source_id "
                             "WHERE c.id=? AND s.status='ready'",
                             (models.EMBEDDING_MODEL, digest, dimension, pack_vector(vector), item["id"]))
                indexed += 1
    return {"status": "ready", "indexed_chunks": indexed}


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
    return json.dumps({"citation": f"C{number}", "source": chunk["source_name"], "page": chunk["page"],
                       "text": chunk["text"]}, ensure_ascii=False)


def budgeted(chunks, budget=None, number_start=1):
    if budget is None:
        budget = EVIDENCE_BUDGET
    selected, used = [], 0
    for chunk in chunks:
        cost = len(evidence_line(chunk, number_start + len(selected))) + (1 if selected else 0)
        if used + cost > budget:
            continue
        if any(_overlap(chunk, old) for old in selected):
            continue
        selected.append(chunk)
        used += cost
    return selected


def _index_models(notebook_id, source_ids):
    params = [notebook_id]
    clause = ""
    if source_ids is not None:
        clause = f" AND c.source_id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    return db.rows("SELECT DISTINCT e.model_name,e.model_digest,e.dimensions FROM chunk_embeddings e "
                   "JOIN chunks c ON c.id=e.chunk_id WHERE c.notebook_id=?" + clause, params)


def _rank_candidates(notebook_id, query, lexical_query, candidates, source_ids, mode,
                     diagnostics=None, model_digest=None):
    index_rows = _index_models(notebook_id, source_ids)
    names = {row["model_name"] for row in index_rows}
    index_model = (models.EMBEDDING_MODEL if any(row["model_name"] == models.EMBEDDING_MODEL and
                                                 row["model_digest"] == model_digest for row in index_rows)
                   else next(iter(names)) if len(names) == 1 else "mixed" if names else None)
    similarities = {c["id"]: 0.0 for c in candidates}
    semantic = []
    query_dimensions = None
    reason = None
    indexed = [c for c in candidates if c["semantic_embedding"] is not None]
    if not model_digest:
        reason = "model_digest_unavailable"
    elif not indexed:
        reason = ("stale_embedding_model" if names and models.EMBEDDING_MODEL not in names else
                  "stale_embedding_digest" if names else "missing_embedding_index")
    else:
        try:
            vectors = models.provider.embed([query])
            query_dimensions = models.validate_vectors(vectors, 1)
            vector = vectors[0]
            if not any(vector):
                reason = "zero_query_vector"
            else:
                compatible = [c for c in indexed if c["index_dimensions"] == query_dimensions and
                              len(c["semantic_embedding"]) == query_dimensions * 4]
                if not compatible:
                    reason = "dimension_mismatch"
                else:
                    semantic = [c for c in compatible if any(unpack_vector(c["semantic_embedding"]))]
                    if not semantic:
                        reason = "zero_document_vectors"
                    for c in semantic:
                        similarities[c["id"]] = cosine(vector, unpack_vector(c["semantic_embedding"]))
                    semantic.sort(key=lambda c: (-similarities[c["id"]], c["id"]))
                    if len(semantic) < len(candidates) and reason is None:
                        reason = "partial_embedding_index"
                    elif semantic and not any(similarities[c["id"]] >= MIN_SIMILARITY for c in semantic):
                        reason = "no_semantic_hits"
        except Exception:
            reason = "query_embedding_failed"
            semantic = []
    semantic_rank = {c["id"]: i for i, c in enumerate(semantic if mode != "focused" else semantic[:80], 1)}
    if diagnostics is not None:
        dimensions = {row["dimensions"] for row in index_rows
                      if row["model_name"] == models.EMBEDDING_MODEL and row["model_digest"] == model_digest}
        diagnostics.update({"embedding_index_model": index_model,
                            "query_embedding_model": models.EMBEDDING_MODEL,
                            "embedding_dimensions": {"index": next(iter(dimensions)) if len(dimensions) == 1 else None,
                                                     "query": query_dimensions},
                            "semantic_candidates": len(semantic),
                            "semantic_hits": sum(similarities[c["id"]] >= MIN_SIMILARITY for c in semantic),
                            "semantic_fallback_reason": reason})
    lexical_rank = {}
    fts = _fts_query(lexical_query)
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
    meaningful = _tokens(lexical_query)
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
    return ranked if mode != "focused" else ranked[:80]


def _focused_selection(ranked, limit):
    # Only actual ranked candidates compete for diversity; no zero-score filler.
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


def _coverage_selection(ranked, limit):
    # Each round visits every represented source before taking another chunk.
    grouped = {}
    for item in ranked:
        grouped.setdefault(item["source_id"], []).append(item)
    chosen = []
    for round_index in range(2):
        for source_chunks in grouped.values():
            if limit is not None and len(chosen) >= limit:
                return chosen
            if sum(c["source_id"] == source_chunks[0]["source_id"] for c in chosen) > round_index:
                continue
            for item in source_chunks:
                if item in chosen or any(_overlap(item, c) for c in chosen):
                    continue
                if len(budgeted(chosen + [item])) == len(chosen) + 1:
                    chosen.append(item)
                    break
    return chosen


def _rank_query(notebook_id, query, candidates, source_ids, mode, prior_user_questions,
                diagnostics=None, model_digest=None):
    search_query = contextualize_retrieval_query(query, prior_user_questions)
    # A new explicit term in the current turn should dominate lexical matching;
    # pronoun-only turns borrow lexical terms from bounded user-question history.
    if search_query != query:
        lexical_query = " ".join(sorted((_tokens(query) - FOLLOWUP_WORDS) or
                                        (_tokens(search_query) - FOLLOWUP_WORDS)))
    else:
        lexical_query = query
    return _rank_candidates(notebook_id, search_query, lexical_query, candidates, source_ids, mode,
                            diagnostics, model_digest)


def retrieve(notebook_id, query, source_ids=None, limit=None, prior_user_questions=(), diagnostics=None):
    digest = models.embedding_model_digest()
    candidates = scoped_chunks(notebook_id, source_ids, digest)
    if not candidates:
        return []
    mode = retrieval_mode(query)
    ranked = _rank_query(notebook_id, query, candidates, source_ids, mode, prior_user_questions,
                         diagnostics, digest)
    if mode != "focused":
        return _coverage_selection(ranked, limit)
    return _focused_selection(ranked, 12 if limit is None else limit)


def exhaustive_budget(source_count):
    """Return per-source hard chunk and character limits for complete coverage."""
    if source_count <= 2:
        return 12, 12_000
    if source_count <= 5:
        return 9, 9_000
    if source_count <= 10:
        return 6, 6_000
    return 4, 4_800


def _section_key(item):
    section = item.get("section")
    if not isinstance(section, str) or section.strip().lower() in {"", "unknown"}:
        return None
    return section.strip().casefold()


def retrieve_by_source(notebook_id, query, source_ids=None, prior_user_questions=(),
                       chunks_per_source=None, source_budget=None, max_sources=None, diagnostics=None):
    """One hybrid ranking pass, then fill each source's bounded evidence budget."""
    sources = scoped_sources(notebook_id, source_ids)
    if not sources:
        return []
    if max_sources is not None and len(sources) > max_sources:
        raise ValueError("Too many selected sources for a source-complete answer")
    default_chunks, default_budget = exhaustive_budget(len(sources))
    chunks_per_source = default_chunks if chunks_per_source is None else chunks_per_source
    source_budget = default_budget if source_budget is None else source_budget
    digest = models.embedding_model_digest()
    candidates = scoped_chunks(notebook_id, source_ids, digest)
    grouped = {source["id"]: [] for source in sources}
    if candidates:
        ranked = _rank_query(notebook_id, query, candidates, source_ids, "coverage", prior_user_questions,
                             diagnostics, digest)
        for item in ranked:
            grouped[item["source_id"]].append(item)
    result = []
    selected_counts, selected_chars, selected_sections = {}, {}, {}
    citation_start = 1
    for source in sources:
        ranked = grouped[source["id"]]
        selected, seen_sections = [], set()
        # First take the best eligible chunk from each named section in rank order.
        # Then continue through the ranking until the character or chunk ceiling
        # is reached, preserving exhaustive coverage within those hard bounds.
        for item in ranked:
            section = _section_key(item)
            if section is None or section in seen_sections:
                continue
            if len(selected) >= chunks_per_source:
                break
            if len(budgeted(selected + [item], source_budget, citation_start)) == len(selected) + 1:
                selected.append(item)
                seen_sections.add(section)
        # Then fill any remaining slots with the original hybrid ranking.
        for item in ranked:
            if len(selected) >= chunks_per_source:
                break
            if item in selected or any(_overlap(item, old) for old in selected):
                continue
            if len(budgeted(selected + [item], source_budget, citation_start)) == len(selected) + 1:
                selected.append(item)
        selected_counts[source["id"]] = len(selected)
        selected_chars[source["id"]] = sum(
            len(evidence_line(item, citation_start + index)) + (1 if index else 0)
            for index, item in enumerate(selected))
        selected_sections[source["id"]] = len({_section_key(item) for item in selected if _section_key(item)})
        result.append((source, selected))
        citation_start += len(selected)
    if diagnostics is not None:
        diagnostics.update({"chunks_per_source_selected": selected_counts,
                            "evidence_chars_per_source": selected_chars,
                            "distinct_sections_per_source": selected_sections,
                            "exhaustive_chunks_per_source_limit": chunks_per_source,
                            "exhaustive_source_evidence_budget_chars": source_budget})
    return result


def citations_for(chunks):
    return [{"number": i, "namespace": "C", "chunk_id": c["id"], "source_id": c["source_id"],
             "source_name": c["source_name"], "page": c["page"], "excerpt": c["text"][:700]}
            for i, c in enumerate(chunks, 1)]


def evidence_text(chunks):
    # JSON string escaping separates untrusted source metadata/content from prompt structure.
    return "\n".join(evidence_line(c, i) for i, c in enumerate(chunks, 1))
