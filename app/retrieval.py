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
    if (re.search(rf"\b(?:one|1)\s+row\s+per\s+{subject}\b", q) or
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


def scoped_chunks(notebook_id, source_ids=None):
    if source_ids == []:
        return []
    params = [notebook_id]
    clause = "c.notebook_id=? AND s.notebook_id=c.notebook_id AND s.status='ready' AND s.enabled=1"
    if source_ids is not None:
        clause += f" AND c.source_id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    return db.rows(f"SELECT c.*,s.name AS source_name FROM chunks c JOIN sources s ON s.id=c.source_id WHERE {clause} ORDER BY s.created_at,s.id,c.ordinal,c.id", params)


def scoped_sources(notebook_id, source_ids=None):
    if source_ids == []:
        return []
    params = [notebook_id]
    clause = "notebook_id=? AND status='ready' AND enabled=1"
    if source_ids is not None:
        clause += f" AND id IN ({','.join('?' for _ in source_ids)})"
        params.extend(source_ids)
    return db.rows(f"SELECT id,name FROM sources WHERE {clause} ORDER BY created_at,id", params)


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


def _rank_candidates(notebook_id, query, lexical_query, candidates, source_ids, mode):
    vector = models.provider.embed([query])
    models.validate_vectors(vector, 1)
    similarities = {c["id"]: cosine(vector[0], unpack_vector(c["embedding"])) for c in candidates}
    semantic = sorted(candidates, key=lambda c: (-similarities[c["id"]], c["id"]))
    semantic_rank = {c["id"]: i for i, c in enumerate(semantic if mode != "focused" else semantic[:80], 1)}
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


def _rank_query(notebook_id, query, candidates, source_ids, mode, prior_user_questions):
    search_query = contextualize_retrieval_query(query, prior_user_questions)
    # A new explicit term in the current turn should dominate lexical matching;
    # pronoun-only turns borrow lexical terms from bounded user-question history.
    if search_query != query:
        lexical_query = " ".join(sorted((_tokens(query) - FOLLOWUP_WORDS) or
                                        (_tokens(search_query) - FOLLOWUP_WORDS)))
    else:
        lexical_query = query
    return _rank_candidates(notebook_id, search_query, lexical_query, candidates, source_ids, mode)


def retrieve(notebook_id, query, source_ids=None, limit=None, prior_user_questions=()):
    candidates = scoped_chunks(notebook_id, source_ids)
    if not candidates:
        return []
    mode = retrieval_mode(query)
    ranked = _rank_query(notebook_id, query, candidates, source_ids, mode, prior_user_questions)
    if mode != "focused":
        return _coverage_selection(ranked, limit)
    return _focused_selection(ranked, 12 if limit is None else limit)


def retrieve_by_source(notebook_id, query, source_ids=None, prior_user_questions=(),
                       chunks_per_source=3, source_budget=5800, max_sources=None):
    """One hybrid ranking pass, then bounded non-overlapping evidence per eligible source."""
    sources = scoped_sources(notebook_id, source_ids)
    if not sources:
        return []
    if max_sources is not None and len(sources) > max_sources:
        raise ValueError("Too many selected sources for a source-complete answer")
    candidates = scoped_chunks(notebook_id, source_ids)
    grouped = {source["id"]: [] for source in sources}
    if candidates:
        ranked = _rank_query(notebook_id, query, candidates, source_ids, "coverage", prior_user_questions)
        for item in ranked:
            grouped[item["source_id"]].append(item)
    result = []
    for source in sources:
        selected = []
        for item in grouped[source["id"]]:
            if len(selected) >= chunks_per_source:
                break
            if len(budgeted(selected + [item], source_budget)) == len(selected) + 1:
                selected.append(item)
        result.append((source, selected))
    return result


def citations_for(chunks):
    return [{"number": i, "chunk_id": c["id"], "source_id": c["source_id"],
             "source_name": c["source_name"], "page": c["page"], "excerpt": c["text"][:700]}
            for i, c in enumerate(chunks, 1)]


def evidence_text(chunks):
    # JSON string escaping separates untrusted source metadata/content from prompt structure.
    return "\n".join(evidence_line(c, i) for i, c in enumerate(chunks, 1))
