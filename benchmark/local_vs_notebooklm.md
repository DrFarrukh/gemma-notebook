# Local vs NotebookLM — NUST_Policies evaluation

Method: the five-question follow-up chain is run through the real local pipeline
(rewrite → prior-source scoping → recursive retrieval → cited generation) on a
private copy of the notebook database. `[Cn]` citations map to notebook chunks
(source + page). Answers are scored against the checkpoints extracted from the
NotebookLM baseline (`benchmark/notebooklm_baseline.md`).

## Runs

| Run | Model | Context | Thinking | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|---|---|
| 1 | qwen3.5:4b | 16384 | auto | partial | strong | good | partial | **failed** (no citations) |
| 2 | qwen3.5:4b | 16384 | off | partial + 1 wrong claim | **beats baseline** | good | partial | **wrong** (10 CH vs 16) |

Thinking `off` also cut generation time roughly in half (20–57 s vs 52–194 s per
question) and was the difference between Q5 failing and Q5 producing a cited
answer at all.

## Scoring, run 2

### Q1 — Ex-Pakistan Leave, probation, pay: partial, one wrong claim
- [x] EPL definition (apply/proceed abroad, specific request) `[C4]`
- [x] 30 days at a time, limited to Earned Leave balance `[C4]`
- [ ] Sanctioning chain: says "Competent Authority = Rector or delegate" `[C13]`;
      baseline's Pro-Rector (Academics) → HoD → Dean → Principal chain is not
      supported by the indexed Leave Policy chunks
- [x] EOL requires removal from probation `[C6]`
- [ ] **Wrong claim:** "No, you cannot take EPL while on probation" — the policy
      restricts only EOL (Leave Policy p.7, 6.4.4.1); EPL against an EL balance
      is not explicitly probation-restricted (baseline gets this nuance right)
- ~ Pay: "with/without pay depending on authorization" `[C12]`; misses the clean
      EL = full pay / beyond EL = without pay distinction
- [ ] Start/termination (leaves office → resumes duty) not stated

### Q2 — 2-year leave: beats baseline
- [x] Long leave = Study Leave (up to 5 years) and EOL (2 years at a stretch)
- [x] EOL: 4 years total, 2 at a stretch, no extension, 3-year gap `[C3]`
- [x] EOL requires removal from probation `[C3]`
- [x] Without pay; no increments; no pension/gratuity credit `[C3]`
- [x] Study Leave ≥ 2 years service, bond, not beyond contract `[C1]`
- Adds Sabbatical (1 year, basic pay) and LPR (365 days) — more complete than the
  baseline, all cited.

### Q3 — EOL timing on probation (no baseline answer): good
- [x] Cannot take EOL while on probation; must be removed first `[C2]`
- [x] 2-year stretch, 4-year total, 3-year gap, Competent Authority `[C2]`
- [x] Correctly states no personal joining-date records exist in the sources;
      treats the policy as general rules. Grounded and cited.

### Q4 — Faculty load min/max: partial
- [x] Minimum ~12 CH per regular semester; 8 CH newly joined PhD `[C1][C8]`
- [x] Teaching formula 12 − Research − Institutional Services; 8 − R − I `[C1]`
- [x] Minimum 3 CH classroom teaching `[C1]`
- [x] Research 0–6 CH; MS 0.75 / PhD 1.0 / co-supervisor 0.5; funding CHEL tiers `[C11]`
- [x] Institutional services up to 3 CH (HoD/EIC 3, SAS 1, top-5% 3) `[C9]`
- [ ] Misses the **4 CH exception for Associate Dean / Director Acad (CEME)**
      (`Faculty load.pdf` p.3) and states 3 CH as the max
- [ ] "No explicit total maximum" framing is only implied ("can be higher")
- [ ] Shared-faculty +3 CH (baseline [7]) not supported by any indexed phrase —
      treat the baseline point as unverified (possible NotebookLM conflation)

### Q5 — Total load: wrong
- [x] Supervision values: 8 MS × 0.75 = 6 CH; 1 PhD = 1 CH → 7 CH `[C4]`
- [ ] **Wrong:** claims "1 CH per course" → teaching 3 CH (a 3-CH MS course
      counts its CH; the policy does not define 1 CH per course)
- [ ] Misses the **+1 CH multiple-course bonus** (`Faculty load.pdf` p.2)
- [ ] Misses the **6 CH research cap** — reports 7 CH uncapped
- Result: 10 CH (or an alternative 15 CH) instead of the baseline's 16 CH

## Evidence availability (verified by probing the indexed chunks)

- Q4 omissions and Q5 rules are **all present in supplied evidence**:
  multi-course +1 CH and teaching formula on `Faculty load.pdf` p.2; research
  0–6 CH and supervision values p.3; Assoc Dean/Director 4 CH p.3; the 3 CH cap
  follows immediately after. These are generation/reasoning failures, not
  retrieval failures.
- The **probation restriction is explicitly EOL-only** (`Leave Policy.pdf` p.7,
  6.4.4.1). Q1's "No EPL on probation" is an over-generalization.
- **Two notebook sources failed local indexing** (`TVF Hiring.pdf`, `AAR.pdf`).
  NotebookLM indexed all sources, so any fact living only there (possibly parts
  of the Q1 sanction matrix) is not reachable locally until reprocessed.
- "Pro-Rector" does not appear in the Leave Policy chunks; EPL sanctioning in the
  indexed text uses "Competent Authority / Sanctioning Authority" language.

## Verdict

With `qwen3.5:4b` at 16k, thinking **off**:

- **Q2 beats NotebookLM** (more complete, fully cited).
- **Q3 is good** (baseline has no answer to compare).
- **Q1 and Q4 are partial** — right structure, missing nuance/exception points.
- **Q5 is wrong** — a 4B arithmetic/policy-composition failure with the evidence
  sitting in context.

Not yet matching overall. The blockers are reasoning-heavy questions (Q1 nuance,
Q5 composition), not retrieval. Next candidates, in user-preference order:
rerun the same harness with **gemma4:e4b** (the app's default; user reports it is
clearly better than Qwen), then `qwen3.5:9b-16k` (fits 100% GPU when loaded
alone), and retry the two failed source imports.
