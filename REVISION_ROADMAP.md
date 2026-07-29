# VotIE — Camera-Ready Revision Roadmap

**Paper:** VotIE: Information Extraction from Meeting Minutes
**Venue:** ACL Findings (short paper) · **Scores:** 3 / 3 / 3 / 2 (overall), Soundness 3.5 / 2 / 2 / 3
**Manuscript:** `ACL_CitiLink___Vote_Identification__José_Evans_/latex/acl_latex.tex`
**Status of rebuttal:** sent; R2 (3fsk) and R3 (iyoP) explicitly confirmed the framing clarification resolves their main concern.

---

## Overview

| | |
|---|---|
| Decision | Camera-ready (Findings) |
| Total distinct comments | 31 |
| By type | 5 Major · 11 Minor · 12 Editorial · 3 Positive |
| Estimated effort | **Moderate** (Tier A only: 2–3 days) → **Substantial** if Tier B adopted (+1 week) |
| Binding constraint | ACL short-paper camera-ready = **5 content pages**; Limitations/Ethics/References/Appendices are exempt |

**Reviewer ID mapping** (reviews.txt order → rebuttal handles):
`Review 1` = R1 = **3fsk** · `Review 2` = R2 = **YpBJ** · `Review 3` = R3 = **iyoP** · `Review 4` = R4 = **4SQ9**
> Note: `my_responses.txt` addresses them in a different order than `reviews.txt` lists them. Confirm before reusing any text.

---

## Cross-Reviewer Patterns (highest leverage)

| Pattern | Raised by | Weight |
|---|---|---|
| **Event-extraction framing vs. flat-span evaluation** | R1, R2, R3, R4 (**all four**) | The single unifying criticism. Drives both Soundness-2 scores. |
| **References must precede Appendices** | R2, R3, R4 | Trivial fix, three reviewers noticed → zero excuse to miss |
| **Low-support types make per-type F1 meaningless** | R1, R3 | Needs explicit caveat at point of use, not just in Limitations |
| **Contribution is benchmark, not new data/architecture** | R3, R4 | Needs a precise contribution statement, not a defensive one |

---

## P1 — Must Fix

| # | Comment | Rev | Type | Section | Action |
|---|---|---|---|---|---|
| 1 | Framing overstates task: "event extraction" but evaluation is flat spans | All 4 | Major | Title/Abstract/§1/§3/§7 | Sweep to **"span-level event-argument extraction"**. Add one explicit scope sentence in Abstract + §1 + §3. Keep the EE grounding (rebuttal defended it, R2/R3 accepted) but state the boundary at every claim site. |
| 2 | References placed after appendices | R2,R3,R4 | Editorial | Structure | Move `\bibliography{custom}` (L1032) **above** `\appendix` (L449). |
| 3 | Build is in review mode | — | Editorial | Preamble | `\usepackage[review]{acl}` → `[final]` (L5). |
| 4 | Author block is template placeholder | — | Editorial | Preamble | L65–67 still reads `Address line` / `Author n`. Fill real authors + affiliations. |
| 5 | Anonymised artifact URLs | — | Editorial | §1 footnotes | De-anonymise 3× `Anonymous3445` links (HF model, GitHub, HF demo) at L167. |
| 6 | Report LLM span-discard rate | R1 | Major | §4 + new appendix | **Needs re-run — see Gap A.** Current `results/` cannot answer this. |
| 7 | Report API cost + latency for Gemini/GPT | R1 | Major | App. C | Latency already recoverable; cost needs estimation or re-run. **See Gap B.** |
| 8 | Contribution scope: annotations come from CitiLink-Minutes | R3,R4 | Major | §1 | State plainly: annotations are from CitiLink-Minutes; *this* paper contributes task formulation, schema, splits, protocol, baselines, models, code, demo. Rebuttal's 8-item list is good — compress to one sentence. |

---

## P2 — Should Fix

| # | Comment | Rev | Type | Section | Action |
|---|---|---|---|---|---|
| 9 | 12 types vs 11-type macro average not explicit in main results | R1 | Minor | §5 + Tab.1 | Currently only in §4 prose (L294) and App. table caption (L757). Add to **Table 1 caption** and first results sentence. |
| 10 | Exact vs relaxed match conflates use cases | R1 | Minor | §5 | Separate **legal traceability** (exact) from **information access / search** (relaxed). The 32→8 pt collapse is the evidence; say what each regime implies. |
| 11 | Low-support types (n=4,5) → near-meaningless F1 | R1,R3 | Minor | App. D | Add caveat directly in Tab. 11 caption; grey out or asterisk n<10 columns. R3 specifically cited the 100% scores. |
| 12 | No concrete error examples | R3 | Minor | §6 | **Data already exists** — see Gap C. Add a qualitative example table. |
| 13 | Char-offset formalism vs token-level BIO implementation | R3 | Minor | §3 | Add 1–2 sentences: spans defined on char offsets; BIO models operate on tokens; conversion + round-trip is exact because gold boundaries align to token boundaries (**verify this claim before asserting it**). |
| 14 | Overlong/opaque sentences | R3 | Minor | §3 | R3 quoted the sentence now at **L201** (ends "…isolated lexical mentions"). Split into 2–3 sentences. Sweep §3 generally. |
| 15 | "theoretically language-agnostic" overclaims | R3,R4 | Minor | §8 | L435 → schema is *conceptually* portable; empirical validation is Portuguese-only; multilingual validation = future work. |
| 16 | LLM conclusions may be prompting-dependent | R1 | Minor | §5/§7 | Add the rebuttal's hedge: within *this* controlled protocol; prompting/chunking/alignment are improvable axes the benchmark exists to measure. |
| 17 | Multi-vote limitation understated | R1,R3 | Minor | §8 | Paper cites 95.3% single-trigger **test** segments (verified: 25/529 = 4.73% multi). But train = 7.99%, dev = 7.59% — test *under-represents* the hard case. Disclose all three, don't cite test alone. |
| 18 | Software rated 1/5 by R2 despite full release | R2 | Minor | §1/App. | R2 marked "No usable software released" — likely missed the links. Make the artifact footnotes prominent; consider a short "Released Artifacts" paragraph. |

---

## P3 — Consider

| # | Item | Rev | Action |
|---|---|---|---|
| 19 | `with he same document` → `within the same document` | R1 | L792 |
| 20 | `the lengthier among` → `the longest among` | R1 | L808 |
| 21 | `costs makes` → `costs make` | R1,R4 | L420 |
| 22 | `in Portugues` → `in Portuguese` | R1 | L554 |
| 23 | `Macro averaged` → `Macro-averaged` | R1 | Tab. 1 caption (L307) |
| 24 | Notation inconsistency | R1 | `VOTER-AGST` (L221 figure legend) vs `VOTER-AGAINST` (L576, L628) vs `V-Agn` (L824, L926). Pick one full form + one table abbreviation; define the abbreviation set once. |
| 25 | Missing terminal periods | R3 | L290 (`…approximate matching`), L433 (`…such procedures`), L843 (`…un-oversampled test set`) |
| 26 | sentence-BERT gap needs explanation | R1 | L363 — say *why* semantic similarity gap is smaller: shared topics, divergent surface form |
| 27 | Verify Table 3 against source JSON | — | Paper: XLM-R-CRF SPU `48±16`; `error_classification_results.json`: `64`. At the interval edge — confirm which seeds/aggregation produced the table. |
| 28 | Page budget check | — | Confirm content ≤5 pages after edits; push all new material to appendices |

---

## Positive Comments (acknowledge only)

| # | Comment | Rev |
|---|---|---|
| 29 | LOMO cross-municipality protocol is an excellent design choice | R1, R3, R4 |
| 30 | Comprehensive baseline coverage across two paradigms | R3 |
| 31 | Thorough limitations section; task is practically valuable | R1, R3, R4 |

---

# Gaps Requiring New Scripts or Data

## Gap A — LLM span-discard rate *(blocks P1 #6)*

**R1 asked:** "How many generated spans were discarded because they could not be matched exactly?"

**Finding: the number currently in `results/` is not the answer.**

Every `*_evaluation.json` reports `alignment_fidelity = 1.0`, `failed_spans = 0`. This is an artifact of the measurement point:

- `src/llm_extraction/span_alignment.py:268` `align_extraction_result()` computes the real discard stats (`failed_alignments`) — but they are only `logger.info`'d (e.g. `gemini/extractor.py:121-124`), never persisted.
- The saved `*.jsonl` contains **post-alignment** entities only.
- `calculate_fidelity_metrics()` (`span_alignment.py:352`) then re-validates those already-aligned spans → necessarily 100%.
- `raw_response` is `None` in every record, so pre-alignment output is unrecoverable.

**Reporting 0% discarded would be wrong.** There are two real drop points, both unmeasured: langextract raw extractions → typed entities (`_convert_result`, gemini/extractor.py:139), and typed entities → aligned spans.

**Work required**
1. Patch the three extractors to persist `n_raw`, pre-alignment entities, and `alignment_stats` per document.
2. Re-run to measure. Minimum credible scope: **Gemini few-shot** (the headline generative system). Ideally + GPT few-shot.
3. New script `scripts/llm_extraction/alignment_report.py` → discard rate overall and per entity type.
4. Add a short appendix subsection + one sentence in §4.

**Cost:** 529 docs × ~1.74k chars ≈ 263k input tokens per config, before few-shot and per-chunk prompt overhead (langextract resends the prompt per chunk — true billed volume is materially higher). Greedy decoding is already configured, so re-runs are near-deterministic but not identical.

> **Fallback if no re-run budget:** report the *mechanism* honestly ("spans that cannot be re-aligned are discarded; the pipeline logs but did not retain per-run counts") and state that all evaluated spans are verbatim-grounded. Weaker, but not false. Do **not** report 0%.

## Gap B — API cost and latency *(blocks P1 #7)*

**Latency: already available**, no re-run needed. `processing_time` per document in each `.jsonl` (mean s/doc over 529 docs):

| Model | zero-shot | few-shot |
|---|---|---|
| Gemini 2.5 Pro | 27.8 | 26.9 |
| GPT 5.5 | 22.8 | 15.0 |
| AMALIA 8B (local) | 52.9 | 89.2 |

AMALIA figures already match Table 10 (52.9 / 89.2 s) — confirms the field is the right one.

**Cost: not recoverable.** No token-usage capture in the extractors. Options:
- (a) Add usage accounting and re-run — exact, combines with Gap A.
- (b) Estimate from character counts × published per-token pricing, clearly labelled as an estimate with the chunking multiplier stated.

Recommend (a) if Gap A is re-run anyway; otherwise (b). Extend Table 10 with a throughput + cost column and drop the current "no GPU costs were measured" sentence (L851), which reads as a gap rather than a design choice.

## Gap C — Concrete error examples *(P2 #12)* — **no new data needed**

`results/error_classification_results.json` already contains `example_errors` per model, per category (MIS/SPU/INC_BOUNDARY/INC_TYPE) with `doc_id`, gold span, predicted span, and `**highlighted**` context. Only a formatting script is needed:

`scripts/make_error_examples_table.py` → LaTeX table, 4–6 examples (prioritise SUBJECT boundary errors — 123 of XLM-R-CRF's 154 boundary errors), with English glosses.

## Gap D — **Gold event grouping already exists in the released data** ⚠️

**This is the most consequential finding of the audit.**

Every gold span in `data/citilink-votie/*.jsonl` carries an **`event_id`**. Verified real, multi-event grouping:

```
Campomaior_cm_010_2024-05-15_20 → events [1, 2]
  ev1  SUBJECT             'ABERTURA DA PISCINA MUNICIPAL/ÉPOCA BALNEAR/2024'
  ev1  VOTER-FAVOR         'A CÂMARA'
  ev1  VOTING              'DELIBEROU'
  ev1  COUNTING-UNANIMITY  'POR UNANIMIDADE'
  ev2  VOTING              'DELIBERADO'
  ev2  COUNTING-UNANIMITY  'POR UNANIMIDADE'
  ev2  SUBJECT             'PREÇO DA VENDA DE TOUCAS NA PISCINA MUNICIPAL…'
```

Segments with >1 event: **train 144 (7.99%) · dev 42 (7.59%) · test 25 (4.73%)**.

**Two consequences:**

**1. A factual correction is mandatory (P1).** §8 currently says the authors leave *"the construction of multi-event gold attachments to future work"* (L427). The attachments are already in the released data. As written this understates the released resource and is checkable by any reader who downloads it. **This sentence must change regardless of which option below is chosen.**

**2. An opportunity (optional, Tier B).** All four reviewers criticised exactly the thing this field enables. A compact event-level evaluation — argument-to-event attachment F1 on the 25 multi-event test segments, even with a trivial nearest-trigger baseline — would convert the paper's central weakness into a contribution.

**Counter-argument, stated fairly:** the rebuttal promised a *narrower* scope, and R2 wrote "*please ensure that this narrower scope is reflected consistently throughout the paper*." Adding an event-level experiment cuts against that promise, costs scarce page budget, and n=25 test segments is thin. A camera-ready is also not the venue for new claims.

**Recommendation:** honour the narrowing commitment in the main text (Tier A), and *if* Tier B is pursued, place it strictly in an appendix framed as a **feasibility probe**, not a headline result.

---

# Locked Decisions

| Question | Decision |
|---|---|
| Gap D — event grouping | **Correct §8 only.** Fix the inaccurate "future work" sentence and note that gold attachments ship with the released data. No event-level experiment; narrowed scope promised to R2 is honoured. Tier B is dropped. |
| Gap A/B — re-run | **Re-run Gemini few-shot + GPT few-shot.** Patch extractors to persist raw output, pre-alignment entities, alignment stats, and token usage. Yields exact per-model discard rates (R1) and true billed cost (Gap B). |

---

# Suggested Revision Order

**Tier A — commitments made in the rebuttal (do all of these)**

1. **Mechanical/structural first** — items 2–5, 19–25. Zero-risk, removes every editorial complaint, gets the build correct.
2. **Framing sweep** (#1) — Abstract → §1 → §3 → §5 → §7 → §8. Single consistent term; scope sentence at each claim site. This is what all four reviewers are waiting to see.
3. **Contribution statement** (#8) and artifact prominence (#18).
4. **Measurement honesty** (#9, #10, #11, #17) — the 11-vs-12 types, exact-vs-relaxed regimes, low-support caveats, multi-vote rates across all splits.
5. **Gap D factual correction** to §8 — mandatory.
6. **Gap C error examples** (#12) — cheap, directly answers R3.
7. **Gap A + B re-runs** (#6, #7) — the only items needing compute; start early since they gate appendix content.
8. **Writing pass** (#13–#16, #26) and final page-budget check (#28).

**Tier B — optional, discuss before starting**

9. Event-attachment feasibility probe (Gap D), appendix-only.

---

# Execution Status

**Done — manuscript builds clean (14 pp, body ends p5, References p7, no undefined refs).**

| Item | Status |
|---|---|
| #2 References before Appendices | ✅ verified in PDF: Conclusion p5 → Limitations p6 → References p7 → Appendices p8+ |
| #3 `[review]` → `[final]` | ✅ |
| #1 Framing sweep | ✅ Abstract, §1, §3, Fig. 1 caption, §7, §8 all now say *span-level*, each with an explicit non-scoring statement |
| #8 Contribution provenance | ✅ §1 now states annotations come from CitiLink-Minutes and are not claimed |
| #9 11-vs-12 types | ✅ added to Table 1 caption |
| #10 Exact vs relaxed use cases | ✅ new §5 paragraph separating traceability from information access |
| #11 Low-support caveat | ✅ Table 11 caption |
| #12 Error examples | ✅ `scripts/make_error_examples_table.py` → new Appendix D + §6 pointer |
| #13 Char-offset vs token BIO | ✅ §3 — **claim corrected after verification, see below** |
| #14 Long sentence | ✅ split into three |
| #15 Language overclaim | ✅ §8 rewritten |
| #16 Prompting hedge | ✅ §4 |
| #17 Multi-vote rates | ✅ all three splits disclosed, incl. that test under-represents the hard case |
| #18 Artifact prominence | ✅ §1 |
| #19–#23, #25 typos/periods | ✅ |
| #24 Notation | ✅ figure legend now uses full forms; table abbreviations defined in Table 11 caption |
| #26 sentence-BERT | ✅ §5 |
| Gap B latency | ✅ Table 10 extended with Gemini/GPT per-doc latency + wall-clock; commensurability paragraph replaces "no GPU costs were measured" |
| Gap D §8 correction | ✅ now states gold event attachments ship with the data |

**Instrumented, awaiting your re-run:**

- `src/llm_extraction/schemas.py` — added `diagnostics` field
- `src/llm_extraction/gemini/extractor.py`, `gpt/extractor.py` — persist raw generations, pre-alignment entities, per-point drop counts, token usage
- `scripts/llm_extraction/alignment_report.py` — reports discard rate + cost; **refuses to emit a number for uninstrumented runs** rather than restating the 1.0 artifact

```bash
export GEMINI_KEY=...   OPENAI_API_KEY=...
python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot --model gemini-2.5-pro
python scripts/llm_extraction/extract_gpt_spans.py    --strategy few_shot --model gpt-5.5
python scripts/llm_extraction/alignment_report.py \
    results/llm_extraction/gemini/few_shot.jsonl \
    results/llm_extraction/gpt5.5/few_shot.jsonl \
    --out results/alignment_report.json
```

Then add one sentence to §4 and a short subsection to App. C with the measured discard rate and cost.

**Still open (needs you):**

| Item | Why blocked |
|---|---|
| #4 Author block | L65–67 still `Address line` / `Author n` — I will not invent authorship |
| #5 De-anonymise URLs | 3× `Anonymous3445` — I don't have the real repo/model/demo URLs |
| #27 Table 3 verification | Paper says XLM-R-CRF SPU `48±16`; `error_classification_results.json` says `64`. Confirm which seeds/aggregation produced the table |
| Gap A/B numbers | Requires the re-run above |

**Latent reproduction bug (not a paper error):** `src/llm_extraction/gemini/config.py` defaults `model_id` to `gemini-2.5-flash-preview-04-17`, while the reported runs used `gemini-2.5-pro` (the CLI default in `extract_gemini_spans.py` overrides it, and the saved records confirm `model: "gemini-2.5-pro"`). Worth aligning the default so a reproducer doesn't silently run Flash.

---

# Verification Log

Claims checked against `results/` and `data/` during this audit:

| Paper claim | Status |
|---|---|
| XLM-R-CRF 93.2 / DeBERTa-CRF 91.5 / BERTimbau-CRF 90.5 | ✅ matches `seed_aggregate_summary.json` |
| CRF gain: +8.0 XLM-R, +13.0 BERTimbau | ✅ from Table 1 |
| 32-pt exact gap → 8-pt relaxed gap | ✅ 93.2−61.3=31.9; 97.0−89.3=7.7 |
| DeBERTa LOMO drop ≈35% | ✅ (91.5−59.1)/91.5 = 35.4% |
| 95.3% of test segments ≤1 trigger | ✅ 504/529 = 95.27% |
| Table 3 XLM-R-CRF SPU `48±16` | ⚠️ source JSON says 64 — at interval edge, verify aggregation |
| LLM alignment fidelity 100% | ❌ measurement artifact — see Gap A |
| "multi-event gold attachments" are future work | ❌ contradicted by `event_id` in released data — see Gap D |
| Reported Gemini model is 2.5 Pro | ✅ saved records confirm `model: "gemini-2.5-pro"` |
| Gold spans align to token boundaries (my own drafted claim) | ❌ **false** — 1.9% of test spans split a word (Portuguese contractions, e.g. gold *os eleitos pelo PS* inside *dos eleitos pelo PS*). Concentrated in voter roles: V-Absent 16.7%, V-Abstention 8.1%, V-Favor 6.0%. Encoder gold is re-projected onto subword offsets by overlap (`scripts/evaluate.py:36`), so encoders are scored at token granularity while LLMs are held to exact characters. §3 now discloses this asymmetry. Too small to explain the 32-pt gap, and the text says so. |
