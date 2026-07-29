# VotIE — Camera-Ready Change Log

**Paper:** VotIE: Information Extraction from Meeting Minutes · ACL Findings (short)
**Base:** submitted `ARR_VotIE.pdf` (scores 3/3/3/2) → camera-ready
**Scope of this log:** every change applied to `acl_latex.tex`, the extraction code, and the new scripts, each traced to the reviewer request it answers. Reviewer IDs follow `reviews.txt` order: R1 = 3fsk, R2 = YpBJ, R3 = iyoP, R4 = 4SQ9.

**Build state:** compiles clean via `latexmk -pdf` (TeX Live 2026). 14 pp total; body ends p5 (short-paper limit), References p7, Appendices p8+. No undefined references or citations. One residual overfull `\hbox` (2.3 pt, Table 8) — cosmetic.

**Convention:** ✅ applied · ⏳ instrumented, awaits your re-run · ⛔ needs you (authorship/URLs/decisions I must not make).

---

## 1. Structural / formatting

| # | Change | Location | Reviewer |
|---|---|---|---|
| ✅ | `\usepackage[review]{acl}` → `[final]` | preamble L5 | build |
| ✅ | Moved `\bibliography{custom}` **above** `\appendix`; removed the trailing duplicate before `\end{document}` | L449 / end | R2, R3, R4 |
| ⛔ | Author block still `Address line` / `Author n` | L65–67 | build — I will not invent authorship |
| ⛔ | 3× `Anonymous3445` artifact URLs still anonymised | L167 footnotes | de-anonymise for camera-ready (real URLs unknown to me) |

Ordering verified in the compiled PDF: Conclusion (p5) → Limitations → Ethical Considerations (p6) → References (p7) → Appendices (p8+).

---

## 2. Framing: "event extraction" → "span-level event-argument extraction"

The unifying criticism of all four reviewers. The EE *grounding* is kept (R2 and R3 accepted it in the rebuttal); at every claim site the manuscript now states explicitly that the benchmark scores **argument identification, not event assembly**.

| # | Change | Location | Reviewer |
|---|---|---|---|
| ✅ | Abstract: task described as "span-level extraction of voting-event arguments"; added a sentence putting trigger binding / role attachment / event grouping out of scope | L135 | all |
| ✅ | §1: "span-level extraction of voting-event arguments" + explicit non-scoring statement | L152 | all |
| ✅ | §3: "operationalizes the problem as span-level event-argument identification rather than full event reconstruction" (already present, retained and reinforced) | L201 | all |
| ✅ | Figure 1 caption: "span-level extraction … scores the identification of these spans, not their assembly into complete events" | L225 | all |
| ✅ | Conclusion: "span-level extraction … scores argument identification rather than complete event reconstruction" | L424 | all |
| ✅ | §8 Task-Formulation Limitations paragraph rewritten (see §5 below) | L431 | all |

Residual "voting event" phrases (L230, L715, L729, table labels) are **descriptive** ("voting-event arguments", "contains at least one voting event"), not task claims — intentionally left.

---

## 3. Contribution scope & artifact prominence

| # | Change | Location | Reviewer |
|---|---|---|---|
| ✅ | §1 contribution sentence now opens by stating the annotations come from CitiLink-Minutes and are **not** claimed as this paper's contribution; reframes the contribution as task + schema + benchmark + protocol + baselines + model + code + demo | L167 | R3, R4 |
| ✅ | Release footnote reworded to "the complete training and evaluation code for **every reported experiment**" (R2 rated software 1/5, likely missed the release) | L167 | R2 |

---

## 4. Results & metrics honesty

| # | Change | Location | Reviewer |
|---|---|---|---|
| ✅ | Table 1 caption: states macro averages are over the **11** types with non-zero test support, Count-Against (12th) excluded | L307 | R1 |
| ✅ | New §5 paragraph separating **legal traceability** (exact match, encoders win) from **information access / search** (relaxed match, LLMs competitive); scopes the encoder recommendation to the traceability setting | after L303 | R1 |
| ✅ | New §4 paragraph: LLM figures are **protocol-relative**, not an upper bound; prompting/chunking/alignment are improvable axes the benchmark exists to measure | L296 | R1 |
| ✅ | Table 11 caption: explicit **caution** that secret-ballot types (n=4,5) and V-Absent (n=18) have low support, 100.0 = 4–5 correct predictions, not to be read as model quality | L916 | R1, R3 |
| ✅ | §5 cross-municipality: added explanation of *why* the sentence-BERT gap is smaller than TF-IDF (shared business, divergent clerk-specific formulae) | L363 | R1 |
| ✅ | §6 error analysis: added that encoders **truncate** long Subjects while Gemini **over-includes** (determiners, honorifics, stance phrases), pointing to new Appendix D | L397 | R3 |
| ⛔ | Table 3 discrepancy **not resolved**: paper reports XLM-R-CRF SPU `48±16`; `results/error_classification_results.json` says `64`. Confirm seeds/aggregation | Table 3 | verify |

---

## 5. Limitations rewrites

| # | Change | Location | Reviewer |
|---|---|---|---|
| ✅ | Multi-vote disclosure now gives **all three splits** (train 8.0%, dev 7.6%, test 4.7%) and notes the test split *under-represents* the hard case → scores are mildly optimistic on multi-vote material. Replaces the single "95.3% of test" figure | L431 | R1, R3 |
| ✅ | **Factual correction (mandatory):** §8 previously called "construction of multi-event gold attachments" future work. The released data already carries per-span `event_id` grouping. Now states the gold attachments ship with the benchmark; only the evaluation protocol/baselines are future work | L431 | data-accuracy |
| ✅ | Language paragraph rewritten: drops "theoretically language-agnostic"; schema portability is a *design expectation*, empirically untested; multilingual validation = future work | L435 | R3, R4 |
| ✅ | §3 new paragraph on the **char-offset vs. token-BIO** mismatch, including the **verified** finding that 1.9% of test spans split an orthographic word (Portuguese contractions) so encoders are scored at token granularity while LLMs are held to exact characters — asymmetry disclosed, noted too small to explain the 32-pt gap | L201 | R3 |

---

## 6. Deployment cost / latency (R1)

| # | Change | Location | Reviewer |
|---|---|---|---|
| ✅ | Table 10 extended with per-document latency (Gemini 27.8/26.9 s, GPT 22.8/15.0 s) and few-shot wall-clock (3.95 h, 2.21 h) — all computed from `processing_time` in the saved `.jsonl`, no re-run needed | Table 10 | R1 |
| ✅ | Replaced "no GPU costs were measured" with a commensurability paragraph: API systems are billed per call and recur on every reprocessing pass, vs. train-once encoders — the basis of the deployment recommendation | L851 | R1 |
| ⏳ | Exact token-usage **cost** and **span-discard rate** await the re-run (§8 below) | App. C / §4 | R1 |

---

## 7. Editorial fixes

| Change | Location | Reviewer |
|---|---|---|
| `within he same document` → `within the same document` | L792 | R1 |
| `the lengthier among` → `the longest among` | L808 | R1 |
| `costs makes` → `costs make` | L420 | R1, R4 |
| `in Portugues` → `in Portuguese` | L554 | R1 |
| `Macro averaged` → `Macro-averaged` (caption) | L307 | R1 |
| Figure-1 legend `VOTER-ABS`/`VOTER-AGST`/`COUNT-MAJ` → full forms `VOTER-ABSTENTION`/`VOTER-AGAINST`/`COUNTING-MAJORITY`; table abbreviations defined in Table 11 caption | L217–221 | R1 |
| Added terminal periods | L290, L433, L843 | R3 |

---

## 8. Code & tooling

### New scripts
- **`scripts/make_error_examples_table.py`** — generates Appendix D's qualitative error table from `results/error_classification_results.json` (curated to show distinct failure modes; English glosses). Output wired via `\input{error_examples_table}`. Directly answers R3's request for concrete examples; **no new data**.
- **`scripts/llm_extraction/alignment_report.py`** — computes span-discard rate + API cost from instrumented runs. **Refuses to emit a number for uninstrumented runs** rather than restating the 1.0 artifact.

### Instrumentation (⏳ effective only on the next extraction run)
- **`src/llm_extraction/schemas.py`** — added `diagnostics: Optional[Dict]` to `SpanExtractionResult`.
- **`src/llm_extraction/gemini/extractor.py`**, **`gpt/extractor.py`** — persist raw generations, pre-alignment entities, per-point drop counts (unknown-type vs. failed-alignment), and best-effort token usage.

> **Why needed:** the saved `*_evaluation.json` all report `alignment_fidelity = 1.0`, `failed_spans = 0`. This is an artifact — fidelity is computed over spans that already survived alignment (`span_alignment.py:352`); the real discard counts were only `logger.info`'d and `raw_response` is `None`. Reporting 0% discarded would be false.

### Re-run to execute (yours — no API keys in this environment)
```bash
export GEMINI_KEY=...   OPENAI_API_KEY=...
python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot --model gemini-2.5-pro
python scripts/llm_extraction/extract_gpt_spans.py    --strategy few_shot --model gpt-5.5
python scripts/llm_extraction/alignment_report.py \
    results/llm_extraction/gemini/few_shot.jsonl \
    results/llm_extraction/gpt5.5/few_shot.jsonl --out results/alignment_report.json
```
Then add one sentence to §4 and a short App. C subsection with the measured discard rate + cost.

### Latent reproduction bug (not a paper error)
`src/llm_extraction/gemini/config.py` defaults `model_id` to `gemini-2.5-flash-preview-04-17`, while the reported runs used `gemini-2.5-pro` (CLI default in `extract_gemini_spans.py` overrides it; saved records confirm `model: "gemini-2.5-pro"`). Align the default so a reproducer doesn't silently run Flash.

---

## 9. Outstanding before submission

| Item | Owner | Blocker |
|---|---|---|
| Fill author block + affiliations | you | authorship |
| De-anonymise 3× artifact URLs | you | real URLs |
| Execute LLM re-run → discard rate + cost | you | API keys |
| Resolve Table 3 SPU `48` vs `64` | you | confirm aggregation |
| Add §4 sentence + App. C subsection with re-run numbers | after re-run | — |
| Align Gemini config default to `2.5-pro` | you | optional but recommended |

---

## Files touched

```
ACL_CitiLink___Vote_Identification__José_Evans_/latex/acl_latex.tex   (edited)
ACL_CitiLink___Vote_Identification__José_Evans_/latex/error_examples_table.tex   (generated)
src/llm_extraction/schemas.py                     (edited)
src/llm_extraction/gemini/extractor.py            (edited)
src/llm_extraction/gpt/extractor.py               (edited)
scripts/make_error_examples_table.py              (new)
scripts/llm_extraction/alignment_report.py        (new)
REVISION_ROADMAP.md                               (new — full plan + verification log)
CHANGELOG_camera_ready.md                         (this file)
```
