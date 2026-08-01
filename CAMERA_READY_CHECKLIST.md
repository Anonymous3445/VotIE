# VotIE — Camera-Ready Master Checklist

**Paper:** VotIE · ACL Findings (short) · scores 3/3/3/2
**Manuscript:** `ACL_CitiLink___Vote_Identification__José_Evans_/latex/acl_latex.tex`
**Page budget:** 5 content pages. Limitations, Ethics, Acknowledgements, References and Appendices are
exempt. Submission body ended on p4 → ~1 full page of new body space available.

**Legend:** `[ ]` todo · `[x]` done · `[~]` partially done · `[!]` blocked on author ·
`[C]` needs compute/API

> Line numbers reflect the file as of this audit and **will shift** once edits are applied. Use the
> quoted text to locate each item.

---

## 0. Blocked on you

- [!] **B1** — Fill the author block. Still `Address line` / `Author n` (L65–67).
- [!] **B2** — De-anonymise artifact URLs. 4 occurrences of `Anonymous3445` (HF model, GitHub, HF Space).
- [x] **B3** — ~~Sync LOMO outputs from Deucalion.~~ **Done / not needed.** `./lomo` was pulled and is
  byte-identical to `results/lomo`. The data was never wrong — see L1. What *is* still missing is seed
  coverage for XLM-R and BERTimbau (L4, L5), which requires new runs, not a sync.
- [!] **B4** — Provide API keys (`GEMINI_KEY`, `OPENAI_API_KEY`) for the generative re-runs.
- [!] **B5** — Decide compute sequencing if AMALIA LOMO (~71 GPU-h) doesn't fit. Recommendation: drop
  AMALIA 0-shot LOMO first.

---

## 1. Build & structure

- [x] **S1** — `\usepackage[review]{acl}` → `[final]` (L5).
- [x] **S2** — Move `\bibliography{custom}` (L1023) **above** `\appendix` (L440). Compiled order must be
  Conclusion → Limitations → Ethics → **References** → Appendices.
  *Raised by Review 2, Review 3 (×2), Review 4 — four separate mentions.*
- [x] **S3** — Fix §1 pasted-PDF damage (L154–158): hard hyphens `assem-\nbly`, `bind-\ning`, and the
  duplicated word *"extraction extraction"*.
- [x] **S4** — Delete the duplicated contributions clause. L154–158 ends with a contributions list that
  **repeats** the full sentence at L161. Keep the scope sentence from L154–158; keep L161 as the single
  contributions statement.
- [ ] **S5** — Final build check: `latexmk -pdf acl_latex.tex`, no undefined refs/citations.
- [ ] **S6** — Final page check: content (§1–§7) ≤ 5 pages.

---

## 2. Framing — "event extraction" vs. flat span identification

*The single unifying criticism: all four reviewers raised it, and it drove both Soundness-2 scores.
Reviews 2 and 3 confirmed in the rebuttal thread that the clarification resolves their concern —
Review 2 wrote "please ensure that this narrower scope is reflected consistently throughout the paper."*

- [x] **F1** — Abstract says "span-level extraction of voting-event arguments".
- [x] **F2** — §1 says "a task for span-level extraction of voting-event arguments" + explicit
  non-scoring statement.
- [x] **F3** — §3 says "operationalizes the problem as span-level event-argument identification rather
  than full event reconstruction".
- [x] **F4** — §8 Task Formulation states the benchmark "evaluates flat span identification only".
- [x] **F5** — **Figure 1 caption** (L214) still reads *"extracting structured voting event arguments"*.
  Change to span-level and add that the benchmark scores identification, not assembly.
- [x] **F6** — **§7 Conclusion** (L411) still reads *"a novel task and benchmark for extracting
  voting-event arguments"* with no scope clause. Add one.
- [?] **F7** — **Title** (L56): *"VotIE: Information Extraction from Meeting Minutes"* is vaguer than the
  task now claimed. Consider *"VotIE: Span-Level Voting-Argument Extraction from Municipal Meeting
  Minutes."* *(Meta-review M1: "describe the addressed task without any ambiguity, and its setup".)*
- [ ] **F8** — Sweep for residual task claims. Descriptive uses ("voting-event arguments", "contains at
  least one voting event") are fine; task claims are not.

---

## 3. Contribution & positioning

- [X] **P1** — **Answer the meta-review's unanswered question**: *"describe how the defined task differs
  from the tasks and experiments explored in the CitiLink-Minutes paper. Is the task a subset of all
  categories (event types) defined in that corpus?"* Add a short §3/§4 paragraph naming the category
  count and what is excluded. **Nothing in the paper currently addresses this.**
- [X] **P2** — State plainly in §1 that annotations come from CitiLink-Minutes and are not claimed here;
  this paper contributes task formulation, schema, splits, protocol, baselines, model, code, demo.
  One sentence, not the rebuttal's eight-item list. *(Reviews 3, 4.)*
- [X] **P3** — Acknowledge the novelty framing: the task is new but is a reformulation over an existing
  corpus plus a benchmark of contemporary methods. *(Meta-review; Review 4 "Limited Technical Novelty".)*
- [X] **P4** — Make artifact footnotes prominent in §1. **Review 2 scored software 1/5 ("No usable
  software released")** and datasets 2/5 despite the full release — they likely missed the links.
- [X] **P5** — Review 2 rated Reproducibility 3/5 citing underspecified parameters. Confirm §4 Training
  Setup + Appendix C fully specify hyperparameters, or point to the config files.

---

## 4. Results & metrics honesty

- [ ] **R1** — Table 1 caption (L298): state macro is over the **11** types with non-zero test support
  (`Count-Against` excluded). Currently only in §4 prose and the Appendix B table caption.
  *(Review 1: "the paper says the schema has twelve argument types, but macro averages are over eleven …
  this should be made very explicit in the main results.")*
- [ ] **R2** — New §5 paragraph separating **legal traceability** (exact match; encoders win) from
  **information access / search** (relaxed match; gap collapses 32 → 8 points). *(Review 1 W3.)*
- [ ] **R3** — New §4/§5 sentence: LLM results are protocol-relative, not an upper bound. Prompting,
  chunking and alignment are improvable axes the benchmark exists to measure. *(Review 1 W2.)*
- [ ] **R4** — Fix misattributed rationale (L285): *"partial match F1 … which requires type agreement and
  boundary overlap, **to ensure legal traceability**"*. Traceability belongs to **exact** match.
- [ ] **R5** — Table 11 caption: add the low-support caveat. `Voting-Method` n=5, `Count-Blank` n=5,
  `Count-Favor` n=4 — a 100.0 means "4 of 4 correct", not model quality. *(Reviews 1, 3; Review 3 cited
  the 100% scores specifically.)*
- [ ] **R6** — §5 (L354): explain *why* the sentence-BERT gap is smaller than TF-IDF — shared subject
  matter, municipality-specific surface form. *(Review 1: "could use a little more explanation".)*
- [ ] **R7** — Report **generative document-level failure rates** (see A3). Not currently in the paper.
- [ ] **R8** — Add one §4 sentence + a short Appendix C subsection with the measured **span-discard rate**
  and **API cost** once the re-runs land. *(Review 1 C2, C3.)*
- [ ] **R9** — State Table 3's seed basis in its caption. `error_classification_results.json` holds a
  single seed (XLM-R-CRF: total 260, MIS 41, SPU 64, BND 154, TYP 1); the table reports 3-seed means
  (247±16, 45±4, 48±16, 152±3, 2±2). Consistent, but unstated.
- [ ] **R10** — Add caution about generalizing from a small, geographically limited benchmark
  (120 docs, 2,885 segments, 6 municipalities) and about macro-F1 with rare labels.
  *(Review 1 W5, Review 3 W2b — Review 1 explicitly said this is acceptable "provided the authors are
  cautious in generalizing claims".)*

---

## 5. LOMO — the meta-review's main technical ask

> *"Focus should be more on the LOMO setup, as it is more practical (and, subjectively, more
> interesting). Consider adding the additional methods included in Table 1 to Table 2."*

### 5a. Table 2 — RESOLVED: the numbers are correct, but the metric is undocumented

- [x] **L1** — **Table 2 reproduces.** The earlier "does not reproduce" finding was wrong: it came from
  reading the `entity_f1` field (seqeval macro, which scores zero-support types as 0.0). The paper's
  actual metric is **macro F1 over entity types with support > 0 in the evaluation set**. Under that
  definition, using `./lomo` (byte-identical to `results/lomo`):

  | Model | Cells reproducing | Notes |
  |---|---|---|
  | BERTimbau | **12 / 12** exact + relaxed | all match to <0.15 |
  | mDeBERTa | **12 / 12** exact + relaxed | all match to <0.15 |
  | XLM-R | M01 (3 seeds) + M03 exact | every cell with all 3 seeds reproduces; every residual is a cell missing seeds (M02 has 2, M03–M06 have 1) |

  Worked example — mDeBERTa M04: per-type F1 averaged over the 8 types with support > 0 = **70.3**,
  matching the paper exactly. Including the two zero-support types (`Count-Blank`, `Voting-Method`) as
  0.0 gives 56.2, which is what `entity_f1` reports. **No data problem, no re-runs needed to fix
  Table 2.** The XLM-R residual disappears once L4 completes the seeds.

- [ ] **L2** — **The real problem: the macro denominator varies per municipality and is undocumented.**

  | Held-out | # types scored | Types absent from that municipality |
  |---|---|---|
  | M01 | 8 | Voting-Method, Count-Blank, Count-Favor, Count-Against |
  | M02 | 9 | Count-Blank, Count-Favor, Count-Against |
  | M03 | 9 | Counting-Majority, Count-Favor, Count-Against |
  | M04 | 8 | Voting-Method, Count-Blank, Count-Favor, Count-Against |
  | M05 | 10 | Count-Favor, Count-Against |
  | M06 | **12** | — none |

  Consequences to fix in the text:
  - Cells in Table 2 are **not strictly commensurable** — M04 is a mean over 8 types, M06 over 12.
  - The **Mean column averages across different denominators**.
  - Table 1 uses 11 types; the "DeBERTa drops 35% relative to the standard benchmark" claim therefore
    compares two different denominators.
  - **M06's cell includes `Count-Against`**, contradicting §4's blanket statement that Count-Against
    "has zero test instances and is therefore excluded from evaluation" — true for the standard test
    split, false for the M06 LOMO fold.

  Decide one policy, apply it everywhere, and state it in the caption. The rule is defensible
  (it is the same "types with support > 0" rule Table 1 uses); it just has to be said out loud.

- [ ] **L3** — Write `scripts/aggregate_lomo.py` as the single source of truth: applies the chosen
  denominator explicitly, reports mean ± std **and per-cell seed count**, emits the Table 2 LaTeX.
  Make it share the denominator logic with `scripts/aggregate_seeds.py`.

- [ ] **L3b** — Table 2 caption claims *"XLM-R/DeBERTa are 3-seed means"* — **false for XLM-R**.
  Actual coverage: mDeBERTa 3×6 ✅; XLM-R 3 for M01, 2 for M02, 1 for M03–M06; BERTimbau 1 everywhere.
  Fixed by L4/L5, or state the per-cell seed count.

### 5b. Complete the seeds

- [C] **L4** — XLM-R-CRF LOMO: 9 missing runs (M02×1, M03–M06×2). ~6 GPU-h.
- [C] **L5** — BERTimbau-CRF LOMO: 12 missing runs (M01–M06 × 2 seeds). ~20 GPU-h.

### 5c. Add the missing Table 1 methods to Table 2

Current LOMO coverage: BERTimbau-CRF, XLM-R-CRF, mDeBERTa-CRF, Gemini-5s only.

- [C] **L6** — 3 linear heads (BERTimbau / mDeBERTa / XLM-R) × 6 municipalities = 18 runs. ~10.4 GPU-h.
- [C] **L7** — BiLSTM-CRF × 6 = 6 runs. ~5.2 GPU-h.
- [C] **L8** — Feature-based CRF × 6 = 6 runs. ~1.7 CPU-h.
- [C] **L9** — GPT-5.5 few-shot LOMO (2,885 docs). ~$14. *High priority — gives a second LLM in the table.*
- [C] **L10** — Gemini few-shot LOMO, instrumented re-run. ~$14. *Verifies the existing row + yields
  discard rate and cost for the setting where LLMs win.*
- [C] **L11** — Gemini 0-shot LOMO. ~$7.
- [C] **L12** — GPT-5.5 0-shot LOMO. ~$7.
- [C] **L13** — AMALIA 8B few-shot LOMO. ~71 GPU-h.
- [C] **L14** — AMALIA 8B 0-shot LOMO. ~42 GPU-h. *Drop first if budget binds.*

> Discriminative additions stay single-seed (seed 42), matching how non-CRF rows are reported in Table 1.
> Say so in the caption.

### 5d. Presentation & analysis

- [ ] **L15** — Rebuild Table 2 from `aggregate_lomo.py` with all methods. Keep **exact match + mean** in
  the body; move relaxed-match and per-seed breakdowns to an appendix.
- [ ] **L16** — Extend the §5 LOMO discussion — this is where the extra body page should mostly go.
- [ ] **L17** — Extend `scripts/statistical_tests.py` to LOMO (currently `in_domain` only), so the §5
  claim *"DeBERTa-CRF significantly outperforms XLM-R-CRF on M02, M04 and M06 (p<0.05)"* is regenerated
  rather than asserted — those cells are among the ones that currently fail to reproduce.

---

## 6. Error analysis

- [ ] **E1** — **Manual analysis over a sample** *(meta-review — cannot be automated)*. Hand-annotate ~50
  errors into linguistic causes, weighted toward SUBJECT boundary errors (123 of XLM-R-CRF's 154), and
  report the distribution.
- [ ] **E2** — **Concrete error examples** *(Review 3: "It would be more informative to show some concrete
  error examples")*. `results/error_classification_results.json` already holds `example_errors` per model
  per category with `doc_id`, gold, prediction and highlighted context;
  `scripts/make_error_examples_table.py` renders it. Wire the output into a new appendix + a §6 pointer.
  **No new data needed.**
- [ ] **E3** — §6: add that encoders truncate long Subjects while Gemini over-includes (determiners,
  honorifics, stance phrases), pointing to the new appendix.

---

## 7. Limitations

- [ ] **X1** — **Multi-vote rates.** §8 cites only *"95.3% of test segments"*. Verified: train **7.99%**
  (144/1803), dev **7.59%** (42/553), test **4.73%** (25/529) are multi-vote. The test split
  **under-represents** the hard case, so scores are mildly optimistic on multi-vote material. Disclose
  all three. *(Reviews 1, 3.)*
- [ ] **X2** — **Drop the false tail of §8** (L418): *"we leave this, together with the construction of
  multi-event gold attachments to future work."* Replace with: an event-level **evaluation protocol and
  baselines** are future work. Do not claim gold attachments must be constructed. **Decision: event
  grouping stays out of scope; do not surface `event_id`.**
- [ ] **X3** — **Language overclaim** (L426): drop *"theoretically language-agnostic"*. Schema portability
  is a design expectation, empirically untested; multilingual validation is future work.
  *(Reviews 3, 4.)*
- [ ] **X4** — **§3 char-offset vs. token-BIO mismatch** *(Review 3 C3)*. The formal definition
  `S = {(s_j, e_j, t_j)}` uses character offsets while BIO models work on tokens. Verified: **1.86% of
  test gold spans (35/1882) have a boundary inside an orthographic word** — Voter-Absent 16.7%,
  Voter-Abstention 8.1%, Voter-Favor 6.0% (Portuguese contractions). Encoder gold is re-projected onto
  subword offsets by overlap (`scripts/evaluate.py:36`), so **encoders are scored at token granularity
  while LLMs are held to exact characters**. Disclose the asymmetry; note it is far too small to explain
  the 32-point gap.
- [ ] **X5** — Add cross-segment coreference to the limitations framing *(Review 2 explicitly listed it)*.
  §8 already has a Coreference Resolution paragraph — verify it covers the point.

---

## 8. Editorial & grammar — every reviewer-flagged item

- [x] **G1** — `in Portugues` → `in Portuguese`. **Fixed** (now L545).
- [x] **G2** — Overlong sentence Review 3 quoted ("Figure 1 illustrates the coupling…isolated lexical
  mentions"). **Fixed** — split into three sentences at L192.
- [ ] **G3** — `high inference costs makes` → `make` (L411). *(Reviews 1 and 4.)*
- [ ] **G4** — `several thousand words within he same document` → `within the same document` (L783).
- [ ] **G5** — `Subject spans are the lengthier among the extracted classes` → `the longest among` (L799).
- [ ] **G6** — `Macro averaged results of all baselines` → `Macro-averaged` (Table 1 caption, L298).
- [ ] **G7** — Missing terminal period: `…introduced by approximate matching` (L281, end of "Extraction
  Paradigms").
- [ ] **G8** — Missing terminal period: `…additional minutes containing such procedures` (L424, end of
  "Dataset limitations").
- [ ] **G9** — Missing terminal period: `…computed on the un-oversampled test set` (L834, end of §B.5).
- [ ] **G10** — **Notation consistency** *(Review 1)*. Currently three schemes coexist:
  - Figure 1 legend (L210–212): `COUNT-MAJ`, `VOTER-ABS`, `VOTER-AGST`
  - Prompt appendix (L567–568, L619–620): `VOTER-AGAINST`, `VOTER-ABSTENTION`
  - Tables (L813, L815, L917–918): `V-Abst`, `V-Agn`

  Use full forms in prose and figures; define the table abbreviation set once, in the Table 11 caption.
  Note `VOTER-ABS` is ambiguous between Absent and Abstention — a real hazard, not just style.
- [ ] **G11** — General readability sweep *(Review 3 W3: "Many sentences are unnecessarily long and
  complex")*. §3 and §5 are the densest.

---

## 9. Experiments — re-runs

All generative re-runs must use the **instrumented** extractors so one pass yields the discard rate,
the token cost, and the results together. Cost estimates from actual character counts (test = 920,730
chars; LOMO = 5,012,751 chars over 2,885 segments) at $1.25/M in, $10/M out, accounting for langextract's
per-chunk prompt resend.

- [ ] **N1** — **Instrument AMALIA.** `src/llm_extraction/amalia/extractor.py` has no `diagnostics`
  block; Gemini and GPT do. Mirror `gemini/extractor.py:119-154`.
- [ ] **N2** — Fix the reproduction trap: `src/llm_extraction/gemini/config.py:13` defaults `model_id` to
  `gemini-2.5-flash-preview-04-17` while every reported run used `gemini-2.5-pro` (the CLI default
  overrides it). A reproducer silently runs Flash.
- [C] **N3** — Gemini 2.5 Pro, standard test, 0-shot + 5-shot. ~$1.20 + ~$2.60.
- [C] **N4** — GPT-5.5, standard test, 0-shot + 5-shot. ~$1.20 + ~$2.60.
- [C] **N5** — AMALIA 8B, standard test, 0-shot + 5-shot. ~7.8 + ~13.1 GPU-h.
- [ ] **N6** — Run the report:
  ```
  python scripts/llm_extraction/alignment_report.py \
      results/llm_extraction/*/{zero,few}_shot.jsonl --out results/alignment_report.json
  ```
  Add a `PRICING` entry per model actually billed.
- [ ] **N7** — **Verify re-run scores match the published ones** before substituting into Table 1
  (temperature is 0.0 throughout, so they should). Report any drift rather than silently replacing.

---

## 10. Code changes

- [ ] **C1** — `scripts/run_municipality_experiments.py:63` hardcodes
  `configs/municipality_experiments/deberta_crf_{muni}.yaml`. Add a `--model` flag selecting the config
  family. **Blocks L6–L8.**
- [ ] **C2** — Generate 30 LOMO configs in `configs/municipality_experiments/` for the 5 missing
  discriminative models × 6 municipalities, following the existing `deberta_crf_M01.yaml` pattern.
- [ ] **C3** — `scripts/llm_extraction/run_cross_municipality.py` is Gemini-only (hardcoded
  `GeminiSpanExtractor`, `gemini_few_*` filenames). Add `--provider {gemini,gpt,amalia}` with
  provider-namespaced output dirs. **Blocks L9–L14.** The protocol needs no change — it evaluates all
  splits of the held-out municipality, and its document counts already match the encoder LOMO test sets.
- [ ] **C4** — `scripts/aggregate_lomo.py` (see L3), implementing the L2 denominator policy.
- [ ] **C5** — Extend `scripts/statistical_tests.py` to LOMO (see L17).

---

## 11. Audit findings — problems no reviewer raised

- [x] **A1** — ~~Table 2 non-reproducibility.~~ **Resolved — Table 2 reproduces.** → **L1**
- [ ] **A1b** — Table 2's macro denominator is municipality-dependent (8–12 types) and undocumented;
  M06 silently includes `Count-Against`. → **L2**
- [ ] **A2** — LOMO seed coverage / false caption. → **L3b**
- [ ] **A3** — **Document-level generative failures are unreported.** Counted from the `error` field in
  the saved predictions:

  | Model | 0-shot | 5-shot |
  |---|---|---|
  | Gemini 2.5 Pro | 1 / 529 | 0 / 529 |
  | GPT-5.5 | 7 / 529 | 2 / 529 |
  | **AMALIA 8B** | **58 / 529 (11.0%)** | **89 / 529 (16.8%)** |

  AMALIA's Table 1 scores are depressed by outright generation failures on up to a sixth of the test set
  — a different phenomenon from poor extraction quality. Report the rate and state how failed documents
  are scored (currently they contribute zero predictions, i.e. total recall misses).

- [ ] **A4** — **The span-discard rate is currently unmeasurable, and reporting 100% fidelity would be
  wrong.** All six `*_evaluation.json` report `alignment_fidelity = 1.0`, `failed_spans = 0` (Gemini
  2094/1979 spans, GPT 1874/1994, AMALIA 1825/2864). This is an artifact: the saved `.jsonl` records hold
  only post-alignment entities — their keys are `[api_time, entities, error, id, model, processing_time,
  strategy, text]`, with **no `diagnostics` field** — so fidelity over them is 1.0 by construction.
  Review 1 asked exactly this question, so it must be measured. → **N1, N3–N6**

- [ ] **A5** — **Appendix B.5 misdescribes the oversampling.** It says only segments containing
  `Count-Blank`, `Count-Favor` or `Voting-Method` are duplicated. Actual composition of the 160
  duplicated training segments (train.jsonl = 2,463 rows over 1,803 unique ids):

  | Category | Segments | Copies each |
  |---|---|---|
  | Multi-vote only | 139 | 5 |
  | Minority-type only | 16 | 5 |
  | Both | 5 | 9 |

  **Multi-vote oversampling is undocumented and accounts for 87% of the duplication.** Rewrite B.5 to
  state both criteria and both factors.

- [ ] **A6** — §1 pasted-text damage + duplicated contributions. → **S3, S4**
- [ ] **A7** — §4 traceability misattached to partial match. → **R4**
- [ ] **A8** — Gemini config default model mismatch. → **N2**

### Verified as correct — no action needed

- [x] **V1** — Table 1 reproduces **exactly** from `results/seed_aggregate_summary.json` (all three
  encoder-CRF rows, P/R/F1, exact and relaxed, including ±std).
- [x] **V2** — Appendix Table 4 (segment counts 1,803 / 553 / 529) reproduces exactly from the data.
- [x] **V3** — Appendix Table 5 (entity counts) reproduces exactly — **all 12 types, all 3 splits**.
- [x] **V4** — LOMO test sets = all segments of the held-out municipality
  (504/398/720/235/556/472 = 2,885), and the Gemini LOMO document counts match. Encoder and LLM LOMO
  protocols are commensurable.
- [x] **V5** — The §8 figure "95.3% of test segments have at most one voting trigger" is correct
  (504/529 = 95.27%).

---

## 12. Artifacts

- [ ] **D1** — **Fix the demo.** The meta-review's only flagged technical issue: *"The demo shows an error
  when opened."* The Space code is not in this repo. Open the HF Space, read the build/runtime log, fix,
  and confirm a clean load from a logged-out browser. Reviewers will re-check this.
- [ ] **D2** — Commit the regenerated `results/` so every table is reproducible from the repo.
- [ ] **D3** — `.gitignore` excludes `data/citilink-votie-lomo-splits/`, which is a released benchmark
  artifact. Track it or publish it separately and link it.
- [ ] **D4** — Update Appendix C's compute budget: the current 47.3 GPU-h total does not include the
  ~42 new encoder GPU-hours from L4–L8.

---

## 13. Reviewer coverage matrix

Every comment in `reviews.txt`, mapped to an item above. Reviews are numbered as they appear in the file.

| Source | Comment | Item(s) |
|---|---|---|
| R1 W1 / C1 | Event-extraction framing vs. flat span evaluation | F1–F8 |
| R1 W2 | LLM comparison may underestimate; prompting/chunking/alignment dependent | R3 |
| R1 W3 | Exact match overemphasizes boundaries vs. downstream utility | R2 |
| R1 W4 / C3 | API cost, latency, throughput for Gemini/GPT | R8, N3–N6, M6 |
| R1 W5 | Small, geographically limited benchmark; rare-label macro-F1 | R10, R5 |
| R1 C2 | LLM prompting detail + realignment failure rate | R8, A4, N1–N6 |
| R1 T1 | "with he same document" | G4 |
| R1 T2 | "the lengthier among" | G5 |
| R1 T3 | "high inference costs makes" | G3 |
| R1 T4 | "in Portugues" | **G1 ✅** |
| R1 T5 | sentence-BERT gap needs explanation | R6 |
| R1 T6 | Notation VOTER-AGST / VOTER-AGAINST / V-AGN | G10 |
| R1 T7 | "macro averaged" | G6 |
| R1 T8 | 12 types vs 11-type macro average | R1 |
| R2 W | Flat span only; no event reconstruction, trigger linking, coreference | F1–F8, X5 |
| R2 C2 | References must precede appendices | S2 |
| R2 (scores) | Software 1/5, Datasets 2/5 | P4 |
| R2 (scores) | Reproducibility 3/5, underspecified parameters | P5 |
| R3 W1 | Task frame misleading, overstates contribution | F1–F8 |
| R3 W2 | Contribution is splits/protocol, not annotations | P2 |
| R3 W2b | Dataset small (120 docs, 2,885 segments, 6 municipalities) | R10 |
| R3 W2c | Low-support types make per-type F1 meaningless | R5 |
| R3 W3 | Writing quality; the quoted sentence | **G2 ✅**, G11 |
| R3 W4 | Single language; "theoretically language-agnostic" | X3 |
| R3 W5 / C1 | Appendices before References | S2 |
| R3 C2 | Error analysis brief; show concrete examples | E2, E3 |
| R3 C3 | Char offsets vs. token-level BIO | X4 |
| R3 C4 | Missing full stops (×3) | G7, G8, G9 |
| R4 W | Limited technical novelty | P3 |
| R4 C1 | "costs makes" | G3 |
| R4 C2 | References before appendix | S2 |
| Meta M1 | Describe the task unambiguously + its setup | F1–F8, F7 |
| Meta M2 | How does the task differ from CitiLink-Minutes? Subset of categories? | **P1** |
| Meta M3 | Error analysis: manual analysis over a sample | E1 |
| Meta M4 | Focus more on the LOMO setup | L15, L16 |
| Meta M5 | Add Table 1's methods to Table 2 | L6–L14 |
| Meta M6 | Novelty is limited | P3 |
| Meta M7 | Incorporate all reviewer comments | this whole file |
| Meta M8 | Demo shows an error when opened | D1 |

---

## 14. Cost & latency reference (no re-run needed)

Computed from `processing_time` in the saved prediction files. Use these to replace Appendix C's
*"Gemini 2.5 Pro and GPT 5.5 were accessed via API and no GPU costs were measured"* (L842) with a
commensurability paragraph, and to extend Table 10.

| Model | 0-shot s/doc | 5-shot s/doc | 0-shot wall-clock | 5-shot wall-clock |
|---|---|---|---|---|
| Gemini 2.5 Pro | 27.8 | 26.9 | 4.09 h | 3.95 h |
| GPT-5.5 | 22.8 | 15.0 | 3.35 h | 2.21 h |
| AMALIA 8B (local) | 52.9 | 89.2 | 7.77 h | 13.11 h |

- [ ] **M6** — Apply the above to Table 10 + the App. C paragraph. Add measured token cost per 1k
  documents once N3–N5 land.

---

## 15. Suggested order

1. **Mechanical pass** — S1–S4, G3–G10, R4. Zero risk, clears every editorial complaint.
2. **Start compute early** (it gates appendix content) — N1, N2 → L4–L8 → N3–N5 → L9–L14.
3. **Table regeneration** — C4/L3, then rebuild Tables 1, 2, 3, 11 from scripts only, never by hand.
   Decide the L2 denominator policy first, since it changes every Table 2 cell.
4. **Framing & positioning prose** — F5–F8, P1–P5, X1–X5.
5. **Results prose** — R1–R10, L15–L17, M6.
6. **Error analysis** — E1–E3.
7. **Demo & artifacts** — D1–D4.
8. **Final build + page check** — S5, S6.

---

## 16. Final verification

- [ ] `latexmk -pdf acl_latex.tex` → no undefined references or citations.
- [ ] Compiled order is Conclusion → Limitations → Ethics → **References** → Appendices.
- [ ] Content (§1–§7) ≤ 5 pages. Limitations/Ethics/References/Appendices exempt.
- [ ] Every number in Tables 1, 2, 3 and 11 traces to a committed file under `results/`; re-running the
  aggregation scripts on a clean checkout reproduces the LaTeX. **Tables 1, 2, 4 and 5 all verified to
  reproduce** (Table 2 once the XLM-R seeds land — L4).
- [ ] Table 2's caption states the macro denominator and the per-cell seed count (L2, L3b).
- [ ] `alignment_report.py` emits real discard rates for all six generative configs — it refuses on
  uninstrumented runs, so a silent 0% cannot slip through.
- [ ] Re-run generative scores match previously reported ones within API noise; drift is reported, not
  hidden.
- [ ] Demo Space loads without error from a logged-out browser.
- [ ] No `Anonymous3445` occurrences remain; author block filled.
