#!/usr/bin/env python3
"""Bootstrap-based statistical tests for VotIE per-seed runs.

For each (model, seed) prediction file in ``results/discriminative_models/{model}/seeds/``
(in-domain) and ``results/lomo/{model}/`` (LOMO), this script computes:

  1. Per-seed bootstrap 95 % CI on macro F1 (exact and relaxed) by
     resampling test documents with replacement.
  2. A pooled CI across the three seeds: the same resampled document
     indices are scored with every seed's model, and the per-iteration
     statistic is the mean F1 across seeds. This integrates seed variance
     while preserving the paired structure across documents.
  3. Paired bootstrap significance tests for every model pair, both
     per-seed and pooled across the three seeds. p-values follow the
     two-sided Berg-Kirkpatrick (2012) formulation
     ``P(|D_b - D_obs| >= |D_obs|)``; a one-sided ``P(A > B)`` is also
     reported.

Per-document TP/FP/FN are cached once per file, so 1 000 bootstrap
iterations on ~500-document test sets run in seconds. Macro F1 is
re-derived from the resampled counts using exactly the same per-type-F1
averaging that ``scripts/evaluate.py`` uses, so the ``observed`` numbers
match the cached values in ``results/seed_aggregate_summary.json``.

Usage
-----
    python scripts/statistical_tests.py
    python scripts/statistical_tests.py --n-bootstrap 10000
    python scripts/statistical_tests.py --in-domain-only
    python scripts/statistical_tests.py --lomo-only --n-bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seqeval.metrics.sequence_labeling import get_entities  # noqa: E402

from scripts.evaluate import (  # noqa: E402  (sys.path tweak above)
    bio_to_spans,
    extract_labels,
    load_predictions,
    relaxed_span_match,
)

# --------------------------------------------------------------------- #
# Configuration                                                         #
# --------------------------------------------------------------------- #

MODELS = ["bert_crf", "xlmr_crf", "deberta_crf"]
MUNICIPALITIES = ["M01", "M02", "M03", "M04", "M05", "M06"]
SEEDS: List[Tuple[str, str]] = [
    ("42", "deucalion"),
    ("13", "seeds_s13"),
    ("123", "seeds_s123"),
]

RESULTS_DIR = ROOT / "results"
DISC_DIR = RESULTS_DIR / "discriminative_models"
# Rebound by --lomo-dir. results/lomo holds the superseded batch-16 runs; the
# camera-ready batch is results/lomo_cr. Mixing the two would compare models
# trained under different hyperparameters, so the root is stated explicitly in
# the output rather than assumed.
LOMO_RESULTS_DIR = RESULTS_DIR / "lomo_cr"
OUT_PATH = RESULTS_DIR / "statistical_tests.json"

EPS = 1e-12


# --------------------------------------------------------------------- #
# Prediction-file paths                                                 #
# --------------------------------------------------------------------- #

def in_domain_pred(model: str, seed_id: str) -> Path:
    return DISC_DIR / model / "seeds" / f"seed_s{seed_id}_predictions.jsonl"


def lomo_pred(model: str, mun: str, seed_id: str) -> Path:
    return LOMO_RESULTS_DIR / model / f"{mun}_seed_s{seed_id}_predictions.jsonl"


# --------------------------------------------------------------------- #
# Per-document TP / FP / FN counts                                      #
# --------------------------------------------------------------------- #

def _exact_counts_from_bio(
    pred_labels: List[str], gold_labels: List[str]
) -> Dict[str, List[int]]:
    """Per-type [tp, fp, fn] for one document using seqeval's IOB2 entities.

    Using seqeval directly here makes the resampled F1 reproduce the
    cached per-seed macro F1 exactly.
    """
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    # get_entities returns [(type, start_tok, end_tok), ...]
    pred_entities = get_entities([pred_labels])
    gold_entities = get_entities([gold_labels])

    gold_set: set = set()
    gold_buckets: Dict[Tuple[str, int, int], List[int]] = defaultdict(list)
    for k, (t, s, e) in enumerate(gold_entities):
        gold_buckets[(t, s, e)].append(k)
        gold_set.add((t, s, e))

    matched_gold: set = set()
    for t, s, e in pred_entities:
        bucket = gold_buckets.get((t, s, e), [])
        j = next((g for g in bucket if g not in matched_gold), None)
        if j is not None:
            counts[t][0] += 1
            matched_gold.add(j)
        else:
            counts[t][1] += 1
    for k, (t, s, e) in enumerate(gold_entities):
        if k not in matched_gold:
            counts[t][2] += 1
    return counts


def _relaxed_counts_from_spans(
    pred_spans: List[dict], gold_spans: List[dict]
) -> Dict[str, List[int]]:
    """Per-type [tp, fp, fn] using overlap-based matching on char spans.

    Mirrors ``scripts/evaluate.py:compute_relaxed_metrics`` so the
    resampled F1 reproduces the cached relaxed numbers exactly.
    """
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    matched_gold: set = set()
    for p in pred_spans:
        matched = False
        for j, g in enumerate(gold_spans):
            if j in matched_gold:
                continue
            if relaxed_span_match(p, g):
                counts[p["type"]][0] += 1
                matched_gold.add(j)
                matched = True
                break
        if not matched:
            counts[p["type"]][1] += 1
    for j, g in enumerate(gold_spans):
        if j not in matched_gold:
            counts[g["type"]][2] += 1
    return counts


def per_doc_counts(
    pred_labels_list: List[List[str]],
    gold_labels_list: List[List[str]],
    tokens_list: List[List[str]],
    relaxed: bool,
) -> Tuple[np.ndarray, List[str]]:
    """Return a (n_docs, n_types, 3) int array and the matching type list."""
    per_doc = []
    all_types: set = set()
    for pl, gl, tk in zip(pred_labels_list, gold_labels_list, tokens_list):
        if relaxed:
            pred_spans = bio_to_spans(tk, pl)
            gold_spans = bio_to_spans(tk, gl)
            tc = _relaxed_counts_from_spans(pred_spans, gold_spans)
        else:
            tc = _exact_counts_from_bio(pl, gl)
        per_doc.append(tc)
        all_types.update(tc.keys())

    types = sorted(all_types)
    type_to_idx = {t: i for i, t in enumerate(types)}
    arr = np.zeros((len(per_doc), len(types), 3), dtype=np.int32)
    for i, tc in enumerate(per_doc):
        for t, (tp, fp, fn) in tc.items():
            j = type_to_idx[t]
            arr[i, j] = (tp, fp, fn)
    return arr, types


# --------------------------------------------------------------------- #
# Macro F1 from aggregated counts                                       #
# --------------------------------------------------------------------- #

def macro_f1_from_counts(agg: np.ndarray) -> float:
    """``agg`` has shape (n_types, 3); average per-type F1 over types with
    gold support in the sample (tp+fn > 0). Matches the macro denominator
    used in Table 3 / §4 of the manuscript: types with zero gold support
    in the held-out municipality are excluded so that spurious predictions
    on absent types do not drag the macro to zero on that axis.
    """
    tp = agg[:, 0].astype(np.float64)
    fp = agg[:, 1].astype(np.float64)
    fn = agg[:, 2].astype(np.float64)
    denom_p = tp + fp
    denom_r = tp + fn
    p = np.where(denom_p > 0, tp / np.maximum(denom_p, EPS), 0.0)
    r = np.where(denom_r > 0, tp / np.maximum(denom_r, EPS), 0.0)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, EPS), 0.0)
    present = (tp + fn) > 0
    if not present.any():
        return 0.0
    return float(f1[present].mean())


# --------------------------------------------------------------------- #
# Bootstrap routines                                                    #
# --------------------------------------------------------------------- #

def _sample_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    """``b`` × ``n`` matrix of resampled document indices."""
    return rng.integers(0, n, size=(b, n))


def bootstrap_ci(
    counts: np.ndarray,
    n_iter: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    n_docs = counts.shape[0]
    idx_mat = _sample_indices(rng, n_docs, n_iter)
    samples = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        samples[b] = macro_f1_from_counts(counts[idx_mat[b]].sum(axis=0))
    observed = macro_f1_from_counts(counts.sum(axis=0))
    return {
        "observed": observed,
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_std": float(samples.std(ddof=1)),
        "ci_low": float(np.percentile(samples, 100 * alpha / 2)),
        "ci_high": float(np.percentile(samples, 100 * (1 - alpha / 2))),
        "n_docs": int(n_docs),
        "n_iter": int(n_iter),
        "alpha": alpha,
    }


def pooled_bootstrap_ci(
    counts_per_seed: List[np.ndarray],
    n_iter: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Pooled CI: a single resampled index vector is scored under every seed
    and the bootstrap statistic is the mean F1 across seeds."""
    n_docs = counts_per_seed[0].shape[0]
    assert all(c.shape[0] == n_docs for c in counts_per_seed), \
        "Pooled CI needs the same document set under every seed."
    idx_mat = _sample_indices(rng, n_docs, n_iter)
    samples = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = idx_mat[b]
        per_seed = [macro_f1_from_counts(c[idx].sum(axis=0))
                    for c in counts_per_seed]
        samples[b] = float(np.mean(per_seed))
    observed = float(np.mean(
        [macro_f1_from_counts(c.sum(axis=0)) for c in counts_per_seed]
    ))
    return {
        "observed": observed,
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_std": float(samples.std(ddof=1)),
        "ci_low": float(np.percentile(samples, 100 * alpha / 2)),
        "ci_high": float(np.percentile(samples, 100 * (1 - alpha / 2))),
        "n_docs": int(n_docs),
        "n_iter": int(n_iter),
        "n_seeds_pooled": len(counts_per_seed),
        "alpha": alpha,
    }


def paired_bootstrap_test(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    assert counts_a.shape[0] == counts_b.shape[0]
    n_docs = counts_a.shape[0]
    idx_mat = _sample_indices(rng, n_docs, n_iter)
    observed = (
        macro_f1_from_counts(counts_a.sum(axis=0))
        - macro_f1_from_counts(counts_b.sum(axis=0))
    )
    deltas = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = idx_mat[b]
        f_a = macro_f1_from_counts(counts_a[idx].sum(axis=0))
        f_b = macro_f1_from_counts(counts_b[idx].sum(axis=0))
        deltas[b] = f_a - f_b
    # Berg-Kirkpatrick (2012) two-sided p-value
    p_two = float(np.mean(np.abs(deltas - observed) >= abs(observed)))
    p_a_gt_b = float(np.mean(deltas <= 0))      # how often A failed to beat B
    return {
        "observed_delta_macro_f1": observed,
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_std_delta": float(deltas.std(ddof=1)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_two_sided": p_two,
        "p_one_sided_a_gt_b": p_a_gt_b,
        "n_docs": int(n_docs),
        "n_iter": int(n_iter),
    }


def paired_bootstrap_test_pooled(
    counts_a_per_seed: List[np.ndarray],
    counts_b_per_seed: List[np.ndarray],
    n_iter: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Per-iteration statistic = mean of per-seed deltas."""
    assert len(counts_a_per_seed) == len(counts_b_per_seed)
    n_docs = counts_a_per_seed[0].shape[0]
    idx_mat = _sample_indices(rng, n_docs, n_iter)
    observed_deltas = [
        macro_f1_from_counts(a.sum(axis=0))
        - macro_f1_from_counts(b.sum(axis=0))
        for a, b in zip(counts_a_per_seed, counts_b_per_seed)
    ]
    observed = float(np.mean(observed_deltas))
    deltas = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        idx = idx_mat[b]
        d_seed = [
            macro_f1_from_counts(a[idx].sum(axis=0))
            - macro_f1_from_counts(bb[idx].sum(axis=0))
            for a, bb in zip(counts_a_per_seed, counts_b_per_seed)
        ]
        deltas[b] = float(np.mean(d_seed))
    p_two = float(np.mean(np.abs(deltas - observed) >= abs(observed)))
    p_a_gt_b = float(np.mean(deltas <= 0))
    return {
        "observed_delta_macro_f1": observed,
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_std_delta": float(deltas.std(ddof=1)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_two_sided": p_two,
        "p_one_sided_a_gt_b": p_a_gt_b,
        "n_docs": int(n_docs),
        "n_iter": int(n_iter),
        "n_seeds_pooled": len(counts_a_per_seed),
    }


# --------------------------------------------------------------------- #
# Loading and aligning prediction files                                 #
# --------------------------------------------------------------------- #

def load_counts(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    preds = load_predictions(str(path))
    pl, gl, tk = extract_labels(preds)
    exact, exact_types = per_doc_counts(pl, gl, tk, relaxed=False)
    relaxed, relaxed_types = per_doc_counts(pl, gl, tk, relaxed=True)
    return {
        "ids": [p["id"] for p in preds],
        "exact": exact,
        "exact_types": exact_types,
        "relaxed": relaxed,
        "relaxed_types": relaxed_types,
    }


def _align_within_group(runs: Dict[str, Optional[dict]]) -> None:
    """Reorder every present run so doc ids match across seeds."""
    present = [r for r in runs.values() if r is not None]
    if len(present) < 2:
        return
    base = present[0]["ids"]
    if all(r["ids"] == base for r in present):
        return
    common = set(base)
    for r in present:
        common &= set(r["ids"])
    common_sorted = [i for i in base if i in common]
    if len(common_sorted) < len(base):
        logging.warning(
            "Seed prediction files disagree on doc ids; restricting to "
            "%d common documents (from %d).",
            len(common_sorted), len(base),
        )
    for r in present:
        pos = {i: k for k, i in enumerate(r["ids"])}
        order = [pos[i] for i in common_sorted]
        r["ids"] = common_sorted
        r["exact"] = r["exact"][order]
        r["relaxed"] = r["relaxed"][order]


def _align_pair(ra: dict, rb: dict) -> Tuple[dict, dict]:
    """Return ra/rb restricted to their common doc ids in matched order."""
    if ra["ids"] == rb["ids"]:
        return ra, rb
    common = set(ra["ids"]) & set(rb["ids"])
    common_sorted = [i for i in ra["ids"] if i in common]
    pos_a = {i: k for k, i in enumerate(ra["ids"])}
    pos_b = {i: k for k, i in enumerate(rb["ids"])}
    order_a = [pos_a[i] for i in common_sorted]
    order_b = [pos_b[i] for i in common_sorted]
    ra_aligned = {**ra, "ids": common_sorted,
                  "exact": ra["exact"][order_a],
                  "relaxed": ra["relaxed"][order_a]}
    rb_aligned = {**rb, "ids": common_sorted,
                  "exact": rb["exact"][order_b],
                  "relaxed": rb["relaxed"][order_b]}
    return ra_aligned, rb_aligned


# --------------------------------------------------------------------- #
# In-domain analysis                                                    #
# --------------------------------------------------------------------- #

def analyse_indomain(
    n_bootstrap: int, alpha: float, rng: np.random.Generator
) -> dict:
    runs: Dict[str, Dict[str, Optional[dict]]] = {}
    for model in MODELS:
        runs[model] = {}
        for seed_id, tag in SEEDS:
            runs[model][seed_id] = load_counts(in_domain_pred(model, seed_id))
        _align_within_group(runs[model])

    per_model: Dict[str, dict] = {}
    for model in MODELS:
        per_seed_exact: Dict[str, Optional[dict]] = {}
        per_seed_relaxed: Dict[str, Optional[dict]] = {}
        counts_e, counts_r = [], []
        for seed_id, _ in SEEDS:
            r = runs[model][seed_id]
            if r is None:
                per_seed_exact[seed_id] = None
                per_seed_relaxed[seed_id] = None
                continue
            per_seed_exact[seed_id] = bootstrap_ci(
                r["exact"], n_bootstrap, alpha, rng
            )
            per_seed_relaxed[seed_id] = bootstrap_ci(
                r["relaxed"], n_bootstrap, alpha, rng
            )
            counts_e.append(r["exact"])
            counts_r.append(r["relaxed"])
        per_model[model] = {
            "per_seed_ci_exact_macro_f1": per_seed_exact,
            "per_seed_ci_relaxed_macro_f1": per_seed_relaxed,
            "pooled_ci_exact_macro_f1": (
                pooled_bootstrap_ci(counts_e, n_bootstrap, alpha, rng)
                if len(counts_e) >= 2 else None
            ),
            "pooled_ci_relaxed_macro_f1": (
                pooled_bootstrap_ci(counts_r, n_bootstrap, alpha, rng)
                if len(counts_r) >= 2 else None
            ),
        }

    pairwise: Dict[str, dict] = {}
    for model_a, model_b in combinations(MODELS, 2):
        per_seed_exact: Dict[str, Optional[dict]] = {}
        per_seed_relaxed: Dict[str, Optional[dict]] = {}
        a_e, b_e, a_r, b_r = [], [], [], []
        for seed_id, _ in SEEDS:
            ra = runs[model_a][seed_id]
            rb = runs[model_b][seed_id]
            if ra is None or rb is None:
                per_seed_exact[seed_id] = None
                per_seed_relaxed[seed_id] = None
                continue
            ra_a, rb_a = _align_pair(ra, rb)
            per_seed_exact[seed_id] = paired_bootstrap_test(
                ra_a["exact"], rb_a["exact"], n_bootstrap, rng
            )
            per_seed_relaxed[seed_id] = paired_bootstrap_test(
                ra_a["relaxed"], rb_a["relaxed"], n_bootstrap, rng
            )
            a_e.append(ra_a["exact"]); b_e.append(rb_a["exact"])
            a_r.append(ra_a["relaxed"]); b_r.append(rb_a["relaxed"])
        pairwise[f"{model_a}_vs_{model_b}"] = {
            "per_seed_exact_macro_f1": per_seed_exact,
            "per_seed_relaxed_macro_f1": per_seed_relaxed,
            "pooled_exact_macro_f1": (
                paired_bootstrap_test_pooled(a_e, b_e, n_bootstrap, rng)
                if len(a_e) >= 2 else None
            ),
            "pooled_relaxed_macro_f1": (
                paired_bootstrap_test_pooled(a_r, b_r, n_bootstrap, rng)
                if len(a_r) >= 2 else None
            ),
        }

    return {"per_model": per_model, "pairwise": pairwise}


# --------------------------------------------------------------------- #
# LOMO analysis                                                         #
# --------------------------------------------------------------------- #

def analyse_lomo(
    n_bootstrap: int, alpha: float, rng: np.random.Generator
) -> dict:
    runs: Dict[str, Dict[str, Dict[str, Optional[dict]]]] = {}
    for model in MODELS:
        runs[model] = {}
        for mun in MUNICIPALITIES:
            runs[model][mun] = {
                seed_id: load_counts(lomo_pred(model, mun, seed_id))
                for seed_id, _ in SEEDS
            }
            _align_within_group(runs[model][mun])

    per_model: Dict[str, dict] = {}
    for model in MODELS:
        muns_out: Dict[str, dict] = {}
        for mun in MUNICIPALITIES:
            per_seed_e, per_seed_r = {}, {}
            ce, cr = [], []
            for seed_id, _ in SEEDS:
                r = runs[model][mun][seed_id]
                if r is None:
                    per_seed_e[seed_id] = None
                    per_seed_r[seed_id] = None
                    continue
                per_seed_e[seed_id] = bootstrap_ci(
                    r["exact"], n_bootstrap, alpha, rng
                )
                per_seed_r[seed_id] = bootstrap_ci(
                    r["relaxed"], n_bootstrap, alpha, rng
                )
                ce.append(r["exact"]); cr.append(r["relaxed"])
            muns_out[mun] = {
                "per_seed_ci_exact_macro_f1": per_seed_e,
                "per_seed_ci_relaxed_macro_f1": per_seed_r,
                "pooled_ci_exact_macro_f1": (
                    pooled_bootstrap_ci(ce, n_bootstrap, alpha, rng)
                    if len(ce) >= 2 else None
                ),
                "pooled_ci_relaxed_macro_f1": (
                    pooled_bootstrap_ci(cr, n_bootstrap, alpha, rng)
                    if len(cr) >= 2 else None
                ),
            }
        per_model[model] = {"per_municipality": muns_out}

    pairwise: Dict[str, dict] = {}
    for model_a, model_b in combinations(MODELS, 2):
        muns_out: Dict[str, dict] = {}
        for mun in MUNICIPALITIES:
            per_seed_e, per_seed_r = {}, {}
            a_e, b_e, a_r, b_r = [], [], [], []
            for seed_id, _ in SEEDS:
                ra = runs[model_a][mun][seed_id]
                rb = runs[model_b][mun][seed_id]
                if ra is None or rb is None:
                    per_seed_e[seed_id] = None
                    per_seed_r[seed_id] = None
                    continue
                ra_a, rb_a = _align_pair(ra, rb)
                per_seed_e[seed_id] = paired_bootstrap_test(
                    ra_a["exact"], rb_a["exact"], n_bootstrap, rng
                )
                per_seed_r[seed_id] = paired_bootstrap_test(
                    ra_a["relaxed"], rb_a["relaxed"], n_bootstrap, rng
                )
                a_e.append(ra_a["exact"]); b_e.append(rb_a["exact"])
                a_r.append(ra_a["relaxed"]); b_r.append(rb_a["relaxed"])
            muns_out[mun] = {
                "per_seed_exact_macro_f1": per_seed_e,
                "per_seed_relaxed_macro_f1": per_seed_r,
                "pooled_exact_macro_f1": (
                    paired_bootstrap_test_pooled(a_e, b_e, n_bootstrap, rng)
                    if len(a_e) >= 2 else None
                ),
                "pooled_relaxed_macro_f1": (
                    paired_bootstrap_test_pooled(a_r, b_r, n_bootstrap, rng)
                    if len(a_r) >= 2 else None
                ),
            }
        pairwise[f"{model_a}_vs_{model_b}"] = {"per_municipality": muns_out}

    return {"per_model": per_model, "pairwise": pairwise}


# --------------------------------------------------------------------- #
# Pretty printing                                                       #
# --------------------------------------------------------------------- #

def fmt_pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{100 * x:5.2f}"


def _fmt_ci(ci: Optional[dict]) -> str:
    if ci is None:
        return f"{'n/a':>22}"
    return f"{fmt_pct(ci['observed'])} [{fmt_pct(ci['ci_low'])}, {fmt_pct(ci['ci_high'])}]"


def _fmt_delta(t: Optional[dict]) -> str:
    if t is None:
        return f"{'n/a':>32}"
    d = 100 * t["observed_delta_macro_f1"]
    lo = 100 * t["ci_low"]
    hi = 100 * t["ci_high"]
    sig = "*" if t["p_two_sided"] < 0.05 else " "
    return (f"{d:+6.2f} [{lo:+6.2f},{hi:+6.2f}] "
            f"p={t['p_two_sided']:.3f}{sig}")


def print_indomain(summary: dict) -> None:
    print("\n=== In-domain bootstrap CIs (macro F1) ===")
    print(f"{'model':<13} {'metric':<6} {'seed':>5} "
          f"{'observed [95% CI]':>22}")
    for model, d in summary["per_model"].items():
        for metric, key_per, key_pooled in (
            ("exact", "per_seed_ci_exact_macro_f1", "pooled_ci_exact_macro_f1"),
            ("relax", "per_seed_ci_relaxed_macro_f1", "pooled_ci_relaxed_macro_f1"),
        ):
            for sid, ci in d[key_per].items():
                print(f"  {model:<11} {metric:<6} {sid:>5} {_fmt_ci(ci)}")
            print(f"  {model:<11} {metric:<6} {'pool':>5} {_fmt_ci(d[key_pooled])}")
    print("\n=== In-domain pairwise paired bootstrap (macro F1) ===")
    print(f"{'pair':<30} {'metric':<6} {'seed':>5} "
          f"{'Δ F1 [95% CI] p':>33}")
    for pair, d in summary["pairwise"].items():
        for metric, key_per, key_pooled in (
            ("exact", "per_seed_exact_macro_f1", "pooled_exact_macro_f1"),
            ("relax", "per_seed_relaxed_macro_f1", "pooled_relaxed_macro_f1"),
        ):
            for sid, t in d[key_per].items():
                print(f"  {pair:<28} {metric:<6} {sid:>5} {_fmt_delta(t)}")
            print(f"  {pair:<28} {metric:<6} {'pool':>5} {_fmt_delta(d[key_pooled])}")


def _first_available(d: dict, keys: List[str]) -> Optional[dict]:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def print_lomo(summary: dict) -> None:
    print("\n=== LOMO bootstrap CIs (macro F1) ===")
    print(f"{'model':<13} {'metric':<6} {'mun':>4} {'source':<6} "
          f"{'observed [95% CI]':>22}  n_docs")
    for model, d in summary["per_model"].items():
        for mun, mun_d in d["per_municipality"].items():
            for metric, pooled_key, per_seed_key in (
                ("exact", "pooled_ci_exact_macro_f1",
                 "per_seed_ci_exact_macro_f1"),
                ("relax", "pooled_ci_relaxed_macro_f1",
                 "per_seed_ci_relaxed_macro_f1"),
            ):
                ci = mun_d.get(pooled_key)
                source = "pool"
                if ci is None:
                    # fall back to whichever single-seed CI we have
                    for sid in ("42", "13", "123"):
                        c = mun_d.get(per_seed_key, {}).get(sid)
                        if c is not None:
                            ci = c
                            source = f"s{sid}"
                            break
                if ci is None:
                    continue
                print(f"  {model:<11} {metric:<6} {mun:>4} {source:<6} "
                      f"{_fmt_ci(ci)}  {ci['n_docs']:>4}")
    print("\n=== LOMO pairwise paired bootstrap (macro F1) ===")
    print(f"{'pair':<30} {'metric':<6} {'mun':>4} {'source':<6} "
          f"{'Δ F1 [95% CI] p':>33}")
    for pair, d in summary["pairwise"].items():
        for mun, mun_d in d["per_municipality"].items():
            for metric, pooled_key, per_seed_key in (
                ("exact", "pooled_exact_macro_f1",
                 "per_seed_exact_macro_f1"),
                ("relax", "pooled_relaxed_macro_f1",
                 "per_seed_relaxed_macro_f1"),
            ):
                t = mun_d.get(pooled_key)
                source = "pool"
                if t is None:
                    for sid in ("42", "13", "123"):
                        c = mun_d.get(per_seed_key, {}).get(sid)
                        if c is not None:
                            t = c
                            source = f"s{sid}"
                            break
                if t is None:
                    continue
                print(f"  {pair:<28} {metric:<6} {mun:>4} {source:<6} "
                      f"{_fmt_delta(t)}")


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> None:
    global LOMO_RESULTS_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples (default: 1000).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Two-sided CI alpha (default: 0.05).")
    parser.add_argument("--seed", type=int, default=20260526,
                        help="RNG seed for reproducibility (default: today).")
    parser.add_argument("--in-domain-only", action="store_true",
                        help="Skip LOMO analysis.")
    parser.add_argument("--lomo-only", action="store_true",
                        help="Skip in-domain analysis.")
    parser.add_argument("--out", type=Path, default=OUT_PATH,
                        help=f"Output JSON path (default: {OUT_PATH.relative_to(ROOT)}).")
    parser.add_argument("--lomo-dir", type=Path, default=LOMO_RESULTS_DIR,
                        help=f"LOMO predictions root (default: "
                             f"{LOMO_RESULTS_DIR.relative_to(ROOT)}). Pass results/lomo "
                             f"only to reproduce the superseded batch-16 numbers.")
    args = parser.parse_args()

    LOMO_RESULTS_DIR = args.lomo_dir
    if not args.in_domain_only and not LOMO_RESULTS_DIR.exists():
        raise SystemExit(f"LOMO predictions root not found: {LOMO_RESULTS_DIR}")

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")

    summary: dict = {
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "rng_seed": args.seed,
            "models": MODELS,
            "seeds": [sid for sid, _ in SEEDS],
            "municipalities": MUNICIPALITIES,
            "lomo_results_dir": str(LOMO_RESULTS_DIR),
            "p_value_definition": (
                "Two-sided Berg-Kirkpatrick (2012): "
                "P(|D_b - D_obs| >= |D_obs|) under the paired bootstrap; "
                "p_one_sided_a_gt_b = P(D_b <= 0)."
            ),
        }
    }

    if not args.lomo_only:
        logging.info("Computing in-domain statistics...")
        rng = np.random.default_rng(args.seed)
        summary["in_domain"] = analyse_indomain(args.n_bootstrap, args.alpha, rng)

    if not args.in_domain_only:
        logging.info("Computing LOMO statistics...")
        rng = np.random.default_rng(args.seed + 1)
        summary["lomo"] = analyse_lomo(args.n_bootstrap, args.alpha, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    try:
        written = args.out.relative_to(ROOT)
    except ValueError:  # --out may point outside the repo
        written = args.out
    print(f"\nWrote {written}")

    if "in_domain" in summary:
        print_indomain(summary["in_domain"])
    if "lomo" in summary:
        print_lomo(summary["lomo"])


if __name__ == "__main__":
    main()
