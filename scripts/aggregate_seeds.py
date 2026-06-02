"""Aggregate macro/entity F1 across the three seeds (42, 13, 123) for the
in-domain standard benchmark and for the LOMO experiments. Writes a JSON
summary to results/seed_aggregate_summary.json and prints a human-readable
table.

File layout under results/:
  In-domain:
    discriminative_models/<model>/evaluation.json             seed = 42
    discriminative_models/<model>/seeds/seed_s13_evaluation.json   seed = 13
    discriminative_models/<model>/seeds/seed_s123_evaluation.json  seed = 123
  LOMO:
    lomo/<model>/<M>_evaluation.json                          seed = 42
    lomo/<model>/<M>_seed_s13_evaluation.json                 seed = 13
    lomo/<model>/<M>_seed_s123_evaluation.json                seed = 123

The "macro" F1 the paper reports is computed as the per-type-F1 average over
the 11 entity types (excluding O). The entity_f1 field in the JSON is a
micro-style aggregate; we recompute the macro average from per_type_metrics
for the headline numbers, and also keep micro/entity_f1 for completeness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DISC_DIR = RESULTS / "discriminative_models"
LOMO_DIR = RESULTS / "lomo"
OUT = RESULTS / "seed_aggregate_summary.json"

MODELS = ["bert_crf", "xlmr_crf", "deberta_crf"]
MUNICIPALITIES = ["M01", "M02", "M03", "M04", "M05", "M06"]
SEEDS = [
    ("42", "deucalion"),
    ("13", "seeds_s13"),
    ("123", "seeds_s123"),
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def macro_f1(report: dict) -> float | None:
    per_type = report.get("entity_level_metrics", {}).get("per_type_metrics", {})
    if not per_type:
        return None
    f1s = [v["f1_score"] for v in per_type.values()]
    return sum(f1s) / len(f1s) if f1s else None


def micro_f1(report: dict) -> float | None:
    return report.get("entity_level_metrics", {}).get("entity_f1")


def exact_pr(report: dict) -> tuple[float | None, float | None]:
    em = report.get("entity_level_metrics", {})
    return em.get("entity_precision"), em.get("entity_recall")


def relaxed_prf(report: dict) -> tuple[float | None, float | None, float | None]:
    rm = report.get("relaxed_boundary_metrics") or {}
    return rm.get("precision"), rm.get("recall"), rm.get("f1")


def aggregate(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "range": None, "values": []}
    n = len(values)
    mu = mean(values)
    sd = pstdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mu,
        "std": sd,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "values": values,
    }


def in_domain_paths(model: str) -> dict[str, Path]:
    seed_map = {
        "42": DISC_DIR / model / "evaluation.json",
        "13": DISC_DIR / model / "seeds" / "seed_s13_evaluation.json",
        "123": DISC_DIR / model / "seeds" / "seed_s123_evaluation.json",
    }
    return seed_map


def lomo_paths(model: str, mun: str) -> dict[str, Path]:
    return {
        "42":  LOMO_DIR / model / f"{mun}_seed_s42_evaluation.json",
        "13":  LOMO_DIR / model / f"{mun}_seed_s13_evaluation.json",
        "123": LOMO_DIR / model / f"{mun}_seed_s123_evaluation.json",
    }


def summarise_indomain() -> dict:
    out = {}
    for model in MODELS:
        per_seed = {}
        macros, micros = [], []
        exact_Ps, exact_Rs = [], []
        relaxed_Ps, relaxed_Rs, relaxed_F1s = [], [], []
        per_type_seed_runs: dict[str, list[float]] = {}
        for seed_id, _ in SEEDS:
            path = in_domain_paths(model)[seed_id]
            rep = load(path)
            if rep is None:
                per_seed[seed_id] = {"path": str(path), "present": False}
                continue
            ma = macro_f1(rep)
            mi = micro_f1(rep)
            eP, eR = exact_pr(rep)
            rP, rR, rF = relaxed_prf(rep)
            per_seed[seed_id] = {
                "path": str(path),
                "present": True,
                "macro_f1": ma,
                "micro_f1": mi,
                "exact_precision": eP,
                "exact_recall": eR,
                "relaxed_precision": rP,
                "relaxed_recall": rR,
                "relaxed_f1": rF,
            }
            if ma is not None:
                macros.append(ma)
            if mi is not None:
                micros.append(mi)
            if eP is not None:
                exact_Ps.append(eP)
            if eR is not None:
                exact_Rs.append(eR)
            if rP is not None:
                relaxed_Ps.append(rP)
            if rR is not None:
                relaxed_Rs.append(rR)
            if rF is not None:
                relaxed_F1s.append(rF)
            for tname, tm in rep["entity_level_metrics"]["per_type_metrics"].items():
                per_type_seed_runs.setdefault(tname, []).append(tm["f1_score"])
        out[model] = {
            "per_seed": per_seed,
            "macro_f1": aggregate(macros),
            "micro_f1": aggregate(micros),
            "exact_precision": aggregate(exact_Ps),
            "exact_recall": aggregate(exact_Rs),
            "relaxed_precision": aggregate(relaxed_Ps),
            "relaxed_recall": aggregate(relaxed_Rs),
            "relaxed_f1": aggregate(relaxed_F1s),
            "per_type_macro": {t: aggregate(v) for t, v in per_type_seed_runs.items()},
        }
    return out


def summarise_lomo() -> dict:
    out = {}
    for model in MODELS:
        model_out = {"per_municipality": {}, "mean_macro_f1_per_seed": {},
                     "mean_micro_f1_per_seed": {}, "mean_relaxed_f1_per_seed": {},
                     "summary": {}}
        per_seed_means_macro: dict[str, list[float]] = {s: [] for s, _ in SEEDS}
        per_seed_means_micro: dict[str, list[float]] = {s: [] for s, _ in SEEDS}
        per_seed_means_relaxed: dict[str, list[float]] = {s: [] for s, _ in SEEDS}
        for mun in MUNICIPALITIES:
            paths = lomo_paths(model, mun)
            macros, micros, relaxed_F1s, per_seed = [], [], [], {}
            for seed_id, _ in SEEDS:
                rep = load(paths[seed_id])
                if rep is None:
                    per_seed[seed_id] = {"path": str(paths[seed_id]), "present": False}
                    continue
                ma = macro_f1(rep)
                mi = micro_f1(rep)
                _, _, rF = relaxed_prf(rep)
                per_seed[seed_id] = {
                    "path": str(paths[seed_id]),
                    "present": True,
                    "macro_f1": ma,
                    "micro_f1": mi,
                    "relaxed_f1": rF,
                }
                if ma is not None:
                    macros.append(ma)
                    per_seed_means_macro[seed_id].append(ma)
                if mi is not None:
                    micros.append(mi)
                    per_seed_means_micro[seed_id].append(mi)
                if rF is not None:
                    relaxed_F1s.append(rF)
                    per_seed_means_relaxed[seed_id].append(rF)
            model_out["per_municipality"][mun] = {
                "per_seed": per_seed,
                "macro_f1": aggregate(macros),
                "micro_f1": aggregate(micros),
                "relaxed_f1": aggregate(relaxed_F1s),
            }
        for seed_id, _ in SEEDS:
            vals_ma = per_seed_means_macro[seed_id]
            vals_mi = per_seed_means_micro[seed_id]
            vals_rx = per_seed_means_relaxed[seed_id]
            model_out["mean_macro_f1_per_seed"][seed_id] = (
                sum(vals_ma) / len(vals_ma) if vals_ma else None
            )
            model_out["mean_micro_f1_per_seed"][seed_id] = (
                sum(vals_mi) / len(vals_mi) if vals_mi else None
            )
            model_out["mean_relaxed_f1_per_seed"][seed_id] = (
                sum(vals_rx) / len(vals_rx) if vals_rx else None
            )
        seed_means_macro = [v for v in model_out["mean_macro_f1_per_seed"].values()
                            if v is not None]
        seed_means_micro = [v for v in model_out["mean_micro_f1_per_seed"].values()
                            if v is not None]
        seed_means_relaxed = [v for v in model_out["mean_relaxed_f1_per_seed"].values()
                              if v is not None]
        model_out["summary"]["macro_f1"] = aggregate(seed_means_macro)
        model_out["summary"]["micro_f1"] = aggregate(seed_means_micro)
        model_out["summary"]["relaxed_f1"] = aggregate(seed_means_relaxed)
        out[model] = model_out
    return out


def fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.2f}"


def print_indomain(summary: dict) -> None:
    print("\n=== In-domain (standard benchmark) — 3 seeds (42, 13, 123) ===\n")
    print("Exact Match (mean ± std over 3 seeds):")
    print(f"  {'model':<14} {'P':>6} {'R':>6} {'F1':>6} {'F1 std':>7} {'F1 range':>9}")
    for model, d in summary.items():
        eP = d["exact_precision"]; eR = d["exact_recall"]; ma = d["macro_f1"]
        print(f"  {model:<14} {fmt_pct(eP['mean']):>6} {fmt_pct(eR['mean']):>6} "
              f"{fmt_pct(ma['mean']):>6} {fmt_pct(ma['std']):>7} {fmt_pct(ma['range']):>9}")
    print("\nRelaxed Match (mean ± std over 3 seeds):")
    print(f"  {'model':<14} {'P':>6} {'R':>6} {'F1':>6} {'F1 std':>7} {'F1 range':>9}")
    for model, d in summary.items():
        rP = d["relaxed_precision"]; rR = d["relaxed_recall"]; rF = d["relaxed_f1"]
        print(f"  {model:<14} {fmt_pct(rP['mean']):>6} {fmt_pct(rR['mean']):>6} "
              f"{fmt_pct(rF['mean']):>6} {fmt_pct(rF['std']):>7} {fmt_pct(rF['range']):>9}")
    print("\nPer-seed F1 (Exact / Relaxed) ordered as 42, 13, 123:")
    for model, d in summary.items():
        rows = []
        for s, _ in SEEDS:
            ps = d["per_seed"][s]
            if ps["present"]:
                rows.append(f"{fmt_pct(ps['macro_f1'])}/{fmt_pct(ps.get('relaxed_f1'))}")
            else:
                rows.append("  --/--  ")
        print(f"  {model:<14}  {'  '.join(rows)}")


def print_lomo(summary: dict) -> None:
    print("\n=== LOMO (per-municipality and overall) — Exact Match F1 ===\n")
    header = f"{'model':<14} " + " ".join(f"{m:>6}" for m in MUNICIPALITIES) + \
             f" {'Mean':>7} {'±std':>6}"
    print(header)
    for model, d in summary.items():
        cells = [fmt_pct(d["per_municipality"][mun]["macro_f1"]["mean"]) for mun in MUNICIPALITIES]
        s = d["summary"]["macro_f1"]
        print(f"{model:<14} " + " ".join(f"{c:>6}" for c in cells) +
              f" {fmt_pct(s['mean']):>7} {fmt_pct(s['std']):>6}")
    print("\n=== LOMO (per-municipality and overall) — Relaxed Match F1 ===\n")
    print(header)
    for model, d in summary.items():
        cells = [fmt_pct(d["per_municipality"][mun]["relaxed_f1"]["mean"]) for mun in MUNICIPALITIES]
        s = d["summary"]["relaxed_f1"]
        print(f"{model:<14} " + " ".join(f"{c:>6}" for c in cells) +
              f" {fmt_pct(s['mean']):>7} {fmt_pct(s['std']):>6}")
    print("\n  Per-seed LOMO means (averaged over the 6 municipalities present):")
    for model, d in summary.items():
        per_seed_exact = ", ".join(
            f"s{sid}={fmt_pct(d['mean_macro_f1_per_seed'].get(sid))}"
            for sid, _ in SEEDS
        )
        per_seed_rx = ", ".join(
            f"s{sid}={fmt_pct(d['mean_relaxed_f1_per_seed'].get(sid))}"
            for sid, _ in SEEDS
        )
        print(f"    {model:<14}  Exact  : {per_seed_exact}")
        print(f"    {model:<14}  Relaxed: {per_seed_rx}")


def main() -> None:
    in_domain = summarise_indomain()
    lomo = summarise_lomo()
    summary = {
        "in_domain": in_domain,
        "lomo": lomo,
        "notes": {
            "macro_f1": "Average of per_type F1 over the 11 entity types (paper headline).",
            "micro_f1": "entity_f1 field from the eval JSON (precision/recall aggregated over all entity spans).",
            "seeds": ["42 (deucalion)", "13 (seeds_s13)", "123 (seeds_s123)"],
        },
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print_indomain(in_domain)
    print_lomo(lomo)


if __name__ == "__main__":
    main()
