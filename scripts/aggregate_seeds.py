"""Aggregate macro/entity F1 across seeds for the in-domain standard benchmark
and for the LOMO experiments. Writes a JSON summary and prints a human-readable
table; optionally emits the Table 1 / Table 2 LaTeX so no cell is typed by hand.

File layouts under results/ (both are read; --standard-dir wins per model):
  In-domain, as published:
    discriminative_models/<model>/seeds/seed_s<SEED>_evaluation.json
  In-domain, camera-ready re-run (run_standard_camera_ready.slurm):
    standard_cr/<model>/seed_s<SEED>_evaluation.json
  LOMO, camera-ready re-run (run_lomo_camera_ready.slurm):
    lomo_cr/<model>/<M>_seed_s<SEED>_evaluation.json

THE MACRO DENOMINATOR IS A CHOICE, AND IT MOVES EVERY LOMO CELL.
The eval JSONs come from seqeval's classification_report, which emits a row only
for types present in gold or prediction — so the number of rows varies by fold
(8 to 12 across M01-M06, 11 on the standard test split). Averaging over
"whatever rows exist" therefore silently changes the denominator per cell, which
is what this script used to do while its docstring claimed 11 types. Two defensible
policies are implemented, and both are always computed so the difference is visible:

  support   average over types with support > 0 in that fold (8-12 types).
            Reproduces the published Tables 1 and 2 exactly. Cells within a
            column share a denominator, so per-municipality model rankings are
            sound, but the Mean column averages across different denominators.
  fixed11   average over the 11 schema types excluding Count-Against, scoring
            absent types 0.0. Every cell shares one denominator, so the Mean and
            the Table 1 vs Table 2 comparison are commensurable — at the cost of
            charging a model 0.0 for types its held-out municipality never uses.

On the standard test split the two coincide (exactly 11 types have support), so
they differ only for LOMO. Count-Against is excluded from both: it has 2 gold
instances corpus-wide, both in M06, and none in the standard test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "seed_aggregate_summary.json"

# The 11 scored types. Mirrors src/llm_extraction/schemas.py:ENTITY_TYPES so the
# generative rows of Tables 1 and 2 share this denominator with the encoders.
ENTITY_TYPES = [
    "COUNT-BLANK", "COUNT-FAVOR", "COUNTING-MAJORITY", "COUNTING-UNANIMITY",
    "SUBJECT", "VOTER-ABSENT", "VOTER-ABSTENTION", "VOTER-AGAINST",
    "VOTER-FAVOR", "VOTING", "VOTING-METHOD",
]

MUNICIPALITIES = ["M01", "M02", "M03", "M04", "M05", "M06"]

# Models reported with 3 seeds; everything else is seed 42 only, matching how
# non-CRF rows are reported in Table 1.
MULTI_SEED_MODELS = ["bert_crf", "xlmr_crf", "deberta_crf"]
SINGLE_SEED_MODELS = ["bert_linear", "deberta_linear", "xlmr_linear",
                      "bilstm_fasttext", "crf"]
MODELS = MULTI_SEED_MODELS + SINGLE_SEED_MODELS

# Paper-facing names. `bilstm_fasttext` is the config's model.name (hence the
# results directory); the paper and the older results tree both call it BiLSTM-CRF.
DISPLAY_NAMES = {
    "bert_crf": "BERTimbau-CRF",
    "bert_linear": "BERTimbau-Lin.",
    "deberta_crf": "DeBERTa-CRF",
    "deberta_linear": "DeBERTa-Lin.",
    "xlmr_crf": "XLM-R-CRF",
    "xlmr_linear": "XLM-R-Lin.",
    "bilstm_fasttext": "BiLSTM-CRF",
    "crf": "CRF",
}

# The published in-domain tree predates the config-derived naming: the BiLSTM
# results sit under bilstm_crf, while the config's model.name (and therefore
# every path run_pipeline.py writes) is bilstm_fasttext.
PUBLISHED_DIR_ALIASES = {"bilstm_fasttext": "bilstm_crf"}

SEEDS = ["42", "13", "123"]

EXACT_BLOCK = "entity_level_metrics"
RELAXED_BLOCK = "relaxed_boundary_metrics"


def _rel(path: Path) -> str:
    """Repo-relative when possible; absolute paths (e.g. a scratch dir) pass through."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def seeds_for(model: str) -> list[str]:
    return SEEDS if model in MULTI_SEED_MODELS else ["42"]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def macro_f1(report: dict, block: str, denominator: str) -> tuple[float | None, int | None]:
    """Macro F1 over per-type scores, plus the denominator actually used.

    Returning the count alongside the value is what stops a caption from drifting
    away from the table: the caller can assert every cell used the same one.
    """
    per_type = (report.get(block) or {}).get("per_type_metrics") or {}
    if not per_type:
        return None, None

    if denominator == "fixed11":
        f1s = [per_type.get(t, {}).get("f1_score", 0.0) for t in ENTITY_TYPES]
    elif denominator == "support":
        f1s = [v["f1_score"] for v in per_type.values() if v.get("support", 0) > 0]
    else:
        raise ValueError(f"unknown denominator {denominator!r}")

    if not f1s:
        return None, 0
    return sum(f1s) / len(f1s), len(f1s)


def micro_f1(report: dict) -> float | None:
    return (report.get(EXACT_BLOCK) or {}).get("entity_f1")


def exact_pr(report: dict) -> tuple[float | None, float | None]:
    em = report.get(EXACT_BLOCK) or {}
    return em.get("entity_precision"), em.get("entity_recall")


def relaxed_pr(report: dict) -> tuple[float | None, float | None]:
    rm = report.get(RELAXED_BLOCK) or {}
    return rm.get("precision"), rm.get("recall")


def aggregate(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "range": None, "values": []}
    n = len(values)
    return {
        "n": n,
        "mean": mean(values),
        "std": pstdev(values) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "values": values,
    }


def in_domain_paths(model: str, disc_dir: Path, standard_dir: Path | None) -> dict[str, Path]:
    """Prefer the camera-ready re-run where it exists, else the published tree.

    Only four of the eight Table 1 rows were re-run, so Table 1 is a merge of two
    roots. Each cell records which one it came from (see `source` below) rather
    than leaving that to memory.
    """
    if standard_dir is not None:
        cr = {s: standard_dir / model / f"seed_s{s}_evaluation.json" for s in SEEDS}
        if any(p.exists() for p in cr.values()):
            return cr
    published = PUBLISHED_DIR_ALIASES.get(model, model)
    return {s: disc_dir / published / "seeds" / f"seed_s{s}_evaluation.json" for s in SEEDS}


def lomo_paths(model: str, mun: str, lomo_dir: Path) -> dict[str, Path]:
    return {s: lomo_dir / model / f"{mun}_seed_s{s}_evaluation.json" for s in SEEDS}


def _cell(report: dict, denominator: str) -> dict:
    exact, n_exact = macro_f1(report, EXACT_BLOCK, denominator)
    relaxed, n_relaxed = macro_f1(report, RELAXED_BLOCK, denominator)
    eP, eR = exact_pr(report)
    rP, rR = relaxed_pr(report)
    return {
        "present": True,
        "macro_f1": exact,
        "relaxed_f1": relaxed,
        "n_types_exact": n_exact,
        "n_types_relaxed": n_relaxed,
        "micro_f1": micro_f1(report),
        "exact_precision": eP,
        "exact_recall": eR,
        "relaxed_precision": rP,
        "relaxed_recall": rR,
    }


def summarise_indomain(disc_dir: Path, standard_dir: Path | None, denominator: str) -> dict:
    out = {}
    for model in MODELS:
        paths = in_domain_paths(model, disc_dir, standard_dir)
        per_seed, keys = {}, {
            "macro_f1": [], "relaxed_f1": [], "micro_f1": [],
            "exact_precision": [], "exact_recall": [],
            "relaxed_precision": [], "relaxed_recall": [],
        }
        denominators = set()
        for seed in seeds_for(model):
            path = paths[seed]
            report = load(path)
            if report is None:
                per_seed[seed] = {"path": str(path), "present": False}
                continue
            cell = _cell(report, denominator)
            cell["path"] = str(path)
            cell["source"] = "camera_ready" if standard_dir and standard_dir in path.parents \
                else "published"
            per_seed[seed] = cell
            for key in keys:
                if cell.get(key) is not None:
                    keys[key].append(cell[key])
            if cell["n_types_exact"]:
                denominators.add(cell["n_types_exact"])
        out[model] = {
            "display_name": DISPLAY_NAMES.get(model, model),
            "per_seed": per_seed,
            "seed_count": sum(1 for v in per_seed.values() if v.get("present")),
            "denominators_used": sorted(denominators),
            **{key: aggregate(vals) for key, vals in keys.items()},
        }
    return out


def summarise_lomo(lomo_dir: Path, denominator: str) -> dict:
    out = {}
    for model in MODELS:
        per_mun, exact_means, relaxed_means = {}, [], []
        denominators = set()
        for mun in MUNICIPALITIES:
            paths = lomo_paths(model, mun, lomo_dir)
            per_seed, exacts, relaxeds = {}, [], []
            for seed in seeds_for(model):
                report = load(paths[seed])
                if report is None:
                    per_seed[seed] = {"path": str(paths[seed]), "present": False}
                    continue
                cell = _cell(report, denominator)
                cell["path"] = str(paths[seed])
                per_seed[seed] = cell
                if cell["macro_f1"] is not None:
                    exacts.append(cell["macro_f1"])
                if cell["relaxed_f1"] is not None:
                    relaxeds.append(cell["relaxed_f1"])
                if cell["n_types_exact"]:
                    denominators.add(cell["n_types_exact"])
            per_mun[mun] = {
                "per_seed": per_seed,
                "seed_count": sum(1 for v in per_seed.values() if v.get("present")),
                "macro_f1": aggregate(exacts),
                "relaxed_f1": aggregate(relaxeds),
            }
            if exacts:
                exact_means.append(mean(exacts))
            if relaxeds:
                relaxed_means.append(mean(relaxeds))
        out[model] = {
            "display_name": DISPLAY_NAMES.get(model, model),
            "per_municipality": per_mun,
            "denominators_used": sorted(denominators),
            "summary": {
                "macro_f1": aggregate(exact_means),
                "relaxed_f1": aggregate(relaxed_means),
            },
        }
    return out


def fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def print_indomain(summary: dict, denominator: str) -> None:
    print(f"\n=== In-domain (standard benchmark) — denominator: {denominator} ===\n")
    print(f"  {'model':<16} {'seeds':>5} {'den':>4} "
          f"{'eP':>6} {'eR':>6} {'eF1':>6} {'±':>5}  {'rP':>6} {'rR':>6} {'rF1':>6} {'±':>5}  src")
    for model, d in summary.items():
        if d["seed_count"] == 0:
            print(f"  {d['display_name']:<16} {'--':>5}  (no results found)")
            continue
        den = ",".join(str(n) for n in d["denominators_used"]) or "?"
        sources = {v.get("source") for v in d["per_seed"].values() if v.get("present")}
        print(f"  {d['display_name']:<16} {d['seed_count']:>5} {den:>4} "
              f"{fmt_pct(d['exact_precision']['mean']):>6} {fmt_pct(d['exact_recall']['mean']):>6} "
              f"{fmt_pct(d['macro_f1']['mean']):>6} {fmt_pct(d['macro_f1']['std']):>5}  "
              f"{fmt_pct(d['relaxed_precision']['mean']):>6} {fmt_pct(d['relaxed_recall']['mean']):>6} "
              f"{fmt_pct(d['relaxed_f1']['mean']):>6} {fmt_pct(d['relaxed_f1']['std']):>5}  "
              f"{'/'.join(sorted(s for s in sources if s))}")


def print_lomo(summary: dict, denominator: str) -> None:
    for label, key in [("Exact Match", "macro_f1"), ("Relaxed Match", "relaxed_f1")]:
        print(f"\n=== LOMO — {label} F1 — denominator: {denominator} ===\n")
        print(f"{'model':<16} " + " ".join(f"{m:>6}" for m in MUNICIPALITIES) +
              f" {'Mean':>7} {'±std':>6}  seeds/cell")
        for model, d in summary.items():
            cells = [fmt_pct(d["per_municipality"][m][key]["mean"]) for m in MUNICIPALITIES]
            counts = "".join(str(d["per_municipality"][m]["seed_count"]) for m in MUNICIPALITIES)
            s = d["summary"][key]
            print(f"{d['display_name']:<16} " + " ".join(f"{c:>6}" for c in cells) +
                  f" {fmt_pct(s['mean']):>7} {fmt_pct(s['std']):>6}  {counts}")

    dens = sorted({n for d in summary.values() for n in d["denominators_used"]})
    if len(dens) > 1:
        print(f"\n  NOTE: cells used differing denominators {dens}. Under 'support' this is"
              f"\n  expected — M01/M04 have 8 scored types, M06 has 12 — but it means the Mean"
              f"\n  column averages across denominators. Use --denominator fixed11 to remove it.")
    elif dens:
        print(f"\n  Every cell used a denominator of {dens[0]} types.")


def emit_latex(lomo: dict, denominator: str, path: Path) -> None:
    """Write the Table 2 body so no cell is transcribed by hand."""
    lines = [
        f"% Generated by scripts/aggregate_seeds.py --denominator {denominator}",
        "% Do not edit by hand; re-run the script instead.",
    ]
    for label, key in [("Exact Match", "macro_f1"), ("Relaxed Match", "relaxed_f1")]:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{8}}{{c}}{{\textbf{{{label}}}}} \\")
        lines.append(r"\midrule")
        for model, d in lomo.items():
            if all(d["per_municipality"][m]["seed_count"] == 0 for m in MUNICIPALITIES):
                continue
            cells = [d["per_municipality"][m][key]["mean"] for m in MUNICIPALITIES]
            rendered = " & ".join("--" if c is None else f"{100 * c:.1f}" for c in cells)
            overall = d["summary"][key]["mean"]
            overall_s = "--" if overall is None else f"{100 * overall:.1f}"
            lines.append(f"{d['display_name']:<16} & {rendered} & {overall_s} \\\\")
    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {_rel(path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--denominator", choices=["fixed11", "support"], default="fixed11",
                    help="Macro-average denominator policy (default: fixed11). Both are "
                         "always written to the JSON; this selects what is printed.")
    ap.add_argument("--lomo-dir", type=Path, default=RESULTS / "lomo_cr",
                    help="LOMO results root (default: results/lomo_cr)")
    ap.add_argument("--disc-dir", type=Path, default=RESULTS / "discriminative_models",
                    help="Published in-domain results root")
    ap.add_argument("--standard-dir", type=Path, default=RESULTS / "standard_cr",
                    help="Camera-ready in-domain re-run root; takes precedence per model")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--latex", type=Path, default=None,
                    help="Also write the Table 2 body to this path")
    args = ap.parse_args()

    if not args.lomo_dir.exists():
        raise SystemExit(
            f"LOMO results root not found: {args.lomo_dir}\n"
            f"Run run_lomo_camera_ready.slurm first, or pass --lomo-dir explicitly. "
            f"results/lomo/ holds the superseded batch-16 runs and is never read by default."
        )
    standard_dir = args.standard_dir if args.standard_dir.exists() else None

    # Compute under both policies so the JSON records the difference, and the
    # choice made for the paper can be checked against the alternative later.
    summary = {"denominator_reported": args.denominator, "by_denominator": {}}
    for denominator in ("fixed11", "support"):
        summary["by_denominator"][denominator] = {
            "in_domain": summarise_indomain(args.disc_dir, standard_dir, denominator),
            "lomo": summarise_lomo(args.lomo_dir, denominator),
        }
    summary["notes"] = {
        "entity_types": ENTITY_TYPES,
        "excluded": "COUNT-AGAINST (2 gold instances corpus-wide, both in M06; "
                    "0 in the standard test split)",
        "denominators": {
            "fixed11": "average over the 11 types above, absent types scored 0.0",
            "support": "average over types with support > 0 in that fold (8-12 for LOMO)",
        },
        "sources": {"lomo": str(args.lomo_dir), "in_domain_published": str(args.disc_dir),
                    "in_domain_camera_ready": str(standard_dir) if standard_dir else None},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {_rel(args.out)}")

    chosen = summary["by_denominator"][args.denominator]
    print_indomain(chosen["in_domain"], args.denominator)
    print_lomo(chosen["lomo"], args.denominator)

    other = "support" if args.denominator == "fixed11" else "fixed11"
    print(f"\n  (The same run under --denominator {other} is in {args.out.name} "
          f"under by_denominator.{other}.)")

    if args.latex:
        emit_latex(chosen["lomo"], args.denominator, args.latex)


if __name__ == "__main__":
    main()
