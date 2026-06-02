#!/usr/bin/env python3
"""
Create Leave-One-Municipality-Out (LOMO) splits at v6 dataset level.

For each of the 6 municipalities, generate a LOMO split where that municipality
is the held-out test set, with the same v6 transforms applied as in
generate_v6_multi_vote_aware.py:

    1. SUB-SUBJECT boundary normalization (strip leading article)
    2. SUB-SUBJECT label -> SUBJECT (train/dev/test main files)
    3. Voting-only filter: spans=[] when source has no VOTING anchor
    4. Per-segment metadata: is_multi_vote, subject_span_lengths
    5. Train-only minority-class oversampling x N
    6. Train-only multi-vote oversampling x N

Input/output:
    data/citilink_spans_v2/{train,dev,test}.jsonl  (merged across splits)
        ↓ [THIS SCRIPT]
    data/citilink-votie-lomo-splits/M0X/{train,dev,test}.jsonl
    data/citilink-votie-lomo-splits/M0X/statistics.json
    data/citilink-votie-lomo-splits/M0X/dev_gold_with_sub_subjects.jsonl
    data/citilink-votie-lomo-splits/M0X/test_gold_with_sub_subjects.jsonl
    data/citilink-votie-lomo-splits/overall_summary.json

Within each LOMO split:
    test:  ALL examples from the target municipality (v6 transforms applied)
    train: 85% of the other 5 municipalities, stratified (v6 transforms + oversample)
    dev:   15% of the other 5 municipalities, stratified (v6 transforms, no oversample)

The shared v6 transform helpers are imported from generate_v6_multi_vote_aware.

Usage:
    python scripts/dataset_generation/create_lomo_splits_v6.py
    python scripts/dataset_generation/create_lomo_splits_v6.py \
        --input-dir data/citilink_spans_v2 \
        --output-dir data/citilink-votie-lomo-splits \
        --oversample-factor 4 --multi-vote-oversample-factor 4 --seed 42
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# Reuse the v6 transform logic exactly — no duplication.
sys.path.insert(0, str(Path(__file__).parent))
from generate_v6_multi_vote_aware import (  # noqa: E402
    MINORITY_LABELS,
    SUBJECT_LENGTH_BINS,
    bin_subject_length,
    transform_example,
    validate_example,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MUNICIPALITIES = {
    "M01": "alandroal",
    "M02": "campomaior",
    "M03": "covilha",
    "M04": "fundao",
    "M05": "guimaraes",
    "M06": "porto",
}

DEV_RATIO = 0.15


def load_all_examples(input_dir: Path) -> List[Dict[str, Any]]:
    """Load and merge train+dev+test from v2-format input directory."""
    all_examples: List[Dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        split_file = input_dir / f"{split}.jsonl"
        if not split_file.exists():
            logger.warning(f"Missing input split: {split_file}")
            continue
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_examples.append(json.loads(line))
    return all_examples


def stratify_key(example: Dict[str, Any]) -> str:
    """Stratify key combining municipality and entity pattern (mirrors v5/v4 LOMO)."""
    municipality = example.get("municipality", "?")
    labels = {s["label"] for s in example.get("spans", [])}
    if labels & {"VOTER-FAVOR", "COUNTING-UNANIMITY"}:
        pattern = "positive_vote"
    elif labels & {"VOTER-AGAINST", "VOTER-ABSENT"}:
        pattern = "negative_vote"
    elif "VOTER-ABSTENTION" in labels:
        pattern = "abstention"
    elif "COUNTING-MAJORITY" in labels:
        pattern = "majority_decision"
    elif "SUBJECT" in labels:
        pattern = "has_subject"
    else:
        pattern = "no_entities"
    return f"{municipality}_{pattern}"


def adaptive_stratify(examples: List[Dict[str, Any]], keys: List[str]) -> List[str]:
    """Merge rare strata (<2 examples) into broader categories so sklearn accepts them."""
    counts = Counter(keys)
    simplified = []
    for i, k in enumerate(keys):
        if counts[k] >= 2:
            simplified.append(k)
        else:
            simplified.append(f"{examples[i].get('municipality', '?')}_consolidated")
    simplified_counts = Counter(simplified)
    if any(c < 2 for c in simplified_counts.values()):
        return [k if simplified_counts[k] >= 2 else "consolidated_mixed" for k in simplified]
    return simplified


def split_pool(
    pool: List[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified train/dev split of the non-target-municipality pool."""
    if not pool:
        return [], []
    keys = adaptive_stratify(pool, [stratify_key(ex) for ex in pool])
    indices = np.arange(len(pool))
    try:
        train_idx, dev_idx = train_test_split(
            indices,
            stratify=[keys[i] for i in indices],
            test_size=DEV_RATIO,
            random_state=seed,
        )
    except ValueError as e:
        logger.warning(f"  Stratified split failed ({e}); falling back to random split.")
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_dev = int(len(shuffled) * DEV_RATIO)
        dev_idx = shuffled[:n_dev]
        train_idx = shuffled[n_dev:]
    return [pool[i] for i in train_idx], [pool[i] for i in dev_idx]


def apply_v6_transforms(
    raw_examples: List[Dict[str, Any]],
    *,
    merge_sub_subject: bool,
    oversample_factor: int,
    multi_vote_oversample_factor: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Run v6 transforms + oversample on a list of raw examples, return (records, stats, errors).

    Oversample_factor / multi_vote_oversample_factor of 0 disables that pile
    (used for dev/test). Final shuffle is deterministic for the given seed.
    """
    stats = {
        "segments_in": len(raw_examples),
        "voting_segments": 0,
        "non_voting_segments_kept": 0,
        "sub_subjects": 0,
        "boundaries_fixed": 0,
        "relabeled": 0,
        "minority_segments_orig": 0,
        "oversample_copies": 0,
        "multi_vote_segments_orig": 0,
        "multi_vote_oversample_copies": 0,
        "entity_counts_after": Counter(),
        "subject_length_histogram": Counter(),
        "per_municipality": {},
    }
    per_mun: Dict[str, Dict[str, int]] = {}
    base: List[Dict[str, Any]] = []
    minority: List[Dict[str, Any]] = []
    multi_vote: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    for ex in raw_examples:
        transformed, ex_stats = transform_example(
            ex, merge_sub_subject=merge_sub_subject, enforce_voting_only=True,
        )
        stats["sub_subjects"] += ex_stats["sub_subjects"]
        stats["boundaries_fixed"] += ex_stats["boundaries_fixed"]
        stats["relabeled"] += ex_stats["relabeled"]
        if ex_stats["is_voting"]:
            stats["voting_segments"] += 1
        else:
            stats["non_voting_segments_kept"] += 1

        mun = transformed.get("municipality", "?")
        mun_stats = per_mun.setdefault(mun, {"segments": 0, "voting": 0, "non_voting": 0,
                                             "entities": 0, "multi_vote": 0})
        mun_stats["segments"] += 1
        mun_stats["voting" if ex_stats["is_voting"] else "non_voting"] += 1
        for s in transformed["spans"]:
            stats["entity_counts_after"][s["label"]] += 1
            mun_stats["entities"] += 1
        if transformed["is_multi_vote"]:
            mun_stats["multi_vote"] += 1

        for length in transformed["subject_span_lengths"]:
            stats["subject_length_histogram"][bin_subject_length(length)] += 1

        all_errors.extend(validate_example(transformed))

        base.append(transformed)
        if any(s["label"] in MINORITY_LABELS for s in transformed["spans"]):
            minority.append(transformed)
        if transformed["is_multi_vote"]:
            multi_vote.append(transformed)

    stats["minority_segments_orig"] = len(minority)
    stats["multi_vote_segments_orig"] = len(multi_vote)
    stats["per_municipality"] = per_mun

    output = list(base)
    did_oversample = False
    if oversample_factor > 0 and minority:
        copies = minority * oversample_factor
        stats["oversample_copies"] = len(copies)
        output.extend(copies)
        did_oversample = True
    if multi_vote_oversample_factor > 0 and multi_vote:
        copies = multi_vote * multi_vote_oversample_factor
        stats["multi_vote_oversample_copies"] = len(copies)
        output.extend(copies)
        did_oversample = True
    if did_oversample:
        rng = random.Random(seed)
        rng.shuffle(output)

    stats["segments_out"] = len(output)
    stats["spans_total_after"] = sum(len(e["spans"]) for e in output)

    return output, stats, all_errors


def serialize_split_stats(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "segments_in": s["segments_in"],
        "segments_out": s.get("segments_out", s["segments_in"]),
        "voting_segments": s["voting_segments"],
        "non_voting_segments_kept": s["non_voting_segments_kept"],
        "spans_total_after_filter": s.get("spans_total_after", 0),
        "sub_subject_normalizations": s["boundaries_fixed"],
        "sub_subject_relabels": s["relabeled"],
        "minority_segments_orig": s["minority_segments_orig"],
        "oversample_copies": s["oversample_copies"],
        "multi_vote_segments_orig": s["multi_vote_segments_orig"],
        "multi_vote_oversample_copies": s["multi_vote_oversample_copies"],
        "multi_vote_fraction": (
            s["multi_vote_segments_orig"] / s["voting_segments"]
            if s["voting_segments"] else 0
        ),
        "entity_counts_after": dict(s["entity_counts_after"]),
        "per_municipality": {k: dict(v) for k, v in s["per_municipality"].items()},
        "subject_length_histogram": dict(s["subject_length_histogram"]),
    }


def write_jsonl(records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in records:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def process_municipality(
    municipality_id: str,
    all_examples: List[Dict[str, Any]],
    output_dir: Path,
    *,
    oversample_factor: int,
    multi_vote_oversample_factor: int,
    seed: int,
) -> Dict[str, Any]:
    """Build one LOMO split (test = municipality_id, train+dev = others)."""
    lomo_dir = output_dir / municipality_id
    test_raw = [ex for ex in all_examples if ex.get("municipality") == municipality_id]
    pool_raw = [ex for ex in all_examples if ex.get("municipality") != municipality_id]
    train_raw, dev_raw = split_pool(pool_raw, seed=seed)

    logger.info(f"  {municipality_id}: pool={len(pool_raw)} -> train={len(train_raw)} dev={len(dev_raw)} | test={len(test_raw)}")

    # Validate no leakage (by example id, mirroring create_lomo_splits.py).
    train_ids = {ex["id"] for ex in train_raw}
    dev_ids = {ex["id"] for ex in dev_raw}
    test_ids = {ex["id"] for ex in test_raw}
    assert not (train_ids & dev_ids), f"{municipality_id}: train/dev id overlap"
    assert not (train_ids & test_ids), f"{municipality_id}: train/test id overlap"
    assert not (dev_ids & test_ids), f"{municipality_id}: dev/test id overlap"
    assert {ex["municipality"] for ex in test_raw} == {municipality_id}, "test set contamination"

    train_out, train_stats, train_errs = apply_v6_transforms(
        train_raw,
        merge_sub_subject=True,
        oversample_factor=oversample_factor,
        multi_vote_oversample_factor=multi_vote_oversample_factor,
        seed=seed,
    )
    dev_out, dev_stats, dev_errs = apply_v6_transforms(
        dev_raw,
        merge_sub_subject=True,
        oversample_factor=0,
        multi_vote_oversample_factor=0,
        seed=seed,
    )
    test_out, test_stats, test_errs = apply_v6_transforms(
        test_raw,
        merge_sub_subject=True,
        oversample_factor=0,
        multi_vote_oversample_factor=0,
        seed=seed,
    )

    write_jsonl(train_out, lomo_dir / "train.jsonl")
    write_jsonl(dev_out, lomo_dir / "dev.jsonl")
    write_jsonl(test_out, lomo_dir / "test.jsonl")

    # Gold-with-sub-subjects references for dev/test (no merge applied).
    dev_gold, _, _ = apply_v6_transforms(
        dev_raw, merge_sub_subject=False, oversample_factor=0,
        multi_vote_oversample_factor=0, seed=seed,
    )
    test_gold, _, _ = apply_v6_transforms(
        test_raw, merge_sub_subject=False, oversample_factor=0,
        multi_vote_oversample_factor=0, seed=seed,
    )
    write_jsonl(dev_gold, lomo_dir / "dev_gold_with_sub_subjects.jsonl")
    write_jsonl(test_gold, lomo_dir / "test_gold_with_sub_subjects.jsonl")

    errors = train_errs + dev_errs + test_errs
    if errors:
        logger.error(f"  {municipality_id}: {len(errors)} validation errors")
        for e in errors[:10]:
            logger.error(f"    {e}")

    stats = {
        "target_municipality": municipality_id,
        "municipality_name": MUNICIPALITIES.get(municipality_id, "?"),
        "train": serialize_split_stats(train_stats),
        "dev": serialize_split_stats(dev_stats),
        "test": serialize_split_stats(test_stats),
        "validation_errors": len(errors),
    }
    with open(lomo_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return {"stats": stats, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", type=Path, default=Path("data/citilink_spans_v2"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/citilink-votie-lomo-splits"),
    )
    parser.add_argument("--oversample-factor", type=int, default=4)
    parser.add_argument("--multi-vote-oversample-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("v6 LOMO splits generation")
    logger.info("=" * 80)
    logger.info(f"Input:                       {args.input_dir}")
    logger.info(f"Output:                      {args.output_dir}")
    logger.info(f"Minority oversample factor:  {args.oversample_factor}")
    logger.info(f"Multi-vote oversample factor:{args.multi_vote_oversample_factor}")
    logger.info(f"Seed:                        {args.seed}")
    logger.info("")

    all_examples = load_all_examples(args.input_dir)
    if not all_examples:
        logger.error(f"No examples loaded from {args.input_dir}")
        sys.exit(1)

    mun_counts = Counter(ex.get("municipality", "?") for ex in all_examples)
    logger.info(f"Loaded {len(all_examples)} examples across {len(mun_counts)} municipalities")
    for mid in sorted(MUNICIPALITIES):
        logger.info(f"  {mid} ({MUNICIPALITIES[mid]:12}): {mun_counts.get(mid, 0)} examples")
    logger.info("")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overall: Dict[str, Any] = {
        "description": "Per-municipality LOMO splits at v6 dataset level.",
        "source": str(args.input_dir),
        "oversampling": {
            "minority_factor": args.oversample_factor,
            "multi_vote_factor": args.multi_vote_oversample_factor,
            "seed": args.seed,
        },
        "subject_length_bins": [
            f"{lo}-{'+' if hi == float('inf') else int(hi)}"
            for lo, hi in SUBJECT_LENGTH_BINS
        ],
        "generated_at": datetime.now().isoformat(),
        "script": Path(__file__).name,
        "splits": {},
    }
    total_errors = 0
    for mid in sorted(MUNICIPALITIES):
        if mun_counts.get(mid, 0) == 0:
            logger.warning(f"Skipping {mid}: no examples in source")
            continue
        result = process_municipality(
            mid, all_examples, args.output_dir,
            oversample_factor=args.oversample_factor,
            multi_vote_oversample_factor=args.multi_vote_oversample_factor,
            seed=args.seed,
        )
        overall["splits"][mid] = result["stats"]
        total_errors += len(result["errors"])

    with open(args.output_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {args.output_dir / 'overall_summary.json'}")

    logger.info("")
    if total_errors:
        logger.error(f"FAILED: {total_errors} validation errors across all splits")
        sys.exit(1)
    logger.info("Validation: all splits clean (no errors).")


if __name__ == "__main__":
    main()
