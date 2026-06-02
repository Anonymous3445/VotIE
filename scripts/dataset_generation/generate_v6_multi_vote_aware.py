#!/usr/bin/env python3
"""
Generate citilink-votie: training-ready span JSONL with structural metadata and multi-vote oversampling.

Applies, in order:
    1. SUB-SUBJECT boundary normalisation (strip leading article).
    2. SUB-SUBJECT → SUBJECT relabel (main files; originals kept in *_gold_with_sub_subjects.jsonl).
    3. Voting-only filter — segments without a VOTING span keep empty spans[].
    4. Minority-class oversampling ×4 (train only): COUNT-BLANK, COUNT-FAVOR, VOTING-METHOD.
    5. Multi-vote oversampling ×4 (train only): segments with ≥2 distinct event_ids among VOTING spans.

Added per-segment fields:
    - `is_multi_vote` (bool) — computed from VOTING spans' event_ids.
    - `subject_span_lengths` (list[int]) — whitespace-token counts per SUBJECT span.

Outputs:
    {output_dir}/train.jsonl                       — training set + oversamples (2,463 segments)
    {output_dir}/dev.jsonl                         — dev set (553 segments)
    {output_dir}/test.jsonl                        — test set (529 segments)
    {output_dir}/dev_gold_with_sub_subjects.jsonl  — dev with original SUB-SUBJECT labels
    {output_dir}/test_gold_with_sub_subjects.jsonl — test with original SUB-SUBJECT labels
    {output_dir}/generation_summary.json           — per-split statistics and per-municipality breakdown

Usage:
    python scripts/dataset_generation/generate_v6_multi_vote_aware.py
    python scripts/dataset_generation/generate_v6_multi_vote_aware.py \
        --output-dir data/citilink-votie --oversample-factor 4 --multi-vote-oversample-factor 4 --seed 42
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LEADING_ARTICLE_RE = re.compile(r'^(o|os|a|as)\s+', re.IGNORECASE)
MINORITY_LABELS = {'COUNT-BLANK', 'COUNT-FAVOR', 'COUNT-AGAINST', 'VOTING-METHOD'}
ALL_ENTITY_LABELS = {
    'SUBJECT', 'SUB-SUBJECT', 'VOTING',
    'VOTER-FAVOR', 'VOTER-AGAINST', 'VOTER-ABSTENTION', 'VOTER-ABSENT',
    'COUNTING-UNANIMITY', 'COUNTING-MAJORITY',
    'COUNT-FAVOR', 'COUNT-AGAINST', 'COUNT-ABSTENTION', 'COUNT-BLANK',
    'VOTING-METHOD',
}

SUBJECT_LENGTH_BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, float('inf'))]


def normalize_boundary(span: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Strip leading article from a SUB-SUBJECT span (inherited from v4/v5)."""
    if span['label'] != 'SUB-SUBJECT':
        return span
    match = LEADING_ARTICLE_RE.match(span['text'])
    if not match:
        return span
    prefix = match.group(0)
    new_text = span['text'][len(prefix):]
    if not new_text.strip():
        return span
    new_start = span['start'] + len(prefix)
    if text[new_start:span['end']] != new_text:
        logger.warning(f"Offset mismatch stripping article from '{span['text'][:60]}', keeping original.")
        return span
    return {**span, 'start': new_start, 'end': span['end'], 'text': new_text}


def has_voting_event(spans: List[Dict[str, Any]]) -> bool:
    return any(s['label'] == 'VOTING' for s in spans)


def compute_is_multi_vote(spans: List[Dict[str, Any]]) -> bool:
    """Multi-vote iff >=2 distinct event_ids appear among VOTING spans."""
    voting_event_ids = {s.get('event_id') for s in spans if s['label'] == 'VOTING'}
    voting_event_ids.discard(None)
    return len(voting_event_ids) >= 2


def compute_subject_span_lengths(spans: List[Dict[str, Any]]) -> List[int]:
    """Whitespace-token count for each SUBJECT span (post-merge of SUB-SUBJECT)."""
    return [len(s['text'].split()) for s in spans if s['label'] == 'SUBJECT']


def bin_subject_length(length: int) -> str:
    for lo, hi in SUBJECT_LENGTH_BINS:
        if lo <= length < hi:
            hi_label = '+' if hi == float('inf') else str(int(hi))
            return f"{lo}-{hi_label}" if hi != float('inf') else f"{lo}+"
    return f"{SUBJECT_LENGTH_BINS[-1][0]}+"


def transform_example(
    example: Dict[str, Any],
    merge_sub_subject: bool,
    enforce_voting_only: bool,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Apply v5 transforms, then attach v6 structural metadata."""
    text = example['text']
    stats = {'sub_subjects': 0, 'boundaries_fixed': 0, 'relabeled': 0,
             'is_voting': 0, 'spans_kept': 0, 'spans_dropped': 0}

    # Stage 1: SUB-SUBJECT boundary normalization (verbatim from v5)
    transformed_spans: List[Dict[str, Any]] = []
    for span in example.get('spans', []):
        s = dict(span)
        if s['label'] == 'SUB-SUBJECT':
            stats['sub_subjects'] += 1
            normalized = normalize_boundary(s, text)
            if normalized['start'] != s['start']:
                stats['boundaries_fixed'] += 1
            s = normalized
            if merge_sub_subject:
                s['label'] = 'SUBJECT'
                stats['relabeled'] += 1
        transformed_spans.append(s)

    # Stage 2: voting-only filter (verbatim from v5)
    if enforce_voting_only and not has_voting_event(transformed_spans):
        stats['spans_dropped'] = len(transformed_spans)
        transformed_spans = []
    else:
        stats['is_voting'] = 1
        stats['spans_kept'] = len(transformed_spans)

    out = dict(example)
    out['spans'] = transformed_spans

    # Stage 3 (v6-specific): structural metadata. Computed on the *post-filter*
    # span list so flags describe what is actually in the emitted record.
    out['is_multi_vote'] = compute_is_multi_vote(transformed_spans)
    out['subject_span_lengths'] = compute_subject_span_lengths(transformed_spans)

    return out, stats


def validate_example(example: Dict[str, Any]) -> List[str]:
    """Per-segment integrity checks (verbatim from v5, plus v6 flag checks)."""
    errors = []
    text = example.get('text', '')
    seen: set = set()
    for s in example.get('spans', []):
        if s['label'] not in ALL_ENTITY_LABELS:
            errors.append(f"{example['id']}: unknown label {s['label']!r}")
            continue
        if s['start'] < 0 or s['end'] > len(text) or s['start'] >= s['end']:
            errors.append(f"{example['id']}: out-of-bounds {s['label']} [{s['start']}:{s['end']}] in len={len(text)}")
            continue
        if text[s['start']:s['end']] != s['text']:
            errors.append(
                f"{example['id']}: text mismatch at [{s['start']}:{s['end']}]: "
                f"{text[s['start']:s['end']]!r} != {s['text']!r}"
            )
        key = (s['start'], s['end'], s['label'], s.get('event_id'))
        if key in seen:
            errors.append(f"{example['id']}: duplicate (start,end,label,event_id) {key}")
        seen.add(key)

    if 'is_multi_vote' not in example or not isinstance(example['is_multi_vote'], bool):
        errors.append(f"{example['id']}: missing or non-bool is_multi_vote")
    subject_count = sum(1 for s in example.get('spans', []) if s['label'] == 'SUBJECT')
    if len(example.get('subject_span_lengths', [])) != subject_count:
        errors.append(
            f"{example['id']}: subject_span_lengths length "
            f"{len(example.get('subject_span_lengths', []))} != SUBJECT span count {subject_count}"
        )
    return errors


def process_split(
    input_path: Path,
    output_path: Path,
    *,
    merge_sub_subject: bool,
    enforce_voting_only: bool,
    oversample_factor: int,
    multi_vote_oversample_factor: int,
    seed: int,
) -> Dict[str, Any]:
    """Transform one split, oversample minority + multi-vote, validate.

    Order: append minority copies (matches v5 ordering for reproducibility),
    then append multi-vote copies, then a single deterministic shuffle. A
    segment that is both minority and multi-vote is oversampled by both
    factors — matches the "harder cases get more weight" intuition.
    """
    stats = {
        'segments_in': 0,
        'segments_out': 0,
        'voting_segments': 0,
        'non_voting_segments_kept': 0,
        'spans_total_before': 0,
        'spans_total_after': 0,
        'sub_subjects': 0,
        'boundaries_fixed': 0,
        'relabeled': 0,
        'minority_segments_orig': 0,
        'oversample_copies': 0,
        'multi_vote_segments_orig': 0,
        'multi_vote_oversample_copies': 0,
        'entity_counts_after': Counter(),
        'per_municipality': defaultdict(lambda: {
            'segments': 0, 'voting': 0, 'non_voting': 0,
            'entities': 0, 'multi_vote': 0,
        }),
        'subject_length_histogram': Counter(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_examples: List[Dict[str, Any]] = []
    minority_examples: List[Dict[str, Any]] = []
    multi_vote_examples: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    with open(input_path, 'r', encoding='utf-8') as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {input_path} at line {line_no}: {e.msg}") from e

            stats['segments_in'] += 1
            stats['spans_total_before'] += len(example.get('spans', []))

            transformed, ex_stats = transform_example(
                example,
                merge_sub_subject=merge_sub_subject,
                enforce_voting_only=enforce_voting_only,
            )
            stats['sub_subjects'] += ex_stats['sub_subjects']
            stats['boundaries_fixed'] += ex_stats['boundaries_fixed']
            stats['relabeled'] += ex_stats['relabeled']

            if ex_stats['is_voting']:
                stats['voting_segments'] += 1
            else:
                stats['non_voting_segments_kept'] += 1

            mun = transformed.get('municipality', '?')
            stats['per_municipality'][mun]['segments'] += 1
            stats['per_municipality'][mun]['voting' if ex_stats['is_voting'] else 'non_voting'] += 1
            for s in transformed['spans']:
                stats['entity_counts_after'][s['label']] += 1
                stats['per_municipality'][mun]['entities'] += 1

            if transformed['is_multi_vote']:
                stats['per_municipality'][mun]['multi_vote'] += 1

            for length in transformed['subject_span_lengths']:
                stats['subject_length_histogram'][bin_subject_length(length)] += 1

            errors = validate_example(transformed)
            all_errors.extend(errors)

            base_examples.append(transformed)

            if any(s['label'] in MINORITY_LABELS for s in transformed['spans']):
                minority_examples.append(transformed)
            if transformed['is_multi_vote']:
                multi_vote_examples.append(transformed)

    stats['minority_segments_orig'] = len(minority_examples)
    stats['multi_vote_segments_orig'] = len(multi_vote_examples)

    output_examples = list(base_examples)
    oversampled = False
    if oversample_factor > 0 and minority_examples:
        copies = minority_examples * oversample_factor
        stats['oversample_copies'] = len(copies)
        output_examples.extend(copies)
        oversampled = True
    if multi_vote_oversample_factor > 0 and multi_vote_examples:
        copies = multi_vote_examples * multi_vote_oversample_factor
        stats['multi_vote_oversample_copies'] = len(copies)
        output_examples.extend(copies)
        oversampled = True
    if oversampled:
        rng = random.Random(seed)
        rng.shuffle(output_examples)

    stats['segments_out'] = len(output_examples)
    stats['spans_total_after'] = sum(len(e['spans']) for e in output_examples)

    with open(output_path, 'w', encoding='utf-8') as fout:
        for example in output_examples:
            fout.write(json.dumps(example, ensure_ascii=False) + '\n')

    return {'stats': stats, 'errors': all_errors}


def write_gold_reference(input_path: Path, output_path: Path, *, enforce_voting_only: bool):
    """Reference file with original SUB-SUBJECT labels preserved.

    Applies boundary normalization but NOT the SUB-SUBJECT->SUBJECT merge.
    v6 metadata (is_multi_vote, subject_span_lengths) is still attached so
    downstream evaluators can stratify gold-reference predictions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            transformed, _ = transform_example(
                example,
                merge_sub_subject=False,
                enforce_voting_only=enforce_voting_only,
            )
            fout.write(json.dumps(transformed, ensure_ascii=False) + '\n')


def write_summary(output_dir: Path, results: Dict[str, Dict[str, Any]], args):
    output_dir.mkdir(parents=True, exist_ok=True)

    def serialize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'segments_in': stats['segments_in'],
            'segments_out': stats['segments_out'],
            'voting_segments': stats['voting_segments'],
            'non_voting_segments_kept': stats['non_voting_segments_kept'],
            'voting_fraction': (
                stats['voting_segments'] / stats['segments_in']
                if stats['segments_in'] else 0
            ),
            'spans_total_before_filter': stats['spans_total_before'],
            'spans_total_after_filter': stats['spans_total_after'],
            'sub_subject_normalizations': stats['boundaries_fixed'],
            'sub_subject_relabels': stats['relabeled'],
            'minority_segments_orig': stats['minority_segments_orig'],
            'oversample_copies': stats['oversample_copies'],
            'multi_vote_segments_orig': stats['multi_vote_segments_orig'],
            'multi_vote_oversample_copies': stats['multi_vote_oversample_copies'],
            'multi_vote_fraction': (
                stats['multi_vote_segments_orig'] / stats['voting_segments']
                if stats['voting_segments'] else 0
            ),
            'entity_counts_after': dict(stats['entity_counts_after']),
            'per_municipality': {k: dict(v) for k, v in stats['per_municipality'].items()},
            'subject_length_histogram': dict(stats['subject_length_histogram']),
        }

    summary = {
        'description': (
            'citilink-votie: SUB-SUBJECT merge + voting-only filter + '
            'minority-class oversampling + per-segment structural metadata '
            '(is_multi_vote, subject_span_lengths) + segment-level multi-vote '
            'oversampling.'
        ),
        'rationale': (
            'Multi-vote segments (~7-8.6% of corpus) are under-represented in '
            'training relative to their difficulty: span-level F1 is comparable '
            'between single-vote and multi-vote segments but event-level '
            'assembly fails disproportionately on multi-vote. Multi-vote '
            'segments are oversampled at the dataset level (same mechanism as '
            'minority-label oversampling) without any model-architecture change.'
        ),
        'source': str(args.input_dir),
        'transformations': [
            'SUB-SUBJECT boundary normalization (strip leading article)',
            'SUB-SUBJECT label -> SUBJECT (train/dev/test main files)',
            'Voting-only filter: spans=[] when source has no VOTING anchor',
            f"Minority-class oversampling x{args.oversample_factor} (train only, seed={args.seed})",
            f"Multi-vote segment oversampling x{args.multi_vote_oversample_factor} (train only, seed={args.seed})",
            'Per-segment metadata: is_multi_vote, subject_span_lengths',
        ],
        'gold_references': {
            'dev': 'dev_gold_with_sub_subjects.jsonl',
            'test': 'test_gold_with_sub_subjects.jsonl',
        },
        'oversampling': {
            'minority_factor': args.oversample_factor,
            'minority_labels': sorted(MINORITY_LABELS),
            'multi_vote_factor': args.multi_vote_oversample_factor,
            'multi_vote_definition': '>=2 distinct event_ids among VOTING spans',
            'seed': args.seed,
            'shuffle': 'deterministic via random.Random(seed) after all oversample appends',
            'note': (
                'A segment that is both minority and multi-vote receives both '
                'oversample factors (multiplied independently).'
            ),
        },
        'schema_additions_over_v5': {
            'is_multi_vote': 'bool — >=2 distinct event_ids among VOTING spans',
            'subject_span_lengths': 'list[int] — whitespace-token count per SUBJECT span (post-merge)',
        },
        'subject_length_bins': [f"{lo}-{'+' if hi == float('inf') else int(hi)}"
                                for lo, hi in SUBJECT_LENGTH_BINS],
        'split_stats': {
            split: serialize_stats(res['stats']) for split, res in results.items()
        },
        'generated_at': datetime.now().isoformat(),
        'script': Path(__file__).name,
    }

    out = output_dir / 'generation_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--input-dir', type=Path, default=Path('data/citilink_spans_v2'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/citilink-votie'))
    parser.add_argument('--oversample-factor', type=int, default=4,
                        help='Extra copies of minority-class segments in train (default 4, matches v5)')
    parser.add_argument('--multi-vote-oversample-factor', type=int, default=4,
                        help='Extra copies of multi-vote segments in train (default 4)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-voting-filter', action='store_true',
                        help='Disable the voting-only filter (for ablation)')
    args = parser.parse_args()

    enforce_voting_only = not args.no_voting_filter

    logger.info('=' * 80)
    logger.info('Generating citilink-votie')
    logger.info('=' * 80)
    logger.info(f"Input:                       {args.input_dir}")
    logger.info(f"Output:                      {args.output_dir}")
    logger.info(f"Voting-only filter:          {enforce_voting_only}")
    logger.info(f"Minority oversample factor:  {args.oversample_factor} (train only)")
    logger.info(f"Multi-vote oversample factor:{args.multi_vote_oversample_factor} (train only)")
    logger.info(f"Seed:                        {args.seed}")
    logger.info('')

    results: Dict[str, Dict[str, Any]] = {}
    total_errors = 0

    for split in ['train', 'dev', 'test']:
        input_path = args.input_dir / f'{split}.jsonl'
        if not input_path.exists():
            logger.warning(f"Skipping {split}: {input_path} not found")
            continue
        output_path = args.output_dir / f'{split}.jsonl'
        oversample = args.oversample_factor if split == 'train' else 0
        mv_oversample = args.multi_vote_oversample_factor if split == 'train' else 0
        result = process_split(
            input_path, output_path,
            merge_sub_subject=True,
            enforce_voting_only=enforce_voting_only,
            oversample_factor=oversample,
            multi_vote_oversample_factor=mv_oversample,
            seed=args.seed,
        )
        results[split] = result
        s = result['stats']
        logger.info(
            f"  {split}: in={s['segments_in']}  voting={s['voting_segments']}  "
            f"multi-vote={s['multi_vote_segments_orig']}  "
            f"non-voting-kept={s['non_voting_segments_kept']}  "
            f"out={s['segments_out']}"
        )
        logger.info(
            f"    oversample: minority_copies={s['oversample_copies']}  "
            f"multi_vote_copies={s['multi_vote_oversample_copies']}"
        )
        logger.info(
            f"    sub-subjects: relabeled={s['relabeled']}  "
            f"boundaries_fixed={s['boundaries_fixed']}"
        )
        logger.info(
            f"    spans: before_filter={s['spans_total_before']}  "
            f"after_filter+oversample={s['spans_total_after']}"
        )

        if result['errors']:
            total_errors += len(result['errors'])
            logger.error(f"    {len(result['errors'])} validation errors:")
            for err in result['errors'][:10]:
                logger.error(f"      {err}")
            if len(result['errors']) > 10:
                logger.error(f"      ... and {len(result['errors']) - 10} more")

        if split in ('dev', 'test'):
            gold_path = args.output_dir / f'{split}_gold_with_sub_subjects.jsonl'
            write_gold_reference(input_path, gold_path, enforce_voting_only=enforce_voting_only)
            logger.info(f"    Wrote gold reference: {gold_path}")

    write_summary(args.output_dir, results, args)

    logger.info('')
    if total_errors:
        logger.error(f'FAILED: {total_errors} validation errors across all splits')
        sys.exit(1)
    logger.info('Validation: all splits clean (no errors).')


if __name__ == '__main__':
    main()
