#!/usr/bin/env python3
"""
[OPTIONAL] Convert Span-Annotated VotIE Data to BIO-Tagged Format

This script is NOT required for any model training. All models (including
BiLSTM-CRF) auto-convert span format to BIO at load time via src/data/span_to_bio.py.

It is provided as a convenience utility for:
  - External researchers who prefer pre-tokenized BIO data
  - Debugging tokenization and span-to-token alignment
  - Pre-computing BIO for faster batch loading in large-scale experiments

The spans format (data/citilink_spans/) is the canonical dataset format.
BIO format = spans format + {"tokens": [...], "labels": ["B-VOTER-FAVOR", ...]}

The tokenization uses a regex-based tokenizer (src/data/span_to_bio.py) that
matches the tokenization used during all model training.

Usage:
    # Convert main dataset (default paths)
    python scripts/dataset_generation/convert_spans_to_bio.py

    # Custom paths
    python scripts/dataset_generation/convert_spans_to_bio.py \\
        --input-dir data/citilink_spans \\
        --output-dir data/citilink_bio

    # Also convert municipality (LOMO) splits
    python scripts/dataset_generation/convert_spans_to_bio.py --convert-municipality-splits
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime

# Add root to path
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.data.span_to_bio import convert_span_to_bio, validate_bio_tags


def load_span_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load span-annotated JSONL file."""
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num} in {file_path}: {e}")
                continue

    return examples


def save_bio_data(examples: List[Dict[str, Any]], output_file: Path):
    """Save BIO-tagged examples to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def compute_statistics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics for converted data."""
    total_examples = len(examples)
    total_tokens = sum(len(ex['tokens']) for ex in examples)
    total_entities = sum(sum(1 for label in ex['labels'] if label.startswith('B-')) for ex in examples)

    # Entity type counts
    entity_counts = Counter()
    for ex in examples:
        for label in ex['labels']:
            if label.startswith('B-'):
                entity_type = label[2:]  # Remove 'B-' prefix
                entity_counts[entity_type] += 1

    # Token length distribution
    token_lengths = [len(ex['tokens']) for ex in examples]
    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0
    min_tokens = min(token_lengths) if token_lengths else 0

    # Examples with/without entities
    examples_with_entities = sum(1 for ex in examples if any(label != 'O' for label in ex['labels']))
    examples_without_entities = total_examples - examples_with_entities

    return {
        'total_examples': total_examples,
        'total_tokens': total_tokens,
        'total_entities': total_entities,
        'avg_tokens_per_example': round(avg_tokens, 2),
        'max_tokens': max_tokens,
        'min_tokens': min_tokens,
        'examples_with_entities': examples_with_entities,
        'examples_without_entities': examples_without_entities,
        'entity_counts': dict(entity_counts),
        'label_distribution': dict(Counter(label for ex in examples for label in ex['labels']))
    }


def validate_converted_data(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate converted BIO-tagged data."""
    validation_results = {
        'total_examples': len(examples),
        'valid_examples': 0,
        'invalid_examples': 0,
        'validation_errors': []
    }

    for ex in examples:
        example_id = ex.get('id', 'unknown')
        tokens = ex.get('tokens', [])
        labels = ex.get('labels', [])

        is_valid = validate_bio_tags(tokens, labels, example_id)

        if is_valid:
            validation_results['valid_examples'] += 1
        else:
            validation_results['invalid_examples'] += 1
            validation_results['validation_errors'].append(example_id)

    validation_results['validation_rate'] = (
        validation_results['valid_examples'] / validation_results['total_examples'] * 100
        if validation_results['total_examples'] > 0 else 0
    )

    return validation_results


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    files_to_convert: List[str] = None
) -> Dict[str, Any]:
    """
    Convert all JSONL files in a directory from span to BIO format.

    Args:
        input_dir: Directory containing span-annotated files
        output_dir: Directory to save BIO-tagged files
        files_to_convert: List of filenames to convert (default: all .jsonl files)

    Returns:
        Dictionary with conversion statistics
    """
    if files_to_convert is None:
        files_to_convert = ['train.jsonl', 'dev.jsonl', 'test.jsonl']

    conversion_stats = {}

    for filename in files_to_convert:
        input_file = input_dir / filename

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue

        print(f"\nConverting {input_file}...")

        # Load span data
        span_examples = load_span_data(input_file)
        print(f"  Loaded {len(span_examples)} examples")

        # Convert to BIO
        bio_examples = []
        conversion_errors = 0

        for example in span_examples:
            try:
                bio_example = convert_span_to_bio(example, validate=True)
                bio_examples.append(bio_example)
            except Exception as e:
                print(f"  Error converting {example.get('id', 'unknown')}: {e}")
                conversion_errors += 1

        print(f"  Converted {len(bio_examples)} examples successfully")
        if conversion_errors > 0:
            print(f"  Failed to convert {conversion_errors} examples")

        # Save BIO data
        output_file = output_dir / filename
        save_bio_data(bio_examples, output_file)
        print(f"  Saved to {output_file}")

        # Compute statistics
        stats = compute_statistics(bio_examples)
        validation = validate_converted_data(bio_examples)

        conversion_stats[filename] = {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'original_count': len(span_examples),
            'converted_count': len(bio_examples),
            'conversion_errors': conversion_errors,
            'statistics': stats,
            'validation': validation
        }

        # Print summary
        print(f"  Statistics:")
        print(f"    Total tokens: {stats['total_tokens']:,}")
        print(f"    Total entities: {stats['total_entities']:,}")
        print(f"    Avg tokens/example: {stats['avg_tokens_per_example']:.1f}")
        print(f"    Examples with entities: {stats['examples_with_entities']}")
        print(f"    Validation rate: {validation['validation_rate']:.1f}%")

    return conversion_stats


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Convert span-annotated VotIE data to BIO format"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/citilink_spans'),
        help='Input directory with span-annotated data (default: data/citilink_spans)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/citilink_bio'),
        help='Output directory for BIO-tagged data (default: data/citilink_bio)'
    )
    parser.add_argument(
        '--convert-municipality-splits',
        action='store_true',
        help='Also convert municipality split data'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VotIE Span-to-BIO Conversion")
    print("=" * 80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Convert main dataset
    print("\n" + "=" * 80)
    print("Converting Main Dataset")
    print("=" * 80)

    main_stats = convert_directory(args.input_dir, args.output_dir)

    # Convert municipality splits if requested
    municipality_stats = {}
    if args.convert_municipality_splits:
        print("\n" + "=" * 80)
        print("Converting Municipality Splits")
        print("=" * 80)

        municipality_input_base = Path('data/citilink-votie-lomo-splits')
        municipality_output_base = Path('data/citilink_bio_municipality_splits')

        if municipality_input_base.exists():
            municipalities = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06']

            for mun in municipalities:
                print(f"\n{'-' * 80}")
                print(f"Converting {mun}")
                print(f"{'-' * 80}")

                mun_input_dir = municipality_input_base / mun
                mun_output_dir = municipality_output_base / mun

                if mun_input_dir.exists():
                    mun_stats = convert_directory(mun_input_dir, mun_output_dir)
                    municipality_stats[mun] = mun_stats
                else:
                    print(f"Warning: {mun_input_dir} not found, skipping")
        else:
            print(f"Warning: {municipality_input_base} not found")
            print("Run scripts/create_municipality_splits.py first")

    # Save overall statistics
    overall_stats = {
        'conversion_timestamp': datetime.now().isoformat(),
        'input_directory': str(args.input_dir),
        'output_directory': str(args.output_dir),
        'main_dataset': main_stats,
        'municipality_splits': municipality_stats
    }

    stats_file = args.output_dir / 'conversion_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"✓ Converted main dataset to: {args.output_dir}")
    if municipality_stats:
        print(f"✓ Converted {len(municipality_stats)} municipality splits")
    print(f"✓ Statistics saved to: {stats_file}")
    print("=" * 80)

    # Print summary table
    print("\nConversion Summary:")
    print(f"{'File':<15} {'Original':<12} {'Converted':<12} {'Entities':<12} {'Valid %':<10}")
    print("-" * 70)

    for filename, stats in main_stats.items():
        file_short = filename.replace('.jsonl', '')
        original = stats['original_count']
        converted = stats['converted_count']
        entities = stats['statistics']['total_entities']
        valid_pct = stats['validation']['validation_rate']

        print(f"{file_short:<15} {original:<12} {converted:<12} {entities:<12} {valid_pct:<10.1f}%")

    print("\nNext steps:")
    print("1. Review conversion statistics in conversion_statistics.json")
    print("2. BIO data is ready for training:")
    print("   - Transformer configs: data/citilink_spans/ (auto-converts at load time)")
    print("   - Traditional model configs: data/citilink_bio/ (pre-tokenized)")
    print("3. To create LOMO splits: python scripts/create_lomo_splits.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
