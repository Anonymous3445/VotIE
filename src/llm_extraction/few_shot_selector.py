"""
Few-shot example selector for span extraction.

Selects diverse examples from training data to use in few-shot prompts.
"""

import logging
import random
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter

from .shared.data_utils import load_jsonl
from .schemas import ENTITY_TYPES


logger = logging.getLogger(__name__)


def count_entity_types(example: Dict[str, Any]) -> Dict[str, int]:
    """
    Count entity types in an example.

    Args:
        example: Example with 'spans' field

    Returns:
        Dictionary mapping entity types to counts
    """
    counts = Counter()
    for span in example.get('spans', []):
        entity_type = span.get('label', span.get('type', 'UNKNOWN'))
        counts[entity_type] += 1
    return dict(counts)


def get_entity_diversity(example: Dict[str, Any]) -> int:
    """
    Get number of unique entity types in an example.

    Args:
        example: Example with 'spans' field

    Returns:
        Number of unique entity types
    """
    entity_types = set()
    for span in example.get('spans', []):
        entity_type = span.get('label', span.get('type', 'UNKNOWN'))
        entity_types.add(entity_type)
    return len(entity_types)


def calculate_diversity_score(example: Dict[str, Any], covered_types: Set[str]) -> float:
    """
    Calculate diversity score for an example.

    Prioritizes examples with:
    - High number of different entity types
    - Entity types not yet covered
    - Moderate text length (not too short or too long)

    Args:
        example: Example to score
        covered_types: Set of entity types already covered

    Returns:
        Diversity score (higher is better)
    """
    score = 0.0

    # Count entity types
    entity_types = set()
    for span in example.get('spans', []):
        entity_type = span.get('label', span.get('type', 'UNKNOWN'))
        entity_types.add(entity_type)

        # Bonus for new types
        if entity_type not in covered_types:
            score += 2.0

    # Bonus for number of different types
    score += len(entity_types) * 1.5

    # Text length penalty (prefer moderate length)
    text_len = len(example.get('text', ''))
    if 100 < text_len < 400:
        score += 1.0
    elif text_len > 600:
        score -= 0.5

    # Number of entities (prefer examples with multiple entities)
    num_entities = len(example.get('spans', []))
    if 3 <= num_entities <= 8:
        score += 1.0

    return score


def select_few_shot_examples(
    train_file: str,
    num_examples: int = 5,
    min_entity_types: int = 3,
    pool_size: int = 100,
    seed: int = 42,
    exclude_municipalities: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Select diverse few-shot examples from training data.

    Strategy:
    1. Load a pool of candidates from training data
    2. Filter for examples with minimum entity diversity
    3. Greedily select examples that maximize coverage of entity types
    4. Ensure moderate text length and entity counts

    Args:
        train_file: Path to training JSONL file
        num_examples: Number of examples to select (default: 5)
        min_entity_types: Minimum number of different entity types per example
        pool_size: Number of examples to consider from training data
        seed: Random seed for reproducibility
        exclude_municipalities: List of municipality codes to exclude (e.g., ['M06'])

    Returns:
        List of selected examples
    """
    random.seed(seed)

    # Load pool of candidates
    logger.info(f"Loading pool of {pool_size} candidates from {train_file}")
    all_examples = load_jsonl(train_file)

    # Take first pool_size examples with entities
    candidates = []
    for example in all_examples:
        # Skip if from excluded municipality
        if exclude_municipalities:
            example_id = example.get('id', '')
            municipality = example_id.split('_')[0] if '_' in example_id else ''
            if municipality in exclude_municipalities:
                continue

        if len(example.get('spans', [])) > 0:
            candidates.append(example)
        if len(candidates) >= pool_size:
            break

    logger.info(f"Found {len(candidates)} candidates with entities")

    # Filter for minimum entity diversity
    filtered = []
    for example in candidates:
        diversity = get_entity_diversity(example)
        if diversity >= min_entity_types:
            filtered.append(example)

    logger.info(f"Filtered to {len(filtered)} examples with {min_entity_types}+ entity types")

    if len(filtered) < num_examples:
        logger.warning(
            f"Only found {len(filtered)} examples meeting criteria, "
            f"relaxing min_entity_types requirement"
        )
        filtered = [ex for ex in candidates if len(ex.get('spans', [])) >= 2]

    # Greedy selection to maximize entity type coverage
    selected = []
    covered_types = set()

    for _ in range(num_examples):
        if not filtered:
            break

        # Score each candidate
        scores = [
            (calculate_diversity_score(ex, covered_types), ex)
            for ex in filtered
        ]

        # Select best
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_example = scores[0]

        selected.append(best_example)
        filtered.remove(best_example)

        # Update covered types
        for span in best_example.get('spans', []):
            entity_type = span.get('label', span.get('type', 'UNKNOWN'))
            covered_types.add(entity_type)

        logger.debug(f"Selected example {len(selected)} (score={best_score:.2f}, "
                    f"entity_types={get_entity_diversity(best_example)})")

    # Log statistics
    logger.info(f"Selected {len(selected)} few-shot examples")
    logger.info(f"Entity type coverage: {len(covered_types)}/{len(ENTITY_TYPES)}")
    logger.info(f"Covered types: {sorted(covered_types)}")

    return selected


def format_few_shot_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format an example for use in prompts (text-only, no positions).

    Args:
        example: Example with 'text' and 'spans'

    Returns:
        Dictionary with 'text' and 'output' (JSON formatted, no start/end)
    """
    import json

    # Convert spans to entity format (text and type only, no positions)
    entities = []
    for span in example.get('spans', []):
        entities.append({
            'text': span['text'],
            'type': span.get('label', span.get('type', 'UNKNOWN'))
            # NOTE: No start/end - positions found by alignment module
        })

    output = {
        'entities': entities
    }

    return {
        'id': example.get('id', ''),
        'text': example['text'],
        'output': json.dumps(output, ensure_ascii=False, indent=2)
    }


def select_lomo_few_shot_examples(
    train_file: str,
    train_municipalities: List[str],
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Select few-shot examples for LOMO: exactly 1 per training municipality.

    Strategy:
    1. For each training municipality, select the best example
    2. Greedily select to maximize entity type coverage across all municipalities
    3. Ensure all 8 entity types are covered if possible

    Args:
        train_file: Path to training JSONL file
        train_municipalities: List of municipality codes to select from (e.g., ['M02', 'M03', 'M04', 'M05', 'M06'])
        seed: Random seed for reproducibility

    Returns:
        List of selected examples (one per municipality)
    """
    random.seed(seed)

    logger.info(f"LOMO few-shot selection: 1 example per training municipality")
    logger.info(f"Training municipalities: {train_municipalities}")

    # Load all training data
    all_examples = load_jsonl(train_file)

    # Group examples by municipality
    by_municipality = defaultdict(list)
    for example in all_examples:
        if len(example.get('spans', [])) == 0:
            continue
        example_id = example.get('id', '')
        municipality = example_id.split('_')[0] if '_' in example_id else ''
        if municipality in train_municipalities:
            by_municipality[municipality].append(example)

    logger.info(f"Found examples per municipality:")
    for muni in train_municipalities:
        logger.info(f"  {muni}: {len(by_municipality.get(muni, []))} examples")

    # Greedy selection: pick one from each municipality to maximize coverage
    selected = []
    covered_types = set()

    for municipality in train_municipalities:
        candidates = by_municipality.get(municipality, [])
        if not candidates:
            logger.warning(f"No examples found for {municipality}")
            continue

        # Score each candidate based on diversity and uncovered types
        scores = [
            (calculate_diversity_score(ex, covered_types), ex)
            for ex in candidates
        ]

        # Select best
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_example = scores[0]

        selected.append(best_example)

        # Update covered types
        for span in best_example.get('spans', []):
            entity_type = span.get('label', span.get('type', 'UNKNOWN'))
            covered_types.add(entity_type)

        logger.info(f"  Selected from {municipality}: {best_example['id']} "
                   f"(types={get_entity_diversity(best_example)}, score={best_score:.2f})")

    logger.info(f"Selected {len(selected)} few-shot examples (1 per municipality)")
    logger.info(f"Entity type coverage: {len(covered_types)}/{len(ENTITY_TYPES)}")
    logger.info(f"Covered types: {sorted(covered_types)}")

    if len(covered_types) < len(ENTITY_TYPES):
        missing = set(ENTITY_TYPES) - covered_types
        logger.warning(f"Missing entity types: {sorted(missing)}")

    return selected


def load_and_format_few_shot_examples(
    train_file: str,
    num_examples: int = 5,
    pool_size: int = 500,  # Increased from 100 to find VOTER-AGAINST
    save_to_file: bool = True,
    exclude_municipalities: List[str] = None,
    lomo_mode: bool = False,
    train_municipalities: List[str] = None
) -> List[Dict[str, str]]:
    """
    Load and format few-shot examples from training data.

    Args:
        train_file: Path to training JSONL file
        num_examples: Number of examples to select (ignored if lomo_mode=True)
        pool_size: Number of candidates to consider (increased to 500)
        save_to_file: Whether to save selected examples to file
        lomo_mode: If True, select exactly 1 example per training municipality
        train_municipalities: List of municipalities for LOMO mode (required if lomo_mode=True)
        exclude_municipalities: List of municipality codes to exclude (e.g., ['M06'])

    Returns:
        List of formatted examples ready for prompts
    """
    import json
    from pathlib import Path

    # Select diverse examples
    if lomo_mode:
        # LOMO mode: exactly 1 example per training municipality
        if not train_municipalities:
            raise ValueError("train_municipalities required when lomo_mode=True")
        examples = select_lomo_few_shot_examples(
            train_file,
            train_municipalities=train_municipalities
        )
    else:
        # Standard mode: greedy selection for diversity
        examples = select_few_shot_examples(
            train_file,
            num_examples=num_examples,
            pool_size=pool_size,
            exclude_municipalities=exclude_municipalities
        )

    # Format for prompts
    formatted = [format_few_shot_example(ex) for ex in examples]

    logger.info(f"Formatted {len(formatted)} few-shot examples for prompts")

    # Save to file if requested
    if save_to_file:
        save_dir = Path(__file__).parent / "few_shot_data"
        save_dir.mkdir(exist_ok=True)

        # Save formatted examples
        examples_file = save_dir / f"examples_{num_examples}shot_pool{pool_size}.json"
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved few-shot examples to {examples_file}")

        # Save metadata
        metadata = {
            "num_examples": num_examples,
            "pool_size": pool_size,
            "train_file": train_file,
            "example_ids": [ex['id'] for ex in formatted],
            "entity_types_covered": list(set(
                entity_type
                for ex in formatted
                for entity in json.loads(ex['output'])['entities']
                for entity_type in [entity['type']]
            ))
        }
        metadata_file = save_dir / f"metadata_{num_examples}shot_pool{pool_size}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_file}")

    return formatted
