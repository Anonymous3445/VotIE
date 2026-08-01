"""Leakage-free few-shot example selection for the LOMO protocol.

`fixed_examples.py` provides one curated set of five demonstrations for the
standard benchmark, where they are safe: they come from the training split and
the benchmark scores the test split.

They are **not** safe under LOMO. A LOMO fold evaluates *every* split of the
held-out municipality, and four of the five curated examples are gold documents
from Porto (M06) and Covilha (M03). For those two folds the prompt would contain
test documents together with their gold answers.

This module builds demonstrations from gold annotations instead, so each one
carries the municipality it came from and can be excluded per fold. For a given
held-out municipality it returns five examples covering the same five phenomena
as the curated set, drawn only from the five municipalities that fold trains on.

Two phenomena — a majority vote with both named dissent types, and a secret
ballot with explicit counts — occur only in Porto across the whole corpus. When
Porto is held out those are represented by a relaxed variant (partial dissent,
or any secret-ballot marker) rather than dropped, so every fold still sees five
examples. `select_lomo_examples` reports which tier each example came from.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import langextract as lx

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/citilink-votie")
SPLITS = ("train", "dev", "test")

# Examples much longer than this crowd out the document being annotated; the
# curated set runs 102-292 characters. Used as a preference, not a hard filter.
MAX_EXAMPLE_CHARS = 600


class Phenomenon(NamedTuple):
    """One demonstration slot, with a fallback for folds where it is unavailable."""

    name: str
    strict: Callable[[Counter], bool]
    relaxed: Callable[[Counter], bool]


# Mirrors the five slots documented in fixed_examples.py:4-12.
PHENOMENA: List[Phenomenon] = [
    Phenomenon(
        "majority_with_dissent",
        # Porto is the only municipality with both named dissent roles.
        lambda c: bool(c["COUNTING-MAJORITY"] and c["VOTER-AGAINST"] and c["VOTER-ABSTENTION"]),
        lambda c: bool(c["COUNTING-MAJORITY"] and (c["VOTER-AGAINST"] or c["VOTER-ABSTENTION"])),
    ),
    Phenomenon(
        "unanimity_with_absent",
        lambda c: bool(c["COUNTING-UNANIMITY"] and c["VOTER-ABSENT"] and c["VOTER-FAVOR"]),
        lambda c: bool(c["COUNTING-UNANIMITY"] and c["VOTER-FAVOR"]),
    ),
    Phenomenon(
        "secret_ballot",
        # Also Porto-only in the strict form.
        lambda c: bool(c["VOTING-METHOD"] and c["COUNT-FAVOR"]),
        lambda c: bool(c["VOTING-METHOD"] or c["COUNT-FAVOR"] or c["COUNT-BLANK"]),
    ),
    Phenomenon(
        "multi_vote",
        lambda c: c["VOTING"] >= 2,
        lambda c: c["VOTING"] >= 2,
    ),
    Phenomenon(
        # Suppresses spurious SUBJECT generation on informational agenda items.
        "empty_extraction",
        lambda c: sum(c.values()) == 0,
        lambda c: sum(c.values()) == 0,
    ),
]


class Selection(NamedTuple):
    """Provenance of one selected demonstration, recorded in the run metadata."""

    phenomenon: str
    tier: str  # "strict" or "relaxed"
    id: str
    municipality: str
    split: str
    n_entities: int


_CACHE: Dict[Path, List[dict]] = {}


def load_corpus(data_dir: Path = DEFAULT_DATA_DIR) -> List[dict]:
    """Load every unique segment across splits, with its municipality.

    The training split contains oversampling duplicates (2,463 rows over 1,803
    unique ids), so segments are de-duplicated by id here.
    """
    data_dir = Path(data_dir)
    if data_dir in _CACHE:
        return _CACHE[data_dir]

    rows: List[dict] = []
    seen = set()
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["id"] in seen:
                    continue
                seen.add(record["id"])
                record["_split"] = split
                rows.append(record)

    _CACHE[data_dir] = rows
    logger.info(f"Loaded {len(rows)} unique segments from {data_dir}")
    return rows


def _label_counts(record: dict) -> Counter:
    return Counter(span["label"] for span in record.get("spans", []))


def _sort_key(record: dict) -> Tuple[int, int, str]:
    """Deterministic ordering: prefer short examples, break ties by id.

    Segments over MAX_EXAMPLE_CHARS sort last rather than being excluded, so a
    fold never loses a slot purely to length.
    """
    length = len(record["text"])
    return (0 if length <= MAX_EXAMPLE_CHARS else 1, length, record["id"])


def _to_example(record: dict) -> lx.data.ExampleData:
    """Build a langextract example from gold spans, in document order.

    Span text is taken as ``text[start:end]`` so the demonstration is verbatim
    with respect to the source, matching what the extractors are asked to produce.
    """
    text = record["text"]
    extractions = [
        lx.data.Extraction(
            extraction_class=span["label"],
            extraction_text=text[span["start"]:span["end"]],
        )
        for span in sorted(record.get("spans", []), key=lambda s: s["start"])
    ]
    return lx.data.ExampleData(text=text, extractions=extractions)


def select_lomo_examples(
    held_out: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    num_examples: int = 5,
) -> Tuple[List[lx.data.ExampleData], List[Selection]]:
    """Pick demonstrations for one LOMO fold, excluding the held-out municipality.

    Selection is deterministic — candidates are fully sorted, no RNG — so a fold
    reproduces exactly. Municipalities already used are deprioritised so the five
    examples do not all come from one place.

    Args:
        held_out: municipality code being evaluated in this fold (e.g. "M06").
        data_dir: corpus directory.
        num_examples: number of demonstrations; slots are filled in PHENOMENA order.

    Returns:
        (examples, provenance) — provenance is written into the evaluation metadata.

    Raises:
        ValueError: if a phenomenon cannot be filled, or if any selected example
            belongs to the held-out municipality.
    """
    corpus = load_corpus(data_dir)
    pool = [r for r in corpus if r.get("municipality") != held_out]
    if not pool:
        raise ValueError(f"No candidate segments outside {held_out}")

    examples: List[lx.data.ExampleData] = []
    provenance: List[Selection] = []
    used_municipalities: set = set()

    for phenomenon in PHENOMENA[:num_examples]:
        chosen = None
        chosen_tier = None

        for tier, predicate in (("strict", phenomenon.strict), ("relaxed", phenomenon.relaxed)):
            candidates = [r for r in pool if predicate(_label_counts(r))]
            if not candidates:
                continue
            # Prefer a municipality not yet represented, then shortest, then id.
            candidates.sort(key=lambda r: (r["municipality"] in used_municipalities, *_sort_key(r)))
            chosen, chosen_tier = candidates[0], tier
            break

        if chosen is None:
            raise ValueError(
                f"No candidate for phenomenon {phenomenon.name!r} with {held_out} held out"
            )

        used_municipalities.add(chosen["municipality"])
        examples.append(_to_example(chosen))
        provenance.append(
            Selection(
                phenomenon=phenomenon.name,
                tier=chosen_tier,
                id=chosen["id"],
                municipality=chosen["municipality"],
                split=chosen["_split"],
                n_entities=len(chosen.get("spans", [])),
            )
        )

    assert_no_leakage(held_out, provenance)
    return examples, provenance


def assert_no_leakage(held_out: str, provenance: List[Selection]) -> None:
    """Fail loudly if any demonstration comes from the municipality under test.

    This is the check whose absence let leaked prompts reach the published
    LOMO results; callers must run it before spending anything on a fold.
    """
    leaked = [s for s in provenance if s.municipality == held_out]
    if leaked:
        raise ValueError(
            f"Few-shot leakage in the {held_out} fold: "
            + ", ".join(f"{s.id} ({s.phenomenon})" for s in leaked)
        )


def scaffold_example(
    held_out: str, data_dir: Path = DEFAULT_DATA_DIR
) -> Tuple[List[lx.data.ExampleData], List[Selection]]:
    """Single example for zero-shot runs.

    langextract requires at least one example unconditionally, so zero-shot is
    really one-scaffold. That scaffold is still a real document and must respect
    the fold boundary too.
    """
    return select_lomo_examples(held_out, data_dir=data_dir, num_examples=1)
