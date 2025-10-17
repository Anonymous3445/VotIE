# VotIE Dataset

## Overview
This directory contains the VotIE corpus of Portuguese municipal meeting minutes with voting information annotations.

## Files

### BIO-Tagged Data (`votie_bio/`)
- `train.jsonl` – 2,015 training examples
- `dev.jsonl` – 431 development examples
- `test.jsonl` – 433 test examples
- `statistics.json` – Dataset statistics

### Event-Level Data (`votie_events/`)
- `train.json` – Training set with event annotations
- `dev.json` – Development set with event annotations
- `test.json` – Test set with event annotations
- `statistics.json` – Dataset statistics

## Format Specification

Each line in JSONL files contains one voting segment:
- `id`: Unique identifier (format: `MXX_cm_XXX_YYYY-MM-DD_segXXX`)
- `text`: Raw text of the voting segment
- `tokens`: List of tokenized words
- `labels`: BIO tags for each token
- `municipality`: Municipality identifier (M01-M06)

## Entity Types

1. **VOTER-FAVOR** – Participants voting in favor
2. **VOTER-AGAINST** – Participants voting against
3. **VOTER-ABSTENTION** – Participants abstaining
4. **VOTER-ABSENT** – Absent participants
5. **VOTING** – Voting action verbs
6. **SUBJECT** – Subject being voted on
7. **COUNTING-UNANIMITY** – Unanimity indicators
8. **COUNTING-MAJORITY** – Majority indicators

## Usage

Load data using the provided utilities:
```python
from src.data.dataset import load_jsonl_file

train_data = load_jsonl_file('data/votie_bio/train.jsonl')
