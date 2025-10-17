# Gemini Event Extraction Module

Direct voting event extraction from Portuguese municipal meeting minutes using Google's Gemini API.

## Overview

This module uses **Gemini 2.5 Flash** with few-shot prompting to extract structured voting events directly from Portuguese text. Unlike traditional NER approaches that first extract entity spans, Gemini predicts complete voting events in JSON format in a single API call.

## Event Structure

The system extracts voting events in the following format:

```json
{
  "id": "M01_cm_003_2023-02-01_seg020",
  "has_voting_event": true,
  "event": {
    "subject": "atribuição de três apoios",
    "participants": [
      {"text": "Executivo Municipal", "position": "Favor"}
    ],
    "counting": [
      {"text": "por unanimidade", "type": "Unanimity"}
    ],
    "voting_expressions": ["deliberou", "aprovado"]
  }
}
```

## Installation

### Prerequisites

```bash
# Install Google Generative AI SDK
pip install google-generativeai

# Optional: for environment variables
pip install python-dotenv
```

### API Key Setup

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then set it:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```bash
# Test mode (10 documents)
cd src/gemini
python run_gemini_events.py --test

# Full evaluation (all 433 test documents)
python run_gemini_events.py

# Custom number of documents
python run_gemini_events.py --max-docs 50
```

### Output

The script generates the following files in `experiments/gemini_events_YYYYMMDD_HHMMSS/`:

- `predictions_events.jsonl` - Predicted events for each document
- `evaluation_results_events.json` - Detailed metrics (precision, recall, F1)
- `evaluation_summary_events.md` - Human-readable evaluation report

### Example Output

```
INFO - Processing document 1/433: M01_cm_003_2023-02-01_seg020
INFO - ✓ Extracted event from M01_cm_003_2023-02-01_seg020 (has_event=True)
...
INFO - Extraction completed: 433 documents in 720.45s (0.60 docs/sec)

============================================================
EVALUATION RESULTS
============================================================
Model: gemini-2.5-flash
Documents processed: 433
Total extraction time: 720.45s
Avg time per document: 1.66s
============================================================
EVENT-LEVEL METRICS:
  Precision: 0.8654
  Recall:    0.8821
  F1-Score:  0.8737
  Correct:   374
  Partial:   43
  Missed:    16
  Spurious:  9
============================================================
```

## Module Structure

```
gemini/
├── run_gemini_events.py          # Main extraction script
├── config.py                      # Configuration (API key, model selection)
├── evaluation_events.py           # Event-level evaluation metrics
├── __init__.py                    # Module exports
├── README.md                      # This file
└── shared/
    ├── model.py                   # GeminiEventExtractor class
    ├── prompts.py                 # System prompts and few-shot examples
    └── few_shot_examples.json     # 10 curated training examples
```

## Configuration

Edit `config.py` to change the model:

```python
@dataclass
class Config:
    model_id: str = "gemini-2.5-flash"  # Change to switch models
```

Available models:
- `gemini-2.5-flash` (recommended, fast and accurate)
- `gemini-2.0-flash-exp` (experimental)
- `gemini-1.5-flash` (stable)
- `gemini-1.5-pro` (highest quality, slower)

## Evaluation Metrics

### Event-Level Metrics

- **Precision**: Correctness of extracted events
- **Recall**: Coverage of all voting events
- **F1-Score**: Harmonic mean of precision and recall
- **Correct**: Events with all components matching
- **Partial**: Events with some components matching
- **Missed**: Voting events not detected
- **Spurious**: False positive detections


## Few-Shot Learning

The system uses 10 curated examples from `shared/few_shot_examples.json`:

- Examples cover all 6 municipalities (M01-M06)
- Include all entity types (SUBJECT, VOTER-*, COUNTING-*, VOTING)
- Cover various voting scenarios (unanimity, majority, abstentions)
- Handle different document styles and lengths

Examples are automatically loaded and included in each API request.

## API Costs

Approximate costs with Gemini 2.5 Flash (as of October 2024):

| Input | Output | Cost per 1M tokens |
|-------|--------|-------------------|
| ~1500 tokens/doc | ~200 tokens/doc | $0.075 input / $0.30 output |

**Estimated cost for full test set (433 docs):** ~$0.15

## Performance

Typical performance on Portuguese test set (433 documents):

| Metric | Score |
|--------|-------|
| Event Detection | ~95% |
| Subject F1 | ~53% |
| Participant F1 | ~70% |
| Complete Event F1 | ~30% |
| Processing Speed | ~0.6 docs/sec |

*Performance varies based on model version and prompt engineering.*

## Troubleshooting

### API Key Not Found

```
Error: No API key found
```

**Solution:** Set `GOOGLE_API_KEY` environment variable or create `.env` file.

### Quota Exceeded

```
Error: Quota limit exceeded
```

**Solution:** The system automatically retries with exponential backoff. If persistent, wait or upgrade your API quota.

### JSON Parsing Errors

```
Error: Failed to parse JSON response
```

**Solution:** This indicates model hallucination. The system logs raw responses for debugging. Consider:
- Using a more capable model (e.g., `gemini-1.5-pro`)
- Reducing the number of few-shot examples
- Simplifying the prompt


## Entity Types

The system extracts 8 entity types:

- **SUBJECT** - Subject/topic being voted on
- **VOTER-FAVOR** - Person/entity voting in favor
- **VOTER-AGAINST** - Person/entity voting against
- **VOTER-ABSTENTION** - Person/entity abstaining
- **VOTER-ABSENT** - Person/entity absent
- **VOTING** - Voting action expressions
- **COUNTING-MAJORITY** - Majority vote indicators
- **COUNTING-UNANIMITY** - Unanimity vote indicators

## License

Part of the VotIE project. See main repository LICENSE for details.
