# Citilink: Voting Information Extraction from Municipal Meeting Minutes

[![License: CC-BY-ND 4.0](https://img.shields.io/badge/License-CC--BY--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official repository for the paper **"Citilink: Voting Information Extraction from Municipal Meeting Minutes"**, submitted to ACL 2026.

This repository provides a comprehensive framework for extracting structured voting information from Portuguese municipal meeting minutes using span extraction with sequence labeling (BIO tagging) and LLM-based extraction.

> **🎯 Try Citilink Now**: Test the model interactively at [huggingface.co/spaces/Anonymous3445/VotIE-demo](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Status](#project-status)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Making Predictions](#making-predictions)
  - [Running Evaluation](#running-evaluation)
- [Dataset](#dataset)
- [Models](#models)
  - [Discriminative Models](#discriminative-models)
  - [LLM-Based Extraction](#llm-based-extraction)
  - [Prompts](#prompts)
- [Experimental Results](#experimental-results)
  - [Main Results](#main-results-test-set)
  - [Leave-One-Municipality-Out (LOMO)](#leave-one-municipality-out-lomo-evaluation)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

![Citilink Overview](VotIE_dataset_diagram.png)

VotIE is a specialized framework designed to extract structured voting information from Portuguese municipal meeting minutes using:

- **Sequence Labeling** – Identifies and classifies voting-related spans using BIO tagging
- **LLM-Based Extraction** – Uses Gemini with few-shot prompting for span extraction

The framework supports 8 entity types:
- **VOTER-FAVOR** – Participants who voted in favor
- **VOTER-AGAINST** – Participants who voted against
- **VOTER-ABSTENTION** – Participants who abstained
- **VOTER-ABSENT** – Participants who were absent
- **VOTING** – Voting action expressions (e.g., "deliberated", "approved")
- **SUBJECT** – The subject matter being voted on
- **COUNTING-UNANIMITY** – Unanimous vote indicators
- **COUNTING-MAJORITY** – Majority vote indicators

---

## Key Features

- **Sequence Labeling**: Token-level F1 metrics using seqeval for strict BIO validation
- **Multiple Architectures**: Traditional baselines (CRF, BiLSTM+FastText); Transformer models (BERTimbau, DeBERTa, XLM-RoBERTa); LLM extraction (Gemini via API)
- **Windowing Support**: Handles long documents that exceed transformer context limits
- **Cross-Municipality Evaluation**: Leave-One-Municipality-Out (LOMO) experiments for domain transfer analysis
- **Reproducible**: Fixed seeds, documented hyperparameters, and detailed training logs

---

## Project Status

The VotIE framework is **fully implemented and validated** for research use. The codebase is actively maintained to ensure reproducibility of published results.

**Dataset Availability**:
- ✅ **Sample Data Available**: 30 representative test examples for demonstration (included in repository)
- ✅ **Full Dataset**: Available for download from [RDM Repository](https://rdm.inesctec.pt/dataset/cs-2025-007)
- ✅ **Document ID Lists**: Train/dev/test split IDs included for reproducibility verification

---

## Technology Stack

### Core Frameworks
- **PyTorch** (2.0+) – Deep learning backend
- **Transformers (Hugging Face)** (4.20+) – Pre-trained language models
- **seqeval** (1.2.2+) – Sequence labeling evaluation

### NLP Libraries
- **sklearn-crfsuite** (0.3.6+) – Traditional CRF baseline
- **pytorch-crf** (0.7.2+) – CRF layer for transformers
- **FastText** – Word embeddings for BiLSTM baseline
- **spaCy** (3.4.0+) – Text preprocessing

### LLM Integration
- **Google Generative AI** – Gemini API for LLM extraction with contrained output.


### Hardware
- All experiments were conducted on a NVIDIA 5070 GPU

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.7+ (for GPU support)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anonymous3445/citilink.git
cd citilink
```

### Step 2: Download the Full Dataset

The complete Citilink dataset must be downloaded separately from the official RDM repository:

**📥 Download Link**: https://rdm.inesctec.pt/dataset/cs-2025-007

After downloading, extract the dataset files into the placeholder directories:

```bash
# Extract span format
unzip citilink_spans.zip -d data/citilink_spans/

# Extract BIO format
unzip citilink_bio.zip -d data/citilink_bio/
```

The extracted files will be placed alongside the existing ID files in each directory.

Verify the download matches the provided ID lists:
```python
import json

# Verify train split
with open('data/splits/train_ids.txt') as f:
    expected_ids = set(line.strip() for line in f)

with open('data/citilink_spans/train.jsonl') as f:
    actual_ids = set(json.loads(line)['id'] for line in f)

assert expected_ids == actual_ids
print(f"✓ Train split verified: {len(expected_ids)} examples")
```

### Step 3: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: (Optional) Set up Gemini API
For LLM-based extraction experiments:
```bash
export GOOGLE_API_KEY="your-api-key"
```

---

## Quick Start

**Option 1: Try the Interactive Demo (Recommended)**

🎯 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)**

Try VotIE directly in your browser without any installation! The interactive demo allows you to:
- Test the model on sample Portuguese municipal texts
- Upload your own text
- Visualize entity extraction results in real-time

**Option 2: Use Pre-trained Model Locally**

For a quick test of the model on your own text, see `scripts/quick_start.py`:

```python
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model_name = "Anonymous3445/XLM-RoBERTa-CRF-VotIE"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Example text
text = "A Câmara deliberou aprovar a proposta por unanimidade."

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
predictions = model.decode(**inputs, tokenizer=tokenizer, text=text)

# Print results
for pred in predictions:
    print(f"{pred['word']:<30} {pred['label']}")
```

**Option 3: Train from Scratch - Full Pipeline** (requires full dataset from RDM)
```bash
# Train, predict, and evaluate in one command
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --name my_experiment
```

**Option 4: Train from Scratch - Step by Step** (requires full dataset from RDM)
```bash
# 1. Train model
python scripts/train.py --config configs/deberta_crf.yaml --experiment-name my_experiment

# 2. Make predictions
python scripts/predict.py \
  models/deberta_crf/my_experiment/best_model \
  data/citilink_spans/test.jsonl \
  predictions/my_predictions.jsonl

# 3. Evaluate
python scripts/evaluate.py \
  predictions/my_predictions.jsonl \
  evaluation/my_results.json
```

---

## Usage

### Training Models

> **📥 Dataset Required**: Training requires the full Citilink dataset from [RDM](https://rdm.inesctec.pt/dataset/cs-2025-007). See [Installation → Step 2](#step-2-download-the-full-dataset) for download instructions. For quick testing without downloading, use the [pre-trained model](https://huggingface.co/Anonymous3445/XLM-RoBERTa-CRF-VotIE) or the [🎯 Interactive Demo](https://huggingface.co/spaces/Anonymous3445/VotIE-demo).

The unified training script `scripts/train.py` supports all model architectures. Model type is automatically determined from the configuration file.

#### Basic Training
```bash
python scripts/train.py --config configs/MODEL_NAME.yaml
```

#### Available Model Configurations

**Traditional Baselines:**
- `configs/crf.yaml` – Conditional Random Fields
- `configs/bilstm_fasttext.yaml` – BiLSTM + FastText embeddings

**Transformer Baselines:**
- `configs/bert_linear.yaml` – BERTimbau + Linear layer
- `configs/bert_crf.yaml` – BERTimbau + CRF layer
- `configs/deberta_linear.yaml` – DeBERTa + Linear layer
- `configs/deberta_crf.yaml` – DeBERTa + CRF layer
- `configs/xlmr_linear.yaml` – XLM-RoBERTa + Linear layer
- `configs/xlmr_crf.yaml` – XLM-RoBERTa + CRF layer ⭐ (BEST)

**Municipality-Specific (LOMO):**
- `configs/municipality_experiments/deberta_crf_M01.yaml` – Leave out M01
- `configs/municipality_experiments/deberta_crf_M02.yaml` – Leave out M02
- ... (M01–M06)

#### Training Examples
```bash
# Train CRF baseline
python scripts/train.py --config configs/crf.yaml

# Train BERTimbau+CRF with custom experiment name
python scripts/train.py --config configs/bert_crf.yaml --experiment-name exp_001
```

#### Training Output
Trained models are saved to `models/MODEL_NAME/EXPERIMENT_NAME/`:
```
models/deberta_crf/run_20251017_120000/
├── best_model/              # Best checkpoint by validation F1
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── training_results.json    # Training metrics and hyperparameters
└── logs/                    # Training logs
```

### Making Predictions

Generate predictions on new data:

```bash
python scripts/predict.py MODEL_PATH INPUT_JSONL OUTPUT_JSONL [DEVICE]
```

**Examples:**
```bash

# Predict on test set (GPU)
python scripts/predict.py \
  models/deberta_crf/run_20251017_120000/best_model \
  data/citilink_spans/test.jsonl \
  predictions/test_predictions.jsonl \
  cuda
```

**Input Format (JSONL):**
```json
{"id": "example_001", "tokens": ["O", "Executivo", "aprovou", "..."], "labels": ["O", "B-VOTER-FAVOR", "B-VOTING", "..."]}
```

**Output Format (JSONL):**
```json
{"id": "example_001", "tokens": ["O", "Executivo", "aprovou", "..."], "pred_labels": ["O", "B-VOTER-FAVOR", "B-VOTING", "..."], "gold_labels": ["O", "B-VOTER-FAVOR", "B-VOTING", "..."]}
```

### Running Evaluation

Evaluate predictions with entity-level metrics:

```bash
python scripts/evaluate.py PREDICTIONS_JSONL [OUTPUT_JSON]
```

**Examples:**
```bash
# Evaluate with entity metrics
python scripts/evaluate.py predictions/test_predictions.jsonl

# Save detailed results to file
python scripts/evaluate.py \
  predictions/test_predictions.jsonl \
  evaluation/detailed_results.json
```

**Metrics Computed:**
- Precision, Recall, F1 (macro averaged and per-entity-type)
- Uses `seqeval` library for strict BIO validation

### Full Pipeline

For convenience, use the pipeline script to run train → predict → evaluate in one command:

```bash
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --name my_experiment
```

### Running All Experiments

To reproduce all paper experiments (requires full dataset):

```bash
# Run all model configurations
bash scripts/run_all_experiments.sh paper_reproduction
```

---

## Dataset

### Citilink Corpus

The Citilink corpus consists of voting segments extracted from Portuguese municipal meeting minutes.

> **📥 DOWNLOAD REQUIRED**
> 
> The **full Citilink dataset is NOT included** in this repository. You must download it from:
> 
> ### **https://rdm.inesctec.pt/dataset/cs-2025-007**
> 
> **What's Included in This Repository:**
> - ✅ Document ID lists for train/dev/test splits (for reproducibility verification)
> - ✅ Empty placeholder folders (`data/citilink_spans/`, `data/citilink_bio/`)
> 
> **What You Must Download:**
> - ⚠️ Full dataset files: `train.jsonl`, `dev.jsonl`, `test.jsonl` (both span and BIO formats)
> 
> **Setup Instructions:** See [Installation → Step 2](#step-2-download-the-full-dataset)
> 
> **Test Without Downloading:** Try our **[🎯 Interactive Demo](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)**

**Full Dataset Statistics**:

| **Attribute** | **Value** |
|---------------|-----------|
| **Language** | Portuguese |
| **Municipalities** | 6 (M01–M06) |
| **Total Examples** | 2,879 voting segments |
| **Total Characters** | ~5,119,217 characters |
| **Total Entities** | 9,721 annotated entities |
| **Train Examples** | 1,798 (62%) |
| **Dev Examples** | 552 (19%) |
| **Test Examples** | 529 (18%) |

### Entity Distribution (Full Dataset)

| **Entity Type** | **Train** | **Dev** | **Test** | **Total** |
|-----------------|-----------|---------|----------|-----------|
| VOTING | 1,709 | 472 | 467 | 2,648 |
| SUBJECT | 1,422 | 406 | 420 | 2,248 |
| VOTER-FAVOR | 1,039 | 311 | 300 | 1,650 |
| COUNTING-UNANIMITY | 970 | 295 | 305 | 1,570 |
| VOTER-ABSTENTION | 730 | 179 | 133 | 1,042 |
| COUNTING-MAJORITY | 175 | 51 | 62 | 288 |
| VOTER-AGAINST | 98 | 43 | 36 | 177 |
| VOTER-ABSENT | 57 | 23 | 18 | 98 |

### Data Format

The Citilink dataset is available in a voting focused format:

**Span Format** (`data/citilink_spans/`):
```json
{
  "id": "M01_cm_003_2023-02-01_seg020",
  "text": "O Executivo Municipal deliberou por unanimidade aprovar...",
  "spans": [
    {"text": "O Executivo Municipal", "type": "VOTER-FAVOR", "start": 0, "end": 21},
    {"text": "deliberou", "type": "VOTING", "start": 22, "end": 31}
  ],
  "municipality": "M01"
}
```

**BIO Format** - A conversion script is used to produce the BIO tags used for the sequence labeling approach(`data/citilink_bio/`):
```json
{
  "id": "M01_cm_003_2023-02-01_seg020",
  "text": "O Executivo Municipal deliberou por unanimidade aprovar...",
  "tokens": ["O", "Executivo", "Municipal", "deliberou", "por", "unanimidade", "aprovar", "..."],
  "labels": ["B-VOTER-FAVOR", "I-VOTER-FAVOR", "I-VOTER-FAVOR", "B-VOTING", "O", "B-COUNTING-UNANIMITY", "O", "..."],
  "municipality": "M01"
}
```

### Directory Structure

```
data/
├── README.md                              # Dataset documentation with download instructions
├── splits/                                # Split document IDs
│   ├── train_ids.txt                      # ✅ Train IDs (1,798)
│   ├── dev_ids.txt                        # ✅ Dev IDs (552)
│   └── test_ids.txt                       # ✅ Test IDs (529)
│
├── citilink_spans/                        # Span format (DOWNLOAD from RDM)
│   ├── train.jsonl                        # ⚠️ DOWNLOAD from RDM
│   ├── dev.jsonl                          # ⚠️ DOWNLOAD from RDM
│   └── test.jsonl                         # ⚠️ DOWNLOAD from RDM
│
└── citilink_bio/                          # BIO format (DOWNLOAD from RDM)
    ├── train.jsonl                        # ⚠️ DOWNLOAD from RDM
    ├── dev.jsonl                          # ⚠️ DOWNLOAD from RDM
    └── test.jsonl                         # ⚠️ DOWNLOAD from RDM
```

---

## Models

### Discriminative Models

All discriminative models follow a token classification architecture for BIO tagging:

**Traditional Baselines:**
1. **CRF** – Conditional Random Fields with hand-crafted features
2. **BiLSTM+FastText** – Bidirectional LSTM with pre-trained FastText embeddings

**Transformer Baselines:**

3. **BERTimbau-Large + Linear** – Portuguese BERT with linear classification head
4. **BERTimbau-Large + CRF** – Portuguese BERT with CRF layer
5. **DeBERTa-V3-Base + Linear** – DeBERTa V3 with linear head
6. **DeBERTa-V3-Base + CRF** – DeBERTa V3 with CRF layer
7. **XLM-RoBERTa-Large + Linear** – Multilingual XLM-R with linear head
8. **XLM-RoBERTa-Large + CRF** ⭐ – Multilingual XLM-R with CRF layer (Best)

### LLM-Based Extraction

The framework includes LLM-based span extraction using Google's Gemini:

**Strategies:**
- **Zero-Shot** – No examples, detailed entity definitions in prompt
- **Few-Shot** – 5 examples selected for entity type coverage

**Few-Shot Example Selection:**

Few-shot examples are automatically selected to maximize entity type diversity:
- **Selection Algorithm**: Greedy coverage of all 8 entity types
- **Pool Size**: 500 candidates from training data
- **Examples**: 5 examples covering all entity types
- **File Location**: [`src/llm_extraction/few_shot_data/examples_5shot_pool500.json`](src/llm_extraction/few_shot_data/examples_5shot_pool500.json)

```python
# Regenerate few-shot examples
from src.llm_extraction.few_shot_selector import load_and_format_few_shot_examples

examples = load_and_format_few_shot_examples(
    'data/citilink_spans/train.jsonl',
    num_examples=5,
    pool_size=500,
    save_to_file=True
)
```

**Few-shot examples:** [`src/llm_extraction/few_shot_data/examples_5shot_pool500.json`](src/llm_extraction/few_shot_data/examples_5shot_pool500.json)

**Running LLM Extraction:**
```bash
# Zero-shot extraction
python scripts/llm_extraction/extract_gemini_spans.py --strategy zero_shot

# Few-shot extraction
python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot

# Evaluate LLM results
python scripts/llm_extraction/evaluate_spans.py \
  results/llm_extraction/gemini_few_standard.jsonl
```

### Prompts

The following prompts are used for Gemini-based extraction. For complete implementation details, see [`src/llm_extraction/gemini/prompts.py`](src/llm_extraction/gemini/prompts.py).

#### Zero-Shot Prompt

```
You are an expert Portuguese NLP system extracting semantic spans from municipal meeting minutes.

Task: Extract all entity spans from the text and return a JSON object with an "entities" array.

Each entity must have:
- text: EXACT character span from the input text (do not paraphrase or modify)
- type: One of [VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT, VOTING, SUBJECT, COUNTING-UNANIMITY, COUNTING-MAJORITY]

Entity Type Definitions:
- VOTER-FAVOR: Person or group voting in favor
- VOTER-AGAINST: Person or group voting against
- VOTER-ABSTENTION: Person or group abstaining
- VOTER-ABSENT: Person or group absent from vote
- VOTING: Verb indicating voting action (e.g., "deliberou", "aprovou")
- SUBJECT: Matter being voted on (NOUN PHRASE ONLY, see rules below)
- COUNTING-UNANIMITY: Expression indicating unanimous decision
- COUNTING-MAJORITY: Expression indicating majority decision

CRITICAL SUBJECT EXTRACTION RULES:
1. Extract the MATTER being voted on (what is being approved/rejected)
2. Do NOT extract section titles, headings, or numbering (e.g., "2. TÍTULO DO PONTO")
3. Extract ONLY the noun phrase, WITHOUT action verbs
4. Remove verbs like "aprovar", "ratificar", "deliberar", "conceder", "autorizar" from the SUBJECT
5. Extract EXACTLY ONE subject per voting decision
6. If the subject appears multiple times in different forms, choose the most specific version (with details)

SUBJECT Examples:
✓ CORRECT: "alteração orçamental permutativa" (noun phrase only)
✗ WRONG: "ALTERAÇÃO ORÇAMENTAL PERMUTATIVA" (section title at beginning)
✗ WRONG: "aprovar a alteração orçamental" (includes verb)
✓ CORRECT: "transporte de dois alunos para a Escola Profissional de Desenvolvimento Rural de Serpa" (specific with details)
✗ WRONG: "transporte de alunos" (too generic when specific version exists)

CRITICAL REQUIREMENTS:
1. Extract the EXACT text as it appears in the input - character-for-character
2. Do not paraphrase, summarize, reword, or modify any text
3. Do not add or remove words
4. Preserve all articles, prepositions, and punctuation exactly as they appear
5. Return empty array if no entities found

Examples of CORRECT extraction:
✓ Input: "o Executivo Municipal" → Extract: "o Executivo Municipal"
✓ Input: "deliberou por unanimidade" → Extract: "deliberou" and "por unanimidade"

Examples of INCORRECT extraction (DO NOT DO THIS):
✗ Input: "o Executivo Municipal" → Extract: "Executivo Municipal" (missing "o")
✗ Input: "deliberou por unanimidade" → Extract: "decidiu por unanimidade" (paraphrased)

Text:
{input_text}

Extract entities as JSON:
```

#### Few-Shot Prompt

```
You are an expert Portuguese NLP system extracting named entities from municipal meeting minutes.

Task: Extract all entity spans from the text and return a JSON object with an "entities" array.

Each entity must have:
- text: EXACT character span from the input text (do not paraphrase or modify)
- type: One of [VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT, VOTING, SUBJECT, COUNTING-UNANIMITY, COUNTING-MAJORITY]

CRITICAL SUBJECT EXTRACTION RULES:
1. Extract the MATTER being voted on (what is being approved/rejected)
2. Do NOT extract section titles, headings, or numbering (e.g., "2. TÍTULO DO PONTO")
3. Extract ONLY the noun phrase, WITHOUT action verbs
4. Remove verbs like "aprovar", "ratificar", "deliberar", "conceder", "autorizar" from SUBJECT
5. Extract EXACTLY ONE subject per voting decision
6. Choose the most specific version when multiple forms exist

CRITICAL: Extract EXACT text as it appears. Do not paraphrase, summarize, or modify any words.

Here are some examples:

Example 1:
Text: {example_1_text}
Output: {example_1_output}

Example 2:
Text: {example_2_text}
Output: {example_2_output}

[... 5 examples total ...]

Now extract from the following text:
Text: {input_text}

Output:
```

**Note**: The few-shot examples are dynamically selected from the training data using a greedy algorithm that ensures coverage of all 8 entity types. The 5 examples are loaded from [`src/llm_extraction/few_shot_data/examples_5shot_pool500.json`](src/llm_extraction/few_shot_data/examples_5shot_pool500.json).

### Pre-trained Models

The best-performing model from the paper is available on Hugging Face:

- **XLM-RoBERTa-Large + CRF** (Best Model): [`Anonymous3445/XLM-RoBERTa-CRF-VotIE`](https://huggingface.co/Anonymous3445/XLM-RoBERTa-CRF-VotIE)
- **🎯 Interactive Demo**: [VotIE-demo on Hugging Face Spaces](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)

**Usage:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Anonymous3445/XLM-RoBERTa-CRF-VotIE", trust_remote_code=True)
model = AutoModel.from_pretrained("Anonymous3445/XLM-RoBERTa-CRF-VotIE", trust_remote_code=True)
```

---

## Experimental Results

### Main Results (Test Set)

All experimental results are available in `results/`. The tables below summarize the main findings.

**Entity-Level Performance (Macro-Averaged F1 Score):**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 84.7% | 72.7% | 75.1% |
| BiLSTM+FastText | 74.3% | 68.3% | 70.2% |
| BERTimbau+Linear | 82.0% | 93.0% | 87.0% |
| DeBERTa+Linear | 86.7% | 94.9% | 90.4% |
| DeBERTa+CRF | 89.4% | 91.6% | 90.4% |
| XLM-R+Linear | 78.4% | 94.0% | 84.9% |
| **XLM-R+CRF** ⭐ | **91.0%** | **95.6%** | **93.2%** |
| Gemini (Zero-Shot) | 55.6% | 50.0% | 52.3% |
| Gemini (Few-Shot) | 62.7% | 65.8% | 64.1% |


**Note**: All metrics are macro-averaged across entity types using strict boundary matching (seqeval).

### Per-Entity Type Performance

The following tables show detailed F1 scores for each entity type across all models:

#### VOTER-FAVOR

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 91.7% | 92.3% | 92.0% |
| BiLSTM+FastText | 91.1% | 92.7% | 91.9% |
| BERTimbau+Linear | 89.2% | 96.7% | 92.8% |
| DeBERTa+Linear | 91.0% | 97.3% | 94.0% |
| DeBERTa+CRF | 93.9% | 97.0% | 95.4% |
| XLM-R+Linear | 91.6% | 98.3% | 94.9% |
| **XLM-R+CRF** | **93.8%** | **96.3%** | **95.1%** |
| Gemini (Zero-Shot) | 80.3% | 65.3% | 72.1% |
| Gemini (Few-Shot) | 84.2% | 97.7% | 90.4% |

#### VOTER-AGAINST

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 63.6% | 19.4% | 29.8% |
| BiLSTM+FastText | 94.4% | 47.2% | 63.0% |
| BERTimbau+Linear | 85.7% | 100.0% | 92.3% |
| DeBERTa+Linear | 100.0% | 100.0% | 100.0% |
| DeBERTa+CRF | 90.0% | 100.0% | 94.7% |
| XLM-R+Linear | 72.3% | 94.4% | 81.9% |
| **XLM-R+CRF** | **92.3%** | **100.0%** | **96.0%** |
| Gemini (Zero-Shot) | 0.0% | 0.0% | 0.0% |
| Gemini (Few-Shot) | 32.6% | 38.9% | 35.4% |

#### VOTER-ABSTENTION

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 75.3% | 94.0% | 83.6% |
| BiLSTM+FastText | 70.7% | 83.5% | 76.6% |
| BERTimbau+Linear | 92.9% | 97.7% | 95.2% |
| DeBERTa+Linear | 96.3% | 97.7% | 97.0% |
| DeBERTa+CRF | 96.2% | 96.2% | 96.2% |
| XLM-R+Linear | 93.7% | 100.0% | 96.7% |
| **XLM-R+CRF** | **95.7%** | **100.0%** | **97.8%** |
| Gemini (Zero-Shot) | 37.9% | 35.3% | 36.6% |
| Gemini (Few-Shot) | 56.6% | 64.7% | 60.4% |

#### VOTER-ABSENT

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 83.3% | 27.8% | 41.7% |
| BiLSTM+FastText | 0.0% | 0.0% | 0.0% |
| BERTimbau+Linear | 50.0% | 72.2% | 59.1% |
| DeBERTa+Linear | 60.0% | 83.3% | 69.8% |
| DeBERTa+CRF | 75.0% | 66.7% | 70.6% |
| XLM-R+Linear | 37.8% | 77.8% | 50.9% |
| **XLM-R+CRF** | **88.9%** | **88.9%** | **88.9%** |
| Gemini (Zero-Shot) | 40.0% | 22.2% | 28.6% |
| Gemini (Few-Shot) | 68.8% | 61.1% | 64.7% |

#### VOTING

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 95.4% | 98.1% | 96.7% |
| BiLSTM+FastText | 95.4% | 97.9% | 96.6% |
| BERTimbau+Linear | 93.1% | 98.5% | 95.7% |
| DeBERTa+Linear | 94.7% | 98.5% | 96.5% |
| DeBERTa+CRF | 95.4% | 97.9% | 96.6% |
| XLM-R+Linear | 93.9% | 98.9% | 96.4% |
| **XLM-R+CRF** | **96.0%** | **97.9%** | **96.9%** |
| Gemini (Zero-Shot) | 72.8% | 77.9% | 75.3% |
| Gemini (Few-Shot) | 91.5% | 94.4% | 92.9% |

#### SUBJECT

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 76.1% | 53.8% | 63.0% |
| BiLSTM+FastText | 52.6% | 45.7% | 48.9% |
| BERTimbau+Linear | 65.2% | 82.1% | 72.7% |
| DeBERTa+Linear | 68.9% | 85.5% | 76.3% |
| DeBERTa+CRF | 77.7% | 79.5% | 78.6% |
| XLM-R+Linear | 64.4% | 82.9% | 72.5% |
| **XLM-R+CRF** | **74.1%** | **83.8%** | **78.7%** |
| Gemini (Zero-Shot) | 20.6% | 21.4% | 21.0% |
| Gemini (Few-Shot) | 38.0% | 44.3% | 40.9% |

#### COUNTING-UNANIMITY

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 93.4% | 98.0% | 95.7% |
| BiLSTM+FastText | 93.8% | 93.8% | 93.8% |
| BERTimbau+Linear | 92.6% | 98.4% | 95.4% |
| DeBERTa+Linear | 92.1% | 99.7% | 95.7% |
| DeBERTa+CRF | 91.6% | 97.0% | 94.3% |
| XLM-R+Linear | 93.3% | 100.0% | 96.5% |
| **XLM-R+CRF** | **93.1%** | **98.0%** | **95.5%** |
| Gemini (Zero-Shot) | 94.7% | 98.0% | 96.3% |
| Gemini (Few-Shot) | 94.6% | 98.0% | 96.3% |

#### COUNTING-MAJORITY

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 98.4% | 98.4% | 98.4% |
| BiLSTM+FastText | 96.4% | 85.5% | 90.6% |
| BERTimbau+Linear | 87.1% | 98.4% | 92.4% |
| DeBERTa+Linear | 90.9% | 96.8% | 93.7% |
| DeBERTa+CRF | 95.3% | 98.4% | 96.8% |
| XLM-R+Linear | 80.5% | 100.0% | 89.2% |
| **XLM-R+CRF** | **93.9%** | **100.0%** | **96.9%** |
| Gemini (Zero-Shot) | 97.6% | 96.8% | 97.2% |
| Gemini (Few-Shot) | 98.3% | 93.5% | 95.9% |

### Leave-One-Municipality-Out (LOMO) Evaluation

To evaluate cross-municipality generalization, we perform LOMO experiments where models are trained on 5 municipalities and tested on the held-out municipality.

**LOMO Configuration:**
- **Training**: Data from 5 municipalities
- **Testing**: Data from 1 held-out municipality
- **Model**: DeBERTa-CRF

**LOMO Results (Macro-Averaged Entity F1):**

| Held-Out Municipality | Test Size | Precision | Recall | F1 Score |
|-----------------------|-----------|-----------|--------|----------|
| M01 | 503 | 45.6% | 47.7% | 46.5% |
| M02 | 396 | 60.1% | 72.1% | 65.2% |
| M03 | 719 | 63.2% | 63.2% | 62.4% |
| M04 | 235 | 59.2% | 68.5% | 61.3% |
| M05 | 554 | 63.9% | 73.3% | 64.1% |
| M06 | 472 | 24.0% | 14.2% | 17.5% |
| **Average** | - | **52.6%** | **56.5%** | **52.8%** |

**Running LOMO Experiments:**
```bash
# Run all LOMO experiments
python scripts/run_municipality_experiments.py

# Run single municipality
python scripts/run_pipeline.py \
  --config configs/municipality_experiments/deberta_crf_M01.yaml \
  --name lomo_M01
```

**LLM Cross-Municipality Results (Macro-Averaged Entity F1):**

| Held-Out | Test Size | Precision | Recall | F1 Score |
|----------|-----------|-----------|--------|----------|
| M01 | 503 | 72.1% | 64.2% | 66.8% |
| M02 | 396 | 60.0% | 81.2% | 69.0% |
| M03 | 719 | 70.6% | 69.0% | 69.7% |
| M04 | 235 | 61.9% | 62.2% | 62.0% |
| M05 | 554 | 73.9% | 80.4% | 76.0% |
| M06 | 472 | 46.1% | 44.8% | 45.4% |
| **Average** | - | **64.1%** | **67.0%** | **64.8%** |

Results are available in `results/discriminative_models/deberta_crf/municipality_experiments/` and `results/llm_extraction/cross_municipality/`.

**Note**: LOMO results show significant performance degradation compared to in-domain evaluation (93.2% → 52.8% for DeBERTa-CRF), indicating domain shift challenges across municipalities. Interestingly, Gemini Few-Shot (64.8%) outperforms DeBERTa-CRF in the cross-municipality setting.

---

## Repository Structure

```
citilink/
├── configs/                               # Model configuration files
│   ├── crf.yaml
│   ├── bilstm_fasttext.yaml
│   ├── bert_linear.yaml
│   ├── bert_crf.yaml
│   ├── deberta_linear.yaml
│   ├── deberta_crf.yaml
│   ├── xlmr_linear.yaml
│   ├── xlmr_crf.yaml                      # Best model config ⭐
│   └── municipality_experiments/          # LOMO experiment configs
│       ├── deberta_crf_M01.yaml
│       └── ... (M01-M06)
│
├── data/                                  # Dataset directory
│   ├── README.md                          # Download instructions
│   ├── sample_data.jsonl                  # 30 sample examples ✅
│   ├── citilink_spans/                    # Span format (download from RDM)
│   │   ├── *_ids.txt                      # Document IDs ✅
│   │   └── *.jsonl                        # Data files (download required)
│   └── citilink_bio/                      # BIO format (download from RDM)
│       ├── *_ids.txt                      # Document IDs ✅
│       └── *.jsonl                        # Data files (download required)
│
├── results/                               # Experimental results
│   ├── discriminative_models/             # Transformer/CRF results
│   ├── llm_extraction/                    # Gemini extraction results
│   │   ├── gemini_zero_standard.jsonl
│   │   ├── gemini_few_standard.jsonl
│   │   └── cross_municipality/            # LOMO results
│   └── error_analysis/                    # Error classification
│
├── scripts/                               # Executable scripts
│   ├── train.py                           # Unified training script
│   ├── predict.py                         # Generate predictions
│   ├── evaluate.py                        # Evaluation metrics
│   ├── run_pipeline.py                    # End-to-end pipeline
│   ├── validate_experiments.py            # Dry-run validation
│   ├── quick_start.py                     # Quick demo script
│   ├── run_all_experiments.sh             # Run all experiments
│   ├── run_municipality_experiments.py    # LOMO experiments
│   ├── error_classification.py            # Error analysis
│   └── llm_extraction/                    # LLM extraction scripts
│       ├── extract_gemini_spans.py
│       ├── evaluate_spans.py
│       ├── run_cross_municipality.py      # LLM LOMO experiments
│       └── analyze_errors.py
│
├── src/                                   # Source code
│   ├── models/                            # Model implementations
│   │   ├── base.py                        # Base model interface
│   │   ├── crf.py                         # Traditional CRF
│   │   ├── bilstm_crf.py                  # BiLSTM+FastText+CRF
│   │   ├── bertimbau_models.py            # BERTimbau models
│   │   ├── deberta_models.py              # DeBERTa models
│   │   └── xlmr_models.py                 # XLM-RoBERTa models
│   ├── data/                              # Data loading and processing
│   │   ├── dataset.py                     # Dataset utilities
│   │   ├── span_to_bio.py                 # Format conversion
│   │   └── postprocessing.py              # Output postprocessing
│   ├── evaluation/                        # Evaluation metrics
│   │   └── entity_metrics.py              # Entity-level (seqeval)
│   ├── llm_extraction/                    # LLM extraction module
│   │   ├── gemini/                        # Gemini extractor
│   │   │   ├── extractor.py
│   │   │   ├── prompts.py                 # Zero/few-shot prompts
│   │   │   └── config.py
│   │   ├── few_shot_data/                 # Few-shot examples
│   │   │   ├── examples_5shot_pool500.json
│   │   │   ├── metadata_5shot_pool500.json
│   │   │   └── README.md
│   │   ├── few_shot_selector.py           # Example selection
│   │   ├── span_alignment.py              # Post-processing alignment
│   │   └── shared/                        # Shared utilities
│   ├── trainer.py                         # Transformer training loop
│   └── bilstm_trainer.py                  # BiLSTM training loop
│
├── models/                                # Trained models (NOT tracked)
├── predictions/                           # Generated predictions (NOT tracked)
├── evaluation/                            # Evaluation outputs (NOT tracked)
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── LICENSE                                # License file (CC-BY-ND 4.0)
```


---

## Reproducibility

### Fixed Random Seeds

All experiments use fixed random seeds for reproducibility:
- PyTorch seed: 42
- NumPy seed: 42
- Python random seed: 42
- CUDA deterministic mode: enabled

### Reproducing Published Results

> **📥 Dataset Required**: Training pipelines require the full dataset from [RDM](https://rdm.inesctec.pt/dataset/cs-2025-007). You can:
> - Download the **full dataset** from RDM and extract to `data/` directory
> - Test the **pre-trained model** from [HuggingFace](https://huggingface.co/Anonymous3445/XLM-RoBERTa-CRF-VotIE)
> - Try the **[🎯 Interactive Demo](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)**

**Option 1: Validate Setup (No Training)**
```bash
# Check all configs compile
python scripts/validate_experiments.py --smoke-test
```

**Option 2: Full Reproduction** (requires full dataset download from RDM)
```bash
# Run all experiments
bash scripts/run_all_experiments.sh paper_reproduction

# Run LOMO experiments
python scripts/run_municipality_experiments.py
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{citilink2026,
  title={VotIE: Voting Information Extraction from Meeting Minutes},
  author={Anonymous},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```

---

## License

This project is licensed under **CC-BY-ND 4.0 (Creative Commons Attribution–NoDerivatives 4.0 International)**.

You are free to:
- **Share:** Copy and redistribute the material in any medium or format

Under the following terms:
- **Attribution:** You must give appropriate credit
- **No Derivatives:** If you remix, transform, or build upon the material, you may not distribute the modified version

For details, see the `LICENSE` file.


---

## Acknowledgments

- The [seqeval](https://github.com/chakki-works/seqeval) project for sequence evaluation metrics
- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting
- [Neuralmind](https://neuralmind.ai/) for BERTimbau pre-trained models
- Google for providing access to the Gemini API

---

**Last Updated**: January 2026
