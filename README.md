# VotIE: Voting Information Extraction from Municipal Meeting Minutes


[![License: CC-BY-ND 4.0](https://img.shields.io/badge/License-CC--BY--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official repository for the paper **"VotIE: Voting Information Extraction from Municipal Meeting Minutes"**.

This repository provides a comprehensive framework for extracting structured voting information from Portuguese municipal meeting minutes using span extraction using sequence labeling, along with voting events construction heuristics.

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
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

VotIE is a specialized pipeline designed to extract structured voting information from Portuguese municipal meeting minutes. The system operates at two levels:

1. **Entity-Level Extraction** – Identifies and classifies voting-related spans using BIO tagging
2. **Event-Level Construction** – Reconstructs complete voting events from extracted entities

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

- **Dual-Level Evaluation**: Entity-level (token-level F1) and Event-level (complete event accuracy) metrics
- **Multiple Architectures**: Traditional baselines (CRF, BiLSTM+FastText); Transformer models (BERTimbau, DeBERTa, XLM-RoBERTa); Gemini 2.5 flash for the event extraction, using gemini API and few-shot approach. 
- **Portuguese-Optimized**: Specialized for Portuguese administrative text with support for BERTimbau
- **Windowing Support**: Handles long documents that exceed transformer context limits
- **Production-Ready**: Modular, well-documented codebase with comprehensive evaluation
- **Reproducible**: Fixed seeds, documented hyperparameters, and detailed training logs

---

## Project Status

The VotIE framework is **fully implemented and validated** for research use. The codebase is actively maintained to ensure reproducibility of published results.

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

### Utilities
- **PyYAML** (6.0+) – Configuration management
- **pandas** – Data analysis and result formatting
- **tqdm** – Progress tracking

### Hardware
- All experiments were conducted on a Nvidia L40 GPU
- Recommended: GPU with 8GB+ VRAM for transformer models

---

## Installation

### Prerequisites
- Python 3.10
- CUDA 11.7+ (for GPU support)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anonymous3445/votie.git
cd votie
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start

**Option 1: Use Pre-trained Model (Fastest)**

For a quick test of the model on your own text, see `scripts/quick_start.py`:

```python
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
model_name = "Anonymous3445/DeBERTa-CRF-VotIE"
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

**Note:** The HuggingFace model is designed for inference on raw text and uses its own tokenization. For evaluation on the pre-tokenized test dataset, use the training pipeline (Option 2 or 3 below).

**Option 2: Train from Scratch - Full Pipeline**
```bash
# Train, predict, and evaluate in one command
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --name my_experiment
```

**Option 3: Train from Scratch - Step by Step**
```bash
# 1. Train model
python scripts/train.py --config configs/deberta_crf.yaml --experiment-name my_experiment

# 2. Make predictions
python scripts/predict.py \
  models/deberta_crf/my_experiment/best_model \
  data/votie_bio/test.jsonl \
  predictions/my_predictions.jsonl

# 3. Evaluate
python scripts/evaluate.py \
  predictions/my_predictions.jsonl \
  evaluation/my_results.json
```

---

## Usage

### Quick Testing with Pre-trained Model

#### Interactive Demo (`quick_start.py`)

Test the model on your own Portuguese text:

```bash
python scripts/quick_start.py
```

This script demonstrates:
- Loading the model from HuggingFace Hub
- Processing a sample Portuguese municipal text
- Extracting voting entities with BIO labels
- Displaying results with character offsets

**Customization:** Edit the `text` variable in the script to test your own content.

**Example output:**
```
Text: 'A Câmara deliberou aprovar a proposta por unanimidade.'

Word                           Label
--------------------------------------------------
A                              B-VOTER-FAVOR
Câmara                         I-VOTER-FAVOR
deliberou                      B-VOTING
aprovar                        O
a                              O
proposta                       B-SUBJECT
por                            O
unanimidade                    B-COUNTING-UNANIMITY
.                              O
```

**Note:** The HuggingFace model is optimized for inference on raw text. For evaluation on the pre-tokenized VotIE test set, use the full training pipeline below.

---

### Training Models

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
- `configs/xlmr_crf.yaml` – XLM-RoBERTa + CRF layer

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

**Note**:
- Training only performs model training and saves to `models/MODEL_NAME/EXPERIMENT_NAME/`
- Model checkpoints are NOT tracked in Git (too large)
- For the complete pipeline (train → predict → evaluate), use `scripts/run_pipeline.py` (see [Full Pipeline](#full-pipeline) section)
- To use the published best model, download from [Hugging Face](https://huggingface.co/Anonymous3445/DeBERTa-CRF-VotIE)

### Making Predictions

Generate predictions on new data:

```bash
python scripts/predict.py MODEL_PATH INPUT_JSONL OUTPUT_JSONL [DEVICE]
```

**Examples:**
```bash
# Predict on test set (CPU)
python scripts/predict.py \
  models/deberta_crf/run_20251017_120000/best_model \
  data/votie_bio/test.jsonl \
  predictions/test_predictions.jsonl \
  cpu

# Predict on test set (GPU)
python scripts/predict.py \
  models/deberta_crf/run_20251017_120000/best_model \
  data/votie_bio/test.jsonl \
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

Evaluate predictions with comprehensive metrics:

```bash
python scripts/evaluate.py PREDICTIONS_JSONL [OUTPUT_JSON] [--no-events]
```

**Examples:**
```bash
# Evaluate with entity and event metrics
python scripts/evaluate.py predictions/test_predictions.jsonl

# Save detailed results to file
python scripts/evaluate.py \
  predictions/test_predictions.jsonl \
  evaluation/detailed_results.json

# Entity-level metrics only (faster)
python scripts/evaluate.py \
  predictions/test_predictions.jsonl \
  --no-events
```

**Metrics Computed:**

*Entity-Level (Span-based):*
- Precision, Recall, F1 (overall and per-entity-type)
- Uses `seqeval` library for strict BIO validation

*Event-Level (Voting Event):*
- Event Detection F1 – Identifying documents with voting events
- Subject F1 – Correct subject extraction
- Counting F1 – Correct counting method (unanimity/majority)
- Participant-Vote F1 – Correct participant-vote pairs
- Complete Event F1 – All event components correct

### Full Pipeline

For convenience, use the pipeline script to run train → predict → evaluate in one command:

```bash
python scripts/run_pipeline.py --config configs/deberta_crf.yaml
```

This will:
1. Train the model
2. Generate predictions on the test set
3. Compute comprehensive evaluation metrics
4. Save all outputs to organized directories

### Running All Experiments

To reproduce all paper experiments:

```bash
# Run all model configurations
bash scripts/run_all_experiments.sh
```

This will train all baselines (CRF, BiLSTM, BERT, DeBERTa, XLM-R) and generate complete results.

---

## Dataset

### VotIE Corpus

The VotIE corpus consists of voting segments extracted from Portuguese municipal meeting minutes.

| **Attribute** | **Value** |
|---------------|-----------|
| **Language** | Portuguese |
| **Municipalities** | 6 (M01–M06) |
| **Total Examples** | 2,879 voting segments |
| **Total Tokens** | ~1,008,324 tokens |
| **Total Entities** | 9,951 annotated entities |
| **Train Examples** | 2,015 (70%) |
| **Dev Examples** | 431 (15%) |
| **Test Examples** | 433 (15%) |

### Entity Distribution

| **Entity Type** | **Train** | **Dev** | **Test** | **Total** |
|-----------------|-----------|---------|----------|-----------|
| VOTING | 1,866 | 385 | 397 | 2,648 |
| SUBJECT | 1,731 | 374 | 373 | 2,478 |
| VOTER-FAVOR | 1,150 | 245 | 255 | 1,650 |
| COUNTING-UNANIMITY | 1,092 | 251 | 227 | 1,570 |
| VOTER-ABSTENTION | 738 | 166 | 138 | 1,042 |
| COUNTING-MAJORITY | 202 | 34 | 52 | 288 |
| VOTER-AGAINST | 116 | 21 | 40 | 177 |
| VOTER-ABSENT | 62 | 14 | 22 | 98 |

### Data Format

**BIO-Tagged JSONL (`data/votie_bio/`):**
```json
{
  "id": "M01_cm_003_2023-02-01_seg020",
  "text": "O Executivo Municipal deliberou por unanimidade aprovar...",
  "tokens": ["O", "Executivo", "Municipal", "deliberou", "por", "unanimidade", "aprovar", "..."],
  "labels": ["O", "B-VOTER-FAVOR", "I-VOTER-FAVOR", "B-VOTING", "O", "B-COUNTING-UNANIMITY", "O", "..."],
  "municipality": "M01"
}
```

**Event-Level JSON (`data/votie_events/`):**
```json
{
  "id": "M01_cm_003_2023-02-01_seg020",
  "text": "...",
  "municipality": "M01",
  "has_voting_event": true,
  "event": {
    "subject": "atribuição de três apoios",
    "participants": [{"text": "Executivo Municipal", "position": "Favor"}],
    "counting": [{"text": "por unanimidade", "type": "Unanimity"}],
    "voting_expressions": ["deliberou"],
    "outcome": "Approved"
  }
}
```
---

## Models

### Model Architectures

All models follow a token classification architecture for BIO tagging:

**Traditional Baselines:**
1. **CRF** – Conditional Random Fields with hand-crafted features
2. **BiLSTM+FastText** – Bidirectional LSTM with pre-trained FastText embeddings

**Transformer Baselines:**

3. **BERTimbau-Large + Linear** – Portuguese BERT with linear classification head
4. **BERTimbau-Large + CRF** – Portuguese BERT with CRF layer
5. **DeBERTa-V3-Base + Linear** – DeBERTa V3 with linear head
6. **DeBERTa-V3-Base + CRF** – DeBERTa V3 with CRF layer
7. **XLM-RoBERTa-Large + Linear** – Multilingual XLM-R with linear head
8. **XLM-RoBERTa-Large + CRF** – Multilingual XLM-R with CRF layer

### Pre-trained Models

The best-performing model from the paper is available on Hugging Face:

- **DeBERTa-V3-Base + CRF** (Best Model): [`Anonymous3445/DeBERTa-CRF-VotIE`](https://huggingface.co/Anonymous3445/DeBERTa-CRF-VotIE)

**Usage:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Anonymous3445/DeBERTa-CRF-VotIE")
model = AutoModel.from_pretrained("Anonymous3445/DeBERTa-CRF-VotIE")
```

To train other models from scratch, use the configurations in `configs/` directory.

### Configuration

All model hyperparameters are specified in YAML configuration files:

```yaml
model:
  name: "deberta_crf"
  model_name: "microsoft/deberta-v3-base"
  architecture: "crf"
  dropout: 0.1

data:
  data_dir: "data/votie_bio"
  train_file: "train.jsonl"
  dev_file: "dev.jsonl"
  test_file: "test.jsonl"

training:
  batch_size: 16
  epochs: 10
  learning_rate: 5e-5
  warmup_proportion: 0.1
  weight_decay: 0.01
  patience: 3
  device: 0  # GPU device ID
  seed: 42

windowing:
  enable_windowing: true
  overlap_tokens: 50
```

---

## Experimental Results

### Main Results (Test Set)

All experimental results are available in `paper_results/evaluation/`. The tables below summarize the main findings.

**Entity-Level Performance (F1 Score):**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CRF | 88.3% | 84.7% | 86.5% |
| BiLSTM+FastText | 80.7% | 77.4% | 79.0% |
| BERTimbau+Linear | 82.5% | 95.3% | 88.5% |
| BERTimbau+CRF | 88.4% | 93.5% | 90.9% |
| DeBERTa+Linear | 87.1% | 95.6% | 91.2% |
| **DeBERTa+CRF** ⭐ | **91.1%** | **95.0%** | **93.0%** |
| XLM-R+Linear | 76.3% | 93.6% | 84.1% |
| XLM-R+CRF | 88.9% | 94.7% | 91.7% |

⭐ **Best model available on Hugging Face**: [`Anonymous3445/DeBERTa-CRF-VotIE`](https://huggingface.co/Anonymous3445/DeBERTa-CRF-VotIE)

**Event-Level Performance:**

| Model | Event Detection F1 | Subject F1 | Counting F1 | Participant-Vote F1 | Complete Event F1 |
|-------|-------------------|------------|-------------|---------------------|-------------------|
| CRF | 97.1% | 78.2% | 99.4% | 90.8% | 69.8% |
| BiLSTM+FastText | 96.2% | 70.1% | 96.2% | 86.0% | 58.4% |
| BERTimbau+Linear | 97.3% | 89.7% | 98.5% | 96.8% | 81.7% |
| BERTimbau+CRF | 96.9% | 89.2% | 99.2% | 97.9% | 85.6% |
| DeBERTa+Linear | 97.0% | 91.0% | 99.2% | 97.9% | 84.4% |
| **DeBERTa+CRF** ⭐ | **97.1%** | **92.2%** | **99.6%** | **97.8%** | **88.3%** |
| XLM-R+Linear | 97.3% | 79.7% | 97.1% | 93.3% | 70.0% |
| XLM-R+CRF | 97.2% | 90.2% | 99.6% | 98.1% | 86.5% |

⭐ **Best overall performance** from DeBERTa+CRF, particularly on complete event extraction (88.3%)

### Gemini 2.0 Flash - Direct Event Extraction

In addition to sequence labeling approaches, we evaluated **Gemini 2.0 Flash** for direct event extraction using few-shot prompting. Unlike the other models that first extract BIO-tagged entities and then construct events, Gemini directly predicts complete voting events in structured JSON format.

**Approach:**
- **Model**: `gemini-2.0-flash-exp`
- **Method**: Few-shot learning with 10 curated examples
- **Task**: Direct extraction of voting events (subject, participants, counting, voting expressions)
- **Prompt**: Structured JSON output with predefined schema (see `src/gemini/shared/prompts.py`)

**Prompt Template:**
```
You are an expert at extracting structured information from Portuguese municipal meeting minutes.

Your task is to identify and extract voting events from city council meeting texts.

Output Format: Return ONLY a valid JSON object with this exact structure:
{
  "has_voting_event": boolean,
  "event": {
    "subject": string or null,
    "participants": [{"text": string, "position": "Favor"|"Against"|"Abstention"|"Absent"}],
    "counting": [{"text": string, "type": "Majority"|"Unanimity"}],
    "voting_expressions": [string]
  }
}
```

**Few-Shot Examples:**
10 examples from `src/gemini/shared/few_shot_examples.json` covering diverse voting scenarios across different municipalities (M01-M06).

**Usage:**
```bash
# Run Gemini extraction
cd src/gemini
python run_gemini_events.py --test  # Test mode (10 documents)
python run_gemini_events.py         # Full evaluation (433 documents)
```

**Note**: Gemini was evaluated solely for event-level extraction and was not compared at the entity-level (span extraction) with other models. See `src/gemini/README.md` for detailed documentation.

### Reproducing Paper Results

Published predictions and evaluation metrics are provided in `paper_results/`:
```bash
# View paper predictions
ls paper_results/predictions/

# View paper evaluation metrics
ls paper_results/evaluation/

# Example: Check DeBERTa-CRF results
cat paper_results/evaluation/deberta_crf.json
```

---

## Repository Structure

```
votie/
├── configs/                      # Model configuration files (tracked in Git)
│   ├── crf.yaml
│   ├── bilstm_fasttext.yaml
│   ├── bert_linear.yaml
│   ├── bert_crf.yaml
│   ├── deberta_linear.yaml
│   ├── deberta_crf.yaml
│   ├── xlmr_linear.yaml
│   └── xlmr_crf.yaml
│
├── data/                        # Dataset files
│   ├── votie_bio/               # BIO-tagged NER data
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   ├── test.jsonl
│   │   └── statistics.json
│   └── votie_events/            # Event-level annotations
│       ├── train.json
│       ├── dev.json
│       ├── test.json
│       └── statistics.json
│
├── paper_results/                # Published results from 
│   ├── predictions/             # Model predictions on 
│   │   ├── crf.jsonl
│   │   ├── deberta_crf.jsonl
│   │   └── ...
│   └── evaluation/              # Evaluation metrics
│       ├── crf.json
│       ├── deberta_crf.json
│       └── ...
│
├── scripts/                     # Executable scripts
│   ├── quick_start.py           # Quick demo using HuggingFace model
│   ├── train.py                 # Unified training script
│   ├── predict.py               # Prediction with trained models
│   ├── evaluate.py              # Comprehensive evaluation
│   ├── run_pipeline.py          # End-to-end pipeline
│   └── run_all_experiments.sh   # Run all experiments sequentially
│
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── base.py              # Base model interface
│   │   ├── crf.py               # Traditional CRF
│   │   ├── bilstm_crf.py        # BiLSTM+FastText+CRF
│   │   ├── bertimbau_models.py  # BERTimbau models
│   │   ├── deberta_models.py    # DeBERTa models
│   │   └── xlmr_models.py       # XLM-RoBERTa models
│   ├── data/                   # Data loading and processing
│   │   └── dataset.py          # Dataset utilities, windowing
│   ├── evaluation/             # Evaluation metrics
│   │   ├── entity_metrics.py   # Entity-level (seqeval)
│   │   └── event_metrics.py    # Event-level metrics
│   ├── utils/                  # Utility functions
│   │   └── event_constructor.py  # Event reconstruction
│   ├── gemini/                  # Gemini API integration
│   │   ├── run_gemini_events.py # Direct event extraction script
│   │   ├── shared/
│   │   │   ├── model.py         # GeminiEventExtractor
│   │   │   ├── prompts.py       # Few-shot prompts
│   │   │   └── few_shot_examples.json  # Curated examples
│   │   └── README.md            # Gemini module documentation
│   ├── trainer.py              # Transformer training loop
│   └── bilstm_trainer.py       # BiLSTM training loop
│
├── models/                      # Local trained models (NOT tracked - too large)
├── predictions/                 # Local predictions (NOT tracked)
├── evaluation/                  # Local evaluations (NOT tracked)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # License file (CC-BY-NC-ND 4.0)
└── .gitignore                   # Git ignore rules
```

**Note on Directory Structure:**
- **Tracked in Git**: Source code, configs, data, and `paper_results/` (for reproducibility)
- **NOT tracked**: `models/`, `predictions/`, `evaluation/` (generated locally, too large for Git)

---

## Reproducibility

### Fixed Random Seeds

All experiments use fixed random seeds for reproducibility:
- PyTorch seed: 42
- NumPy seed: 42
- Python random seed: 42
- CUDA deterministic mode: enabled

### Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA  L40 GPU |
| VRAM | 48GB |
| CUDA Version | 11.7+ |



### Reproducing Published Results

**Option 1: Using the pipeline script for each model**
```bash
# Run complete pipeline for each configuration
python scripts/run_pipeline.py --config configs/crf.yaml --experiment-name paper_reproduction
python scripts/run_pipeline.py --config configs/bilstm_fasttext.yaml --experiment-name paper_reproduction
python scripts/run_pipeline.py --config configs/bert_crf.yaml --experiment-name paper_reproduction
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --experiment-name paper_reproduction
python scripts/run_pipeline.py --config configs/xlmr_crf.yaml --experiment-name paper_reproduction
```

**Option 2: Using the batch script**
```bash
# Run all experiments at once
bash scripts/run_all_experiments.sh
```

**Option 3: Manual step-by-step**
```bash
# 1. Train all baselines
for config in configs/*.yaml; do
  python scripts/train.py --config $config --experiment-name paper_reproduction
done

# 2. Generate predictions on test set
for model_dir in models/*/paper_reproduction/best_model; do
  model_name=$(basename $(dirname $(dirname $model_dir)))
  python scripts/predict.py $model_dir data/votie_bio/test.jsonl predictions/${model_name}_predictions.jsonl
done

# 3. Evaluate all predictions
for pred_file in predictions/*_predictions.jsonl; do
  python scripts/evaluate.py $pred_file evaluation/$(basename $pred_file .jsonl)_results.json
done
```

---

## License

This project is licensed under **CC-BY-ND 4.0 (Creative Commons Attribution–NoDerivatives 4.0 International)**.

You are free to:

- **Share:** Copy and redistribute the material in any medium or format  

Under the following terms:

- **Attribution:** You must give appropriate credit.  
- **No Derivatives:** If you remix, transform, or build upon the material, you may not distribute the modified version.

For details, see the `LICENSE` file.

### Dataset License

The **Council Metadata Corpus** is derived from public municipal meeting minutes and is provided strictly for **research purposes only**.  
Original documents remain the copyright of their respective municipal governments.


### Dataset License

The VotIE corpus is derived from public municipal meeting minutes and is provided strictly for **research purposes only**. Original documents remain the copyright of their respective municipal governments.

By using this dataset, you agree to:
- Use it only for non-commercial research
- Cite the original paper
- Not redistribute without permission

---

## Acknowledgments

- The six Portuguese municipalities (M01–M06) for providing access to meeting minutes
- The [seqeval](https://github.com/chakki-works/seqeval) project for sequence evaluation metrics
- [Hugging Face](https://huggingface.co/) for the Transformers library and model hosting
- [Neuralmind](https://neuralmind.ai/) for BERTimbau pre-trained models

---



**Last Updated**: October 2025
