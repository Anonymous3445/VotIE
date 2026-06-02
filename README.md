# VotIE: Information Extraction from Meeting Minutes

[![License: CC-BY-ND 4.0](https://img.shields.io/badge/License-CC--BY--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official repository for the paper **"VotIE: Information Extraction from Meeting Minutes"**, submitted to EMNLP 2026.

VotIE is a research framework for extracting structured voting information from Portuguese municipal meeting minutes. It implements two complementary approaches: discriminative sequence labeling with BIO tagging across five model families, and LLM-based extraction using Gemini 2.5 Pro, GPT 5.5, and AMALIA 8B in zero/few-shot configurations via langextract.

> **🎯 Try it now**: Test the model interactively at [huggingface.co/spaces/Anonymous3445/VotIE-demo](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)
> **📦 Pre-trained model**: [Anonymous3445/XLM-RoBERTa-CRF-VotIE](https://huggingface.co/Anonymous3445/XLM-RoBERTa-CRF-VotIE)

<div align="center">
    <img width="1000" alt="VotIE dataset diagram" src="VotIE_dataset_diagram.png" />
</div>

---

## Table of Contents

1. [Description](#description)
2. [Project Status](#1-project-status)
3. [Technology Stack](#2-technology-stack)
4. [Dependencies](#3-dependencies)
5. [Installation](#4-installation)
6. [Usage](#5-usage)
7. [Dataset](#6-dataset)
8. [Architecture](#7-architecture)
9. [Evaluation Metrics](#8-evaluation-metrics)
10. [Experimental Results](#9-experimental-results)
11. [Known Issues](#10-known-issues)
12. [License](#11-license)
13. [Resources](#12-resources)
14. [Acknowledgments](#13-acknowledgments)
15. [Citation](#14-citation)

---

## Description

VotIE addresses the task of identifying voting entities in Portuguese municipal council meeting minutes, a domain-specific challenge where voting-related spans must be identified and typed across highly formal, structured text.

The framework supports **12 entity types** across four categories:

**Voter roles:** VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT

**Voting event:** VOTING (the act of voting), SUBJECT (the matter under deliberation)

**Counting expressions:** COUNTING-UNANIMITY, COUNTING-MAJORITY (natural-language outcomes); COUNT-FAVOR, COUNT-BLANK (numeric secret-ballot tallies); COUNT-AGAINST (schema element, zero test instances)

**Procedure:** VOTING-METHOD (e.g., "escrutínio secreto")

### Key Features

- **Unified training interface**: `scripts/train.py` auto-detects model type from YAML config; all configs live in `configs/`
- **Multiple architectures**: Traditional (CRF, BiLSTM+FastText), Transformer (BERTimbau, mDeBERTa-v3, XLM-RoBERTa), and generative LLMs (Gemini 2.5 Pro, GPT 5.5, AMALIA 8B)
- **Single canonical data format**: Span-annotated JSONL; BIO conversion is applied automatically at load time — no pre-processing required
- **Windowing support**: Documents exceeding 512 tokens are split into overlapping windows and reassembled at inference
- **Cross-municipality evaluation**: Leave-One-Municipality-Out (LOMO) experiments across 6 Portuguese municipalities
- **Reproducible**: Three seeds (42, 13, 123) for encoder-CRF models; seed 42 for all other configurations.

---

## 1. Project Status

**Status**: ✅ Completed Research Prototype — Under Review

The VotIE framework is fully implemented and validated. All experiments from the paper are reproducible using the provided code and dataset. The codebase is ready for research use.

**Dataset Availability**:

- ✅ **Full dataset**: Available at [rdm.inesctec.pt/dataset/cs-2025-007](https://rdm.inesctec.pt/dataset/cs-2025-007).
- ✅ **Pre-trained model**: Available on [Hugging Face](https://huggingface.co/Anonymous3445/XLM-RoBERTa-CRF-VotIE)
- ✅ **Interactive demo**: Available at [Hugging Face Spaces](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)

---

## 2. Technology Stack

**Language**: Python 3.10+

**Core Frameworks**:
- **PyTorch** (2.0+) – Deep learning backend for all discriminative models
- **Hugging Face Transformers** (4.20+) – Pre-trained language model loading and fine-tuning
- **seqeval** (1.2.2+) – Sequence labeling evaluation with strict BIO validation

**Key Libraries**:
- `pytorch-crf` (0.7.2+): CRF layer used on top of transformer encoders
- `sklearn-crfsuite` (0.3.6+): Traditional CRF baseline with hand-crafted features
- `fasttext`: Pre-trained word embeddings for the BiLSTM baseline
- `spaCy` (3.4.0+): Text preprocessing in dataset generation scripts
- `google-generativeai` / `google-genai`: Gemini API for LLM-based extraction

**Hardware**: All transformer experiments were conducted on NVIDIA A100-SXM4-40GB GPUs on the Deucalion EuroHPC supercomputer. The CRF baseline runs on CPU. GPT 5.5 and Gemini 2.5 Pro were accessed via API.

---

## 3. Dependencies

All dependencies are in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- **torch** (>=1.10.0) – PyTorch deep learning framework
- **transformers** (>=4.20.0) – Hugging Face pre-trained models
- **pytorch-crf** (>=0.7.2) – CRF layer for transformer NER models
- **seqeval** (>=1.2.2) – Entity-level NER evaluation metrics
- **sklearn-crfsuite** (>=0.3.6) – Traditional CRF baseline
- **scikit-learn** (>=1.0.0) – Stratified splitting and utilities
- **numpy** (>=1.21.0) / **pandas** (>=1.3.0) – Data manipulation
- **pyyaml** (>=6.0) – Configuration file parsing
- **pydantic** (>=2.0.0) – Output validation for LLM extraction

### Optional Dependencies

- **fasttext** – Required only for the BiLSTM+FastText baseline; also requires downloading `cc.pt.300.bin` embeddings separately
- **google-genai** (>=0.1.0) – Required only for LLM extraction experiments (Gemini)
- **spaCy** (>=3.4.0) – Required only for dataset generation scripts

### API Keys (LLM experiments only)

```bash
export GEMINI_KEY="your-gemini-api-key"       # Gemini 2.5 Pro
export IAEDU_API_KEY="your-iaedu-api-key"     # GPT 5.5 via IAEDU API
# AMALIA 8B: requires a running vLLM server endpoint (see src/llm_extraction/amalia/config.py)
```

---

## 4. Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended; CPU inference supported)

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Anonymous3445/citilink.git
cd citilink
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Dataset**: `data/citilink-votie/` (standard splits) is pre-generated and included. For LOMO experiments, generate the splits locally:

```bash
python scripts/dataset_generation/create_lomo_splits_v6.py
```

   The raw source annotations are in `data/citilink-dataset/` and also available at [rdm.inesctec.pt/dataset/cs-2025-007](https://rdm.inesctec.pt/dataset/cs-2025-007).

5. **Verify installation**
```bash
python scripts/quick_start.py
```

**Expected output:**
```
Loading model from HuggingFace: Anonymous3445/XLM-RoBERTa-CRF-VotIE
Running inference on sample text...
Word                           Label
...
```

---

## 5. Usage

### Quick Start

**Option 1: Interactive Demo (no installation)**

🎯 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)**

Test on Portuguese municipal text in your browser — no setup required.

**Option 2: Run the demo script locally**

```bash
python scripts/quick_start.py
```

**Option 3: Use pre-trained model locally**

```python
from transformers import AutoTokenizer, AutoModel

model_name = "Anonymous3445/XLM-RoBERTa-CRF-VotIE"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

text = "A Câmara deliberou aprovar a proposta por unanimidade."
inputs = tokenizer(text, return_tensors="pt")
predictions = model.decode(**inputs, tokenizer=tokenizer, text=text)

for pred in predictions:
    print(f"{pred['word']:<30} {pred['label']}")
```

See `scripts/quick_start.py` for a complete example.

### Training Models

The unified training script auto-detects the model type from the YAML config:

```bash
python scripts/train.py --config configs/MODEL_NAME.yaml
python scripts/train.py --config configs/xlmr_crf.yaml --experiment-name my_run
```

**Available configurations:**

| Config | Architecture |
|--------|-------------|
| `configs/crf.yaml` | Conditional Random Fields |
| `configs/bilstm_fasttext.yaml` | BiLSTM + FastText embeddings |
| `configs/bert_linear.yaml` | BERTimbau + Linear |
| `configs/bert_crf.yaml` | BERTimbau + CRF |
| `configs/deberta_linear.yaml` | DeBERTa-V3 + Linear |
| `configs/deberta_crf.yaml` | DeBERTa-V3 + CRF |
| `configs/xlmr_linear.yaml` | XLM-RoBERTa + Linear |
| `configs/xlmr_crf.yaml` | XLM-RoBERTa + CRF ⭐ (Best) |
| `configs/municipality_experiments/xlmr_crf_M0X.yaml` | XLM-R + CRF LOMO configs |

Trained models are saved to `models/MODEL_NAME/EXPERIMENT_NAME/`:
```
models/xlmr_crf/my_run/
├── best_model/              # Best checkpoint by validation F1
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── training_results.json    # Metrics and hyperparameters
└── logs/
```

### Making Predictions

```bash
python scripts/predict.py MODEL_PATH INPUT_JSONL OUTPUT_JSONL [DEVICE]

# Example
python scripts/predict.py \
  models/xlmr_crf/my_run/best_model \
  data/citilink-votie/test.jsonl \
  predictions/test_predictions.jsonl \
  cuda
```

**Input format (span JSONL)**:
```json
{"id": "M01_cm_003_2023-02-01_seg020", "text": "A Câmara deliberou aprovar...", "spans": [...], "municipality": "M01"}
```

**Output format**:
```json
{"id": "M01_cm_003_2023-02-01_seg020", "tokens": ["A", "Câmara", ...], "pred_labels": ["O", "B-VOTER-FAVOR", ...], "gold_labels": ["O", "B-VOTER-FAVOR", ...]}
```

### Running Evaluation

```bash
python scripts/evaluate.py PREDICTIONS_JSONL [OUTPUT_JSON]

# Example
python scripts/evaluate.py \
  predictions/test_predictions.jsonl \
  results/my_experiment_evaluation.json
```

Outputs precision, recall, and F1 (macro-averaged and per entity type) using strict boundary matching via `seqeval`.

### Full Pipeline

Train → predict → evaluate in one command:

```bash
python scripts/run_pipeline.py --config configs/xlmr_crf.yaml --name my_experiment

# Predict or evaluate only
python scripts/run_pipeline.py --config configs/xlmr_crf.yaml --name my_experiment --predict-only
python scripts/run_pipeline.py --config configs/xlmr_crf.yaml --name my_experiment --evaluate-only
```

### LLM Extraction

All three LLM baselines use the langextract library with the same fixed prompts and few-shot examples.

```bash
# Gemini 2.5 Pro (requires GEMINI_KEY)
python scripts/llm_extraction/extract_gemini_spans.py --strategy zero_shot
python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot

# GPT 5.5 via IAEDU API (requires IAEDU_API_KEY)
python scripts/llm_extraction/extract_gpt_spans.py --strategy zero_shot
python scripts/llm_extraction/extract_gpt_spans.py --strategy few_shot

# AMALIA 8B (requires running vLLM endpoint)
python scripts/llm_extraction/extract_amalia_spans.py --strategy zero_shot
python scripts/llm_extraction/extract_amalia_spans.py --strategy few_shot

# Evaluate any LLM predictions
python scripts/llm_extraction/evaluate_spans.py \
  results/llm_extraction/gemini/few_shot.jsonl
```

### LOMO Experiments

```bash
# Run all leave-one-municipality-out experiments
python scripts/run_municipality_experiments.py

# Run single municipality
python scripts/run_pipeline.py \
  --config configs/municipality_experiments/xlmr_crf_M01.yaml \
  --name lomo_M01
```

### Advanced Usage

**Custom configuration**: copy any YAML from `configs/`, edit fields, and pass with `--config`. Key settings:

```yaml
model:
  name: "xlmr_crf"
  architecture: "crf"   # "crf" or "linear"

data:
  data_dir: "data/citilink-votie"
  train_file: "train.jsonl"

training:
  learning_rate: 5e-5
  batch_size: 16
  patience: 3
  seed: 42
```

---

## 6. Dataset

> **📦 Dataset included**
>
> The training-ready dataset (`data/citilink-votie/`) is pre-generated and included in this repository. The raw source annotations are in `data/citilink-dataset/` and also available at:
>
> ### **https://rdm.inesctec.pt/dataset/cs-2025-007**
>
> Run `python scripts/quick_start.py` to verify your installation without downloading the full dataset.

### Overview

The CitiLink-Minutes corpus consists of council meeting minutes from six Portuguese municipalities, annotated for 12 voting-related entity types. A subset of 2,885 segments were annotated, totalling 10,694 spans. All personal identifiers were manually anonymised.

### Dataset Statistics (before oversampling)

| Attribute | Value |
|-----------|-------|
| **Language** | Portuguese |
| **Municipalities** | 6 (Alandroal, Campo Maior, Covilhã, Fundão, Guimarães, Porto) |
| **Total segments** | 2,885 |
| **w/ voting event** | 2,516 (87.2%) |
| **Total spans** | 10,694 |
| **Train** | 1,803 segments (→ 2,463 after oversampling) |
| **Dev** | 553 segments |
| **Test** | 529 segments |

### Data Format

The canonical format is **span-annotated JSONL**. BIO tags are computed automatically at load time — no pre-tokenized files are needed or distributed.

```json
{
  "id": "M01_cm_003_2023-02-01_seg020",
  "text": "O Executivo Municipal deliberou por unanimidade aprovar...",
  "spans": [
    {"text": "O Executivo Municipal", "type": "VOTER-FAVOR", "start": 0, "end": 21},
    {"text": "deliberou", "type": "VOTING", "start": 22, "end": 31},
    {"text": "por unanimidade", "type": "COUNTING-UNANIMITY", "start": 32, "end": 47}
  ],
  "municipality": "M01",
  "document_id": "M01_cm_003_2023-02-01",
  "segment_id": "M01_cm_003_2023-02-01_seg020"
}
```

### Entity Distribution (before oversampling)

| Entity Type | Train | Dev | Test | Total |
|-------------|-------|-----|------|-------|
| VOTING | 1,749 | 524 | 489 | 2,762 |
| SUBJECT | 1,730 | 523 | 486 | 2,739 |
| VOTER-FAVOR | 1,085 | 336 | 319 | 1,740 |
| COUNTING-UNANIMITY | 1,008 | 313 | 326 | 1,647 |
| VOTER-ABSTENTION | 747 | 221 | 135 | 1,103 |
| COUNTING-MAJORITY | 162 | 58 | 59 | 279 |
| VOTER-AGAINST | 99 | 60 | 36 | 195 |
| VOTER-ABSENT | 61 | 24 | 18 | 103 |
| VOTING-METHOD | 39 | 1 | 5 | 45 |
| COUNT-BLANK | 39 | 1 | 5 | 45 |
| COUNT-FAVOR | 32 | 0 | 4 | 36 |
| COUNT-AGAINST | 2 | 0 | 0 | 2 |
| **Total** | **6,753** | **2,059** | **1,882** | **10,694** |

### Directory Structure

```
data/
├── split_info.json                        # Document-level split assignments ✅
├── DATASETS.md                            # Full pipeline documentation ✅
│
├── citilink-dataset/                      # Raw source annotations ✅ (from RDM)
│   ├── Alandroal.json / Campomaior.json / ... / Porto.json
│   └── all_municipalities.json
│
├── citilink-votie/                        # Training-ready dataset ✅ (pre-generated)
│   ├── train.jsonl                        # 2,463 segments (with oversampling)
│   ├── dev.jsonl                          # 553 segments
│   ├── test.jsonl                         # 529 segments
│   └── generation_summary.json
│
└── citilink-votie-lomo-splits/            # LOMO splits (generate locally — 67 MB)
    └── M01/ … M06/                        # run create_lomo_splits_v6.py
```

To generate the LOMO splits: `python scripts/dataset_generation/create_lomo_splits_v6.py`

---

## 7. Architecture

### Data Flow

```
JSONL (span format)
      │
      ▼
src/data/dataset.py          ← auto-detects span format
      │  span_to_bio.py       ← regex tokenizer + BIO alignment
      ▼
Token sequences + BIO labels
      │
      ├── [Transformer models]   → src/trainer.py         → models/{xlmr,deberta,bertimbau}_models.py
      └── [BiLSTM model]         → src/bilstm_trainer.py  → src/models/bilstm_crf.py
                                                             (FastText embeddings)
      │
      ▼
BIO predictions → seqeval entity-level evaluation
```

Key implementation details:
- **Class imbalance**: O-tag loss weight reduced to 0.01 to handle severe class imbalance
- **Windowing**: Documents >512 tokens are split into 50-token overlapping windows; windows are merged after inference
- **Base class**: `src/models/base.py` (`BaseVotIEModel`) provides shared weighted loss, BIO validation, and bias initialization

### Model Families

| Model | File | Notes |
|-------|------|-------|
| CRF | `src/models/crf.py` | Hand-crafted features, sklearn-crfsuite |
| BiLSTM+FastText+CRF | `src/models/bilstm_crf.py` | Requires `cc.pt.300.bin` |
| BERTimbau + Linear/CRF | `src/models/bertimbau_models.py` | Portuguese BERT |
| mDeBERTa-V3 + Linear/CRF | `src/models/deberta_models.py` | |
| XLM-RoBERTa + Linear/CRF | `src/models/xlmr_models.py` | Best model ⭐ |

### LLM Extraction Module

`src/llm_extraction/` is self-contained and provider-agnostic:
- `shared/base_extractor.py` – Base class for all extractors
- `gemini/extractor.py` – Gemini 2.5 Pro via langextract `GeminiLanguageModel`
- `gpt/extractor.py` – GPT 5.5 via langextract custom `GptLanguageModel`
- `amalia/extractor.py` – AMALIA 8B via langextract + vLLM OpenAI-compatible endpoint
- `amalia/prompts.py` – Shared prompt description (system instructions + taxonomy) used by all three extractors
- `fixed_examples.py` – Five fixed few-shot examples shared by all extractors (Appendix A.2)
- `span_alignment.py` – Post-processing to align LLM output spans to input text

All three generative baselines use the langextract library (Goel, 2025) with greedy generation and schema-constrained decoding for deterministic output.

### Dataset Generation Pipeline

For reproducibility, the raw Citilink annotation files can be regenerated from `private/citilink/`:

```bash
# Full 2-step pipeline
python scripts/dataset_generation/generate_dataset.py

# Step 1: Convert raw Citilink JSON → span JSONL
python scripts/dataset_generation/convert_citilink_to_spans.py

# Step 2: Create LOMO splits
python scripts/dataset_generation/create_lomo_splits.py

# Optional: pre-compute BIO tokens (not required for training)
python scripts/dataset_generation/convert_spans_to_bio.py
```

---

## 8. Evaluation Metrics

All evaluation uses **entity-level F1** via the `seqeval` library with strict boundary matching.

### Entity-Level F1

An entity is counted as correct only if both its **type** and **exact character span** match the gold annotation. Partial matches are not credited.

- **Precision**: correct / predicted
- **Recall**: correct / gold
- **F1**: harmonic mean of precision and recall
- **Macro F1**: unweighted average across 11 entity types (COUNT-AGAINST is excluded — zero test instances)

### Computing Metrics

```python
from src.evaluation.entity_metrics import compute_entity_metrics

metrics = compute_entity_metrics(
    gold_file="data/citilink-votie/test.jsonl",
    pred_file="predictions/test_predictions.jsonl"
)
print(f"Macro F1: {metrics['macro_f1']:.1f}%")
```

Or via CLI:
```bash
python scripts/evaluate.py predictions/test_predictions.jsonl results/my_experiment_evaluation.json
```

---

## 9. Experimental Results

### Standard Benchmark (Test Set)

Macro-averaged F1 (%). For encoder-CRF models, values are means over seeds 42, 13, 123. All other rows report seed 42 only.

| Model | Exact Match F1 | Relaxed Match F1 |
|-------|---------------|-----------------|
| CRF | 81.0 | 84.1 |
| BiLSTM-CRF | 69.7 | 74.6 |
| BERTimbau-Linear | 85.2 | 91.0 |
| BERTimbau-CRF | 90.5 ±1.5 | 96.2 ±0.8 |
| DeBERTa-Linear | 89.8 | 98.3 |
| DeBERTa-CRF | 91.5 ±1.3 | 96.1 ±1.2 |
| XLM-R-Linear | 85.2 | 91.0 |
| **XLM-R-CRF** ⭐ | **93.2 ±1.5** | **97.0 ±0.8** |
| GPT 5.5 (0-shot) | 52.6 | 84.0 |
| GPT 5.5 (5-shot) | 60.2 | 87.0 |
| AMALIA 8B (0-shot) | 43.3 | 46.0 |
| AMALIA 8B (5-shot) | 46.0 | 58.2 |
| Gemini 2.5 (0-shot) | 58.7 | 87.0 |
| Gemini 2.5 (5-shot) | **61.3** | **89.3** |

### Cross-Municipality Generalisation (LOMO, Exact Match Macro F1 %)

Models trained on 5 municipalities, evaluated on the held-out sixth.

| Model | M01 | M02 | M03 | M04 | M05 | M06 | Mean |
|-------|-----|-----|-----|-----|-----|-----|------|
| BERTimbau-CRF | 56.2 | 44.4 | 54.9 | 46.1 | 51.8 | 33.4 | 47.8 |
| XLM-R-CRF | 57.8 | 59.1 | 61.4 | 64.5 | 57.0 | 28.2 | 54.7 |
| DeBERTa-CRF | 57.4 | **73.7** | 62.2 | **70.3** | 59.7 | 31.1 | 59.1 |
| Gemini 2.5 (5s) | **66.8** | 69.0 | **69.7** | 62.0 | **76.0** | **45.4** | **64.8** |

XLM-R and DeBERTa values are 3-seed means. Full per-entity results are in `results/`.

Full per-entity F1 breakdown is in `results/discriminative_models/` and `results/llm_extraction/`.

---

## 10. Known Issues

### Current Limitations

1. **LOMO performance degradation**: Cross-municipality F1 drops from 93.2% (in-domain) to 52.8% (LOMO), particularly severe for M06 (Porto), indicating significant domain shift. Gemini few-shot outperforms discriminative models in this cross-domain setting (+12% average).
   - **Future work**: Domain adaptation, larger municipality-specific fine-tuning data.

3. **LLM extraction cost**: Gemini experiments require a Google API key and incur per-token costs. Few-shot extraction processes ~2,879 examples × 5 examples context per call. AMALIA is not yet distributed to the public, so these experiments cannot be reproduced yet.

4. **SUBJECT entity difficulty**: SUBJECT is the hardest entity type across all models (XLM-R+CRF: 78.7% F1 vs 97.8% for VOTER-ABSTENTION), reflecting the challenges of extracting long spans with different structures.

### Reporting Issues

Please open an issue on GitHub with:
- Python version and OS
- Full error traceback
- Minimal reproducible example

---

## 11. License

This project is licensed under **CC-BY-ND 4.0 (Creative Commons Attribution–NoDerivatives 4.0 International)**.

You are free to:
- **Share**: Copy and redistribute the material in any medium or format

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **No Derivatives**: If you remix, transform, or build upon the material, you may not distribute the modified version

See the [LICENSE](LICENSE) file for full details.

---

## 12. Resources

### Pre-trained Models

- **XLM-RoBERTa + CRF** (best model): [Anonymous3445/XLM-RoBERTa-CRF-VotIE](https://huggingface.co/Anonymous3445/XLM-RoBERTa-CRF-VotIE)
- **Interactive demo**: [VotIE-demo on Hugging Face Spaces](https://huggingface.co/spaces/Anonymous3445/VotIE-demo)

### Dataset

- **Full Citilink dataset**: [rdm.inesctec.pt/dataset/cs-2025-007](https://rdm.inesctec.pt/dataset/cs-2025-007) — span-annotated JSONL, 2,879 voting segments from 6 municipalities

### External Resources

- **FacebookAI/xlm-roberta-base** and **xlm-roberta-large**: [Hugging Face](https://huggingface.co/FacebookAI)
- **microsoft/deberta-v3-base**: [Hugging Face](https://huggingface.co/microsoft/deberta-v3-base)
- **neuralmind/bert-large-portuguese-cased** (BERTimbau): [Hugging Face](https://huggingface.co/neuralmind/bert-large-portuguese-cased)
- **Portuguese FastText embeddings** (`cc.pt.300.bin`): [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html)

---

## 13. Acknowledgments

- The [seqeval](https://github.com/chakki-works/seqeval) project for sequence labeling evaluation
- [Hugging Face](https://huggingface.co/) for the Transformers library and model/demo hosting
- [Neuralmind](https://neuralmind.ai/) for BERTimbau pre-trained models
- Google for Gemini API access
- INESCTEC for dataset hosting via RDM

---

## 14. Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{votie2026emnlp,
  title     = {VotIE: Information Extraction from Meeting Minutes},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```

---

## Appendix: Repository Structure

```
citilink/
├── configs/                               # YAML model configurations
│   ├── crf.yaml
│   ├── bilstm_fasttext.yaml
│   ├── bert_linear.yaml  / bert_crf.yaml
│   ├── deberta_linear.yaml / deberta_crf.yaml
│   ├── xlmr_linear.yaml  / xlmr_crf.yaml  
│   └── municipality_experiments/
│       └── xlmr_crf_M01.yaml … xlmr_crf_M06.yaml
│
├── data/
│   ├── citilink-dataset/                  # Raw source annotations 
│   ├── citilink-votie/                    # Training-ready dataset 
│   │   ├── train.jsonl / dev.jsonl / test.jsonl
│   │   └── generation_summary.json
│   ├── citilink-votie-lomo-splits/        # LOMO splits (generate locally)
│   └── split_info.json                    # Document-level split assignments 
│
├── scripts/
│   ├── dataset_generation/                # Data pipeline
│   │   ├── convert_new_citilink_to_spans.py   # Raw → intermediate span JSONL
│   │   ├── generate_v6_multi_vote_aware.py    # → citilink-votie/ (with oversampling)
│   │   ├── create_lomo_splits_v6.py           # → citilink-votie-lomo-splits/
│   │   └── convert_spans_to_bio.py            # [Optional] pre-tokenize BIO
│   ├── train.py                           # Unified training
│   ├── predict.py                         # Inference
│   ├── evaluate.py                        # Evaluation metrics
│   ├── run_pipeline.py                    # train → predict → evaluate
│   ├── run_standard_benchmark.sh          # Run all model configs
│   ├── run_municipality_experiments.py    # LOMO experiments
│   ├── aggregate_seeds.py                 # Multi-seed result aggregation
│   ├── statistical_tests.py              # Significance testing
│   ├── error_classification.py            # Error analysis (§6)
│   ├── quick_start.py                     # Demo script
│   └── llm_extraction/                    # LLM scripts
│       ├── extract_gemini_spans.py        # Gemini 2.5 Pro extraction
│       ├── extract_gpt_spans.py           # GPT 5.5 extraction
│       ├── extract_amalia_spans.py        # AMALIA 8B extraction
│       ├── evaluate_spans.py              # Evaluate any LLM predictions
│       └── run_cross_municipality.py      # LOMO for Gemini
│
├── src/
│   ├── models/                            # Model implementations
│   │   ├── base.py                        # Shared base class
│   │   ├── crf.py / bilstm_crf.py
│   │   ├── bertimbau_models.py
│   │   ├── deberta_models.py
│   │   └── xlmr_models.py
│   ├── data/
│   │   ├── dataset.py                     # Data loading + auto span→BIO conversion
│   │   ├── span_to_bio.py                 # Regex tokenizer + BIO alignment
│   │   └── postprocessing.py
│   ├── evaluation/
│   │   └── entity_metrics.py
│   ├── llm_extraction/                    # LLM extraction module
│   │   ├── gemini/                        # GeminiSpanExtractor
│   │   ├── gpt/                           # GptSpanExtractor + GptLanguageModel provider
│   │   ├── amalia/                        # AmaliaSpanExtractor + vllm_openai.py + prompts.py
│   │   ├── shared/                        # base_extractor.py, data_utils.py, evaluation.py
│   │   ├── fixed_examples.py             # Fixed few-shot examples shared by all extractors
│   │   └── span_alignment.py             # Post-processing span alignment
│   ├── trainer.py                         # Transformer training loop
│   └── bilstm_trainer.py                  # BiLSTM training loop
│
├── results/                               # All experimental results (tracked)
│   ├── discriminative_models/
│   │   └── {model}/
│   │       └── seeds/                     # Predictions + evaluations per seed
│   │           ├── seed_s42_predictions.jsonl    # Deucalion run (canonical)
│   │           ├── seed_s42_evaluation.json
│   │           ├── seed_s13_predictions.jsonl    # bert_crf, deberta_crf, xlmr_crf only
│   │           ├── seed_s13_evaluation.json
│   │           ├── seed_s123_predictions.jsonl
│   │           └── seed_s123_evaluation.json
│   ├── lomo/
│   │   ├── bert_crf/    M{01-06}_{predictions.jsonl,evaluation.json}
│   │   ├── deberta_crf/ M{01-06} × {seed_s42,seed_s13,seed_s123} × {predictions,evaluation}
│   │   └── xlmr_crf/    M{01-06} × available seeds × {predictions,evaluation}
│   ├── llm_extraction/
│   │   ├── gemini/      {zero,few}_shot.jsonl + _evaluation.json
│   │   │   └── lomo/    M{01-06}.jsonl + M{01-06}_evaluation.json
│   │   ├── gpt5.5/      {zero,few}_shot.jsonl + _evaluation.json
│   │   └── amalia/      {zero,few}_shot.jsonl + _evaluation.json
│   ├── seed_aggregate_summary.json        # Aggregated multi-seed results
│   ├── statistical_tests.json             # Bootstrap significance tests
│   └── error_classification_{report,results}.{md,json}
│
├── evaluation/                            # Runtime output dir (gitignored)
├── models/                                # Trained model checkpoints (gitignored)
├── requirements.txt
├── README.md
└── LICENSE
```

---

**Last Updated**: February 2026
