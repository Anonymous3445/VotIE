#!/bin/bash
# Simple script to run all VotIE experiments sequentially
# For parallel execution across GPUs, use run_all_experiments.sh

set -e

EXPERIMENT_NAME="${1:-paper_reproduction}"

echo "=========================================="
echo "VotIE: Running All Experiments"
echo "Experiment name: $EXPERIMENT_NAME"
echo "=========================================="
echo ""

# List of all configurations
CONFIGS=(
    "configs/crf.yaml"
    "configs/bilstm_fasttext.yaml"
    "configs/bert_linear.yaml"
    "configs/bert_crf.yaml"
    "configs/deberta_linear.yaml"
    "configs/deberta_crf.yaml"
    "configs/xlmr_linear.yaml"
    "configs/xlmr_crf.yaml"
)

# Run each configuration
total=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    num=$((i+1))

    echo ""
    echo "[$num/$total] Running: $config"
    echo "------------------------------------------"

    python scripts/run_pipeline.py \
        --config "$config" \
        --name "$EXPERIMENT_NAME" \
        --force

    echo "✓ Completed: $config"
done

echo ""
echo "=========================================="
echo "✓ All experiments completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Models:      models/*/paper_reproduction/"
echo "  - Predictions: predictions/*_paper_reproduction.jsonl"
echo "  - Evaluation:  evaluation/*_paper_reproduction.json"
echo ""
