#!/usr/bin/env python3
"""
Unified training script for all NER baselines (traditional + transformers).

The model type is automatically determined from the config file's 'model.name' field.

Supports:
Traditional Baselines:
- crf: Conditional Random Fields
- bilstm_fasttext: BiLSTM with FastText embeddings

Transformer Baselines:
- bert_linear: BERT + Linear layer
- bert_crf: BERT + CRF layer
- deberta_linear: DeBERTa + Linear layer
- deberta_crf: DeBERTa + CRF layer
- xlmr_linear: XLM-RoBERTa + Linear layer
- xlmr_crf: XLM-RoBERTa + CRF layer

Usage:
    python scripts/train.py --config configs/main_experiment/bert_crf.yaml
    python scripts/train.py --config configs/main_experiment/crf.yaml --experiment-name exp1
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add root directory to Python path (we're in scripts/ so go up one level)
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Configure logging to show INFO level messages with clean formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# All available baselines
TRADITIONAL_BASELINES = {
    "crf": "Conditional Random Fields",
    "bilstm_fasttext": "BiLSTM + FastText Embeddings"
}

TRANSFORMER_BASELINES = {
    "bert_linear": "BERT + Linear",
    "bert_crf": "BERT + CRF",
    "deberta_linear": "DeBERTa + Linear",
    "deberta_crf": "DeBERTa + CRF",
    "xlmr_linear": "XLM-RoBERTa + Linear",
    "xlmr_crf": "XLM-RoBERTa + CRF"
}

ALL_BASELINES = {**TRADITIONAL_BASELINES, **TRANSFORMER_BASELINES}


def train_crf(baseline_name: str, config_file: str = None, experiment_name: str = None):
    """Train CRF baseline."""
    from src.models.crf import TraditionalCRFModel
    from src.data.dataset import load_jsonl_file
    import json
    import yaml

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    data_config = config['data']

    # Load data from BIO-tagged JSONL format
    data_dir = Path(data_config['data_dir'])

    train_data = load_jsonl_file(data_dir / data_config['train_file'])
    dev_data = load_jsonl_file(data_dir / data_config['dev_file'])

    print(f"   Train: {len(train_data)}, Dev: {len(dev_data)} examples")

    # Train CRF
    output_dir = Path("models") / baseline_name / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TraditionalCRFModel(
        algorithm=model_config.get('algorithm', 'lbfgs'),
        c1=model_config.get('c1', 0.1),
        c2=model_config.get('c2', 0.1),
        max_iterations=model_config.get('max_iterations', 100),
        all_possible_transitions=model_config.get('all_possible_transitions', True)
    )

    print("\n🚀 Training...")
    start = datetime.now()
    model.train(train_data)
    training_time = (datetime.now() - start).total_seconds()

    print("\n📊 Evaluating on validation set...")
    dev_metrics = model.evaluate(dev_data)

    model.save_model(output_dir)

    results = {
        'model_config': model_config,
        'training_time': training_time,
        'dev_results': dev_metrics
    }

    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Dev F1: {dev_metrics.get('entity_f1_strict', 0.0):.4f}")
    print("💡 Use scripts/predict.py and scripts/evaluate.py to test the model")
    
    return output_dir


def train_bilstm(baseline_name: str, config_file: str = None, experiment_name: str = None):
    """Train BiLSTM+FastText baseline."""
    from src.bilstm_trainer import BiLSTMTrainer
    import yaml

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']

    output_dir = Path("models") / baseline_name / experiment_name

    # Filter out 'name' field from model_config as it's not a BiLSTMTrainer parameter
    model_params_filtered = {k: v for k, v in model_config.items() if k != 'name'}

    trainer = BiLSTMTrainer(
        config=config,
        output_dir=output_dir,
        **model_params_filtered,
        **training_config
    )

    data_dir = Path(data_config['data_dir'])

    results = trainer.train(
        train_file=data_dir / data_config['train_file'],
        val_file=data_dir / data_config['dev_file']
    )

    best_dev_f1 = results.get('best_eval_score', 0.0)

    print(f"\n✅ Best Dev F1: {best_dev_f1:.4f}")
    print("💡 Use scripts/predict.py and scripts/evaluate.py to test the model")
    
    return output_dir


def train_transformer(baseline_name: str, config_file: str = None, experiment_name: str = None):
    """Train transformer baseline (BERT, DeBERTa, XLM-R)."""
    import torch
    import yaml
    from transformers import AutoTokenizer
    from src.trainer import VotIETrainer
    from src.models.bertimbau_models import BertimbauLinearVotIE, BertimbauCRFVotIE
    from src.models.deberta_models import DebertaLinearVotIE, DebertaCRFVotIE
    from src.models.xlmr_models import XLMRLinearVotIE, XLMRCRFVotIE
    from src.data.dataset import load_ner_dataset_with_dynamic_windowing

    MODEL_CLASSES = {
        "bert_linear": BertimbauLinearVotIE,
        "bert_crf": BertimbauCRFVotIE,
        "deberta_linear": DebertaLinearVotIE,
        "deberta_crf": DebertaCRFVotIE,
        "xlmr_linear": XLMRLinearVotIE,
        "xlmr_crf": XLMRCRFVotIE,
    }

    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    windowing_config = config.get('windowing', {})
    augmentation_config = config.get('augmentation', {})

    # Load data
    data_dir = Path(data_config['data_dir'])

    model_name = model_config['model_name']
    train_dataset, dev_dataset, _, label_to_id, id_to_label, tokenizer = load_ner_dataset_with_dynamic_windowing(
        data_dir=data_dir,
        train_file=data_config.get('train_file', 'train.jsonl'),
        dev_file=data_config.get('dev_file', 'dev.jsonl'),
        test_file=data_config.get('test_file', 'test.jsonl'),  # Still needed for label mapping
        tokenizer_name=model_name,
        max_length=augmentation_config.get('max_length', 512),
        overlap_tokens=windowing_config.get('overlap_tokens', 50),
        enable_windowing=windowing_config.get('enable_windowing', True)
    )

    # Create model
    print(f"\n🔧 Creating {baseline_name} model...")
    model_class = MODEL_CLASSES[baseline_name]
    model = model_class(model_name=model_name, num_labels=len(label_to_id))
    print(f"✓ Model created with {len(label_to_id)} labels")

    # Train
    output_dir = Path("models") / baseline_name / experiment_name
    print(f"\n📦 Initializing trainer...")

    trainer = VotIETrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        output_dir=output_dir,
        id_to_label=id_to_label,
        num_train_epochs=int(training_config.get('epochs', 10)),
        train_batch_size=int(training_config.get('batch_size', 16)),
        eval_batch_size=int(training_config.get('batch_size', 16)),
        learning_rate=float(training_config.get('learning_rate', 2e-5)),
        warmup_proportion=float(training_config.get('warmup_proportion', 0.1)),
        max_grad_norm=float(config.get('optimization', {}).get('grad_clip', 1.0)),
        patience=int(training_config.get('patience', 3)),
        seed=int(training_config.get('seed', 42)),
        tokenizer=tokenizer,
        base_model=model_name,
        device=int(training_config.get('device', 0)),
        apply_bio_validation=bool(training_config.get('apply_bio_validation', True))
    )
    print(f"✓ Trainer initialized\n")

    print(f"🚀 Starting training for {int(training_config.get('epochs', 10))} epochs...")
    results = trainer.train()
    
    trainer._save_training_results(results)
    
    print(f"\n✅ Best Dev F1: {results.get('best_eval_score', 0.0):.4f}")
    print("💡 Use scripts/predict.py and scripts/evaluate.py to test the model")
    
    return output_dir


def main(config_file: str, experiment_name: str = None):
    """Main training entry point."""
    import yaml
    from pathlib import Path
    
    # Check if config file exists
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_file}")
        print("\nAvailable configs in configs/main_experiment/:")
        main_exp_dir = Path("configs/main_experiment")
        if main_exp_dir.exists():
            for cfg in sorted(main_exp_dir.glob("*.yaml")):
                print(f"  - {cfg}")
        print("\nExample usage:")
        print("  python scripts/train.py --config configs/main_experiment/bert_crf.yaml")
        sys.exit(1)
    
    # Load config to get baseline name
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    baseline_name = config['model']['name']
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestamp}"

    print(f"🚀 Training {baseline_name} baseline")
    print(f"📝 Config: {config_file}")
    print(f"📝 Experiment: {experiment_name}\n")

    # Validate baseline
    if baseline_name not in ALL_BASELINES:
        print(f"❌ Unknown baseline: {baseline_name}")
        print(f"\nAvailable baselines:")
        print("\nTraditional:")
        for name, desc in TRADITIONAL_BASELINES.items():
            print(f"  - {name}: {desc}")
        print("\nTransformers:")
        for name, desc in TRANSFORMER_BASELINES.items():
            print(f"  - {name}: {desc}")
        sys.exit(1)

    # Route to appropriate trainer
    try:
        output_dir = None
        if baseline_name == "crf":
            output_dir = train_crf(baseline_name, config_file, experiment_name)
        elif baseline_name == "bilstm_fasttext":
            output_dir = train_bilstm(baseline_name, config_file, experiment_name)
        else:
            output_dir = train_transformer(baseline_name, config_file, experiment_name)

        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py --config CONFIG_FILE [--experiment-name NAME]")
        print("\nTraditional Baselines:")
        for name, desc in TRADITIONAL_BASELINES.items():
            print(f"  - {name}: {desc}")
        print("\nTransformer Baselines:")
        for name, desc in TRANSFORMER_BASELINES.items():
            print(f"  - {name}: {desc}")
        print("\nOptions:")
        print("  --config: Path to config file (required)")
        print("  --experiment-name: Name for this experiment run (optional)")
        print("\nExamples:")
        print("  python scripts/train.py --config configs/crf.yaml")
        print("  python scripts/train.py --config configs/bert_linear.yaml --experiment-name exp1")
        print("  python scripts/train.py --config configs/deberta_crf.yaml --experiment-name paper_final")
        print("\nFor full pipeline (train + predict + evaluate), use:")
        print("  python scripts/run_pipeline.py --config configs/deberta_crf.yaml")
        sys.exit(1)

    # Parse arguments
    config_file = None
    experiment_name = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--experiment-name" and i + 1 < len(sys.argv):
            experiment_name = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if not config_file:
        print("❌ Error: --config argument is required")
        print("Usage: python scripts/train.py --config CONFIG_FILE [--experiment-name NAME]")
        sys.exit(1)

    main(config_file, experiment_name)
