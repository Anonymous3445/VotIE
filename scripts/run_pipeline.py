#!/usr/bin/env python3
"""
Complete ML Pipeline Orchestrator

This script chains training → prediction → evaluation for a complete ML pipeline
with proper separation of concerns.

Usage:
    python scripts/run_pipeline.py --config configs/main_experiment/bert_crf.yaml --name my_experiment
    
    # Skip training if model already exists
    python scripts/run_pipeline.py --config configs/main_experiment/bert_crf.yaml --name my_experiment --predict-only
    
    # Skip training and prediction if predictions already exist
    python scripts/run_pipeline.py --config configs/main_experiment/bert_crf.yaml --name my_experiment --evaluate-only
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_training(config_path: str, baseline_name: str, experiment_name: str) -> Path:
    """Run training and return model directory."""
    print("\n" + "="*80)
    print("🚀 STEP 1: TRAINING")
    print("="*80)
    
    from scripts.train import train_crf, train_bilstm, train_transformer
    
    if baseline_name == "crf":
        model_dir = train_crf(baseline_name, config_path, experiment_name)
    elif baseline_name == "bilstm_fasttext":
        model_dir = train_bilstm(baseline_name, config_path, experiment_name)
    else:
        model_dir = train_transformer(baseline_name, config_path, experiment_name)
    
    print(f"✅ Training completed. Model saved to: {model_dir}")
    return model_dir

def run_prediction(model_dir: Path, test_data_path: str, predictions_file: Path) -> None:
    """Run prediction using the trained model."""
    print("\n" + "="*80) 
    print("🔮 STEP 2: PREDICTION")
    print("="*80)
    
    from scripts.predict import predict
    
    predict(
        model_path=str(model_dir),
        test_data_path=test_data_path,
        output_path=str(predictions_file)
    )
    
    print(f"✅ Predictions saved to: {predictions_file}")

def run_evaluation(predictions_file: Path, results_file: Path) -> dict:
    """Run evaluation using the predictions."""
    print("\n" + "="*80)
    print("📊 STEP 3: EVALUATION") 
    print("="*80)
    
    from scripts.evaluate import evaluate
    
    results = evaluate(
        predictions_path=str(predictions_file),
        output_path=str(results_file)
    )
    
    print(f"✅ Evaluation results saved to: {results_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Run complete ML pipeline: train → predict → evaluate')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--baseline', help='Override baseline name (default: derived from config file)')
    parser.add_argument('--predict-only', action='store_true', help='Skip training, start from prediction')
    parser.add_argument('--evaluate-only', action='store_true', help='Skip training and prediction, only evaluate')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine baseline name
    if args.baseline:
        baseline_name = args.baseline
    else:
        # Always use the model name from config to avoid mismatches
        baseline_name = config['model']['name']
    
    experiment_name = args.name
    data_config = config['data']
    
    # Set up paths following the repository structure
    model_dir = Path("models") / baseline_name / experiment_name
    predictions_dir = Path("predictions")
    evaluation_dir = Path("evaluation")

    # Create output directories
    predictions_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    test_data_path = Path(data_config['data_dir']) / data_config['test_file']
    predictions_file = predictions_dir / f"{baseline_name}_{experiment_name}.jsonl"
    results_file = evaluation_dir / f"{baseline_name}_{experiment_name}.json"
    
    print(f"🎯 Running pipeline for {baseline_name} experiment '{experiment_name}'")
    print(f"   Config: {args.config}")
    print(f"   Model dir: {model_dir}")
    print(f"   Test data: {test_data_path}")
    print(f"   Predictions: {predictions_file}")
    print(f"   Results: {results_file}")
    
    pipeline_start = datetime.now()
    
    try:
        # Step 1: Training
        if not args.predict_only and not args.evaluate_only:
            if model_dir.exists() and not args.force:
                print(f"⚠️  Model directory already exists: {model_dir}")
                print("   Use --force to overwrite or --predict-only to skip training")
                sys.exit(1)
            
            model_dir = run_training(args.config, baseline_name, experiment_name)
        else:
            if not model_dir.exists():
                print(f"❌ Model directory not found: {model_dir}")
                print("   Run without --predict-only to train the model first")
                sys.exit(1)
            print(f"⏭️  Skipping training. Using existing model: {model_dir}")
        
        # Step 2: Prediction
        if not args.evaluate_only:
            if predictions_file.exists() and not args.force:
                print(f"⚠️  Predictions file already exists: {predictions_file}")
                print("   Use --force to overwrite or --evaluate-only to skip prediction")
                sys.exit(1)
            
            run_prediction(model_dir, str(test_data_path), predictions_file)
        else:
            if not predictions_file.exists():
                print(f"❌ Predictions file not found: {predictions_file}")
                print("   Run without --evaluate-only to generate predictions first")
                sys.exit(1)
            print(f"⏭️  Skipping prediction. Using existing predictions: {predictions_file}")
        
        # Step 3: Evaluation
        if results_file.exists() and not args.force:
            print(f"⚠️  Results file already exists: {results_file}")
            print("   Use --force to overwrite")
            if not args.force:
                sys.exit(1)
        
        results = run_evaluation(predictions_file, results_file)
        
        # Pipeline summary
        pipeline_time = (datetime.now() - pipeline_start).total_seconds()
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"📁 Model: {model_dir}")
        print(f"🔮 Predictions: {predictions_file}")
        print(f"📊 Results: {results_file}")
        print(f"⏱️  Total time: {pipeline_time:.2f}s")
        entity_f1 = results.get('entity_level_metrics', {}).get('entity_f1', 0.0)
        print(f"🎯 Test Entity F1: {entity_f1:.4f}")

        # Save pipeline metadata
        metadata_file = model_dir / "pipeline_metadata.json"
        pipeline_metadata = {
            'experiment_name': experiment_name,
            'baseline_name': baseline_name,
            'config_file': args.config,
            'pipeline_time': pipeline_time,
            'completed_at': datetime.now().isoformat(),
            'files': {
                'model_dir': str(model_dir),
                'predictions': str(predictions_file),
                'evaluation': str(results_file)
            },
            'final_results': results
        }

        with open(metadata_file, 'w') as f:
            json.dump(pipeline_metadata, f, indent=2)

        print(f"📋 Pipeline metadata saved to: {metadata_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()