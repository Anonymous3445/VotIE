#!/usr/bin/env python
"""
Gemini Event Extraction - Direct Event-Based Extraction

Run event extraction on Portuguese test dataset and evaluate with event-level metrics.

Usage:
    python run_gemini_events.py
    python run_gemini_events.py --test  # Test mode (10 documents)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).parent))

from shared.model import GeminiEventExtractor
from shared.prompts import get_extraction_config
from evaluation_events import evaluate_gemini_events
from config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)


def load_events_dataset(events_file: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from votie_events JSON or JSONL file

    Args:
        events_file: Path to events .json or .jsonl file (train/dev/test)

    Returns:
        List of event dictionaries
    """
    logger.info(f"Loading events dataset from: {events_file}")

    # Handle both .json and .jsonl formats
    if events_file.suffix == '.json':
        # Load as JSON array
        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
    else:
        # Load as JSONL (line-by-line)
        events = []
        with open(events_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line.strip())
                    events.append(event)

    logger.info(f"Loaded {len(events)} events")
    return events


def extract_events_with_gemini(documents: List[Dict[str, Any]],
                               model: GeminiEventExtractor,
                               config: Dict[str, Any],
                               max_documents: int = None) -> List[Dict[str, Any]]:
    """
    Extract events from documents using Gemini

    Args:
        documents: List of documents with text
        model: GeminiEventExtractor instance
        config: Extraction configuration
        max_documents: Maximum number of documents to process

    Returns:
        List of predicted event dictionaries
    """
    logger.info(f"Starting event extraction with {model.model_id}")

    predictions = []
    total_docs = len(documents) if max_documents is None else min(max_documents, len(documents))

    start_time = time.time()

    for i, doc in enumerate(documents[:total_docs]):
        doc_id = doc['id']
        text = doc['text']

        logger.info(f"Processing document {i+1}/{total_docs}: {doc_id}")

        # Extract event using Gemini
        result = model.extract(
            text=text,
            system_prompt=config['system_prompt'],
            examples=config['examples'],
            document_id=doc_id
        )

        if result.success:
            # Create event dict in expected format
            pred_event = {
                "id": doc_id,
                "has_voting_event": result.has_voting_event,
                "event": result.event
            }
            predictions.append(pred_event)
            logger.info(f"✓ Extracted event from {doc_id} (has_event={result.has_voting_event})")
        else:
            logger.error(f"✗ Failed to extract from {doc_id}: {result.error}")
            # Add failed prediction
            predictions.append({
                "id": doc_id,
                "has_voting_event": False,
                "event": None,
                "error": result.error
            })

        # Rate limiting - small delay between requests
        if i < total_docs - 1:
            time.sleep(1)

    elapsed_time = time.time() - start_time
    docs_per_sec = total_docs / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Extraction completed: {total_docs} documents in {elapsed_time:.2f}s ({docs_per_sec:.2f} docs/sec)")

    return predictions


def save_results(predictions: List[Dict[str, Any]],
                evaluation_results: Dict[str, Any],
                output_dir: Path,
                model_id: str):
    """
    Save predictions and evaluation results

    Args:
        predictions: List of predicted events
        evaluation_results: Evaluation metrics
        output_dir: Output directory
        model_id: Model identifier
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_path = output_dir / "predictions_events.jsonl"
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    logger.info(f"Saved predictions to: {predictions_path}")

    # Save evaluation results
    eval_path = output_dir / "evaluation_results_events.json"
    results_dict = {
        "metadata": {
            "model": model_id,
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(predictions),
            "evaluation_method": "event_extraction"
        },
        "event_level_metrics": evaluation_results["event_level"]
    }

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation results to: {eval_path}")

    # Save markdown summary
    md_path = output_dir / "evaluation_summary_events.md"
    summary = f"""# Gemini Event Extraction Evaluation Report

## Model Information
- **Model**: {model_id}
- **Language**: Portuguese
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Evaluation Method**: Event-based extraction

## Event-Level Performance

| Metric | Value |
|--------|-------|
| **Precision** | {evaluation_results['event_level']['precision']:.4f} |
| **Recall** | {evaluation_results['event_level']['recall']:.4f} |
| **F1-Score** | {evaluation_results['event_level']['f1']:.4f} |
| **Correct** | {evaluation_results['event_level']['correct']} |
| **Partial** | {evaluation_results['event_level']['partial']} |
| **Missed** | {evaluation_results['event_level']['missed']} |
| **Spurious** | {evaluation_results['event_level']['spurious']} |

## Interpretation

- **Event-level metrics** measure how well the model extracts complete voting events with all components
- **Correct**: Events where all components match perfectly
- **Partial**: Events where some components match
- **Missed**: Gold events that were not detected
- **Spurious**: Predicted events that are false positives
- Higher F1 scores indicate better structured extraction of complete voting information
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    logger.info(f"Saved summary to: {md_path}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Gemini Event Extraction for Portuguese Voting')
    parser.add_argument('--test', action='store_true', help='Run in test mode (10 documents only)')
    parser.add_argument('--max-docs', type=int, default=None, help='Maximum number of documents to process')
    args = parser.parse_args()

    # Setup paths - go from src/gemini/run_gemini_events.py to repository root
    # Path(__file__).parent -> src/gemini
    # .parent -> src
    # .parent -> VotIE (repository root)
    project_root = Path(__file__).parent.parent.parent
    
    # Try both .json and .jsonl formats
    test_file_json = project_root / "data/votie_events/test.json"
    test_file_jsonl = project_root / "data/votie_events/test.jsonl"
    
    if test_file_json.exists():
        test_file = test_file_json
    elif test_file_jsonl.exists():
        test_file = test_file_jsonl
    else:
        logger.error(f"Test file not found: tried {test_file_json} and {test_file_jsonl}")
        return
    
    output_dir = Path(__file__).parent / "experiments" / f"gemini_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Load test dataset
    test_events = load_events_dataset(test_file)

    # Determine number of documents to process
    if args.test:
        max_docs = 10
        logger.info("Running in TEST MODE - processing 10 documents")
    elif args.max_docs:
        max_docs = args.max_docs
        logger.info(f"Processing {max_docs} documents")
    else:
        max_docs = len(test_events)
        logger.info(f"Running FULL EVALUATION - processing all {max_docs} documents")

    test_events = test_events[:max_docs]

    # Load configuration
    cfg = get_config()

    if not cfg.api_key:
        logger.error("No API key found. Set GOOGLE_API_KEY or LANGEXTRACT_API_KEY environment variable.")
        return

    # Create model
    logger.info(f"Initializing Gemini model: {cfg.model_id}")
    try:
        model = GeminiEventExtractor(model_id=cfg.model_id, api_key=cfg.api_key)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    # Get extraction configuration
    config = get_extraction_config()

    # Extract events
    extraction_start_time = time.time()
    predictions = extract_events_with_gemini(test_events, model, config, max_documents=max_docs)
    total_extraction_time = time.time() - extraction_start_time

    # Prepare for evaluation
    gold_events = [
        {
            "has_voting_event": event.get("has_voting_event", False),
            "event": event.get("event", None)
        }
        for event in test_events
    ]

    # Evaluate
    logger.info("Evaluating predictions...")
    evaluation_results = evaluate_gemini_events(
        test_data=test_events,
        predictions=predictions,
        gold_events=gold_events
    )

    # Print results
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_id}")
    logger.info(f"Documents processed: {max_docs}")
    logger.info(f"Total extraction time: {total_extraction_time:.2f}s")
    logger.info(f"Avg time per document: {total_extraction_time / max_docs:.2f}s")
    logger.info("=" * 80)
    logger.info("EVENT-LEVEL METRICS:")
    logger.info(f"  Precision: {evaluation_results['event_level']['precision']:.4f}")
    logger.info(f"  Recall:    {evaluation_results['event_level']['recall']:.4f}")
    logger.info(f"  F1-Score:  {evaluation_results['event_level']['f1']:.4f}")
    logger.info(f"  Correct:   {evaluation_results['event_level']['correct']}")
    logger.info(f"  Partial:   {evaluation_results['event_level']['partial']}")
    logger.info(f"  Missed:    {evaluation_results['event_level']['missed']}")
    logger.info(f"  Spurious:  {evaluation_results['event_level']['spurious']}")
    logger.info("=" * 80)

    # Save results
    save_results(predictions, evaluation_results, output_dir, cfg.model_id)

    logger.info(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
