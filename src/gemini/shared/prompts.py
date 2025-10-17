"""
Portuguese Voting Event Extraction Prompts for Gemini 2.0 Flash

Event-based extraction using structured JSON output.
"""

import json
from typing import List, Dict, Any

# System prompt for event extraction
SYSTEM_PROMPT = """You are an expert at extracting structured information from Portuguese municipal meeting minutes.

Your task is to identify and extract voting events from city council meeting texts. A voting event occurs when the council formally votes on a proposal or decision.

You must extract:
1. Subject: What is being voted on
2. Participants: Who voted and their positions (Favor/Against/Abstention/Absent)
3. Counting: Whether the vote was by majority or unanimity
4. Voting expression: The phrase indicating a vote occurred

Output Format: Return ONLY a valid JSON object with this exact structure:
{
  "has_voting_event": boolean,
  "event": {
    "subject": string or null,
    "participants": [
      {"text": string, "position": "Favor"|"Against"|"Abstention"|"Absent"}
    ],
    "counting": [{"text": string, "type": "Majority"|"Unanimity"}],
    "voting_expressions": [string]
  }
}

Rules:
- If no voting event exists, set has_voting_event to false and event to null
- Extract exact text spans as they appear in the original
- Position must be one of: Favor, Against, Abstention, Absent
- Type must be one of: Majority, Unanimity
- Return ONLY the JSON object, no additional text or explanation
"""


def load_few_shot_examples() -> List[Dict[str, Any]]:
    """
    Load few-shot examples from selected_event_examples.json

    Returns 10 curated examples covering all entity types from different municipalities.
    """
    import os
    from pathlib import Path

    # File is in the same directory as this script
    examples_file = Path(__file__).parent / "selected_event_examples.json"

    if examples_file.exists():
        with open(examples_file, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            return examples[:10]  # Return first 10

    # Fallback: return empty list if file not found
    return []


def format_example_for_prompt(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a single example for few-shot learning.

    Converts from votie_events format to the desired output format.

    Args:
        example: Example from votie_events/train.jsonl

    Returns:
        Dict with 'input' text and 'output' JSON string
    """
    text = example.get('text', '').strip()

    if not example.get('has_voting_event'):
        output = {
            "has_voting_event": False,
            "event": None
        }
    else:
        event_data = example.get('event', {})

        # Use hybrid format: subject singular, others lists
        subject = event_data.get('subject')
        if subject is None:
            # Handle old format where subject might be in 'subjects' list
            subjects_list = event_data.get('subjects', [])
            subject = subjects_list[0] if subjects_list else None

        voting_expressions = event_data.get('voting_expressions', [])
        if not voting_expressions:
            # Handle old format where it might be singular 'voting_expression'
            single_expr = event_data.get('voting_expression')
            voting_expressions = [single_expr] if single_expr else []

        counting = event_data.get('counting', [])
        if counting and not isinstance(counting, list):
            # Handle old format where counting might be a dict
            counting = [counting]

        # Keep participants as-is (already a list)
        participants = event_data.get('participants', [])

        output = {
            "has_voting_event": True,
            "event": {
                "subject": subject,
                "participants": participants,
                "counting": counting,
                "voting_expressions": voting_expressions
            }
        }

    return {
        "input": text,
        "output": json.dumps(output, ensure_ascii=False, indent=2)
    }


def get_few_shot_prompt() -> str:
    """
    Generate few-shot prompt with examples.

    Returns:
        Complete prompt string with system instructions and examples
    """
    examples = load_few_shot_examples()

    prompt_parts = [SYSTEM_PROMPT]
    prompt_parts.append("\n\n# Examples\n")

    for i, example in enumerate(examples, 1):
        formatted = format_example_for_prompt(example)

        prompt_parts.append(f"\n## Example {i}\n")
        prompt_parts.append(f"\n**Input:**\n{formatted['input'][:500]}...\n")  # Truncate long texts
        prompt_parts.append(f"\n**Output:**\n```json\n{formatted['output']}\n```\n")

    prompt_parts.append("\n\n# Your Task\n")
    prompt_parts.append("Given the input text below, extract the voting event and return ONLY the JSON object as specified.\n")

    return "\n".join(prompt_parts)


def get_extraction_config() -> Dict[str, Any]:
    """
    Get complete extraction configuration for event-based extraction.

    Returns:
        Configuration dictionary with system prompt and examples
    """
    examples = load_few_shot_examples()

    return {
        "system_prompt": SYSTEM_PROMPT,
        "examples": examples,
        "output_format": "json",
        "language": "pt"
    }
