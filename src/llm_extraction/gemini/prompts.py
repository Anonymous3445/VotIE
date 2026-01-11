"""
Prompts for Gemini span extraction.
"""

from typing import List, Dict, Any


SYSTEM_PROMPT_ZERO_SHOT = """You are an expert Portuguese NLP system extracting semantic spans from municipal meeting minutes.

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
"""


SYSTEM_PROMPT_FEW_SHOT = """You are an expert Portuguese NLP system extracting named entities from municipal meeting minutes.

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
"""


def build_zero_shot_prompt(text: str) -> str:
    """
    Build zero-shot prompt for Gemini.

    Args:
        text: Source text to extract from

    Returns:
        Formatted prompt
    """
    return f"""{SYSTEM_PROMPT_ZERO_SHOT}

Text:
{text}

Extract entities as JSON:"""


def build_few_shot_prompt(text: str, examples: List[Dict[str, Any]]) -> str:
    """
    Build few-shot prompt for Gemini.

    Args:
        text: Source text to extract from
        examples: List of example dictionaries with 'text' and 'output'

    Returns:
        Formatted prompt
    """
    prompt_parts = [SYSTEM_PROMPT_FEW_SHOT]

    # Add examples
    for i, example in enumerate(examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Text: {example['text']}")
        prompt_parts.append(f"Output: {example['output']}")

    # Add task
    prompt_parts.append(f"\n\nNow extract from the following text:")
    prompt_parts.append(f"Text: {text}")
    prompt_parts.append("\nOutput:")

    return "\n".join(prompt_parts)
