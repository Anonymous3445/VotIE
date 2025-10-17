#!/usr/bin/env python
"""
Event Constructor for Voting Event Extraction.

This module constructs structured voting events from BIO-tagged predictions.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EventConstructor:
    """
    Constructs structured voting events from BIO-tagged predictions: stage 2 of the VotIE pipeline. 

    Event Structure (PAPER-ALIGNED FORMAT):
    {
        'id': str,
        'has_voting_event': bool,
        'event': {
            'subject': str or None,                       # SINGULAR: |S| = 1 (first one if multiple)
            'participants': [{'text': str, 'position': str}],  # LIST: |P| ≥ 0 (multiple allowed)
            'counting': [{'text': str, 'type': str}],     # LIST: |C| ≥ 0 (multiple allowed)
            'voting_expressions': [str],                  # LIST: |V| ≥ 1 (at least one required)
            'outcome': str or None                        # SINGULAR: O ∈ {Approved, Rejected} or None
        }
    }

    Cardinality Constraints (from paper):
    - |S| = 1: Exactly one subject per voting event
    - |V| ≥ 1: At least one voting expression required
    - |C| ≥ 0: Zero or more counting expressions
    - |P| ≥ 0: Zero or more participants
    - O: Deterministically inferred outcome
    """
    
    def __init__(self):
        """Initialize the event constructor."""
        pass
    
    def construct_event(
        self,
        tokens: List[str],
        labels: List[str],
        example_id: str
    ) -> Dict[str, Any]:
        """
        Construct structured event from BIO-tagged sequence.
        
        Args:
            tokens: List of word tokens
            labels: List of BIO labels (e.g., 'B-SUBJECT', 'I-SUBJECT', 'O')
            example_id: Unique identifier for this example
            
        Returns:
            Event dictionary with has_voting_event and structured event
        """
        # Check if this segment contains a voting event
        has_voting = self._has_voting_event(labels)
        
        if not has_voting:
            return {
                'id': example_id,
                'has_voting_event': False,
                'event': None
            }
        
        # Extract all entity spans
        subject = self._extract_subject(tokens, labels)
        participants = self._extract_participants(tokens, labels)
        counting = self._extract_counting_list(tokens, labels)  # Returns LIST
        voting_expressions = self._extract_voting_expressions_list(tokens, labels)  # Returns LIST
        
        # Infer outcome deterministically from voting expressions, counting, and participants
        outcome = self._infer_outcome(voting_expressions, counting, participants)

        return {
            'id': example_id,
            'has_voting_event': True,
            'event': {
                'subject': subject,
                'participants': participants,
                'counting': counting,
                'voting_expressions': voting_expressions,
                'outcome': outcome
            }
        }
    
    def construct_events(
        self,
        examples: List[Dict[str, Any]],
        predictions: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Construct events for multiple examples.
        
        Args:
            examples: List of examples with 'tokens' and 'id' fields
            predictions: List of predicted label sequences
            
        Returns:
            List of constructed events
        """
        events = []
        
        for example, pred_labels in zip(examples, predictions):
            event = self.construct_event(
                example['tokens'],
                pred_labels,
                example['id']
            )
            events.append(event)
        
        return events
    
    def _has_voting_event(self, labels: List[str]) -> bool:
        """
        Check if labels contain a voting event.
        
        According to paper: A voting event is defined by |V| ≥ 1 (at least one voting expression).
        The presence of a VOTING entity (B-VOTING tag) is the sole criterion for determining
        whether a segment contains a voting event.
        
        Returns:
            True if at least one B-VOTING tag is found, False otherwise
        """
        return any(label.startswith('B-VOTING') for label in labels)
    
    def _extract_subject(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> Optional[str]:
        """
        Extract subject span (single string).
        
        If multiple SUBJECT spans exist, takes the first one (or could concatenate).
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            Subject text string or None
        """
        spans = self._extract_spans_by_type(tokens, labels, 'SUBJECT')
        
        if not spans:
            return None
        
        # Take first subject span
        subject_text = ' '.join(spans[0])
        
        return subject_text
    
    def _extract_participants(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract participant spans with positions.
        
        Parses VOTER-* labels to extract position (FAVOR, AGAINST, ABSTENTION, ABSENT).
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            
        Returns:
            List of {'text': str, 'position': str} dictionaries
        """
        participants = []
        
        # Find all VOTER-* spans
        i = 0
        while i < len(labels):
            label = labels[i]
            
            if label.startswith('B-VOTER-'):
                # Extract position from label (e.g., B-VOTER-FAVOR -> FAVOR)
                span_type = label[2:]  # Remove 'B-' prefix (e.g., 'VOTER-FAVOR')
                position = span_type.split('-', 1)[1]  # Extract position part
                
                # Collect span tokens
                span_tokens = [tokens[i]]
                j = i + 1
                
                # Continue with I-VOTER-* tags
                while j < len(labels) and labels[j] == f'I-{span_type}':
                    span_tokens.append(tokens[j])
                    j += 1
                
                participant_text = ' '.join(span_tokens)
                
                participants.append({
                    'text': participant_text,
                    'position': position.capitalize()
                })
                
                i = j
            else:
                i += 1
        
        return participants
    
    def _extract_counting_list(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[Dict[str, str]]:
        """
        Extract all counting spans with aggregation types (LIST of dicts).

        Parses COUNTING-* labels to extract type (UNANIMITY, MAJORITY).
        Returns all counting expressions found.

        Args:
            tokens: Word tokens
            labels: BIO labels

        Returns:
            List of {'text': str, 'type': str} dictionaries (empty list if none found)
        """
        counting_list = []
        i = 0

        while i < len(labels):
            label = labels[i]

            if label.startswith('B-COUNTING-'):
                # Extract counting type (e.g., B-COUNTING-UNANIMITY -> UNANIMITY)
                span_type = label[2:]  # Remove 'B-' prefix
                counting_type = span_type.split('-', 1)[1]  # Extract type part

                # Collect span tokens
                span_tokens = [tokens[i]]
                j = i + 1

                # Continue with I-COUNTING-* tags
                while j < len(labels) and labels[j] == f'I-{span_type}':
                    span_tokens.append(tokens[j])
                    j += 1

                counting_text = ' '.join(span_tokens)

                counting_list.append({
                    'text': counting_text,
                    'type': counting_type.capitalize()
                })

                i = j
            else:
                i += 1

        return counting_list
    
    def _extract_voting_expressions_list(
        self,
        tokens: List[str],
        labels: List[str]
    ) -> List[str]:
        """
        Extract all voting expression spans (LIST of strings).

        Returns all VOTING spans found.

        Args:
            tokens: Word tokens
            labels: BIO labels

        Returns:
            List of voting expression text strings (empty list if none found)
        """
        spans = self._extract_spans_by_type(tokens, labels, 'VOTING')

        # Convert all spans to strings and return as list
        return [' '.join(span) for span in spans]
    
    def _extract_spans_by_type(
        self,
        tokens: List[str],
        labels: List[str],
        span_type: str
    ) -> List[List[str]]:
        """
        Extract all spans of a given type.
        
        Args:
            tokens: Word tokens
            labels: BIO labels
            span_type: Type to extract (e.g., 'SUBJECT', 'VOTING')
            
        Returns:
            List of token lists, one per span
        """
        spans = []
        i = 0
        
        while i < len(labels):
            label = labels[i]
            
            if label == f'B-{span_type}':
                # Start of span
                span_tokens = [tokens[i]]
                j = i + 1
                
                # Collect continuation tokens
                while j < len(labels) and labels[j] == f'I-{span_type}':
                    span_tokens.append(tokens[j])
                    j += 1
                
                spans.append(span_tokens)
                i = j
            else:
                i += 1
        
        return spans
    
    def _infer_outcome(
        self,
        voting_expressions: List[str],
        counting: List[Dict[str, str]],
        participants: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Deterministically infer voting outcome from expressions, counting, and participants.
        
        As per paper: O ∈ {Approved, Rejected} is inferred from V, C, and P using rule-based heuristics.
        
        Heuristic rules (Portuguese municipal context):
        1. Look for approval keywords in voting expressions
        2. Consider counting type (unanimity often implies approval)
        3. Consider participant positions (more favor than against suggests approval)
        
        Args:
            voting_expressions: List of voting expression texts
            counting: List of counting dicts with 'text' and 'type'
            participants: List of participant dicts with 'text' and 'position'
            
        Returns:
            'Approved', 'Rejected', or None if outcome cannot be determined
        """
        if not voting_expressions and not counting and not participants:
            return None
        
        # Approval keywords (Portuguese)
        approval_keywords = [
            'aprovad', 'deliberad', 'deferido', 'autorizado', 'ratificad',
            'homologad', 'sancionad', 'concordad'
        ]
        
        # Rejection keywords (Portuguese)
        rejection_keywords = [
            'rejeitad', 'indeferido', 'recusad', 'negad', 'chumbad',
            'não aprovad', 'não deferido'
        ]
        
        # Check voting expressions for outcome keywords
        for expr in voting_expressions:
            expr_lower = expr.lower()
            
            # Check for rejection first (more specific)
            if any(keyword in expr_lower for keyword in rejection_keywords):
                return 'Rejected'
            
            # Check for approval
            if any(keyword in expr_lower for keyword in approval_keywords):
                return 'Approved'
        
        # If unanimity counting exists, likely approved (Portuguese municipal convention)
        if any(c.get('type', '').lower() in ['unanimity', 'unanimidade'] for c in counting):
            return 'Approved'
        
        # Count participant positions
        if participants:
            favor_count = sum(1 for p in participants if p.get('position') == 'Favor')
            against_count = sum(1 for p in participants if p.get('position') == 'Against')
            
            if favor_count > against_count:
                return 'Approved'
            elif against_count > favor_count:
                return 'Rejected'
        
        # Cannot determine outcome
        return None


# Convenience function
def construct_events_from_predictions(
    examples: List[Dict[str, Any]],
    predictions: List[List[str]]
) -> List[Dict[str, Any]]:
    """
    Construct events from BIO predictions.
    
    Args:
        examples: List of examples with 'tokens' and 'id'
        predictions: List of predicted BIO label sequences
    
    Returns:
        List of structured event dictionaries
    """
    constructor = EventConstructor()
    return constructor.construct_events(examples, predictions)