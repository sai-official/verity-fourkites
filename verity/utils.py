"""
Utility classes and functions for VERITY framework
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class DetectionResult:
    """
    Result of hallucination detection

    Attributes:
        score: Overall factuality score [0, 1]
        signals: Individual signal contributions
        verdict: Human-readable interpretation
        confidence_interval: (lower, upper) bounds
        risk_factors: List of detected risk indicators
        metadata: Additional information
    """
    score: float
    signals: Dict[str, float]
    verdict: str
    confidence_interval: tuple = (0.0, 1.0)
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set verdict based on score"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")

        # Set verdict if not provided
        if not self.verdict:
            self.verdict = self._compute_verdict()

    def _compute_verdict(self) -> str:
        """Compute verdict from score"""
        if self.score >= 0.9:
            return "HIGH_CONFIDENCE_FACTUAL"
        elif self.score >= 0.7:
            return "LIKELY_FACTUAL"
        elif self.score >= 0.5:
            return "UNCERTAIN"
        elif self.score >= 0.3:
            return "LIKELY_HALLUCINATED"
        else:
            return "HIGH_CONFIDENCE_HALLUCINATED"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'score': float(self.score),
            'signals': {k: float(v) for k, v in self.signals.items()},
            'verdict': self.verdict,
            'confidence_interval': self.confidence_interval,
            'risk_factors': self.risk_factors,
            'metadata': self.metadata
        }

    def __str__(self) -> str:
        """Human-readable string representation"""
        lines = [
            f"Factuality Score: {self.score:.3f}",
            f"Verdict: {self.verdict}",
            f"Confidence Interval: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]",
            "\nSignal Breakdown:"
        ]
        for signal, value in self.signals.items():
            lines.append(f"  {signal}: {value:.3f}")

        if self.risk_factors:
            lines.append("\nRisk Factors:")
            for factor in self.risk_factors:
                lines.append(f"  - {factor}")

        return "\n".join(lines)


def compute_semantic_similarity(text1: str, text2: str, embeddings) -> float:
    """
    Compute semantic similarity between two texts using embeddings

    Args:
        text1: First text
        text2: Second text
        embeddings: Sentence embedding model

    Returns:
        Similarity score [0, 1]
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if not text1.strip() or not text2.strip():
        return 0.0

    emb1 = embeddings.encode([text1])
    emb2 = embeddings.encode([text2])

    similarity = cosine_similarity(emb1, emb2)[0][0]
    # Convert from [-1, 1] to [0, 1]
    return (similarity + 1) / 2


def normalize_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a value to [0, 1] range

    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy of probability distribution

    Args:
        probabilities: Probability distribution (must sum to 1)

    Returns:
        Entropy value
    """
    # Filter out zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def extract_named_entities(text: str) -> List[str]:
    """
    Extract named entities and numbers from text

    Args:
        text: Input text

    Returns:
        List of entities/numbers
    """
    import re

    # Simple extraction: capitalized words and numbers
    # In production, use spaCy or similar NLP library
    entities = []

    # Extract capitalized words (potential named entities)
    entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    # Extract numbers (including decimals)
    entities.extend(re.findall(r'\b\d+(?:\.\d+)?\b', text))

    return entities


def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for scores

    Args:
        scores: List of score values
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 1.0)

    from scipy import stats

    mean = np.mean(scores)
    std_error = stats.sem(scores)
    interval = std_error * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)

    lower = np.clip(mean - interval, 0.0, 1.0)
    upper = np.clip(mean + interval, 0.0, 1.0)

    return (float(lower), float(upper))


class TokenAnalyzer:
    """Analyzes token-level information"""

    @staticmethod
    def get_important_token_indices(tokens: List[str]) -> List[int]:
        """
        Identify indices of important tokens (content words, entities, numbers)

        Args:
            tokens: List of tokens

        Returns:
            List of important token indices
        """
        # Function words to ignore
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }

        important_indices = []
        for i, token in enumerate(tokens):
            token_lower = token.lower().strip('.,!?;:')

            # Check if it's a function word
            if token_lower in function_words:
                continue

            # Check if it's capitalized (potential named entity)
            if token[0].isupper():
                important_indices.append(i)
                continue

            # Check if it contains numbers
            if any(c.isdigit() for c in token):
                important_indices.append(i)
                continue

            # Check if it's a content word (length > 3)
            if len(token_lower) > 3:
                important_indices.append(i)

        return important_indices


class PromptTemplates:
    """Templates for generating prompts"""

    CROSS_EXAM_TEMPLATE = """Based on this response: "{response}"

Generate 3 specific probing questions that would help verify the factual accuracy of this information.
Focus on details, numbers, names, and relationships mentioned.

Questions:
1."""

    ADVERSARIAL_QUESTION_TEMPLATE = """You are a skeptical fact-checker. The following claim was made:

"{claim}"

Generate 2 challenging questions that could reveal if this claim is incorrect or hallucinated.
Be specific and focus on verifiable details.

Questions:
1."""

    CONSISTENCY_CHECK_TEMPLATE = """Original answer: "{original}"

Follow-up question: "{followup}"
Follow-up answer: "{followup_answer}"

Are these answers consistent? Reply with just "yes" or "no"."""

    REPHRASE_TEMPLATE = """Rephrase the following question while preserving its meaning:

Original: "{question}"

Rephrased:"""
