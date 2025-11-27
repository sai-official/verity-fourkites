"""
Signal extraction modules for hallucination detection
Each signal provides a different perspective on factuality
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

from .utils import (
    compute_semantic_similarity,
    compute_entropy,
    normalize_score,
    extract_named_entities,
    TokenAnalyzer,
    PromptTemplates
)


class BaseSignal(ABC):
    """Base class for hallucination detection signals"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute(self, question: str, response: str, **kwargs) -> float:
        """
        Compute signal score

        Args:
            question: Input question
            response: Model response
            **kwargs: Additional parameters

        Returns:
            Score in [0, 1] where 1 = high confidence factual
        """
        pass


class SelfConsistencySignal(BaseSignal):
    """
    Signal 1: Self-Consistency Score (σ_SC)

    Measures consistency across multiple sampled responses.
    Uses clustered sampling to avoid mode collapse.
    """

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        super().__init__("self_consistency")
        self.embeddings = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')

    def compute(
        self,
        question: str,
        response: str,
        model_fn=None,
        num_samples: int = 5,
        temperature_range: Tuple[float, float] = (0.7, 1.1),
        **kwargs
    ) -> float:
        """
        Compute self-consistency by sampling multiple responses

        Args:
            question: Input question
            response: Original response (included in analysis)
            model_fn: Function to generate responses: model_fn(question, temperature) -> str
            num_samples: Number of additional samples to generate
            temperature_range: (min_temp, max_temp) for sampling

        Returns:
            Consistency score [0, 1]
        """
        if model_fn is None:
            # If no model function, cannot compute - return neutral score
            return 0.5

        # Generate responses with varying temperatures
        responses = [response]  # Include original
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_samples)

        for temp in temperatures:
            try:
                sampled_response = model_fn(question, temperature=temp)
                responses.append(sampled_response)
            except Exception as e:
                # If sampling fails, continue with available samples
                pass

        if len(responses) < 2:
            return 0.5  # Not enough samples

        # Compute pairwise semantic similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = compute_semantic_similarity(
                    responses[i],
                    responses[j],
                    self.embeddings
                )
                similarities.append(sim)

        # High variance in similarities indicates inconsistency
        mean_sim = np.mean(similarities)
        variance_sim = np.var(similarities)

        # Score: high mean and low variance = consistent = high score
        consistency = mean_sim * (1 - np.clip(variance_sim, 0, 1))

        return float(np.clip(consistency, 0, 1))


class SemanticEntropySignal(BaseSignal):
    """
    Signal 2: Semantic Entropy (σ_SE)

    Measures uncertainty in semantic space by clustering responses
    and computing entropy over cluster distribution.

    Based on: Kuhn et al. "Semantic Uncertainty" (2023)
    """

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        super().__init__("semantic_entropy")
        self.embeddings = embedding_model or SentenceTransformer('all-MiniLM-L6-v2')

    def compute(
        self,
        question: str,
        response: str,
        model_fn=None,
        num_samples: int = 5,
        **kwargs
    ) -> float:
        """
        Compute semantic entropy over sampled responses

        Args:
            question: Input question
            response: Original response
            model_fn: Function to generate responses
            num_samples: Number of samples

        Returns:
            Score [0, 1] where low entropy = high confidence = high score
        """
        if model_fn is None:
            return 0.5

        # Generate multiple responses
        responses = [response]
        for _ in range(num_samples):
            try:
                sampled = model_fn(question, temperature=0.9)
                responses.append(sampled)
            except:
                pass

        if len(responses) < 2:
            return 0.5

        # Get embeddings for all responses
        embeddings = self.embeddings.encode(responses)

        # Cluster responses using DBSCAN for bidirectional entailment approximation
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Compute cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()

        # Compute entropy
        entropy = compute_entropy(probabilities)

        # Normalize: max entropy = log2(num_samples)
        max_entropy = np.log2(len(responses))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Convert to score: low entropy = high confidence = high score
        score = 1 - normalized_entropy

        return float(np.clip(score, 0, 1))


class TokenConfidenceSignal(BaseSignal):
    """
    Signal 3: Token Confidence Score (σ_TC)

    Analyzes token-level probabilities during generation.
    Applies contextual weighting to important tokens.
    """

    def __init__(self):
        super().__init__("token_confidence")
        self.analyzer = TokenAnalyzer()
        self.low_prob_threshold = 0.1  # Threshold for "uncertain" tokens

    def compute(
        self,
        question: str,
        response: str,
        token_probs: Optional[List[float]] = None,
        tokens: Optional[List[str]] = None,
        **kwargs
    ) -> float:
        """
        Compute token confidence score

        Args:
            question: Input question
            response: Model response
            token_probs: List of token probabilities (if available)
            tokens: List of tokens (if available)

        Returns:
            Confidence score [0, 1]
        """
        if token_probs is None or len(token_probs) == 0:
            # No token probabilities available - return neutral
            return 0.5

        if tokens is None:
            # Simple tokenization if not provided
            tokens = response.split()

        # Ensure we have matching lengths
        if len(tokens) != len(token_probs):
            token_probs = token_probs[:len(tokens)]

        # Get important token indices
        important_indices = self.analyzer.get_important_token_indices(tokens)

        # Apply higher weight to important tokens
        weighted_probs = []
        for i, prob in enumerate(token_probs):
            weight = 2.0 if i in important_indices else 1.0
            weighted_probs.extend([prob] * int(weight))

        # Geometric mean of probabilities (more sensitive to low values)
        if len(weighted_probs) == 0:
            return 0.5

        geometric_mean = np.exp(np.mean(np.log(np.clip(weighted_probs, 1e-10, 1.0))))

        # Count low-probability tokens (potential hallucinations)
        low_prob_count = sum(1 for p in token_probs if p < self.low_prob_threshold)
        low_prob_ratio = low_prob_count / len(token_probs)

        # Penalty for low-probability tokens
        penalty = np.exp(-2.0 * low_prob_ratio)

        # Final score
        score = geometric_mean * penalty

        return float(np.clip(score, 0, 1))


class CrossExaminerSignal(BaseSignal):
    """
    Signal 4: Cross-Examiner Score (σ_CE)

    Generates probing questions about the response and checks consistency.
    Uses adversarial question generation to test factuality.
    """

    def __init__(self):
        super().__init__("cross_examiner")

    def compute(
        self,
        question: str,
        response: str,
        model_fn=None,
        num_questions: int = 3,
        **kwargs
    ) -> float:
        """
        Generate probing questions and check consistency

        Args:
            question: Original question
            response: Model response
            model_fn: Function to query model
            num_questions: Number of probing questions

        Returns:
            Consistency score [0, 1]
        """
        if model_fn is None:
            return 0.5

        # Extract key claims from response
        entities = extract_named_entities(response)

        # Generate probing questions
        probe_questions = self._generate_probing_questions(
            response, entities, model_fn, num_questions
        )

        if not probe_questions:
            return 0.5

        # Ask each probing question and check consistency
        consistencies = []
        for probe_q in probe_questions:
            try:
                probe_answer = model_fn(probe_q, temperature=0.3)

                # Check if probe answer is consistent with original response
                consistency = self._check_consistency(
                    response, probe_answer, model_fn
                )
                consistencies.append(consistency)
            except:
                continue

        if not consistencies:
            return 0.5

        # Average consistency
        score = np.mean(consistencies)
        return float(np.clip(score, 0, 1))

    def _generate_probing_questions(
        self,
        response: str,
        entities: List[str],
        model_fn,
        num_questions: int
    ) -> List[str]:
        """Generate probing questions about the response"""
        prompt = PromptTemplates.CROSS_EXAM_TEMPLATE.format(response=response)

        try:
            # Generate questions
            questions_text = model_fn(prompt, temperature=0.7)

            # Parse questions (simple extraction)
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                # Look for numbered questions
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    q = line.lstrip('0123456789.-) ').strip()
                    if q and len(q) > 10:  # Valid question
                        questions.append(q)

            return questions[:num_questions]
        except:
            return []

    def _check_consistency(
        self,
        original: str,
        probe_answer: str,
        model_fn
    ) -> float:
        """Check if probe answer is consistent with original"""
        # Use model to judge consistency
        prompt = PromptTemplates.CONSISTENCY_CHECK_TEMPLATE.format(
            original=original,
            followup="(probing question)",
            followup_answer=probe_answer
        )

        try:
            judgment = model_fn(prompt, temperature=0.1).strip().lower()

            if 'yes' in judgment:
                return 1.0
            elif 'no' in judgment:
                return 0.0
            else:
                return 0.5
        except:
            return 0.5


class PerplexityVarianceSignal(BaseSignal):
    """
    Signal 5: Perplexity Variance (σ_PV)

    Analyzes perplexity variance across model layers.
    High variance indicates uncertainty in generation.
    """

    def __init__(self):
        super().__init__("perplexity_variance")

    def compute(
        self,
        question: str,
        response: str,
        layer_perplexities: Optional[List[float]] = None,
        **kwargs
    ) -> float:
        """
        Compute perplexity variance score

        Args:
            question: Input question
            response: Model response
            layer_perplexities: Perplexity from each layer (if available)

        Returns:
            Score [0, 1] where low variance = high confidence
        """
        if layer_perplexities is None or len(layer_perplexities) < 2:
            # No layer information - return neutral
            return 0.5

        # Compute variance across layers
        mean_perp = np.mean(layer_perplexities)
        variance = np.var(layer_perplexities)

        # Coefficient of variation (normalized variance)
        cv = np.sqrt(variance) / mean_perp if mean_perp > 0 else 0

        # Weight middle layers more heavily (they contain semantic info)
        # This is a simplified version - in practice, extract layer-specific perplexities
        num_layers = len(layer_perplexities)
        middle_start = num_layers // 3
        middle_end = 2 * num_layers // 3
        middle_variance = np.var(layer_perplexities[middle_start:middle_end])

        # Combine overall and middle-layer variance
        combined_variance = 0.6 * cv + 0.4 * middle_variance

        # Convert to score: low variance = high confidence = high score
        score = 1 / (1 + combined_variance)

        return float(np.clip(score, 0, 1))


class ExpertiseWeightedSignal(BaseSignal):
    """
    Signal 6: Expertise-Weighted Agreement (σ_EWA)

    Based on FEWL framework from the paper.
    Uses multiple reference LLMs weighted by expertise.
    """

    def __init__(self, reference_models: Optional[List] = None):
        super().__init__("expertise_weighted")
        self.reference_models = reference_models or []
        self.expertise_weights = {}
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')

    def compute(
        self,
        question: str,
        response: str,
        reference_responses: Optional[List[str]] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Compute expertise-weighted agreement

        Args:
            question: Input question
            response: Model response to evaluate
            reference_responses: Responses from reference models
            domain: Domain/topic for dynamic expertise weighting

        Returns:
            Agreement score [0, 1]
        """
        if not reference_responses or len(reference_responses) == 0:
            return 0.5

        # Compute expertise weights for this domain
        weights = self._compute_expertise_weights(
            question, reference_responses, domain
        )

        # Compute weighted similarity
        weighted_similarities = []
        for ref_response, weight in zip(reference_responses, weights):
            similarity = compute_semantic_similarity(
                response, ref_response, self.embeddings
            )
            weighted_similarities.append(weight * similarity)

        # Weighted average
        agreement_score = sum(weighted_similarities) / sum(weights) if sum(weights) > 0 else 0.5

        # Apply laziness penalty
        laziness_penalty = self._compute_laziness_penalty(
            question, reference_responses
        )

        # Final score
        score = agreement_score - 0.1 * laziness_penalty

        return float(np.clip(score, 0, 1))

    def _compute_expertise_weights(
        self,
        question: str,
        reference_responses: List[str],
        domain: Optional[str]
    ) -> List[float]:
        """
        Compute expertise weights for reference models

        In full implementation: test on probe questions for this domain
        Here: simplified uniform weighting with slight variation
        """
        # Simplified: weight by response length and specificity
        weights = []
        for response in reference_responses:
            # Longer, more specific responses get higher weight
            length_factor = min(len(response.split()) / 50, 1.0)

            # Count specific entities (indicates expertise)
            entities = extract_named_entities(response)
            specificity_factor = min(len(entities) / 5, 1.0)

            weight = 0.5 + 0.3 * length_factor + 0.2 * specificity_factor
            weights.append(weight)

        # Normalize
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

    def _compute_laziness_penalty(
        self,
        question: str,
        reference_responses: List[str]
    ) -> float:
        """
        Penalize reference models showing superficial knowledge

        Check if they give similar answers to different but related questions
        """
        # Simplified: check if responses are too similar to each other
        if len(reference_responses) < 2:
            return 0.0

        similarities = []
        for i in range(len(reference_responses)):
            for j in range(i + 1, len(reference_responses)):
                sim = compute_semantic_similarity(
                    reference_responses[i],
                    reference_responses[j],
                    self.embeddings
                )
                similarities.append(sim)

        # High similarity among all references = potential laziness
        mean_similarity = np.mean(similarities)

        # Penalty increases with similarity
        penalty = mean_similarity if mean_similarity > 0.9 else 0.0

        return float(penalty)
