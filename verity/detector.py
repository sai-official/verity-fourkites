"""
Main hallucination detector class
"""

from typing import Dict, List, Optional, Callable
import numpy as np
from sentence_transformers import SentenceTransformer

from .signals import (
    SelfConsistencySignal,
    SemanticEntropySignal,
    TokenConfidenceSignal,
    CrossExaminerSignal,
    PerplexityVarianceSignal,
    ExpertiseWeightedSignal
)
from .fusion import SignalFusion
from .utils import DetectionResult, compute_confidence_interval


class HallucinationDetector:
    """
    Main VERITY hallucination detection system

    Combines multiple signals to detect and quantify hallucinations
    in LLM-generated content.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        model_fn: Optional[Callable] = None,
        enable_all_signals: bool = True,
        signal_weights: Optional[Dict[str, float]] = None,
        num_samples: int = 5,
        temperature_range: tuple = (0.7, 1.1),
        embedding_model: Optional[str] = None
    ):
        """
        Initialize hallucination detector

        Args:
            model_name: Name of the LLM being evaluated
            model_fn: Function to query model: fn(question, temperature) -> response
            enable_all_signals: Whether to enable all signals
            signal_weights: Custom weights for signals
            num_samples: Number of samples for consistency checks
            temperature_range: (min, max) temperature for sampling
            embedding_model: Name of sentence embedding model
        """
        self.model_name = model_name
        self.model_fn = model_fn
        self.num_samples = num_samples
        self.temperature_range = temperature_range

        # Initialize embedding model
        embedding_model_name = embedding_model or 'all-MiniLM-L6-v2'
        self.embeddings = SentenceTransformer(embedding_model_name)

        # Initialize signals
        self.signals = {}
        if enable_all_signals:
            self._initialize_all_signals()

        # Initialize fusion module
        signal_names = list(self.signals.keys())
        self.fusion = SignalFusion(
            signal_names=signal_names,
            custom_weights=signal_weights
        )

        # Cache for storing signal scores
        self._cache = {}

    def _initialize_all_signals(self):
        """Initialize all available signals"""
        self.signals['self_consistency'] = SelfConsistencySignal(self.embeddings)
        self.signals['semantic_entropy'] = SemanticEntropySignal(self.embeddings)
        self.signals['token_confidence'] = TokenConfidenceSignal()
        self.signals['cross_examiner'] = CrossExaminerSignal()
        self.signals['perplexity_variance'] = PerplexityVarianceSignal()
        self.signals['expertise_weighted'] = ExpertiseWeightedSignal()

    def detect(
        self,
        question: str,
        response: str,
        token_probs: Optional[List[float]] = None,
        tokens: Optional[List[str]] = None,
        layer_perplexities: Optional[List[float]] = None,
        reference_responses: Optional[List[str]] = None,
        fusion_mode: str = 'ensemble',
        verbose: bool = False
    ) -> DetectionResult:
        """
        Detect hallucinations in a response

        Args:
            question: Input question
            response: Model response to evaluate
            token_probs: Token probabilities (if available)
            tokens: Tokens (if available)
            layer_perplexities: Layer-wise perplexities (if available)
            reference_responses: Responses from reference models
            fusion_mode: How to combine signals ('weighted', 'learned', 'ensemble')
            verbose: Print detailed information

        Returns:
            DetectionResult with score, signals, and metadata
        """
        if verbose:
            print(f"Evaluating response for: {question[:100]}...")

        # Compute all signal scores
        signal_scores = self._compute_all_signals(
            question=question,
            response=response,
            token_probs=token_probs,
            tokens=tokens,
            layer_perplexities=layer_perplexities,
            reference_responses=reference_responses,
            verbose=verbose
        )

        # Fuse signals into final score
        final_score = self.fusion.fuse(signal_scores, mode=fusion_mode)

        # Compute confidence interval
        ci = self.fusion.compute_confidence_interval(signal_scores)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(signal_scores)

        # Get explanation
        explanation = self.fusion.explain_decision(signal_scores, final_score)

        # Create result
        result = DetectionResult(
            score=final_score,
            signals=signal_scores,
            verdict="",  # Will be auto-computed
            confidence_interval=ci,
            risk_factors=risk_factors,
            metadata={
                'question': question,
                'response_length': len(response),
                'fusion_mode': fusion_mode,
                'explanation': explanation
            }
        )

        if verbose:
            print(f"\n{result}")

        return result

    def _compute_all_signals(
        self,
        question: str,
        response: str,
        token_probs: Optional[List[float]],
        tokens: Optional[List[str]],
        layer_perplexities: Optional[List[float]],
        reference_responses: Optional[List[str]],
        verbose: bool
    ) -> Dict[str, float]:
        """Compute all signal scores"""
        signal_scores = {}

        for signal_name, signal in self.signals.items():
            try:
                if verbose:
                    print(f"Computing {signal_name}...", end=" ")

                # Prepare kwargs for signal
                kwargs = {
                    'model_fn': self.model_fn,
                    'num_samples': self.num_samples,
                    'temperature_range': self.temperature_range,
                    'token_probs': token_probs,
                    'tokens': tokens,
                    'layer_perplexities': layer_perplexities,
                    'reference_responses': reference_responses
                }

                # Compute signal
                score = signal.compute(question, response, **kwargs)
                signal_scores[signal_name] = score

                if verbose:
                    print(f"{score:.3f}")

            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                # On error, use neutral score
                signal_scores[signal_name] = 0.5

        return signal_scores

    def _identify_risk_factors(self, signal_scores: Dict[str, float]) -> List[str]:
        """Identify specific risk factors from signal scores"""
        risk_factors = []

        # Check each signal for concerning scores
        if signal_scores.get('self_consistency', 1.0) < 0.5:
            risk_factors.append("Inconsistent responses across sampling")

        if signal_scores.get('semantic_entropy', 1.0) < 0.5:
            risk_factors.append("High semantic uncertainty")

        if signal_scores.get('token_confidence', 1.0) < 0.5:
            risk_factors.append("Low token-level confidence")

        if signal_scores.get('cross_examiner', 1.0) < 0.5:
            risk_factors.append("Failed consistency probing")

        if signal_scores.get('perplexity_variance', 1.0) < 0.5:
            risk_factors.append("High layer-wise perplexity variance")

        if signal_scores.get('expertise_weighted', 1.0) < 0.5:
            risk_factors.append("Low agreement with reference models")

        return risk_factors

    def batch_detect(
        self,
        questions: List[str],
        responses: List[str],
        **kwargs
    ) -> List[DetectionResult]:
        """
        Detect hallucinations for multiple Q&A pairs

        Args:
            questions: List of questions
            responses: List of responses
            **kwargs: Additional arguments passed to detect()

        Returns:
            List of DetectionResults
        """
        results = []
        for q, r in zip(questions, responses):
            result = self.detect(q, r, **kwargs)
            results.append(result)
        return results

    def calibrate(
        self,
        questions: List[str],
        responses: List[str],
        labels: List[int]
    ):
        """
        Calibrate the detector using labeled data

        Args:
            questions: List of questions
            responses: List of responses
            labels: List of labels (1 = factual, 0 = hallucinated)
        """
        # Compute signal scores for all examples
        X = []
        for q, r in zip(questions, responses):
            scores = self._compute_all_signals(
                question=q,
                response=r,
                token_probs=None,
                tokens=None,
                layer_perplexities=None,
                reference_responses=None,
                verbose=False
            )
            score_vector = [scores[name] for name in self.fusion.signal_names]
            X.append(score_vector)

        X = np.array(X)
        y = np.array(labels)

        # Train fusion model
        self.fusion.train(X, y)

    def get_signal_importance(self) -> Dict[str, float]:
        """Get importance of each signal"""
        return self.fusion.get_signal_importance()

    def set_model_function(self, model_fn: Callable):
        """
        Set the model function for querying the LLM

        Args:
            model_fn: Function with signature fn(question, temperature) -> response
        """
        self.model_fn = model_fn

    def add_signal(self, name: str, signal):
        """
        Add a custom signal to the detector

        Args:
            name: Signal name
            signal: Signal instance (must inherit from BaseSignal)
        """
        self.signals[name] = signal
        # Update fusion module
        self.fusion.signal_names.append(name)
        if name not in self.fusion.weights:
            # Add with default weight, then normalize
            self.fusion.weights[name] = 1.0
            self.fusion.weights = self.fusion._normalize_weights(self.fusion.weights)

    def remove_signal(self, name: str):
        """Remove a signal from the detector"""
        if name in self.signals:
            del self.signals[name]
            self.fusion.signal_names.remove(name)
            if name in self.fusion.weights:
                del self.fusion.weights[name]
                self.fusion.weights = self.fusion._normalize_weights(self.fusion.weights)

    def benchmark(
        self,
        dataset: List[Dict],
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Benchmark the detector on a labeled dataset

        Args:
            dataset: List of dicts with 'question', 'response', 'label' keys
            metric: Evaluation metric ('accuracy', 'f1', 'auc')

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        predictions = []
        labels = []

        for item in dataset:
            result = self.detect(item['question'], item['response'])
            # Threshold at 0.5
            pred = 1 if result.score >= 0.5 else 0
            predictions.append(pred)
            labels.append(item['label'])

        scores = predictions  # Raw scores for AUC
        predictions = np.array(predictions)
        labels = np.array(labels)

        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='binary'),
        }

        try:
            metrics['auc'] = roc_auc_score(labels, scores)
        except:
            metrics['auc'] = 0.0

        return metrics
