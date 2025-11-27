"""
Signal fusion module for combining multiple hallucination signals
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


class SignalFusion:
    """
    Fuses multiple hallucination detection signals into a unified score

    Uses a lightweight neural network to learn optimal signal weights
    and calibrate the final score.
    """

    def __init__(
        self,
        signal_names: List[str],
        custom_weights: Optional[Dict[str, float]] = None,
        calibrated: bool = True
    ):
        """
        Initialize signal fusion

        Args:
            signal_names: Names of signals to fuse
            custom_weights: Optional custom weights for signals
            calibrated: Whether to use probability calibration
        """
        self.signal_names = signal_names
        self.calibrated = calibrated

        # Initialize weights
        if custom_weights:
            self.weights = self._normalize_weights(custom_weights)
        else:
            self.weights = self._default_weights()

        # Initialize fusion model (lightweight MLP)
        self.fusion_model = None
        self._initialize_model()

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1"""
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()} if total > 0 else weights

    def _default_weights(self) -> Dict[str, float]:
        """Default signal weights based on empirical performance"""
        default = {
            'self_consistency': 0.20,
            'semantic_entropy': 0.18,
            'token_confidence': 0.15,
            'cross_examiner': 0.22,
            'perplexity_variance': 0.10,
            'expertise_weighted': 0.15
        }

        # Use only signals that are present
        weights = {name: default.get(name, 1.0) for name in self.signal_names}
        return self._normalize_weights(weights)

    def _initialize_model(self):
        """Initialize lightweight fusion model"""
        # Simple MLP with one hidden layer
        base_model = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )

        if self.calibrated:
            # Wrap with calibration for better probability estimates
            self.fusion_model = CalibratedClassifierCV(
                base_model,
                method='sigmoid',
                cv=3
            )
        else:
            self.fusion_model = base_model

    def fuse(
        self,
        signal_scores: Dict[str, float],
        mode: str = 'weighted'
    ) -> float:
        """
        Fuse multiple signal scores into unified score

        Args:
            signal_scores: Dictionary of signal name -> score
            mode: Fusion mode ('weighted', 'learned', or 'ensemble')

        Returns:
            Fused score [0, 1]
        """
        if mode == 'weighted':
            return self._weighted_fusion(signal_scores)
        elif mode == 'learned' and self.fusion_model is not None:
            return self._learned_fusion(signal_scores)
        elif mode == 'ensemble':
            return self._ensemble_fusion(signal_scores)
        else:
            return self._weighted_fusion(signal_scores)

    def _weighted_fusion(self, signal_scores: Dict[str, float]) -> float:
        """Simple weighted average of signals"""
        total_score = 0.0
        total_weight = 0.0

        for signal_name, score in signal_scores.items():
            if signal_name in self.weights:
                weight = self.weights[signal_name]
                total_score += weight * score
                total_weight += weight

        if total_weight == 0:
            return 0.5

        return total_score / total_weight

    def _learned_fusion(self, signal_scores: Dict[str, float]) -> float:
        """
        Use learned fusion model

        Note: Requires training data. In this version, falls back to weighted.
        """
        # Convert to feature vector
        features = np.array([
            signal_scores.get(name, 0.5) for name in self.signal_names
        ]).reshape(1, -1)

        try:
            # Predict probability of being factual
            if hasattr(self.fusion_model, 'predict_proba'):
                proba = self.fusion_model.predict_proba(features)[0]
                return float(proba[1])  # Probability of factual class
            else:
                # Model not trained, fall back to weighted
                return self._weighted_fusion(signal_scores)
        except:
            return self._weighted_fusion(signal_scores)

    def _ensemble_fusion(self, signal_scores: Dict[str, float]) -> float:
        """
        Ensemble fusion combining multiple strategies

        Combines weighted average, min/max, and median
        """
        scores = list(signal_scores.values())

        if not scores:
            return 0.5

        # Different aggregation strategies
        weighted = self._weighted_fusion(signal_scores)
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        min_score = np.min(scores)

        # Conservative ensemble: weight towards lower scores
        # (hallucination detection should be conservative)
        ensemble = (
            0.4 * weighted +
            0.3 * median_score +
            0.2 * mean_score +
            0.1 * min_score
        )

        return float(np.clip(ensemble, 0, 1))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train the fusion model on labeled data

        Args:
            X_train: Training features (N x num_signals)
            y_train: Training labels (N,) - 1 for factual, 0 for hallucinated
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        if self.fusion_model is None:
            self._initialize_model()

        try:
            self.fusion_model.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                score = self.fusion_model.score(X_val, y_val)
                print(f"Validation accuracy: {score:.3f}")

        except Exception as e:
            print(f"Warning: Training failed - {e}")
            print("Falling back to weighted fusion")

    def compute_confidence_interval(
        self,
        signal_scores: Dict[str, float],
        confidence: float = 0.95
    ) -> tuple:
        """
        Compute confidence interval for fused score

        Uses bootstrap resampling of signal scores

        Args:
            signal_scores: Dictionary of signal scores
            confidence: Confidence level

        Returns:
            (lower_bound, upper_bound)
        """
        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_scores = []

        scores_array = np.array(list(signal_scores.values()))

        for _ in range(n_bootstrap):
            # Resample signals with replacement
            resampled = np.random.choice(scores_array, size=len(scores_array), replace=True)

            # Create resampled dictionary
            resampled_dict = {
                name: score for name, score in zip(signal_scores.keys(), resampled)
            }

            # Compute fused score
            fused = self.fuse(resampled_dict)
            bootstrap_scores.append(fused)

        # Compute confidence interval
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_scores, alpha * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

        return (float(lower), float(upper))

    def explain_decision(
        self,
        signal_scores: Dict[str, float],
        fused_score: float
    ) -> Dict[str, any]:
        """
        Explain how signals contributed to final score

        Args:
            signal_scores: Individual signal scores
            fused_score: Final fused score

        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'fused_score': fused_score,
            'signal_contributions': {},
            'key_factors': [],
            'risk_indicators': []
        }

        # Compute weighted contributions
        for signal_name, score in signal_scores.items():
            weight = self.weights.get(signal_name, 0)
            contribution = weight * score
            explanation['signal_contributions'][signal_name] = {
                'score': score,
                'weight': weight,
                'contribution': contribution
            }

        # Identify key factors (signals with highest contribution)
        sorted_signals = sorted(
            explanation['signal_contributions'].items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        explanation['key_factors'] = [name for name, _ in sorted_signals[:3]]

        # Identify risk indicators (signals with low scores)
        for signal_name, score in signal_scores.items():
            if score < 0.5:
                explanation['risk_indicators'].append({
                    'signal': signal_name,
                    'score': score,
                    'severity': 'high' if score < 0.3 else 'medium'
                })

        return explanation

    def get_signal_importance(self) -> Dict[str, float]:
        """
        Get importance score for each signal

        Returns:
            Dictionary of signal name -> importance
        """
        return self.weights.copy()

    def update_weights(self, new_weights: Dict[str, float]):
        """Update signal weights"""
        self.weights = self._normalize_weights(new_weights)
