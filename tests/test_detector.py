"""
Unit tests for VERITY hallucination detector
"""

import unittest
import sys
sys.path.append('..')

from verity import HallucinationDetector
from verity.utils import DetectionResult


class TestHallucinationDetector(unittest.TestCase):
    """Test cases for HallucinationDetector"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = HallucinationDetector(
            model_name="test-model",
            enable_all_signals=False  # Disable signals requiring model_fn
        )

        # Add only signals that don't require model_fn
        self.detector.add_signal('token_confidence', self.detector.signals.get('token_confidence'))

    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.model_name, "test-model")
        self.assertIsNotNone(self.detector.fusion)

    def test_detect_basic(self):
        """Test basic detection without model function"""
        question = "What is 2+2?"
        response = "2+2 equals 4."

        result = self.detector.detect(question, response)

        self.assertIsInstance(result, DetectionResult)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIsNotNone(result.verdict)

    def test_detect_with_token_probs(self):
        """Test detection with token probabilities"""
        question = "What is the capital of France?"
        response = "The capital of France is Paris."
        tokens = ["The", "capital", "of", "France", "is", "Paris", "."]
        token_probs = [0.9, 0.85, 0.95, 0.92, 0.88, 0.94, 1.0]

        result = self.detector.detect(
            question, response,
            token_probs=token_probs,
            tokens=tokens
        )

        self.assertIsInstance(result, DetectionResult)
        self.assertIn('token_confidence', result.signals)

    def test_batch_detect(self):
        """Test batch detection"""
        questions = [
            "What is 2+2?",
            "What is the capital of France?",
            "Who invented the telephone?"
        ]
        responses = [
            "2+2 equals 4.",
            "Paris is the capital of France.",
            "Alexander Graham Bell invented the telephone."
        ]

        results = self.detector.batch_detect(questions, responses)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, DetectionResult)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_verdict_categories(self):
        """Test verdict categorization"""
        # Test different score ranges
        test_scores = [0.95, 0.8, 0.6, 0.4, 0.2]
        expected_verdicts = [
            "HIGH_CONFIDENCE_FACTUAL",
            "LIKELY_FACTUAL",
            "UNCERTAIN",
            "LIKELY_HALLUCINATED",
            "HIGH_CONFIDENCE_HALLUCINATED"
        ]

        for score, expected in zip(test_scores, expected_verdicts):
            result = DetectionResult(
                score=score,
                signals={},
                verdict=""
            )
            self.assertEqual(result.verdict, expected)

    def test_signal_importance(self):
        """Test getting signal importance"""
        importance = self.detector.get_signal_importance()

        self.assertIsInstance(importance, dict)
        # Weights should sum to approximately 1.0
        total_weight = sum(importance.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    def test_add_remove_signal(self):
        """Test adding and removing signals"""
        from verity.signals import BaseSignal

        class DummySignal(BaseSignal):
            def compute(self, question, response, **kwargs):
                return 0.75

        # Add signal
        self.detector.add_signal('dummy', DummySignal())
        self.assertIn('dummy', self.detector.signals)

        # Remove signal
        self.detector.remove_signal('dummy')
        self.assertNotIn('dummy', self.detector.signals)

    def test_confidence_interval(self):
        """Test confidence interval computation"""
        question = "Test question"
        response = "Test response"

        result = self.detector.detect(question, response)

        lower, upper = result.confidence_interval
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLessEqual(lower, upper)
        self.assertLessEqual(lower, result.score)
        self.assertGreaterEqual(upper, result.score)

    def test_risk_factors(self):
        """Test risk factor identification"""
        question = "Test question"
        response = "Test response"

        result = self.detector.detect(question, response)

        self.assertIsInstance(result.risk_factors, list)

    def test_detection_result_to_dict(self):
        """Test conversion of DetectionResult to dictionary"""
        result = DetectionResult(
            score=0.85,
            signals={'signal1': 0.9, 'signal2': 0.8},
            verdict="LIKELY_FACTUAL",
            risk_factors=["test risk"]
        )

        result_dict = result.to_dict()

        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['score'], 0.85)
        self.assertEqual(result_dict['verdict'], "LIKELY_FACTUAL")
        self.assertIn('signals', result_dict)
        self.assertIn('risk_factors', result_dict)


class TestSignals(unittest.TestCase):
    """Test cases for individual signals"""

    def test_token_confidence_signal(self):
        """Test token confidence signal"""
        from verity.signals import TokenConfidenceSignal

        signal = TokenConfidenceSignal()

        tokens = ["The", "capital", "is", "Paris"]
        token_probs = [0.9, 0.85, 0.88, 0.92]

        score = signal.compute(
            question="What is the capital?",
            response="The capital is Paris",
            tokens=tokens,
            token_probs=token_probs
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_token_confidence_with_low_probs(self):
        """Test token confidence with low probability tokens"""
        from verity.signals import TokenConfidenceSignal

        signal = TokenConfidenceSignal()

        # Include some low-probability tokens (potential hallucinations)
        tokens = ["The", "capital", "is", "Xyzabc"]
        token_probs = [0.9, 0.85, 0.88, 0.05]  # Last token has very low prob

        score = signal.compute(
            question="What is the capital?",
            response="The capital is Xyzabc",
            tokens=tokens,
            token_probs=token_probs
        )

        # Score should be lower due to low-probability token
        self.assertLess(score, 0.7)

    def test_perplexity_variance_signal(self):
        """Test perplexity variance signal"""
        from verity.signals import PerplexityVarianceSignal

        signal = PerplexityVarianceSignal()

        # Low variance = consistent = high score
        low_variance_perps = [2.5, 2.6, 2.4, 2.5, 2.6]
        score_low = signal.compute(
            question="Test",
            response="Test response",
            layer_perplexities=low_variance_perps
        )

        # High variance = uncertain = low score
        high_variance_perps = [1.0, 5.0, 2.0, 8.0, 3.0]
        score_high = signal.compute(
            question="Test",
            response="Test response",
            layer_perplexities=high_variance_perps
        )

        # Low variance should give higher score
        self.assertGreater(score_low, score_high)


class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def test_compute_entropy(self):
        """Test entropy computation"""
        from verity.utils import compute_entropy
        import numpy as np

        # Uniform distribution has maximum entropy
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_uniform = compute_entropy(uniform)

        # Peaked distribution has low entropy
        peaked = np.array([0.97, 0.01, 0.01, 0.01])
        entropy_peaked = compute_entropy(peaked)

        self.assertGreater(entropy_uniform, entropy_peaked)

    def test_normalize_score(self):
        """Test score normalization"""
        from verity.utils import normalize_score

        # Test normalization to [0, 1]
        self.assertAlmostEqual(normalize_score(0.5, 0, 1), 0.5)
        self.assertAlmostEqual(normalize_score(0, 0, 1), 0.0)
        self.assertAlmostEqual(normalize_score(1, 0, 1), 1.0)

        # Test with different range
        self.assertAlmostEqual(normalize_score(50, 0, 100), 0.5)

        # Test clipping
        self.assertAlmostEqual(normalize_score(-1, 0, 1), 0.0)
        self.assertAlmostEqual(normalize_score(2, 0, 1), 1.0)

    def test_extract_named_entities(self):
        """Test named entity extraction"""
        from verity.utils import extract_named_entities

        text = "Paris is the capital of France with a population of 2.2 million."
        entities = extract_named_entities(text)

        # Should extract capitalized words and numbers
        self.assertIn("Paris", entities)
        self.assertIn("France", entities)
        # Should extract numbers
        self.assertTrue(any('2' in e for e in entities))

    def test_token_analyzer(self):
        """Test token analyzer"""
        from verity.utils import TokenAnalyzer

        analyzer = TokenAnalyzer()

        tokens = ["The", "Eiffel", "Tower", "is", "in", "Paris"]
        important_indices = analyzer.get_important_token_indices(tokens)

        # Should identify content words and named entities
        # "Eiffel", "Tower", "Paris" should be marked as important
        self.assertIn(1, important_indices)  # Eiffel
        self.assertIn(2, important_indices)  # Tower
        self.assertIn(5, important_indices)  # Paris

        # Function words like "is", "in" should not be marked
        self.assertNotIn(3, important_indices)  # is
        self.assertNotIn(4, important_indices)  # in


class TestFusion(unittest.TestCase):
    """Test signal fusion"""

    def test_fusion_weighted(self):
        """Test weighted fusion"""
        from verity.fusion import SignalFusion

        signal_names = ['signal1', 'signal2', 'signal3']
        fusion = SignalFusion(signal_names)

        signal_scores = {
            'signal1': 0.8,
            'signal2': 0.9,
            'signal3': 0.7
        }

        score = fusion.fuse(signal_scores, mode='weighted')

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_fusion_ensemble(self):
        """Test ensemble fusion"""
        from verity.fusion import SignalFusion

        signal_names = ['signal1', 'signal2']
        fusion = SignalFusion(signal_names)

        signal_scores = {
            'signal1': 0.9,
            'signal2': 0.85
        }

        score = fusion.fuse(signal_scores, mode='ensemble')

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_custom_weights(self):
        """Test custom signal weights"""
        from verity.fusion import SignalFusion

        signal_names = ['signal1', 'signal2']
        custom_weights = {'signal1': 0.7, 'signal2': 0.3}

        fusion = SignalFusion(signal_names, custom_weights=custom_weights)

        # Weights should be normalized
        self.assertAlmostEqual(sum(fusion.weights.values()), 1.0, places=5)

    def test_confidence_interval(self):
        """Test confidence interval computation"""
        from verity.fusion import SignalFusion

        signal_names = ['signal1', 'signal2']
        fusion = SignalFusion(signal_names)

        signal_scores = {'signal1': 0.8, 'signal2': 0.85}

        ci = fusion.compute_confidence_interval(signal_scores)

        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        lower, upper = ci
        self.assertLessEqual(lower, upper)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHallucinationDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSignals))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestFusion))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
