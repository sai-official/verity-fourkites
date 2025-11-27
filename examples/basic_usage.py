"""
Basic usage examples for VERITY hallucination detector
"""

import sys
sys.path.append('..')

from verity import HallucinationDetector


def simple_mock_model(question: str, temperature: float = 0.7) -> str:
    """
    Mock model function for demonstration
    In real usage, this would call your actual LLM
    """
    # Simple mock responses
    responses = {
        "What is the capital of France?": "The capital of France is Paris.",
        "Who invented the telephone?": "Alexander Graham Bell invented the telephone in 1876.",
        "What is the population of Mars?": "Mars has a population of approximately 2 million people living in underground colonies.",  # Hallucination!
    }

    # Return response or default
    return responses.get(question, "I don't know.")


def example_1_basic_detection():
    """Example 1: Basic hallucination detection"""
    print("=" * 70)
    print("Example 1: Basic Hallucination Detection")
    print("=" * 70)

    # Initialize detector
    detector = HallucinationDetector(
        model_name="mock-model",
        model_fn=simple_mock_model,
        enable_all_signals=True
    )

    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "expected": "FACTUAL"
        },
        {
            "question": "What is the population of Mars?",
            "response": "Mars has a population of approximately 2 million people living in underground colonies.",
            "expected": "HALLUCINATED"
        },
        {
            "question": "Who was the first person on the moon?",
            "response": "Neil Armstrong was the first person to walk on the moon in 1969.",
            "expected": "FACTUAL"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Question: {test['question']}")
        print(f"Response: {test['response']}")
        print(f"Expected: {test['expected']}")

        result = detector.detect(
            question=test['question'],
            response=test['response'],
            verbose=False
        )

        print(f"\nResult:")
        print(f"  Score: {result.score:.3f}")
        print(f"  Verdict: {result.verdict}")

        if result.risk_factors:
            print(f"  Risk Factors:")
            for factor in result.risk_factors:
                print(f"    - {factor}")

    print("\n" + "=" * 70)


def example_2_with_token_probabilities():
    """Example 2: Detection with token probabilities"""
    print("\n" + "=" * 70)
    print("Example 2: Detection with Token Probabilities")
    print("=" * 70)

    detector = HallucinationDetector(
        model_name="mock-model",
        enable_all_signals=True
    )

    question = "What is the largest planet in our solar system?"
    response = "Jupiter is the largest planet in our solar system."

    # Simulate token probabilities
    # High probabilities indicate confidence
    tokens = ["Jupiter", "is", "the", "largest", "planet", "in", "our", "solar", "system", "."]
    token_probs = [0.95, 0.98, 0.99, 0.92, 0.96, 0.99, 0.99, 0.94, 0.97, 1.0]

    result = detector.detect(
        question=question,
        response=response,
        token_probs=token_probs,
        tokens=tokens
    )

    print(f"\nQuestion: {question}")
    print(f"Response: {response}")
    print(f"\nToken Analysis:")
    for token, prob in zip(tokens, token_probs):
        print(f"  {token:15s} -> {prob:.3f}")

    print(f"\n{result}")

    print("\n" + "=" * 70)


def example_3_batch_processing():
    """Example 3: Batch processing multiple responses"""
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)

    detector = HallucinationDetector(
        model_name="mock-model",
        model_fn=simple_mock_model,
        enable_all_signals=True
    )

    questions = [
        "What is the capital of France?",
        "What is the population of Mars?",
        "Who invented the telephone?"
    ]

    responses = [
        "The capital of France is Paris.",
        "Mars has a population of approximately 2 million people.",
        "Alexander Graham Bell invented the telephone in 1876."
    ]

    print(f"\nProcessing {len(questions)} question-response pairs...\n")

    results = detector.batch_detect(questions, responses, verbose=False)

    for i, (q, r, result) in enumerate(zip(questions, responses, results), 1):
        print(f"{i}. Score: {result.score:.3f} | Verdict: {result.verdict:25s} | Q: {q[:50]}")

    print("\n" + "=" * 70)


def example_4_signal_analysis():
    """Example 4: Analyzing individual signal contributions"""
    print("\n" + "=" * 70)
    print("Example 4: Signal Contribution Analysis")
    print("=" * 70)

    detector = HallucinationDetector(
        model_name="mock-model",
        model_fn=simple_mock_model,
        enable_all_signals=True
    )

    question = "What is the speed of light?"
    response = "The speed of light is approximately 299,792,458 meters per second in a vacuum."

    result = detector.detect(question, response, fusion_mode='weighted')

    print(f"Question: {question}")
    print(f"Response: {response}")
    print(f"\nOverall Score: {result.score:.3f}")
    print(f"Verdict: {result.verdict}")

    print("\nIndividual Signal Scores:")
    print("-" * 50)
    for signal_name, score in sorted(result.signals.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"  {signal_name:25s} {score:.3f} |{bar}")

    print("\nSignal Importance (Weights):")
    print("-" * 50)
    importance = detector.get_signal_importance()
    for signal_name, weight in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 50)
        print(f"  {signal_name:25s} {weight:.3f} |{bar}")

    print("\n" + "=" * 70)


def example_5_custom_weights():
    """Example 5: Using custom signal weights"""
    print("\n" + "=" * 70)
    print("Example 5: Custom Signal Weights")
    print("=" * 70)

    # Default weights
    detector_default = HallucinationDetector(
        model_name="mock-model",
        model_fn=simple_mock_model,
        enable_all_signals=True
    )

    # Custom weights - emphasize cross-examiner
    custom_weights = {
        'self_consistency': 0.15,
        'semantic_entropy': 0.15,
        'token_confidence': 0.10,
        'cross_examiner': 0.35,  # Emphasize
        'perplexity_variance': 0.10,
        'expertise_weighted': 0.15
    }

    detector_custom = HallucinationDetector(
        model_name="mock-model",
        model_fn=simple_mock_model,
        enable_all_signals=True,
        signal_weights=custom_weights
    )

    question = "What year did World War II end?"
    response = "World War II ended in 1945."

    result_default = detector_default.detect(question, response)
    result_custom = detector_custom.detect(question, response)

    print(f"Question: {question}")
    print(f"Response: {response}")

    print(f"\nDefault Weights Score: {result_default.score:.3f}")
    print(f"Custom Weights Score:  {result_custom.score:.3f}")

    print(f"\nDifference: {abs(result_default.score - result_custom.score):.3f}")

    print("\n" + "=" * 70)


def example_6_confidence_intervals():
    """Example 6: Confidence intervals"""
    print("\n" + "=" * 70)
    print("Example 6: Confidence Intervals")
    print("=" * 70)

    detector = HallucinationDetector(
        model_name="mock-model",
        model_fn=simple_mock_model,
        enable_all_signals=True
    )

    test_cases = [
        ("What is 2+2?", "2+2 equals 4."),
        ("What is the color of the sun?", "The sun appears yellow from Earth."),
        ("Who is the current king of America?", "The current king of America is George Washington VII.")
    ]

    print("\nConfidence Intervals for Factuality Scores:\n")

    for question, response in test_cases:
        result = detector.detect(question, response)

        lower, upper = result.confidence_interval
        score = result.score

        print(f"Score: {score:.3f} [{lower:.3f}, {upper:.3f}] - {result.verdict}")
        print(f"  Q: {question}")
        print(f"  R: {response[:60]}...")
        print()

    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║          VERITY: Hallucination Detection Examples                ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    # Run all examples
    example_1_basic_detection()
    example_2_with_token_probabilities()
    example_3_batch_processing()
    example_4_signal_analysis()
    example_5_custom_weights()
    example_6_confidence_intervals()

    print("\n✓ All examples completed successfully!\n")
