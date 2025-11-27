#!/usr/bin/env python3
"""
VERITY Quick Start Demo
=======================

A minimal example showing how to use VERITY for hallucination detection.
"""

from verity import HallucinationDetector


def mock_llm(question: str, temperature: float = 0.7) -> str:
    """
    Mock LLM for demonstration purposes.
    Replace this with your actual LLM API call.
    """
    responses = {
        "What is the capital of France?": "The capital of France is Paris, which is located in the north-central part of the country.",
        "What is the population of the Moon?": "The Moon has a permanent population of about 500,000 people living in lunar colonies.",
        "Who wrote 'To Kill a Mockingbird'?": "Harper Lee wrote 'To Kill a Mockingbird', which was published in 1960.",
        "What is the speed of light?": "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
    }
    return responses.get(question, "I don't have information about that.")


def main():
    print("=" * 70)
    print("VERITY Hallucination Detector - Quick Start Demo")
    print("=" * 70)
    print()

    # Initialize detector
    print("Initializing VERITY detector...")
    detector = HallucinationDetector(
        model_name="demo-model",
        model_fn=mock_llm,
        enable_all_signals=True
    )
    print("✓ Detector initialized\n")

    # Test cases
    test_cases = [
        {
            "question": "What is the capital of France?",
            "expected": "✓ FACTUAL"
        },
        {
            "question": "What is the population of the Moon?",
            "expected": "✗ HALLUCINATED"
        },
        {
            "question": "Who wrote 'To Kill a Mockingbird'?",
            "expected": "✓ FACTUAL"
        },
        {
            "question": "What is the speed of light?",
            "expected": "✓ FACTUAL"
        }
    ]

    # Run detection
    print("-" * 70)
    print("Running Hallucination Detection")
    print("-" * 70)
    print()

    for i, test in enumerate(test_cases, 1):
        question = test['question']
        response = mock_llm(question)

        print(f"[{i}/{len(test_cases)}] Question: {question}")
        print(f"     Response: {response[:80]}...")

        # Detect hallucination
        result = detector.detect(
            question=question,
            response=response,
            verbose=False
        )

        # Print result
        score_bar = "█" * int(result.score * 30)
        print(f"     Score: {result.score:.3f} |{score_bar}")
        print(f"     Verdict: {result.verdict}")
        print(f"     Expected: {test['expected']}")

        # Show signal breakdown
        print("     Signals:", end="")
        for signal, value in sorted(result.signals.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f" {signal}={value:.2f}", end="")
        print()

        if result.risk_factors:
            print(f"     Risk Factors: {', '.join(result.risk_factors[:2])}")

        print()

    print("-" * 70)
    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("  1. Replace mock_llm() with your actual LLM API")
    print("  2. See examples/basic_usage.py for more examples")
    print("  3. Run tests: python tests/test_detector.py")
    print("  4. Read README.md for detailed documentation")
    print()


if __name__ == "__main__":
    main()
