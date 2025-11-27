"""
Benchmark tests for VERITY on synthetic hallucination dataset
"""

import sys
sys.path.append('..')

import numpy as np
from typing import List, Dict
from verity import HallucinationDetector


class SyntheticHallucinationDataset:
    """
    Generate synthetic hallucination examples for testing
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_dataset(self, n_samples: int = 100) -> List[Dict]:
        """
        Generate synthetic dataset with hallucinations

        Returns:
            List of dicts with 'question', 'response', 'label', 'category'
        """
        dataset = []

        # Category 1: Factual responses (label=1)
        factual_examples = [
            {
                "question": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "label": 1,
                "category": "factual"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "response": "Romeo and Juliet was written by William Shakespeare.",
                "label": 1,
                "category": "factual"
            },
            {
                "question": "What is 2+2?",
                "response": "2+2 equals 4.",
                "label": 1,
                "category": "factual"
            },
            {
                "question": "What is the speed of light?",
                "response": "The speed of light is approximately 299,792,458 meters per second.",
                "label": 1,
                "category": "factual"
            },
            {
                "question": "What is the largest planet in our solar system?",
                "response": "Jupiter is the largest planet in our solar system.",
                "label": 1,
                "category": "factual"
            },
        ]

        # Category 2: Full hallucinations (label=0)
        hallucinated_examples = [
            {
                "question": "What is the population of Mars?",
                "response": "Mars has a permanent population of approximately 2 million people living in underground colonies.",
                "label": 0,
                "category": "full_hallucination"
            },
            {
                "question": "Who is the current king of America?",
                "response": "The current king of America is George Washington VII, who inherited the throne in 2010.",
                "label": 0,
                "category": "full_hallucination"
            },
            {
                "question": "What is the capital of Atlantis?",
                "response": "The capital of Atlantis is Poseidonia, located at the center of the island.",
                "label": 0,
                "category": "full_hallucination"
            },
            {
                "question": "When did the moon landing hoax get revealed?",
                "response": "The moon landing hoax was officially revealed by NASA in 1999.",
                "label": 0,
                "category": "full_hallucination"
            },
            {
                "question": "What is the average lifespan of a unicorn?",
                "response": "Unicorns typically live for 200-300 years in the wild.",
                "label": 0,
                "category": "full_hallucination"
            },
        ]

        # Category 3: Partial hallucinations (label=0)
        partial_hallucinations = [
            {
                "question": "When was the Eiffel Tower built?",
                "response": "The Eiffel Tower was built in 1889 by Leonardo da Vinci.",
                "label": 0,
                "category": "partial_hallucination"
            },
            {
                "question": "What is the atomic number of gold?",
                "response": "Gold has an atomic number of 79 and was discovered in ancient Egypt in 3000 BC.",
                "label": 0,  # Date is wrong/misleading
                "category": "partial_hallucination"
            },
            {
                "question": "How many continents are there?",
                "response": "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, and Atlantis.",
                "label": 0,
                "category": "partial_hallucination"
            },
        ]

        # Category 4: Uncertain/Vague responses (label=0)
        uncertain_examples = [
            {
                "question": "What causes gravity?",
                "response": "Gravity is caused by something.",
                "label": 0,
                "category": "uninformative"
            },
            {
                "question": "How does photosynthesis work?",
                "response": "Photosynthesis is a process that happens in plants.",
                "label": 0,
                "category": "uninformative"
            },
        ]

        # Combine all examples
        all_examples = (
            factual_examples * (n_samples // 20) +
            hallucinated_examples * (n_samples // 20) +
            partial_hallucinations * (n_samples // 20) +
            uncertain_examples * (n_samples // 20)
        )

        # Shuffle and return
        np.random.shuffle(all_examples)
        return all_examples[:n_samples]


def run_benchmark(n_samples: int = 50, verbose: bool = True):
    """
    Run benchmark on synthetic dataset

    Args:
        n_samples: Number of samples to test
        verbose: Print detailed results
    """
    print("=" * 70)
    print("VERITY Benchmark on Synthetic Hallucination Dataset")
    print("=" * 70)

    # Generate dataset
    dataset_gen = SyntheticHallucinationDataset(seed=42)
    dataset = dataset_gen.generate_dataset(n_samples)

    print(f"\nDataset size: {len(dataset)} samples")

    # Count categories
    category_counts = {}
    for item in dataset:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat:25s}: {count:3d} samples")

    # Initialize detector
    print("\nInitializing VERITY detector...")
    detector = HallucinationDetector(
        model_name="benchmark-model",
        enable_all_signals=False  # Disable signals requiring model_fn for speed
    )

    # Run detection on all samples
    print("\nRunning hallucination detection...")

    predictions = []
    labels = []
    scores = []
    results_by_category = {cat: {'scores': [], 'labels': []} for cat in category_counts.keys()}

    for i, item in enumerate(dataset):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples...")

        result = detector.detect(
            question=item['question'],
            response=item['response'],
            verbose=False
        )

        # Threshold at 0.5 for binary classification
        pred = 1 if result.score >= 0.5 else 0

        predictions.append(pred)
        labels.append(item['label'])
        scores.append(result.score)

        # Store by category
        cat = item['category']
        results_by_category[cat]['scores'].append(result.score)
        results_by_category[cat]['labels'].append(item['label'])

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score
    )

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = 0.0

    print("\nOverall Performance:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                  Halluc.  Factual")
    print(f"  Actual Halluc.  {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"         Factual  {cm[1][0]:4d}     {cm[1][1]:4d}")

    # Performance by category
    print("\nPerformance by Category:")
    print("-" * 70)

    for cat in sorted(results_by_category.keys()):
        cat_scores = results_by_category[cat]['scores']
        cat_labels = results_by_category[cat]['labels']

        if not cat_scores:
            continue

        cat_preds = [1 if s >= 0.5 else 0 for s in cat_scores]
        cat_acc = accuracy_score(cat_labels, cat_preds)
        mean_score = np.mean(cat_scores)

        print(f"  {cat:25s}: Acc={cat_acc:.3f}, Mean Score={mean_score:.3f}")

    # Score distribution
    print("\nScore Distribution:")
    print("-" * 70)

    factual_scores = [s for s, l in zip(scores, labels) if l == 1]
    halluc_scores = [s for s, l in zip(scores, labels) if l == 0]

    if factual_scores:
        print(f"  Factual responses:      Mean={np.mean(factual_scores):.3f}, "
              f"Std={np.std(factual_scores):.3f}")
    if halluc_scores:
        print(f"  Hallucinated responses: Mean={np.mean(halluc_scores):.3f}, "
              f"Std={np.std(halluc_scores):.3f}")

    # Examples
    if verbose:
        print("\n" + "=" * 70)
        print("Example Detections (Top 5 Highest Confidence)")
        print("=" * 70)

        sorted_indices = np.argsort(scores)[::-1][:5]

        for i in sorted_indices:
            item = dataset[i]
            score = scores[i]
            pred = predictions[i]
            label = labels[i]

            print(f"\nScore: {score:.3f} | Pred: {'Factual' if pred == 1 else 'Halluc'} | "
                  f"True: {'Factual' if label == 1 else 'Halluc'}")
            print(f"Q: {item['question']}")
            print(f"R: {item['response'][:80]}...")

        print("\n" + "=" * 70)
        print("Example Detections (Top 5 Lowest Confidence)")
        print("=" * 70)

        sorted_indices = np.argsort(scores)[:5]

        for i in sorted_indices:
            item = dataset[i]
            score = scores[i]
            pred = predictions[i]
            label = labels[i]

            print(f"\nScore: {score:.3f} | Pred: {'Factual' if pred == 1 else 'Halluc'} | "
                  f"True: {'Factual' if label == 1 else 'Halluc'}")
            print(f"Q: {item['question']}")
            print(f"R: {item['response'][:80]}...")

    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


if __name__ == "__main__":
    # Run benchmark
    metrics = run_benchmark(n_samples=50, verbose=True)

    print("\n✓ Benchmark completed successfully!")
    print(f"\nKey Metrics:")
    print(f"  - Accuracy: {metrics['accuracy']:.1%}")
    print(f"  - F1 Score: {metrics['f1']:.3f}")
    print(f"  - AUC-ROC:  {metrics['auc']:.3f}")
