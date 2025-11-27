# Related Work: Hallucination Detection in LLMs

A comprehensive survey of research papers related to hallucination detection, organized by approach and methodology.


##  Table of Contents

1. [Foundational Papers](#foundational-papers)
2. [Intrinsic Detection Methods](#intrinsic-detection-methods)
3. [Survey Papers](#survey-papers)

---

## Foundational Papers

### 1. **FEWL: Expertise-Weighted Evaluation** (2024)

**Paper**: [Measuring and Reducing LLM Hallucination without Gold-Standard Answers](https://arxiv.org/html/2402.10412v1)

**Authors**: Cheng, Jiaheng et al.

**Key Contributions**:
- First metric operating without ground-truth labels
- Expertise-weighted agreement across reference LLMs
- Laziness penalty for superficial knowledge
- Achieved 72.66% accuracy on TruthfulQA

**Methodology**:
```
FEWL = Σ(λᵢ × similarity(response, refᵢ)) - β × laziness_penalty
```

**Limitations**:
- Requires multiple reference models
- Computationally expensive
- Single signal approach

**Impact**: Foundational for VERITY's expertise-weighted signal

---

### 2. **SelfCheckGPT: Zero-Resource Detection** (2023)

**Paper**: [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection](https://arxiv.org/abs/2303.08896)

**Authors**: Manakul, Potsawee et al.

**Key Idea**: Sample multiple responses; inconsistency indicates hallucination.

**Approach**:
- Generate N responses with temperature > 0
- Measure consistency across responses
- No external knowledge required

**Variants**:
1. **BERTScore**: Semantic similarity scoring
2. **QA-based**: Question-answering consistency
3. **N-gram**: Token overlap consistency

**Results**: ~68% accuracy on factual QA tasks

**Limitation**: Cannot detect confident hallucinations (model consistently wrong)

**VERITY Enhancement**: Combines with 5 other signals to catch consistent hallucinations


**Paper**: [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)

**Purpose**: Questions where models often hallucinate

**Size**: 817 questions across 38 categories

**Format**: Multiple-choice + generation

**Categories**:
- Misconceptions
- Fiction
- Conspiracies
- Misinformation

**Example**:
```
Q: "What happens if you crack your knuckles?"
Truthful: "Nothing harmful"
Common hallucination: "You'll get arthritis"
```

**Usage**: Standard benchmark for VERITY



## Survey Papers

### 3. **Comprehensive Survey of Hallucination** (2024) [Read only the summary of it which is provide below]

**Paper**: [A Comprehensive Survey of Hallucination in LLMs](https://arxiv.org/html/2510.06265v1)

**Taxonomy**:

**By Cause**:
- Data-related (noisy training data)
- Model-related (capacity, attention)
- Inference-related (decoding strategy)

**By Type**:
- Factual (incorrect facts)
- Faithfulness (unfaithful to source)
- Instruction-following (ignores constraints)

**Detection Categories**:
1. Intrinsic (self-consistency, perplexity)
2. External (retrieval, tool-use)
3. Hybrid (VERITY approach)







