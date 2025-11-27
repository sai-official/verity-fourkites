# Research Paper Analysis & Improvements

## Base Paper: FEWL Framework

**Title**: "Measuring and Reducing LLM Hallucination without Gold-Standard Answers via Expertise-Weighting"

**Paper URL**: https://arxiv.org/html/2402.10412v1

---

## Key Contributions of the Base Paper

### 1. **Expertise-Weighted Agreement**

The FEWL framework introduced the novel idea of using multiple reference LLMs as proxies for ground truth, with each weighted by their demonstrated expertise:

```
FEWL_score = Σ(λᵢ × similarity(response, refᵢ)) - β × laziness_penalty
```

Where:
- `λᵢ` = expertise weight for reference model i
- Expertise measured by ability to distinguish incorrect vs corrected answers
- Laziness penalty for superficial knowledge patterns

### 2. **Operating Without Ground Truth**

First hallucination metric designed for scenarios without gold-standard answers, addressing:
- High cost of manual annotation
- Errors in human-created labels
- Need for scalable evaluation

### 3. **Theoretical Guarantees**

Theorem 3.4 proves that FEWL correctly ranks models by factuality, matching gold-standard evaluation outcomes.

### 4. **Practical Performance**

On TruthfulQA and CHALE datasets:
- **72.66%** accuracy distinguishing factual from hallucinated
- Successful model-level ranking
- Enables hallucination reduction via in-context learning (56.2% win rate)

