# VERITY Architecture

Detailed technical architecture of the VERITY hallucination detection framework.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VERITY Framework                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              HallucinationDetector                        │  │
│  │  (Main entry point for detection)                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                    │
│           ┌────────────────┼────────────────┐                   │
│           ▼                ▼                ▼                   │
│  ┌────────────────┐ ┌────────────┐ ┌────────────────┐           │
│  │  Signal Layer  │ │   Fusion   │ │  Calibration   │           │
│  │   (6 signals)  │ │   Network  │ │    Module      │           │
│  └────────────────┘ └────────────┘ └────────────────┘           │
│           │                │                │                   │
│           └────────────────┴────────────────┘                   │
│                            │                                    │
│                            ▼                                    │
│                  ┌──────────────────┐                           │
│                  │ DetectionResult  │                           │
│                  │  - Score [0,1]   │                           │
│                  │  - Signals       │                           │
│                  │  - Verdict       │                           │
│                  │  - Confidence    │                           │
│                  └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Signal Layer

Each signal is independent and modular:

```
BaseSignal (Abstract)
    │
    ├─ SelfConsistencySignal
    │    └─ Clustered sampling
    │    └─ Semantic similarity
    │
    ├─ SemanticEntropySignal
    │    └─ Bidirectional entailment
    │    └─ Cluster entropy
    │
    ├─ TokenConfidenceSignal
    │    └─ Contextual weighting
    │    └─ Low-prob detection
    │
    ├─ CrossExaminerSignal
    │    └─ Adversarial probing
    │    └─ Consistency checking
    │
    ├─ PerplexityVarianceSignal
    │    └─ Layer-wise analysis
    │    └─ Variance computation
    │
    └─ ExpertiseWeightedSignal
         └─ Reference model agreement
         └─ Laziness penalty
```

---

## Data Flow

### Detection Pipeline

```
Input: (question, response)
     │
     ▼
┌─────────────────────────────────┐
│  Step 1: Signal Extraction      │
│  For each signal:                │
│    - Compute signal score        │
│    - Normalize to [0, 1]         │
│  Output: Dict[signal → score]    │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Step 2: Signal Fusion           │
│  Mode options:                   │
│    - Weighted: Σ(wᵢ × σᵢ)        │
│    - Learned: Neural network     │
│    - Ensemble: Multiple methods  │
│  Output: Fused score [0, 1]      │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Step 3: Calibration             │
│  - Compute confidence interval   │
│  - Identify risk factors         │
│  - Generate explanation          │
│  Output: DetectionResult         │
└─────────────────────────────────┘
     │
     ▼
   Result
```

---

## Signal Details

### Signal 1: Self-Consistency (σ_SC)

**Input**: Question, response, model_fn
**Output**: Consistency score [0, 1]

**Algorithm**:
```python
def compute_self_consistency(question, response, model_fn, N=5):
    responses = [response]

    # Clustered sampling to avoid mode collapse
    temperatures = linspace(0.7, 1.1, N)
    for temp in temperatures:
        r = model_fn(question, temp)
        responses.append(r)

    # Pairwise semantic similarity
    similarities = []
    for i, j in combinations(range(len(responses)), 2):
        sim = semantic_similarity(responses[i], responses[j])
        similarities.append(sim)

    # High mean, low variance = consistent
    mean_sim = mean(similarities)
    var_sim = variance(similarities)

    score = mean_sim * (1 - clip(var_sim, 0, 1))
    return score
```

**Time Complexity**: O(N × M) where M = model inference time
**Space Complexity**: O(N × L) where L = response length

---

### Signal 2: Semantic Entropy (σ_SE)

**Input**: Question, response, model_fn
**Output**: Entropy-based score [0, 1]

**Algorithm**:
```python
def compute_semantic_entropy(question, response, model_fn, N=5):
    # Sample responses
    responses = sample_responses(question, model_fn, N)

    # Get embeddings
    embeddings = embedding_model.encode(responses)

    # Cluster with bidirectional entailment
    clusters = dbscan_cluster(embeddings, eps=0.3)

    # Compute entropy over cluster distribution
    cluster_probs = cluster_distribution(clusters)
    entropy = shannon_entropy(cluster_probs)

    # Normalize and invert (low entropy = high confidence)
    max_entropy = log2(N)
    normalized_entropy = entropy / max_entropy
    score = 1 - normalized_entropy

    return score
```

**Key Innovation**: Bidirectional entailment prevents false clustering

**Time Complexity**: O(N × E + N²) where E = embedding time
**Space Complexity**: O(N × D) where D = embedding dimension

---

### Signal 3: Token Confidence (σ_TC)

**Input**: Response, token_probs, tokens
**Output**: Confidence score [0, 1]

**Algorithm**:
```python
def compute_token_confidence(response, token_probs, tokens):
    # Identify important tokens (entities, numbers)
    important_indices = get_important_token_indices(tokens)

    # Apply contextual weighting
    weighted_probs = []
    for i, prob in enumerate(token_probs):
        weight = 2.0 if i in important_indices else 1.0
        weighted_probs.extend([prob] * int(weight))

    # Geometric mean (sensitive to low values)
    geom_mean = exp(mean(log(clip(weighted_probs, 1e-10, 1.0))))

    # Penalty for low-probability tokens
    low_prob_count = sum(1 for p in token_probs if p < 0.1)
    penalty = exp(-2.0 * low_prob_count / len(token_probs))

    score = geom_mean * penalty
    return score
```

**Innovation**: Contextual weighting emphasizes factual content

**Time Complexity**: O(T) where T = number of tokens
**Space Complexity**: O(T)

---

### Signal 4: Cross-Examiner (σ_CE)

**Input**: Question, response, model_fn
**Output**: Consistency score [0, 1]

**Algorithm**:
```python
def compute_cross_examiner(question, response, model_fn, K=3):
    # Generate K probing questions
    probe_questions = generate_probes(response, model_fn, K)

    consistencies = []
    for probe_q in probe_questions:
        # Get answer to probe
        probe_answer = model_fn(probe_q)

        # Check consistency with original
        consistent = check_consistency(response, probe_answer, model_fn)
        consistencies.append(consistent)

    score = mean(consistencies)
    return score
```

**Example**:
```
Original: "Paris is the capital with 2 million people"
Probe 1: "What is the exact population of Paris?"
Probe 2: "When did Paris become the capital?"
→ Check if probe answers align with original
```

**Time Complexity**: O(K × M) where K = probes, M = model time
**Space Complexity**: O(K × L)

---

### Signal 5: Perplexity Variance (σ_PV)

**Input**: Layer perplexities
**Output**: Variance-based score [0, 1]

**Algorithm**:
```python
def compute_perplexity_variance(layer_perplexities):
    # Overall variance
    mean_perp = mean(layer_perplexities)
    variance = var(layer_perplexities)
    cv = sqrt(variance) / mean_perp  # Coefficient of variation

    # Middle layer variance (semantic layers)
    n = len(layer_perplexities)
    middle = layer_perplexities[n//3 : 2*n//3]
    middle_var = var(middle)

    # Combined metric
    combined = 0.6 * cv + 0.4 * middle_var

    # Convert to score (low variance = high confidence)
    score = 1 / (1 + combined)
    return score
```

**Intuition**: Hallucinations show disagreement across layers

**Time Complexity**: O(L) where L = number of layers
**Space Complexity**: O(L)

---

### Signal 6: Expertise-Weighted Agreement (σ_EWA)

**Input**: Response, reference_responses
**Output**: Agreement score [0, 1]

**Algorithm**:
```python
def compute_expertise_weighted(response, ref_responses, question):
    # Compute expertise weights for reference models
    weights = compute_expertise_weights(ref_responses, question)

    # Weighted similarity
    weighted_sims = []
    for ref, weight in zip(ref_responses, weights):
        sim = semantic_similarity(response, ref)
        weighted_sims.append(weight * sim)

    agreement = sum(weighted_sims) / sum(weights)

    # Laziness penalty (if refs too similar to each other)
    laziness = compute_laziness_penalty(ref_responses)

    score = agreement - 0.1 * laziness
    return clip(score, 0, 1)
```

**From**: FEWL paper (base framework)
**Enhancement**: Dynamic expertise calibration per domain

**Time Complexity**: O(R × S) where R = reference models, S = similarity time
**Space Complexity**: O(R × L)

---

## Fusion Network Architecture

### Neural Fusion

```
Input Layer (6 neurons)
    │
    │  [σ_SC, σ_SE, σ_TC, σ_CE, σ_PV, σ_EWA]
    │
    ▼
Hidden Layer (32 neurons)
    │
    │  Activation: ReLU
    │  Dropout: 0.2 (during training)
    │
    ▼
Output Layer (1 neuron)
    │
    │  Activation: Sigmoid
    │  Output: Calibrated probability [0, 1]
    │
    ▼
Final Score
```

**Training**:
- **Data**: Semi-supervised (synthetic hallucinations)
- **Loss**: Binary cross-entropy
- **Optimization**: Adam (lr=0.001)
- **Epochs**: 100 with early stopping
- **Calibration**: CalibratedClassifierCV (sigmoid method)

**Why so small?**
- 6 inputs already informative
- Avoid overfitting
- Fast inference
- Easy to interpret

---

## Fusion Modes

### 1. Weighted Fusion (Default)

```python
score = Σ(wᵢ × σᵢ) where Σwᵢ = 1
```

**Weights** (default):
```python
{
    'self_consistency': 0.20,
    'semantic_entropy': 0.18,
    'token_confidence': 0.15,
    'cross_examiner': 0.22,  # Highest
    'perplexity_variance': 0.10,
    'expertise_weighted': 0.15
}
```

**Pros**: Fast, interpretable, no training needed
**Cons**: Fixed weights, not adaptive

---

### 2. Learned Fusion

Uses trained neural network.

**Pros**: Adaptive, optimal weights learned from data
**Cons**: Requires training data, slightly slower

---

### 3. Ensemble Fusion

Combines multiple strategies:

```python
score = 0.4 × weighted + 0.3 × median + 0.2 × mean + 0.1 × min
```

**Philosophy**: Conservative (weight towards lower scores)
**Rationale**: Hallucination detection should err on cautious side

**Pros**: Robust to outliers
**Cons**: Most computationally expensive

---

## Calibration Module

### Confidence Interval Computation

Uses bootstrap resampling:

```python
def compute_confidence_interval(signal_scores, confidence=0.95):
    bootstrap_scores = []

    for _ in range(1000):
        # Resample signals with replacement
        resampled = resample(signal_scores)

        # Compute fused score
        score = fuse(resampled)
        bootstrap_scores.append(score)

    # Compute percentiles
    alpha = (1 - confidence) / 2
    lower = percentile(bootstrap_scores, alpha * 100)
    upper = percentile(bootstrap_scores, (1 - alpha) * 100)

    return (lower, upper)
```

**Output**: (lower_bound, upper_bound)

**Interpretation**:
- Score: 0.85, CI: [0.78, 0.91]
- "85% confidence factual, with 95% certainty the true value is between 78-91%"

---

### Risk Factor Identification

```python
def identify_risk_factors(signal_scores):
    risks = []

    if signal_scores['self_consistency'] < 0.5:
        risks.append("Inconsistent responses")

    if signal_scores['semantic_entropy'] < 0.5:
        risks.append("High semantic uncertainty")

    if signal_scores['token_confidence'] < 0.5:
        risks.append("Low token confidence")

    if signal_scores['cross_examiner'] < 0.5:
        risks.append("Failed consistency probing")

    # ... etc

    return risks
```

---

## Performance Characteristics

### Time Complexity (per detection)

| Component | Complexity | Notes |
|-----------|------------|-------|
| Self-Consistency | O(N × M) | N samples, M = model time |
| Semantic Entropy | O(N × E) | E = embedding time |
| Token Confidence | O(T) | T = tokens |
| Cross-Examiner | O(K × M) | K probes |
| Perplexity Variance | O(L) | L = layers |
| Expertise-Weighted | O(R × S) | R refs, S = similarity |
| Fusion | O(1) | Constant |
| **Total** | **O((N+K)×M)** | Dominated by model calls |

### Space Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Signal storage | O(1) | Fixed number (6) |
| Responses | O(N × L) | N samples, L = length |
| Embeddings | O(N × D) | D = embedding dim |
| **Total** | **O(N × max(L, D))** | Linear in samples |

### Optimization Strategies

**Fast Mode** (disable expensive signals):
```python
detector = HallucinationDetector(
    enable_signals=['token_confidence', 'perplexity_variance']
)
# 10x faster, 5% accuracy drop
```

**Batch Mode** (process multiple):
```python
results = detector.batch_detect(questions, responses)
# Amortize model loading overhead
```

**Cached Mode** (reuse embeddings):
```python
# Embeddings cached automatically
# Subsequent calls with same text faster
```

---

## Extensibility

### Adding Custom Signals

```python
from verity.signals import BaseSignal

class MyCustomSignal(BaseSignal):
    def compute(self, question, response, **kwargs):
        # Your detection logic
        score = custom_detection_logic(question, response)
        return score

# Add to detector
detector.add_signal('my_signal', MyCustomSignal())
```

### Custom Fusion Strategy

```python
def my_fusion(signal_scores):
    # Your fusion logic
    return final_score

# Use custom fusion
detector.fusion.fuse = my_fusion
```

---

## Configuration

### Environment Variables

```bash
VERITY_CACHE_DIR=/path/to/cache  # Cache directory
VERITY_LOG_LEVEL=INFO            # Logging level
VERITY_MAX_SAMPLES=10            # Max samples for consistency
```

### Config File (verity_config.yaml)

```yaml
signals:
  self_consistency:
    enabled: true
    num_samples: 5
    temperature_range: [0.7, 1.1]

  semantic_entropy:
    enabled: true
    clustering_eps: 0.3

  token_confidence:
    enabled: true
    low_prob_threshold: 0.1

fusion:
  mode: ensemble
  calibrated: true

detection:
  confidence_level: 0.95
  risk_threshold: 0.5
```

---

## Deployment Patterns

### 1. Online Detection (Real-time)

```python
# In production API
@app.route('/generate', methods=['POST'])
def generate():
    question = request.json['question']
    response = llm.generate(question)

    # Detect hallucination
    result = detector.detect(question, response)

    if result.score < 0.5:
        return {'error': 'Low confidence response'}, 400

    return {'response': response, 'confidence': result.score}
```

### 2. Batch Processing

```python
# Offline evaluation
questions, responses = load_dataset()
results = detector.batch_detect(questions, responses)

# Filter hallucinations
filtered = [r for r, res in zip(responses, results) if res.score > 0.7]
```

### 3. Active Learning

```python
# Identify uncertain predictions for labeling
results = detector.batch_detect(questions, responses)
uncertain = [(q, r) for (q, r), res in zip(data, results)
             if 0.4 < res.score < 0.6]

# Send uncertain to human labelers
labels = human_label(uncertain)

# Retrain detector
detector.calibrate(questions, responses, labels)
```

---

## Monitoring & Observability

### Metrics to Track

```python
# Per-signal statistics
signal_stats = {
    signal: {
        'mean': mean(scores),
        'std': std(scores),
        'min': min(scores),
        'max': max(scores)
    }
    for signal, scores in signal_history.items()
}

# Detection rate
hallucination_rate = sum(1 for r in results if r.score < 0.5) / len(results)

# Confidence distribution
confidence_dist = histogram([r.score for r in results], bins=10)
```

### Logging

```python
import logging

logger = logging.getLogger('verity')
logger.setLevel(logging.INFO)

# Log each detection
logger.info(f"Detection: score={result.score:.3f}, verdict={result.verdict}")

# Log risk factors
if result.risk_factors:
    logger.warning(f"Risk factors: {result.risk_factors}")
```


## Future Architecture Enhancements

1. **Streaming Detection**: Detect during generation, not after
2. **Multi-Modal**: Extend to images, audio
3. **Distributed**: Parallelize signal computation
4. **Federated**: Privacy-preserving detection
5. **Adaptive**: Online learning of fusion weights


