# Trust-Aware Query Optimization for AI-Native Data Systems

A reference implementation of a **trust-aware query optimizer** for AI-native data
systems, featuring Bayesian trust inference, TOPSIS multi-criteria ranking, graph-based
trust propagation, and online adaptive trust learning.

## Scientific contributions

| Module | Contribution | Reference |
|---|---|---|
| `models.py` | Bayesian trust estimation via Beta-Binomial model; posterior LCB/UCB for risk-aware source selection | DeGroot (1970) |
| `scoring.py` | TOPSIS multi-criteria decision making; Bayesian UCB exploration-exploitation | Hwang & Yoon (1981); Auer et al. (2002) |
| `trust_graph.py` | Personalized PageRank trust propagation over endorsement graphs | Page et al. (1999) |
| `adaptive.py` | Online Bayesian trust updates with temporal decay; quality-gated feedback | Thompson (1933); Russo et al. (2018) |
| `optimizer.py` | Pareto frontier analysis; margin-based confidence estimation; adaptive execution strategy selection | — |

## Architecture

```
trust_aware/
├── models.py        # TrustEvidence, DataSource, QueryRequest, QueryPlan
├── scoring.py       # LinearWeightedScorer, TOPSISScorer, BayesianUCBScorer
├── trust_graph.py   # TrustEdge, TrustGraph (PageRank propagation)
├── adaptive.py      # ExecutionFeedback, AdaptiveTrustManager
└── optimizer.py     # TrustAwareQueryOptimizer, compute_pareto_front
```

## Quick start

```bash
python -m unittest discover -s tests -v
```

## Usage

### Basic query optimization

```python
from trust_aware import DataSource, QueryRequest, TrustAwareQueryOptimizer

sources = [
    DataSource("vector-db-a", trust_score=0.92, latency_ms=100,
               cost_per_query=0.3, freshness_score=0.95, supports_vector=True),
    DataSource("vector-db-b", trust_score=0.55, latency_ms=30,
               cost_per_query=0.1, freshness_score=0.90, supports_vector=True),
]
request = QueryRequest(requires_vector=True, max_latency_ms=150)
optimizer = TrustAwareQueryOptimizer(strategy="topsis")
plan = optimizer.optimize(sources, request)

print(plan.primary_source.name)       # best source
print(plan.confidence)                 # margin-based confidence in [0, 1]
print(plan.execution_strategy)        # SINGLE / FALLBACK / ENSEMBLE
print(plan.pareto_front)              # non-dominated Pareto-optimal sources
```

### Scoring strategies

```python
# Linear weighted sum (fast baseline)
TrustAwareQueryOptimizer(strategy="linear")

# TOPSIS — vector-normalized multi-criteria ranking
TrustAwareQueryOptimizer(strategy="topsis")

# Bayesian UCB — exploration-exploitation trade-off
TrustAwareQueryOptimizer(strategy="bayesian-ucb", exploration_bonus=0.1)
```

### Adaptive trust learning

```python
from trust_aware import AdaptiveTrustManager, ExecutionFeedback

manager = AdaptiveTrustManager(decay_factor=0.95, quality_threshold=0.7)
manager.register_all(sources)

# After each query execution:
manager.observe(ExecutionFeedback(
    source_name="vector-db-a",
    success=True,
    latency_ms=95,
    quality_score=0.88,
))

# Augment sources with updated posteriors before the next optimization
updated_sources = manager.augment_all(sources)
plan = optimizer.optimize(updated_sources, request)
```

### Graph-based trust propagation

```python
from trust_aware import TrustGraph

graph = TrustGraph(damping=0.85)
for source in sources:
    graph.add_source(source)
graph.add_trust_edge("vector-db-a", "vector-db-b", weight=0.8)

# Sources augmented with propagated PageRank trust
augmented = graph.augmented_sources()
plan = optimizer.optimize(augmented, request)
```

### Pareto frontier analysis

```python
from trust_aware import compute_pareto_front

front = compute_pareto_front(sources)
# front contains only non-dominated sources across trust, latency, freshness, cost
```

## Core models

### `TrustEvidence` — Bayesian Beta-Binomial model

```
T ~ Beta(prior_alpha + successes, prior_beta + failures)
```

Properties: `.mean`, `.std`, `.ucb` (mean + 1.96σ), `.lcb` (mean − 1.96σ), `.n`.
Supports incremental `.update(success)` and temporal `.decay(factor)`.

### `DataSource` — Bayesian-shrinkage effective trust

```
w(n) = n / (n + 10)
T_eff = (1 - w) * T_static + w * posterior_mean
```

The effective trust blends the static label with accumulated evidence,
converging to the empirical success rate as n → ∞.

### `QueryPlan` — rich execution plan

| Field | Description |
|---|---|
| `ranked_sources` | Sources ordered by composite score |
| `breakdowns` | Per-criterion score decompositions |
| `confidence` | Margin-based confidence in top-1 selection |
| `execution_strategy` | SINGLE / FALLBACK / ENSEMBLE |
| `pareto_front` | Non-dominated sources across all objectives |
| `fallback_chain` | Ordered fallback sources after primary |

## References

- DeGroot, M. H. (1970). *Optimal Statistical Decisions*. McGraw-Hill.
- Hwang, C.-L., & Yoon, K. (1981). *Multiple attribute decision making: methods and
  applications*. Springer-Verlag.
- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation
  ranking: Bringing order to the web. *Stanford InfoLab*.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds
  another. *Biometrika*, 25(3/4), 285–294.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the
  multiarmed bandit problem. *Machine Learning*, 47(2), 235–256.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on
  Thompson sampling. *Foundations and Trends in Machine Learning*, 11(1), 1–96.
- Kaufmann, E., Cappé, O., & Garivier, A. (2012). On Bayesian upper confidence bounds
  for bandit problems. *AISTATS 2012*.
