# Trust-Aware Query Optimization for AI-Native Data Systems

This repository is a reference implementation for the paper idea
**Trust-Aware Query Optimization for AI-Native Data Systems**. It models a new
optimizer layer for systems where a "data source" may be a table, vector index,
retrieval service, model-backed tool, memory shard, or grounded generation
operator.

The core claim is simple: AI-native query planners should optimize for trust as
a first-class objective, not as an application-level afterthought.

## What Is Included

- Constraint-aware source selection with capability, latency, freshness, trust,
  cost, privacy, and hallucination-risk gates.
- Calibrated trust scoring from beta-binomial evidence streams.
- Explainable `QueryPlan` output with rejection reasons, score breakdowns, and
  Pareto annotations.
- Named policy profiles for high-assurance, balanced, latency-critical,
  cost-efficient, and trust-only studies.
- Trust certificates that turn optimizer decisions into portable audit records.
- Budgeted source-portfolio planning for redundant high-stakes retrieval.
- Multi-stage AI query pipeline planning for retrieve-generate-verify workflows.
- Online feedback ledgers for updating trust evidence from observed outcomes.
- Deterministic synthetic benchmarks with trust-aware, cost-first, fast-first,
  and trust-only ablations.
- Real-data study on the UCI Wisconsin Diagnostic Breast Cancer dataset with
  cross-validation, Wilson intervals, and bootstrap confidence intervals.
- A compact source catalog and end-to-end research demo.
- Unit tests covering legacy behavior, calibration, explainability, catalogs,
  certificates, portfolios, pipelines, feedback, and benchmark determinism.

## Quick Start

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the benchmark ablation:

```bash
python -m trust_aware --seed 7
```

Run the multi-seed policy study:

```bash
python -m trust_aware --study
```

Run the real-data WDBC study:

```bash
python -m trust_aware --real-study
```

Generate publication-quality SVG figures:

```bash
python -m trust_aware --plots figures --seed 7
```

Run the end-to-end planning demo:

```bash
python examples/research_demo.py
```

## Minimal Example

```python
from trust_aware import Capability, DataSource, QueryRequest, TrustAwareQueryOptimizer

sources = [
    DataSource(
        name="curated-vector-index",
        trust_score=0.91,
        latency_ms=85,
        cost_per_query=0.20,
        freshness_score=0.88,
        capabilities={Capability.VECTOR_SEARCH, Capability.POLICY_AUDIT},
        privacy_risk=0.08,
    ),
    DataSource(
        name="cheap-cache",
        trust_score=0.65,
        latency_ms=25,
        cost_per_query=0.03,
        freshness_score=0.80,
        supports_vector=True,
        privacy_risk=0.20,
    ),
]

request = QueryRequest(
    required_capabilities={Capability.VECTOR_SEARCH},
    max_latency_ms=150,
    min_trust_score=0.70,
    max_privacy_risk=0.25,
    trust_weight=0.55,
    latency_weight=0.15,
    freshness_weight=0.15,
    cost_weight=0.10,
    risk_weight=0.05,
)

plan = TrustAwareQueryOptimizer().optimize(sources, request)
print(plan.primary_source.name)
print(plan.explain())
```

## Research Shape

The implementation is intentionally dependency-free and small enough to audit,
but it is not a toy ranking script. It separates:

- `trust_aware.models`: sources, requests, evidence, score breakdowns, and plans.
- `trust_aware.optimizer`: hard constraints, trust lower bounds, weighted
  expected utility, uncertainty penalties, Pareto marking, and explanations.
- `trust_aware.policies`: named profiles that make experiments repeatable.
- `trust_aware.certificates`: decision certificates for audit and reproducibility.
- `trust_aware.portfolio`: redundant source portfolios with trust, latency, and
  cost budgets.
- `trust_aware.pipeline`: multi-stage AI query planning.
- `trust_aware.feedback`: online trust-evidence updates.
- `trust_aware.evaluation`: multi-seed policy studies and markdown reports.
- `trust_aware.realdata`: WDBC ingestion, cross-validation, and real-data
  policy evaluation.
- `trust_aware.stats`: dependency-free confidence intervals and bootstrap
  summaries.
- `trust_aware.visualization`: high-quality dependency-free SVG figures.
- `trust_aware.catalog`: an in-memory source catalog for experiments.
- `trust_aware.benchmark`: deterministic workload generation and ablation.
- `docs/architecture.md`: the design map for paper readers and reviewers.

## Scoring

For every admissible source, the optimizer computes:

```text
score =
  trust_weight     * calibrated_trust_lower_bound
+ latency_weight   * latency_component
+ freshness_weight * freshness_score
+ cost_weight      * (1 / (1 + cost_per_query))
+ coverage_weight  * coverage
+ risk_weight      * (1 - max(privacy_risk, hallucination_risk))
- uncertainty_weight * (1 - trust_confidence)
```

Hard constraints are enforced before scoring. This keeps policy boundaries
auditable: a cheap source that violates privacy risk or trust floors does not win
by compensating with cost.

## Trust Certificates

Every plan can be certified:

```python
from trust_aware import certify_plan

certificate = certify_plan(plan, request)
print(certificate.certificate_id)
print(certificate.to_markdown())
```

The certificate contains the query, objective, constraints, ranked candidates,
evidence summaries, Pareto status, and rejection reasons. It is stable for the
same decision, which makes it useful for paper artifacts, regression tests, and
deployment audit logs.

## Pipeline Planning

AI-native queries are often workflows rather than one operator. The repository
therefore includes a `TrustAwarePipelinePlanner` that applies a shared policy to
stages such as retrieval, grounding, generation, verification, and policy audit.
It reports end-to-end latency, cost, completeness, and the pipeline trust floor.

## Portfolio Planning

For high-stakes queries, the artifact can now select a small redundant portfolio
instead of a single winning source:

```python
from trust_aware import PortfolioBudget, TrustAwarePortfolioPlanner, certify_portfolio

portfolio_plan = TrustAwarePortfolioPlanner().plan(
    sources,
    request,
    PortfolioBudget(
        min_sources=2,
        max_sources=3,
        max_total_latency_ms=180,
        max_total_cost=0.75,
        min_portfolio_trust=0.86,
        latency_model="parallel",
    ),
)
print(portfolio_plan.explain())
print(certify_portfolio(portfolio_plan, request).to_markdown())
```

Portfolio trust uses a redundancy-aware success model with a configurable
correlation penalty, so independent evidence can raise confidence without
pretending that sources are perfectly independent. This gives the paper artifact
a concrete mechanism for triangulation and budgeted verification.

## Real-Data Study

The repository now ships the UCI Wisconsin Diagnostic Breast Cancer dataset in
`data/`. The real-data path evaluates feature-family diagnostic sources with
deterministic stratified cross-validation, converts observed accuracy,
sensitivity, and specificity into trust evidence, and then asks the optimizer to
choose under the named policy profiles.

The report includes:

- Wilson score intervals for binomial reliability metrics.
- Bootstrap confidence intervals over fold-level balanced accuracy.
- Policy-selected sources with trust lower bound, latency, cost, sensitivity,
  specificity, and false-negative rate.

## Paper Figures

The plot generator emits vector SVG figures suitable for papers, talks, and
artifact documentation:

- `figure_01_policy_atlas.svg`: policy tradeoff atlas.
- `figure_02_trust_latency_frontier.svg`: Pareto skyline over source choices.
- `figure_03_decision_certificate_waterfall.svg`: score contribution waterfall.
- `figure_04_ai_query_pipeline.svg`: multi-stage AI query workflow.
- `figure_05_calibration_lens.svg`: posterior trust and uncertainty lens.
- `figure_06_real_dataset_statistics.svg`: real-data confidence interval study.

## Paper Artifact Checklist

- Reproducible command-line benchmark.
- Deterministic seeds for synthetic catalogs and workloads.
- Public, typed Python API.
- Explainable plan objects for qualitative inspection.
- Trust certificates for decision audit.
- Redundant portfolio certificates for high-stakes source triangulation.
- Real-data statistical validation on WDBC with confidence intervals.
- Multi-policy evaluation reports.
- Multi-stage planning for AI query workflows.
- Online trust updates from feedback events.
- Publication-quality SVG figures generated from the artifact itself.
- Focused tests that validate optimizer behavior.
- No external runtime dependencies.
