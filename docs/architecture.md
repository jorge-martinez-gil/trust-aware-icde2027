# Architecture

This repository is organized as a small but complete research artifact for
trust-aware query optimization in AI-native data systems.

## Planning Model

The optimizer treats every candidate source as a queryable unit: a table, vector
index, retrieval service, tool, memory shard, or model-backed operator. Each
`DataSource` exposes:

- Capability metadata for hard feasibility checks.
- Trust priors and calibrated `TrustEvidence` streams.
- Latency, freshness, cost, coverage, privacy risk, hallucination risk, and
  schema reliability estimates.

Each `QueryRequest` combines:

- Hard constraints, such as required capabilities, latency ceiling, trust floor,
  privacy ceiling, and freshness floor.
- Soft objective weights over trust, latency, freshness, cost, coverage, and
  risk.
- A risk tolerance that determines the trust lower bound used during scoring.

## Optimization Flow

1. Normalize capability labels.
2. Reject incompatible candidates with explicit reasons.
3. Convert trust evidence into a beta-binomial posterior lower bound.
4. Score admissible candidates with weighted expected utility.
5. Penalize trust uncertainty when requested by policy.
6. Rank candidates deterministically.
7. Mark Pareto-optimal candidates for audit and ablation.
8. Return an explainable `QueryPlan`.

For high-stakes settings, `TrustAwarePortfolioPlanner` can lift the admissible
single-source ranking into a ranked set of redundant source portfolios. It
enumerates small candidate sets, applies latency and cost budgets, and computes
a portfolio trust lower bound with a configurable correlation penalty. This
models triangulated retrieval and verification without assuming that evidence
streams are perfectly independent.

## Policy Profiles

The package includes named `TrustPolicy` profiles:

- `high-assurance`: maximizes calibrated trust under strict privacy and
  hallucination-risk ceilings.
- `balanced`: keeps trust central while retaining operational efficiency.
- `latency-critical`: models low-latency serving paths.
- `cost-efficient`: models traditional cost-first pressure.
- `trust-only`: isolates the contribution of calibrated trust.

These profiles let experiments compare optimizer behavior without rewriting
query requests by hand.

## Trust Certificates

`certify_plan(plan, request)` produces a stable audit record for a decision. The
certificate contains the selected source, ranked alternatives, evidence-derived
trust summaries, hard constraints, Pareto status, and rejected-source reasons.
The certificate hash is deterministic over the payload, which makes it suitable
for artifact regression tests and deployment logs.

`certify_portfolio(portfolio_plan, request)` applies the same audit idea to
redundant source sets. The portfolio certificate records the selected source
set, budget settings, portfolio-level score components, constituent source
snapshots, and the number of candidate portfolios rejected by budget.

## Pipeline Planning

`TrustAwarePipelinePlanner` lifts source ranking into multi-stage AI workflows.
A query can be decomposed into stages such as retrieval, grounding, generation,
verification, and policy audit. Each stage receives its own capability and trust
constraints while sharing a global policy profile. The resulting `PipelinePlan`
reports completeness, selected sources, total latency, total cost, and the
minimum trust lower bound across the workflow.

## Online Trust Updates

`TrustLedger` records accepted and rejected feedback events and folds them back
into source evidence. This models the loop needed by AI-native systems whose
retrievers, tools, and model operators change behavior over time.

## Real-Data Evaluation

`trust_aware.realdata` adds a reproducible bridge from real observations to trust
evidence. The artifact ships the UCI Wisconsin Diagnostic Breast Cancer dataset,
builds deterministic stratified folds, evaluates feature-family diagnostic
sources, and converts observed accuracy, malignant sensitivity, and benign
specificity into beta-binomial evidence streams.

The companion `trust_aware.stats` module provides Wilson score intervals for
binomial metrics, bootstrap confidence intervals over fold-level balanced
accuracy, and paired bootstrap differences for comparative analysis. These
statistics are intentionally dependency-free so the artifact remains portable.

## Figure Generation

`trust_aware.visualization` emits dependency-free SVG figures from live artifact
objects. The plots cover policy tradeoffs, source Pareto frontiers, decision
waterfalls, AI query pipelines, trust calibration, and real-data statistical
intervals. Because the figures are vector graphics, they can be used directly in
papers, slides, and artifact documentation.

## Why This Matters

Classic cost-based optimizers assume stable operators and deterministic data
quality. AI-native systems add retrieval indexes, tools, model calls, generated
answers, and evolving memory. Those units differ not only in cost and latency
but also in trustworthiness, calibration confidence, privacy exposure, and
hallucination risk.

The core design goal is to make trust auditable and tunable without hiding it in
application code. A reviewer can inspect why a source was rejected, why another
source won, and how the choice changes under an ablation such as cost-first
planning.

## Artifact Entry Points

- `trust_aware.models`: domain model and calibrated evidence primitives.
- `trust_aware.optimizer`: constraint checking, scoring, ranking, and
  explanations.
- `trust_aware.policies`: named policy profiles and ablations.
- `trust_aware.certificates`: stable audit records for optimizer decisions.
- `trust_aware.portfolio`: budgeted redundant source-set planning.
- `trust_aware.pipeline`: stage-wise AI query workflow planning.
- `trust_aware.feedback`: incremental evidence updates from outcomes.
- `trust_aware.evaluation`: multi-seed study reports.
- `trust_aware.realdata`: real WDBC ingestion and cross-validated evidence.
- `trust_aware.stats`: confidence intervals and bootstrap summaries.
- `trust_aware.visualization`: publication SVG generation.
- `trust_aware.catalog`: a minimal source catalog for experiments.
- `trust_aware.benchmark`: deterministic synthetic workload and ablation.
- `examples/research_demo.py`: a compact end-to-end planning example.
