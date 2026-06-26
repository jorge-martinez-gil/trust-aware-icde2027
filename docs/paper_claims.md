# Paper Claims Supported by the Artifact

This file maps paper-level claims to executable code.

## Claim 1: Trust Should Be a First-Class Optimizer Objective

Supported by:

- `trust_aware.models.TrustEvidence`
- `trust_aware.models.DataSource.trust_lower_bound`
- `trust_aware.optimizer.TrustAwareQueryOptimizer`
- `tests/test_trust_planning.py`

The optimizer uses calibrated lower bounds rather than raw trust claims, which
prevents uncertain high-scoring sources from winning solely on optimistic priors.

## Claim 2: Policy Boundaries Should Be Hard Constraints

Supported by:

- Capability, freshness, latency, trust, cost, privacy, and hallucination gates.
- `RejectedSource` records with explicit reasons.
- `QueryPlan.explain`
- `TrustCertificate`

This makes policy violations auditable rather than hidden in a scalar objective.

## Claim 3: AI-Native Queries Need Workflow-Level Planning

Supported by:

- `trust_aware.pipeline.QueryStage`
- `trust_aware.pipeline.TrustAwarePipelinePlanner`
- `tests/test_award_artifact.py`

The planner composes trusted retrieval and generation decisions while reporting
end-to-end latency, cost, completeness, and trust floor.

## Claim 4: High-Stakes Queries Need Redundant Evidence Portfolios

Supported by:

- `trust_aware.portfolio.PortfolioBudget`
- `trust_aware.portfolio.TrustAwarePortfolioPlanner`
- `trust_aware.certificates.certify_portfolio`
- `tests/test_portfolio.py`

The portfolio planner selects small source sets under latency, cost, and
portfolio-trust budgets. Its redundancy model discounts gains by a configurable
correlation penalty, which lets the artifact study triangulation without making
an unrealistic full-independence assumption.

## Claim 5: Trust Evidence Can Be Derived From Real Outcomes

Supported by:

- `data/wdbc.data`
- `trust_aware.realdata.load_wdbc_dataset`
- `trust_aware.realdata.evaluate_wdbc_sources`
- `trust_aware.stats.wilson_interval`
- `trust_aware.stats.bootstrap_mean_ci`
- `tests/test_realdata.py`

The WDBC study evaluates diagnostic feature-family sources on real labeled
instances with deterministic stratified cross-validation. Accuracy, malignant
sensitivity, and benign specificity become calibrated evidence streams, while
Wilson and bootstrap intervals expose uncertainty for reviewers.

## Claim 6: Trust-Aware Optimization Has Measurable Tradeoffs

Supported by:

- `trust_aware.policies`
- `trust_aware.benchmark`
- `trust_aware.evaluation`
- `python -m trust_aware --study`

The study compares high-assurance, balanced, latency-critical, cost-efficient,
and trust-only policies over deterministic synthetic catalogs and workloads.

## Claim 7: Trust Must Evolve Online

Supported by:

- `trust_aware.feedback.FeedbackEvent`
- `trust_aware.feedback.TrustLedger`
- feedback tests in `tests/test_award_artifact.py`

Feedback events update beta-binomial evidence streams without changing the
optimizer API.

## Claim 8: Trust-Aware Planning Is Visually Auditable

Supported by:

- `trust_aware.visualization.generate_all_figures`
- `python -m trust_aware --plots figures --seed 7`
- `docs/figures.md`

The generated SVG figures expose policy tradeoffs, Pareto frontiers, score
contributions, composed AI query pipelines, calibration uncertainty, and
real-data statistical intervals.

## Claim 9: Trust Is Essential for Provenance-Robust Source Routing (Headline)

Supported by:

- `trust_aware.federated` (real 5-collection IR testbed, BM25, CORI/ReDDE
  baselines, trust-aware router, statistics, figures, report)
- `python -m trust_aware --federated-study`
- `tests/test_federated.py::TestHeadlineClaim`
- `docs/federated_study.md`, `results/federated/RESULTS.md`,
  `figures/federated/`

On a federation of five real, heterogeneous IR collections (CACM, MED, NPL,
CRAN, CISI; 18,526 docs, 476 judged queries), content-based resource selection
(CORI, ReDDE) is *blind to provenance*: when untrusted exact-duplicate
("evil-twin") sources are injected, their routing accuracy collapses (CORI R@1
0.874 -> 0.424 at 6 twins). The trust-aware router, which learns per-source
calibrated beta-binomial trust online from observed outcomes, stays robust
(R@1 0.832 -> 0.777) and improves nDCG@10 over CORI by +0.141 and over ReDDE by
+0.178 at 6 twins. Query-clustered paired Wilcoxon tests over 476 independent
queries, with Holm correction across the four confirmatory comparisons, give
p_Holm <= 1.5e-31 and large paired rank-biserial effects (0.726 and 0.793).
Results average the within-query observations across seeds {7, 11, 13}. On the
clean federation it is competitive but
not better than CORI — the cost of the mechanism is reported transparently.
