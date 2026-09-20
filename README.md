# Trust-Aware Query Optimization for AI-Native Data Systems

A research-oriented Python implementation for trust-aware source selection and planning in AI-native data systems. This repository formalizes the idea that trust should be treated as a first-class optimization objective, alongside latency, cost, freshness, privacy risk, and coverage.

The project provides a reproducible reference implementation for the paper "Trust-Aware Query Optimization for AI-Native Data Systems", including source models, optimizer logic, explainable plan generation, trust certificates, portfolio planning, pipeline planning, and empirical evaluation on synthetic and real datasets.

## Overview

Modern AI-native data systems integrate heterogeneous sources such as:

- relational tables
- vector indexes
- retrieval services
- model-backed tools
- memory shards
- grounded generation operators

In these settings, choosing a source based only on raw relevance or latency is insufficient. A source may be fast and cheap yet untrusted, stale, or overly risky. This project addresses that gap by making trust a measurable, auditable, and optimizable property.

The optimizer supports:

- constraint-aware source selection
- calibrated trust scoring from evidence streams
- explainable decision records
- named policy profiles for different operating regimes
- redundant portfolio planning for high-stakes workloads
- multi-stage query pipeline planning
- online trust evidence updates from observed outcomes
- reproducible benchmark and real-data studies

## Key Features

- Trust-aware optimization with explicit constraint checks and weighted utility
- Evidence-based scoring using beta-binomial confidence estimates
- Explainable plans with rejection reasons, score components, and Pareto annotations
- Policy profiles for high-assurance, balanced, latency-critical, cost-efficient, and trust-only settings
- Trust certificates for portable audit and reproducibility artifacts
- Redundant source portfolios for robust high-stakes retrieval and verification
- End-to-end AI query workflow planning across retrieval, grounding, generation, and verification stages
- Real-data evaluation on the Wisconsin Diagnostic Breast Cancer dataset
- Deterministic synthetic benchmarks and benchmarking ablations
- Dependency-free core implementation for easy auditing and reproducibility

## Repository Structure

```text
.
├── data/                     # benchmark and dataset files
├── docs/                     # project documentation and architecture notes
├── examples/                 # end-to-end usage examples
├── figures/                  # generated SVG/PDF/PNG figure outputs
├── results/                  # experiment outputs and summaries
├── script/                   # utility scripts and task runners
├── tests/                    # unit and validation tests
├── trust_aware/              # core package
│   ├── adaptive.py           # adaptive trust management logic
│   ├── benchmark.py          # synthetic benchmark generation
│   ├── catalog.py            # in-memory source catalog
│   ├── certificates.py      # plan and portfolio certificates
│   ├── evaluation.py        # benchmark evaluation/reporting
│   ├── feedback.py           # online feedback and ledger updates
│   ├── models.py             # data models for sources, requests, and plans
│   ├── optimizer.py          # trust-aware optimizer logic
│   ├── pipeline.py           # multi-stage pipeline planning
│   ├── policies.py           # named optimization policies
│   ├── portfolio.py          # portfolio planner
│   ├── realdata.py           # real-data evaluation logic
│   ├── scoring.py            # scoring strategies
│   ├── stats.py              # confidence interval utilities
│   ├── trust_graph.py        # trust graph data structures
│   ├── visualization.py      # publication-quality figure generation
│   └── __init__.py           # public API exports
├── CITATION.cff              # citation metadata
├── CONTRIBUTING.md           # contribution guidelines
├── LICENSE                   # MIT license
├── pyproject.toml            # package configuration
├── README.md                 # project overview and usage guide
└── .gitignore
```

## Installation

The core package is intended to be lightweight and dependency-free. Python 3.10 or newer is required.

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/jorge-martinez-gil/trust-aware-icde2027.git
cd trust-aware-icde2027
pip install -e .
```

If you want to run the federated retrieval study and associated plotting/statistical utilities, install the optional dependencies:

```bash
pip install -e .[federated]
```

## Quick Start

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```

Run the benchmark ablation study:

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

Generate publication-quality figures:

```bash
python -m trust_aware --plots figures --seed 7
```

Run the end-to-end research demo:

```bash
python examples/research_demo.py
```

## Minimal Example

```python
from trust_aware import (
    Capability,
    DataSource,
    QueryRequest,
    TrustAwareQueryOptimizer,
)

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

## Research Contributions

This repository is more than a conventional optimizer prototype. It implements a complete experimental framework for studying trust-aware planning in AI-native ecosystems.

The core components include:

- `trust_aware.models`: data structures for sources, evidence, requests, and explanations
- `trust_aware.optimizer`: hard constraints, weighted utility, uncertainty penalties, and Pareto-based decisions
- `trust_aware.policies`: named policy profiles for different operating conditions
- `trust_aware.certificates`: audit-ready certificates for optimizer decisions
- `trust_aware.portfolio`: budgeted redundant source portfolios
- `trust_aware.pipeline`: workflow-aware planning for multi-stage AI queries
- `trust_aware.feedback`: online evidence updates from observed outcomes
- `trust_aware.evaluation`: repeatable policy studies and markdown reports
- `trust_aware.realdata`: real-data evaluation on the WDBC dataset
- `trust_aware.stats`: statistical utilities for confidence intervals and uncertainty quantification
- `trust_aware.visualization`: figure generation for publications and artifacts
- `trust_aware.catalog`: in-memory source catalogs for experiments
- `trust_aware.benchmark`: deterministic workload generation and ablations

## Trust Certificates

Every optimizer decision can be certified for auditability and reproducibility.

```python
from trust_aware import certify_plan

certificate = certify_plan(plan, request)
print(certificate.certificate_id)
print(certificate.to_markdown())
```

A certificate documents the query, objective, constraints, ranked candidates, evidence summaries, Pareto status, and rejection reasons. This makes the system useful not only for optimization, but also for deployment auditing, paper artifact generation, and regression testing.

## Portfolio Planning

For high-stakes workflows, the system can plan a small redundant source portfolio instead of selecting a single source:

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

This redundancy-aware planner accounts for uncertainty and source correlation, enabling more robust decision-making under adversarial or incomplete information.

## Real Data and Empirical Evaluation

The repository includes the UCI Wisconsin Diagnostic Breast Cancer dataset in `data/`. The evaluation pipeline uses feature-family diagnostic sources, stratified cross-validation, and calibrated trust evidence to study how the optimizer behaves under realistic conditions.

Key outputs include:

- Wilson score intervals for reliability estimates
- bootstrap confidence intervals over performance metrics
- source-level policy selection under trust and risk constraints
- reproducible benchmark and study reports

The project also contains a federated IR study focused on trust-aware routing in heterogeneous retrieval settings, with comparisons to established baselines such as CORI, ReDDE, Random, Search-All, and Oracle.

## Paper Figures and Artifacts

The project generates publication-ready figures, including:

- policy tradeoff atlas
- trust-latency frontier
- decision certificate waterfall
- AI query pipeline diagram
- calibration lens
- real-data statistical visualizations

These figures are generated directly from the artifact, supporting reproducible scientific reporting.

## Citation

If you use this software or its results, please cite the accompanying work.

```bibtex
@inproceedings{martinezgil2026trustaware,
  title     = {Trust-Aware Query Optimization for AI-Native Data Systems},
  author    = {Martinez-Gil, Jorge},
  year      = {2026},
  note      = {Update with final venue, pages, and DOI upon publication}
}
```

Machine-readable metadata is also available in [`CITATION.cff`](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).

The bundled datasets are redistributed under their own terms; please see [`data/README.md`](data/README.md) and [`data/federated/raw/PROVENANCE.md`](data/federated/raw/PROVENANCE.md) for provenance and licensing details.

## Contributing

Contributions are welcome. For contribution guidelines, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgments

This work is designed for research, reproducible experimentation, and auditability in AI-native data systems. It is intended to support both academic evaluation and practical deployment-oriented trust policies.
