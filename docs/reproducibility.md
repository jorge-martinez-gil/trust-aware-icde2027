# Reproducibility Guide

This artifact is designed to run without external dependencies.

## Environment

- Python 3.10 or newer.
- No required third-party packages.
- Deterministic pseudo-random seeds for all synthetic experiments.

## Commands

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Run the single-seed ablation:

```bash
python -m trust_aware --seed 7
```

Run the multi-seed study:

```bash
python -m trust_aware --study
```

Run the real-data WDBC study:

```bash
python -m trust_aware --real-study
```

Generate paper figures:

```bash
python -m trust_aware --plots figures --seed 7
```

Run the end-to-end demo:

```bash
python examples/research_demo.py
```

## Expected Study Shape

The multi-seed study prints a markdown table with five policy profiles:

- `high-assurance`
- `balanced`
- `latency-critical`
- `cost-efficient`
- `trust-only`

The expected qualitative pattern is:

- `high-assurance` should produce the highest average trust.
- `latency-critical` should produce the lowest average latency.
- `cost-efficient` should produce the lowest average cost.
- `trust-only` isolates the utility of trust-heavy ranking.

The exact values are deterministic for a given code revision and seed list.

## Real-Data Study

The artifact includes `data/wdbc.data` and `data/wdbc.names` from the UCI
Wisconsin Diagnostic Breast Cancer dataset. The command above evaluates audited
feature-family diagnostic sources with five deterministic stratified folds.

The report should include:

- 569 rows, 30 features, and malignant / benign class counts.
- Wilson 95% confidence intervals for sensitivity, specificity, and accuracy.
- Bootstrap 95% confidence intervals for fold-level balanced accuracy.
- Policy decisions showing selected source, trust lower bound, sensitivity,
  specificity, false-negative rate, latency, and cost.

## Audit Workflow

The demo prints four artifacts:

1. An explainable single-stage plan.
2. A stable trust certificate with ranked alternatives and rejection reasons.
3. A redundant portfolio plan and portfolio certificate.
4. A multi-stage pipeline plan with total latency, total cost, and trust floor.

These are the main qualitative objects to inspect when reviewing the system.

The portfolio API supports high-stakes settings:
`TrustAwarePortfolioPlanner` ranks redundant source sets under latency, cost, and
portfolio-trust budgets, and `certify_portfolio` emits a deterministic
certificate for the selected set.

## Figure Workflow

The figure generator writes five SVG files:

- `figure_01_policy_atlas.svg`
- `figure_02_trust_latency_frontier.svg`
- `figure_03_decision_certificate_waterfall.svg`
- `figure_04_ai_query_pipeline.svg`
- `figure_05_calibration_lens.svg`
- `figure_06_real_dataset_statistics.svg`

The figures are vector graphics generated directly from optimizer, policy,
certificate, pipeline, calibration, and real-data statistical objects. They
should remain crisp in a PDF paper and can be regenerated after changing
optimizer logic.
