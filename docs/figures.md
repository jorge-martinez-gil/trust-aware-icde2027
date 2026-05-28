# Figure Guide

Generate all paper figures with:

```bash
python -m trust_aware --plots figures --seed 7
```

## Figure 1: Policy Atlas

`figures/figure_01_policy_atlas.svg`

Shows policy profiles as operating points across trust, latency, and cost. This
is the high-level "why trust-aware planning matters" figure.

## Figure 2: Trust-Latency Frontier

`figures/figure_02_trust_latency_frontier.svg`

Shows rejected candidates, admissible candidates, Pareto-optimal candidates, and
the selected source. This is the optimizer figure.

## Figure 3: Decision Certificate Waterfall

`figures/figure_03_decision_certificate_waterfall.svg`

Shows how trust, freshness, coverage, risk, latency, cost, and uncertainty
contribute to a selected plan. This is the auditability figure.

## Figure 4: AI-Native Query Pipeline

`figures/figure_04_ai_query_pipeline.svg`

Shows a composed retrieve-generate pipeline with trust lower bounds, total cost,
latency, and completeness. This is the systems workflow figure.

## Figure 5: Calibration Lens

`figures/figure_05_calibration_lens.svg`

Shows posterior trust means and lower bounds across sources. This is the
calibration and uncertainty figure.

## Figure 6: Real Dataset Statistics

`figures/figure_06_real_dataset_statistics.svg`

Shows WDBC cross-validated source reliability with bootstrap confidence
intervals, sensitivity, false-negative rate, and policy selections. This is the
real-data empirical validation figure.
