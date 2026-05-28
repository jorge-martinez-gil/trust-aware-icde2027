from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Iterable, Sequence

from .benchmark import synthetic_sources
from .certificates import certify_plan
from .evaluation import EvaluationReport, run_reproducible_study
from .models import Capability, DataSource, QueryRequest, TrustEvidence
from .optimizer import TrustAwareQueryOptimizer
from .pipeline import QueryStage, TrustAwarePipelinePlanner
from .policies import BALANCED, HIGH_ASSURANCE
from .realdata import RealDatasetReport, run_wdbc_real_study


PALETTE = {
    "ink": "#172033",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "paper": "#f8fafc",
    "panel": "#ffffff",
    "trust": "#0f766e",
    "latency": "#2563eb",
    "cost": "#dc8a00",
    "risk": "#be123c",
    "purple": "#6d28d9",
    "gold": "#f4b400",
    "green": "#16a34a",
    "cyan": "#0891b2",
    "gray": "#94a3b8",
}

POLICY_COLORS = {
    "high-assurance": PALETTE["trust"],
    "balanced": PALETTE["latency"],
    "latency-critical": PALETTE["purple"],
    "cost-efficient": PALETTE["cost"],
    "trust-only": PALETTE["risk"],
}


def generate_all_figures(output_dir: str | Path = "figures", seed: int = 7) -> tuple[Path, ...]:
    """Generate publication-quality SVG figures for the paper artifact."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = run_reproducible_study(seeds=(7, 17, 29, 41, 53))
    real_report = run_wdbc_real_study(seed=seed, bootstrap_resamples=400)
    sources = synthetic_sources(seed=seed, count=60)
    demo_sources = _demo_sources()
    demo_request = _demo_request()
    demo_plan = TrustAwareQueryOptimizer().optimize(demo_sources, demo_request)
    pipeline = TrustAwarePipelinePlanner().plan(
        query=demo_request.query,
        stages=(
            QueryStage(
                name="retrieve evidence",
                required_capabilities=(Capability.VECTOR_SEARCH,),
                max_latency_ms=300,
                min_trust_score=0.70,
            ),
            QueryStage(
                name="ground answer",
                required_capabilities=(Capability.LLM_INFERENCE,),
                max_latency_ms=350,
                min_trust_score=0.72,
                max_hallucination_risk=0.12,
            ),
        ),
        sources=demo_sources,
        policy=HIGH_ASSURANCE,
    )

    figures = {
        "figure_01_policy_atlas.svg": policy_atlas_svg(report),
        "figure_02_trust_latency_frontier.svg": trust_latency_frontier_svg(sources),
        "figure_03_decision_certificate_waterfall.svg": decision_waterfall_svg(
            demo_plan, demo_request
        ),
        "figure_04_ai_query_pipeline.svg": pipeline_svg(pipeline),
        "figure_05_calibration_lens.svg": calibration_lens_svg(sources),
        "figure_06_real_dataset_statistics.svg": real_dataset_statistics_svg(
            real_report
        ),
    }

    written = []
    for filename, svg in figures.items():
        target = output_path / filename
        target.write_text(svg, encoding="utf-8")
        written.append(target)
    return tuple(written)


def policy_atlas_svg(report: EvaluationReport) -> str:
    width, height = 1600, 1000
    plot = Plot(width, height, left=170, right=160, top=175, bottom=160)
    results = list(report.results)
    latencies = [result.average_latency_ms for result in results]
    trusts = [result.average_trust for result in results]
    costs = [result.average_cost for result in results]

    x = scale(min(latencies) * 0.85, max(latencies) * 1.10, plot.left, plot.right_edge)
    y = scale(min(trusts) - 0.025, max(trusts) + 0.025, plot.bottom_edge, plot.top)
    c = scale(min(costs), max(costs), 16, 44)

    body = [
        _background(width, height),
        _headline(
            "Policy Atlas",
            "Trust-aware policies form distinct operating points across trust, latency, and cost.",
        ),
        _panel(plot),
        _axis_grid(plot, x_ticks=5, y_ticks=5),
        _axis_labels(plot, "Average latency (ms)", "Average trust lower bound"),
    ]

    ordered = sorted(results, key=lambda item: item.average_latency_ms)
    line_points = " ".join(
        f"{x(result.average_latency_ms):.1f},{y(result.average_trust):.1f}"
        for result in ordered
    )
    body.append(
        f'<polyline points="{line_points}" fill="none" stroke="{PALETTE["grid"]}" '
        'stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>'
    )

    for result in results:
        color = POLICY_COLORS.get(result.policy_name, PALETTE["trust"])
        cx, cy = x(result.average_latency_ms), y(result.average_trust)
        radius = c(result.average_cost)
        body.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.88" stroke="{PALETTE["ink"]}" '
            'stroke-width="3"/>'
        )
        body.append(
            _label(
                cx + radius + 16,
                cy - 8,
                result.policy_name,
                size=27,
                weight=800,
                color=PALETTE["ink"],
            )
        )
        body.append(
            _label(
                cx + radius + 16,
                cy + 24,
                f"trust {result.average_trust:.3f} | cost {result.average_cost:.2f}",
                size=20,
                color=PALETTE["muted"],
            )
        )

    body.extend(_numeric_x_ticks(plot, x, latencies, suffix=""))
    body.extend(_numeric_y_ticks(plot, y, trusts, precision=2))
    body.append(
        _callout(
            980,
            250,
            "High-assurance wins trust; cost-efficient wins spend. The optimizer exposes this tradeoff instead of burying it.",
            PALETTE["trust"],
        )
    )
    return _svg(width, height, "Trust-Aware Policy Atlas", body)


def trust_latency_frontier_svg(sources: Sequence[DataSource]) -> str:
    width, height = 1600, 1000
    plot = Plot(width, height, left=170, right=150, top=170, bottom=155)
    request = BALANCED.request(
        query="frontier analysis",
        required_capabilities=(Capability.VECTOR_SEARCH,),
        max_latency_ms=420,
        min_freshness_score=0.55,
        max_cost_per_query=2.50,
    )
    optimizer = TrustAwareQueryOptimizer()
    plan = optimizer.optimize(sources, request)

    all_sources = list(sources)
    latencies = [source.latency_ms for source in all_sources]
    trusts = [source.trust_lower_bound(request.risk_tolerance) for source in all_sources]
    x = scale(min(latencies) * 0.9, max(latencies) * 1.05, plot.left, plot.right_edge)
    y = scale(0.35, max(trusts) + 0.04, plot.bottom_edge, plot.top)

    admitted_names = {step.source.name for step in plan.steps}
    pareto_names = {source.name for source in plan.pareto_sources}
    selected_name = plan.primary_source.name if plan.primary_source else ""

    body = [
        _background(width, height),
        _headline(
            "Trust-Latency Frontier",
            "The planner separates rejected sources, admissible candidates, and the Pareto skyline.",
        ),
        _panel(plot),
        _axis_grid(plot, x_ticks=6, y_ticks=5),
        _axis_labels(plot, "Latency (ms)", "Calibrated trust lower bound"),
    ]

    frontier = sorted(
        [step.source for step in plan.steps if step.source.name in pareto_names],
        key=lambda source: source.latency_ms,
    )
    if len(frontier) > 1:
        points = " ".join(
            f"{x(source.latency_ms):.1f},{y(source.trust_lower_bound(request.risk_tolerance)):.1f}"
            for source in frontier
        )
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{PALETTE["gold"]}" '
            'stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    for source in all_sources:
        trust = source.trust_lower_bound(request.risk_tolerance)
        cx, cy = x(source.latency_ms), y(trust)
        radius = 8 + 18 * source.coverage
        if source.name not in admitted_names:
            color, opacity, stroke = PALETTE["gray"], 0.35, PALETTE["paper"]
        elif source.name in pareto_names:
            color, opacity, stroke = PALETTE["trust"], 0.90, PALETTE["gold"]
        else:
            color, opacity, stroke = PALETTE["latency"], 0.68, PALETTE["panel"]
        body.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="{opacity}" stroke="{stroke}" '
            'stroke-width="4"/>'
        )
        if source.name == selected_name:
            body.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius + 10:.1f}" '
                f'fill="none" stroke="{PALETTE["risk"]}" stroke-width="5"/>'
            )
            body.append(
                _label(cx + 28, cy - 24, "selected", size=25, weight=900, color=PALETTE["risk"])
            )

    body.extend(_numeric_x_ticks(plot, x, latencies))
    body.extend(_numeric_y_ticks(plot, y, trusts, precision=2))
    body.append(_legend(1020, 770, [("rejected", PALETTE["gray"]), ("admissible", PALETTE["latency"]), ("pareto", PALETTE["trust"])]))
    return _svg(width, height, "Trust-Latency Pareto Frontier", body)


def decision_waterfall_svg(plan, request: QueryRequest) -> str:
    width, height = 1600, 1000
    step = plan.steps[0]
    components = [
        ("trust", step.breakdown.contribution("trust"), PALETTE["trust"]),
        ("freshness", step.breakdown.contribution("freshness"), PALETTE["green"]),
        ("coverage", step.breakdown.contribution("coverage"), PALETTE["cyan"]),
        ("risk", step.breakdown.contribution("risk"), PALETTE["risk"]),
        ("latency", step.breakdown.contribution("latency"), PALETTE["latency"]),
        ("cost", step.breakdown.contribution("cost"), PALETTE["cost"]),
        (
            "uncertainty",
            -step.breakdown.weights.get("uncertainty", 0.0)
            * (1.0 - step.breakdown.components.get("confidence", 1.0)),
            PALETTE["purple"],
        ),
    ]

    plot = Plot(width, height, left=210, right=120, top=195, bottom=165)
    max_abs = max(abs(value) for _, value, _ in components) * 1.2
    x = scale(-max_abs, max_abs, plot.left, plot.right_edge)
    row_gap = 76

    body = [
        _background(width, height),
        _headline(
            "Decision Certificate Waterfall",
            f"Why `{step.source.name}` wins: each bar is an auditable score contribution.",
        ),
        _panel(plot),
        f'<line x1="{x(0):.1f}" y1="{plot.top}" x2="{x(0):.1f}" y2="{plot.bottom_edge}" '
        f'stroke="{PALETTE["ink"]}" stroke-width="3" stroke-opacity="0.35"/>',
    ]

    for index, (name, value, color) in enumerate(components):
        cy = plot.top + 72 + index * row_gap
        x0, x1 = x(0), x(value)
        left, width_bar = min(x0, x1), abs(x1 - x0)
        body.append(_label(plot.left - 26, cy + 8, name, size=25, anchor="end", weight=800))
        body.append(
            f'<rect x="{left:.1f}" y="{cy - 22:.1f}" width="{width_bar:.1f}" '
            f'height="44" rx="8" fill="{color}" fill-opacity="0.90"/>'
        )
        body.append(
            _label(
                x1 + (16 if value >= 0 else -16),
                cy + 8,
                f"{value:+.3f}",
                size=23,
                anchor="start" if value >= 0 else "end",
                weight=800,
                color=PALETTE["ink"],
            )
        )

    cert = certify_plan(plan, request)
    body.append(
        _callout(
            940,
            710,
            f"Certificate {cert.certificate_id}: reproducible decision record with constraints, evidence, and rejections.",
            PALETTE["purple"],
        )
    )
    body.append(
        _metric_strip(
            210,
            835,
            [
                ("score", f"{step.score:.3f}", PALETTE["trust"]),
                ("trust lb", f"{step.breakdown.components['trust']:.3f}", PALETTE["gold"]),
                ("confidence", f"{step.breakdown.components['confidence']:.3f}", PALETTE["latency"]),
            ],
        )
    )
    return _svg(width, height, "Trust-Aware Decision Waterfall", body)


def pipeline_svg(pipeline) -> str:
    width, height = 1600, 1000
    body = [
        _background(width, height),
        _headline(
            "AI-Native Query Pipeline",
            "Trust-aware planning composes retrieval and generation while preserving an end-to-end trust floor.",
        ),
    ]
    stage_x = [260, 760, 1260]
    y = 430
    body.append(
        f'<path d="M {stage_x[0] + 145} {y} C {stage_x[0] + 300} {y - 90}, '
        f'{stage_x[1] - 300} {y - 90}, {stage_x[1] - 145} {y}" '
        f'fill="none" stroke="{PALETTE["trust"]}" stroke-width="18" '
        'stroke-linecap="round" stroke-opacity="0.72"/>'
    )
    body.append(
        f'<path d="M {stage_x[1] + 145} {y} C {stage_x[1] + 300} {y + 90}, '
        f'{stage_x[2] - 300} {y + 90}, {stage_x[2] - 145} {y}" '
        f'fill="none" stroke="{PALETTE["latency"]}" stroke-width="18" '
        'stroke-linecap="round" stroke-opacity="0.72"/>'
    )

    labels = ["query", *[step.stage.name for step in pipeline.steps]]
    selected = ["user intent", *[
        step.selected_source.name if step.selected_source else "no admissible source"
        for step in pipeline.steps
    ]]
    colors = [PALETTE["ink"], PALETTE["trust"], PALETTE["latency"]]
    for index, x_pos in enumerate(stage_x):
        body.append(
            f'<circle cx="{x_pos}" cy="{y}" r="132" fill="{colors[index]}" '
            'fill-opacity="0.96"/>'
        )
        body.append(_label(x_pos, y - 30, labels[index], size=29, anchor="middle", weight=900, color="#ffffff"))
        body.append(_label(x_pos, y + 12, selected[index], size=22, anchor="middle", color="#ffffff"))
        if index > 0:
            score = pipeline.steps[index - 1].plan.steps[0]
            body.append(
                _label(
                    x_pos,
                    y + 58,
                    f"trust lb {score.breakdown.components['trust']:.3f}",
                    size=20,
                    anchor="middle",
                    color="#e0f2fe",
                )
            )

    body.append(
        _metric_strip(
            220,
            710,
            [
                ("complete", str(pipeline.is_complete), PALETTE["green"]),
                ("latency", f"{pipeline.total_latency_ms:.0f} ms", PALETTE["latency"]),
                ("cost", f"{pipeline.total_cost:.2f}", PALETTE["cost"]),
                ("trust floor", f"{pipeline.trust_floor:.3f}", PALETTE["trust"]),
            ],
        )
    )
    body.append(
        _callout(
            1010,
            720,
            "The paper figure should show that trust survives composition, not just single-source ranking.",
            PALETTE["gold"],
        )
    )
    return _svg(width, height, "Trust-Aware AI Query Pipeline", body)


def calibration_lens_svg(sources: Sequence[DataSource]) -> str:
    width, height = 1600, 1000
    plot = Plot(width, height, left=160, right=110, top=175, bottom=160)
    ranked = sorted(
        [source for source in sources if source.trust_evidence],
        key=lambda source: source.calibrated_trust(),
        reverse=True,
    )[:24]
    x = scale(0, len(ranked) - 1, plot.left + 30, plot.right_edge - 30)
    y = scale(0.45, 0.98, plot.bottom_edge, plot.top)

    body = [
        _background(width, height),
        _headline(
            "Calibration Lens",
            "Raw trust claims are narrowed by evidence volume, confidence, and risk tolerance.",
        ),
        _panel(plot),
        _axis_grid(plot, x_ticks=6, y_ticks=5),
        _axis_labels(plot, "Sources ranked by calibrated trust", "Trust estimate"),
    ]

    for index, source in enumerate(ranked):
        cx = x(index)
        mean = source.calibrated_trust()
        lower = source.trust_lower_bound(0.05)
        confidence = source.trust_confidence()
        body.append(
            f'<line x1="{cx:.1f}" y1="{y(lower):.1f}" x2="{cx:.1f}" y2="{y(mean):.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="7" stroke-linecap="round"/>'
        )
        body.append(
            f'<circle cx="{cx:.1f}" cy="{y(mean):.1f}" r="{10 + 13 * confidence:.1f}" '
            f'fill="{PALETTE["trust"]}" fill-opacity="0.86" stroke="{PALETTE["ink"]}" '
            'stroke-width="2"/>'
        )
        body.append(
            f'<circle cx="{cx:.1f}" cy="{y(lower):.1f}" r="7" '
            f'fill="{PALETTE["gold"]}" stroke="{PALETTE["ink"]}" stroke-width="1.5"/>'
        )

    body.extend(_numeric_y_ticks(plot, y, [0.50, 0.65, 0.80, 0.95], precision=2))
    body.append(_legend(1030, 755, [("posterior mean", PALETTE["trust"]), ("lower bound", PALETTE["gold"])]))
    body.append(
        _callout(
            235,
            715,
            "Vertical gaps are the uncertainty tax: impressive priors need evidence before they become optimizer trust.",
            PALETTE["trust"],
        )
    )
    return _svg(width, height, "Trust Calibration Lens", body)


def real_dataset_statistics_svg(report: RealDatasetReport) -> str:
    width, height = 1800, 1100
    plot = Plot(width, height, left=410, right=180, top=205, bottom=330)
    evaluations = sorted(
        report.source_evaluations,
        key=lambda item: item.counts.balanced_accuracy,
        reverse=True,
    )
    intervals = [item.balanced_accuracy_summary.interval for item in evaluations]
    domain_min = min(interval.lower for interval in intervals) - 0.025
    domain_max = min(1.0, max(interval.upper for interval in intervals) + 0.025)
    x = scale(domain_min, domain_max, plot.left, plot.right_edge)
    row_top = plot.top + 48
    row_bottom = plot.bottom_edge - 48
    row_gap = (row_bottom - row_top) / max(1, len(evaluations) - 1)

    body = [
        _background(width, height),
        _headline(
            "Real-Data Statistical Study",
            "WDBC cross-validation turns measured reliability into optimizer trust evidence.",
        ),
        _panel(plot),
        _axis_grid(plot, x_ticks=5, y_ticks=max(1, len(evaluations) - 1)),
        _label(
            (plot.left + plot.right_edge) / 2,
            plot.bottom_edge + 78,
            "Balanced accuracy with bootstrap 95% CI",
            size=25,
            anchor="middle",
            weight=850,
        ),
        f'<text x="54" y="{(plot.top + plot.bottom_edge) / 2:.1f}" '
        f'font-size="25" font-weight="850" fill="{PALETTE["ink"]}" '
        'text-anchor="middle" transform="rotate(-90 54 '
        f'{(plot.top + plot.bottom_edge) / 2:.1f})">Audited source</text>',
    ]

    ticks = _nice_ticks(domain_min, domain_max, 6)
    for value in ticks:
        body.append(
            _label(
                x(value),
                plot.bottom_edge + 34,
                f"{value:.2f}",
                size=20,
                anchor="middle",
                color=PALETTE["muted"],
            )
        )

    selected_by_source: dict[str, list[str]] = {}
    for decision in report.policy_decisions:
        selected_by_source.setdefault(decision.selected_source, []).append(
            decision.policy_name
        )
    for index, evaluation in enumerate(evaluations):
        interval = evaluation.balanced_accuracy_summary.interval
        y = row_top + index * row_gap
        color = PALETTE["trust"] if index == 0 else PALETTE["latency"]
        body.append(
            _label(
                plot.left - 26,
                y + 8,
                evaluation.profile.name,
                size=24,
                anchor="end",
                weight=850,
            )
        )
        body.append(
            f'<line x1="{x(interval.lower):.1f}" y1="{y:.1f}" '
            f'x2="{x(interval.upper):.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="8" stroke-linecap="round"/>'
        )
        body.append(
            f'<circle cx="{x(interval.estimate):.1f}" cy="{y:.1f}" r="15" '
            f'fill="{color}" stroke="{PALETTE["ink"]}" stroke-width="3"/>'
        )
        body.append(
            _label(
                x(interval.upper) + 18,
                y + 8,
                f"sens {evaluation.counts.sensitivity:.3f} | fnr {evaluation.counts.false_negative_rate:.3f}",
                size=20,
                color=PALETTE["muted"],
            )
        )
        if evaluation.profile.name in selected_by_source:
            body.append(
                _label(
                    x(interval.estimate),
                    y - 26,
                    ", ".join(selected_by_source[evaluation.profile.name]),
                    size=16,
                    anchor="middle",
                    weight=800,
                    color=PALETTE["risk"],
                )
            )

    body.append(
        _metric_strip(
            150,
            905,
            [
                ("rows", str(report.row_count), PALETTE["ink"]),
                ("features", str(report.feature_count), PALETTE["cyan"]),
                ("malignant", str(report.malignant_count), PALETTE["risk"]),
                ("benign", str(report.benign_count), PALETTE["green"]),
                ("folds", str(report.folds), PALETTE["purple"]),
            ],
            cell_width=265,
        )
    )
    body.append(
        _label(
            150,
            1072,
            "Wilson binomial intervals and bootstrap fold intervals expose measured uncertainty before optimizer selection.",
            size=22,
            color=PALETTE["muted"],
        )
    )
    return _svg(width, height, "Trust-Aware Real Dataset Statistical Study", body)


class Plot:
    def __init__(self, width: int, height: int, left: int, right: int, top: int, bottom: int) -> None:
        self.width = width
        self.height = height
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.right_edge = width - right
        self.bottom_edge = height - bottom


def scale(domain_min: float, domain_max: float, range_min: float, range_max: float):
    if math.isclose(domain_min, domain_max):
        domain_max = domain_min + 1.0

    def mapper(value: float) -> float:
        ratio = (value - domain_min) / (domain_max - domain_min)
        return range_min + ratio * (range_max - range_min)

    return mapper


def _svg(width: int, height: int, title: str, body: Iterable[str]) -> str:
    title = html.escape(title)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f"<title id=\"title\">{title}</title>\n"
        f'<desc id="desc">Publication figure generated by trust-aware.</desc>\n'
        "<style>"
        "text{font-family:Inter,Segoe UI,Arial,sans-serif}"
        ".small{font-size:20px}.axis{font-size:22px;font-weight:700}"
        "</style>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _background(width: int, height: int) -> str:
    return (
        f'<rect width="{width}" height="{height}" fill="{PALETTE["paper"]}"/>'
        f'<rect x="0" y="0" width="{width}" height="10" fill="{PALETTE["trust"]}"/>'
    )


def _headline(title: str, subtitle: str) -> str:
    return (
        _label(92, 82, title, size=52, weight=950, color=PALETTE["ink"])
        + _label(96, 122, "Trust-Aware Query Optimization for AI-Native Data Systems", size=21, weight=800, color=PALETTE["trust"])
        + _label(96, 155, subtitle, size=24, color=PALETTE["muted"])
    )


def _panel(plot: Plot) -> str:
    return (
        f'<rect x="{plot.left}" y="{plot.top}" width="{plot.right_edge - plot.left}" '
        f'height="{plot.bottom_edge - plot.top}" rx="8" fill="{PALETTE["panel"]}" '
        f'stroke="{PALETTE["grid"]}" stroke-width="2"/>'
    )


def _axis_grid(plot: Plot, x_ticks: int, y_ticks: int) -> str:
    lines = []
    for index in range(x_ticks + 1):
        x = plot.left + index * (plot.right_edge - plot.left) / x_ticks
        lines.append(
            f'<line x1="{x:.1f}" y1="{plot.top}" x2="{x:.1f}" y2="{plot.bottom_edge}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1.4" stroke-opacity="0.8"/>'
        )
    for index in range(y_ticks + 1):
        y = plot.top + index * (plot.bottom_edge - plot.top) / y_ticks
        lines.append(
            f'<line x1="{plot.left}" y1="{y:.1f}" x2="{plot.right_edge}" y2="{y:.1f}" '
            f'stroke="{PALETTE["grid"]}" stroke-width="1.4" stroke-opacity="0.8"/>'
        )
    return "\n".join(lines)


def _axis_labels(plot: Plot, x_label: str, y_label: str) -> str:
    return (
        _label((plot.left + plot.right_edge) / 2, plot.height - 62, x_label, size=25, anchor="middle", weight=850)
        + f'<text x="54" y="{(plot.top + plot.bottom_edge) / 2:.1f}" '
        f'font-size="25" font-weight="850" fill="{PALETTE["ink"]}" '
        'text-anchor="middle" transform="rotate(-90 54 '
        f'{(plot.top + plot.bottom_edge) / 2:.1f})">{html.escape(y_label)}</text>'
    )


def _numeric_x_ticks(plot: Plot, mapper, values: Sequence[float], suffix: str = "") -> list[str]:
    ticks = _nice_ticks(min(values), max(values), 5)
    return [
        _label(mapper(value), plot.bottom_edge + 34, f"{value:.0f}{suffix}", size=19, anchor="middle", color=PALETTE["muted"])
        for value in ticks
    ]


def _numeric_y_ticks(plot: Plot, mapper, values: Sequence[float], precision: int = 2) -> list[str]:
    ticks = _nice_ticks(min(values), max(values), 5)
    return [
        _label(plot.left - 22, mapper(value) + 7, f"{value:.{precision}f}", size=19, anchor="end", color=PALETTE["muted"])
        for value in ticks
    ]


def _nice_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    return [min_value + index * (max_value - min_value) / (count - 1) for index in range(count)]


def _label(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 22,
    anchor: str = "start",
    weight: int = 500,
    color: str | None = None,
) -> str:
    color = color or PALETTE["ink"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def _callout(x: float, y: float, text: str, color: str) -> str:
    escaped = html.escape(text)
    words = escaped.split()
    lines = []
    current = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > 58:
            lines.append(" ".join(current[:-1]))
            current = [word]
    if current:
        lines.append(" ".join(current))
    height = 54 + 30 * len(lines)
    body = [
        f'<rect x="{x}" y="{y}" width="455" height="{height}" rx="8" '
        f'fill="#ffffff" stroke="{color}" stroke-width="4"/>'
    ]
    for index, line in enumerate(lines):
        body.append(_label(x + 24, y + 42 + index * 30, line, size=21, color=PALETTE["ink"]))
    return "\n".join(body)


def _legend(x: float, y: float, entries: Sequence[tuple[str, str]]) -> str:
    body = [
        f'<rect x="{x}" y="{y}" width="330" height="{58 + 42 * len(entries)}" rx="8" '
        f'fill="#ffffff" stroke="{PALETTE["grid"]}" stroke-width="2"/>',
        _label(x + 22, y + 38, "Legend", size=22, weight=900),
    ]
    for index, (label, color) in enumerate(entries):
        cy = y + 76 + index * 42
        body.append(f'<circle cx="{x + 32}" cy="{cy}" r="12" fill="{color}"/>')
        body.append(_label(x + 56, cy + 7, label, size=20, color=PALETTE["muted"]))
    return "\n".join(body)


def _metric_strip(
    x: float,
    y: float,
    metrics: Sequence[tuple[str, str, str]],
    *,
    cell_width: int = 270,
) -> str:
    body = []
    for index, (label, value, color) in enumerate(metrics):
        cell_x = x + index * (cell_width + 18)
        body.append(
            f'<rect x="{cell_x}" y="{y}" width="{cell_width}" height="112" rx="8" '
            f'fill="#ffffff" stroke="{color}" stroke-width="4"/>'
        )
        body.append(_label(cell_x + 22, y + 39, label, size=21, weight=850, color=PALETTE["muted"]))
        body.append(_label(cell_x + 22, y + 83, value, size=35, weight=950, color=color))
    return "\n".join(body)


def _demo_request() -> QueryRequest:
    return QueryRequest(
        query="Find grounded evidence for model drift in EU support tickets.",
        required_capabilities={Capability.VECTOR_SEARCH},
        max_latency_ms=300,
        min_freshness_score=0.80,
        min_trust_score=0.70,
        max_privacy_risk=0.25,
        trust_weight=0.48,
        latency_weight=0.13,
        freshness_weight=0.12,
        cost_weight=0.07,
        coverage_weight=0.12,
        risk_weight=0.08,
        uncertainty_weight=0.03,
        risk_tolerance=0.05,
    )


def _demo_sources() -> tuple[DataSource, ...]:
    return (
        DataSource(
            name="curated-vector-index",
            trust_score=0.90,
            latency_ms=85,
            cost_per_query=0.22,
            freshness_score=0.88,
            capabilities={Capability.VECTOR_SEARCH, Capability.POLICY_AUDIT},
            trust_evidence=(
                TrustEvidence("human_acceptance", positive=420, total=470, weight=0.6),
                TrustEvidence("citation_support", positive=310, total=350, weight=0.4),
            ),
            coverage=0.84,
            privacy_risk=0.08,
            hallucination_risk=0.10,
        ),
        DataSource(
            name="fast-raw-embedding-cache",
            trust_score=0.73,
            latency_ms=28,
            cost_per_query=0.04,
            freshness_score=0.81,
            supports_vector=True,
            trust_evidence=(TrustEvidence("human_acceptance", positive=92, total=130),),
            coverage=0.62,
            privacy_risk=0.18,
            hallucination_risk=0.17,
        ),
        DataSource(
            name="expensive-grounded-agent",
            trust_score=0.94,
            latency_ms=260,
            cost_per_query=1.40,
            freshness_score=0.93,
            capabilities={Capability.VECTOR_SEARCH, Capability.LLM_INFERENCE},
            trust_evidence=(
                TrustEvidence("human_acceptance", positive=280, total=300, weight=0.5),
                TrustEvidence("policy_compliance", positive=294, total=300, weight=0.5),
            ),
            coverage=0.91,
            privacy_risk=0.05,
            hallucination_risk=0.07,
        ),
    )
