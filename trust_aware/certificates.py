from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .models import DataSource, QueryPlan, QueryRequest
from .portfolio import PortfolioPlan, SourcePortfolio


@dataclass(frozen=True)
class TrustCertificate:
    """Portable audit record for one optimizer decision."""

    certificate_id: str
    payload: Mapping[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        if "ranked_portfolios" in self.payload:
            return _portfolio_markdown(self)

        selected = self.payload["selected"]
        lines = [
            f"# Trust Certificate {self.certificate_id}",
            "",
            f"- Query: `{self.payload['query']}`",
            f"- Objective: `{self.payload['objective']}`",
            f"- Selected source: `{selected['name'] if selected else 'none'}`",
            f"- Admissible sources: {len(self.payload['ranked_sources'])}",
            f"- Rejected sources: {len(self.payload['rejected_sources'])}",
            "",
            "## Ranked Sources",
        ]
        if not self.payload["ranked_sources"]:
            lines.append("No admissible sources.")
        for source in self.payload["ranked_sources"]:
            lines.append(
                f"- `{source['name']}` score={source['score']:.3f} "
                f"trust_lb={source['trust_lower_bound']:.3f} "
                f"latency_ms={source['latency_ms']:.1f} "
                f"cost={source['cost_per_query']:.3f}"
            )

        if self.payload["rejected_sources"]:
            lines.extend(["", "## Rejections"])
            for rejected in self.payload["rejected_sources"]:
                lines.append(
                    f"- `{rejected['name']}`: {', '.join(rejected['reasons'])}"
                )
        return "\n".join(lines)


def certify_plan(plan: QueryPlan, request: QueryRequest) -> TrustCertificate:
    payload = {
        "query": request.query,
        "objective": plan.objective,
        "risk_tolerance": request.risk_tolerance,
        "constraints": _request_constraints(request),
        "selected": None,
        "ranked_sources": [],
        "pareto_sources": [source.name for source in plan.pareto_sources],
        "rejected_sources": [
            {"name": rejected.source.name, "reasons": list(rejected.reasons)}
            for rejected in plan.rejected_sources
        ],
    }

    if plan.primary_source is not None:
        payload["selected"] = _source_snapshot(plan.primary_source, request)

    ranked_sources = []
    for step in plan.steps:
        snapshot = _source_snapshot(step.source, request)
        snapshot.update(
            {
                "score": step.score,
                "pareto_optimal": step.is_pareto_optimal,
                "components": dict(step.breakdown.components),
                "weights": dict(step.breakdown.weights),
                "rationale": list(step.rationale),
            }
        )
        ranked_sources.append(snapshot)
    payload["ranked_sources"] = ranked_sources

    certificate_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    payload["certificate_id"] = certificate_id
    return TrustCertificate(certificate_id=certificate_id, payload=payload)


def certify_portfolio(
    portfolio_plan: PortfolioPlan, request: QueryRequest
) -> TrustCertificate:
    """Produce a stable audit certificate for a redundant source portfolio."""

    selected = portfolio_plan.primary_portfolio
    payload = {
        "query": request.query,
        "objective": portfolio_plan.objective,
        "risk_tolerance": request.risk_tolerance,
        "constraints": _request_constraints(request),
        "budget": _portfolio_budget_snapshot(portfolio_plan),
        "selected_portfolio": None
        if selected is None
        else _portfolio_snapshot(selected, request),
        "ranked_portfolios": [
            _portfolio_snapshot(portfolio, request)
            for portfolio in portfolio_plan.ranked_portfolios
        ],
        "rejected_sources": [
            {"name": rejected.source.name, "reasons": list(rejected.reasons)}
            for rejected in portfolio_plan.rejected_sources
        ],
        "candidate_count": portfolio_plan.candidate_count,
        "evaluated_count": portfolio_plan.evaluated_count,
        "budget_rejected_count": portfolio_plan.budget_rejected_count,
    }
    certificate_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    payload["certificate_id"] = certificate_id
    return TrustCertificate(certificate_id=certificate_id, payload=payload)


def _request_constraints(request: QueryRequest) -> dict[str, Any]:
    return {
        "required_capabilities": sorted(request.required_capabilities),
        "max_latency_ms": request.max_latency_ms,
        "min_freshness_score": request.min_freshness_score,
        "min_trust_score": request.min_trust_score,
        "max_cost_per_query": request.max_cost_per_query,
        "max_privacy_risk": request.max_privacy_risk,
        "max_hallucination_risk": request.max_hallucination_risk,
    }


def _source_snapshot(source: DataSource, request: QueryRequest) -> dict[str, Any]:
    return {
        "name": source.name,
        "source_type": source.source_type,
        "capabilities": sorted(source.capabilities),
        "trust_score": source.trust_score,
        "calibrated_trust": source.calibrated_trust(),
        "trust_lower_bound": source.trust_lower_bound(request.risk_tolerance),
        "trust_confidence": source.trust_confidence(),
        "latency_ms": source.latency_ms,
        "cost_per_query": source.cost_per_query,
        "freshness_score": source.freshness_score,
        "coverage": source.coverage,
        "privacy_risk": source.privacy_risk,
        "hallucination_risk": source.hallucination_risk,
        "schema_reliability": source.schema_reliability,
        "evidence": [
            {
                "dimension": evidence.dimension,
                "positive": evidence.positive,
                "total": evidence.total,
                "weight": evidence.weight,
                "posterior_mean": evidence.posterior_mean,
                "confidence": evidence.confidence,
            }
            for evidence in source.trust_evidence
        ],
    }


def _portfolio_budget_snapshot(plan: PortfolioPlan) -> dict[str, Any]:
    budget = plan.budget
    return {
        "min_sources": budget.min_sources,
        "max_sources": budget.max_sources,
        "max_total_latency_ms": budget.max_total_latency_ms,
        "max_total_cost": budget.max_total_cost,
        "min_portfolio_trust": budget.min_portfolio_trust,
        "top_k": budget.top_k,
        "latency_model": budget.latency_model,
        "correlation_penalty": budget.correlation_penalty,
        "source_penalty": budget.source_penalty,
        "candidate_limit": budget.candidate_limit,
    }


def _portfolio_snapshot(
    portfolio: SourcePortfolio, request: QueryRequest
) -> dict[str, Any]:
    return {
        "names": list(portfolio.names),
        "score": portfolio.score,
        "trust_lower_bound": portfolio.trust_lower_bound,
        "latency_ms": portfolio.latency_ms,
        "cost_per_query": portfolio.cost_per_query,
        "coverage": portfolio.coverage,
        "components": dict(portfolio.breakdown.components),
        "weights": dict(portfolio.breakdown.weights),
        "rationale": list(portfolio.rationale),
        "sources": [_source_snapshot(source, request) for source in portfolio.sources],
    }


def _portfolio_markdown(certificate: TrustCertificate) -> str:
    payload = certificate.payload
    selected = payload["selected_portfolio"]
    selected_names = " + ".join(selected["names"]) if selected else "none"
    lines = [
        f"# Trust Certificate {certificate.certificate_id}",
        "",
        f"- Query: `{payload['query']}`",
        f"- Objective: `{payload['objective']}`",
        f"- Selected portfolio: `{selected_names}`",
        f"- Ranked portfolios: {len(payload['ranked_portfolios'])}",
        f"- Candidate sources: {payload['candidate_count']}",
        f"- Evaluated portfolios: {payload['evaluated_count']}",
        f"- Budget-rejected portfolios: {payload['budget_rejected_count']}",
        "",
        "## Ranked Portfolios",
    ]
    if not payload["ranked_portfolios"]:
        lines.append("No admissible portfolios.")
    for portfolio in payload["ranked_portfolios"]:
        names = " + ".join(portfolio["names"])
        lines.append(
            f"- `{names}` score={portfolio['score']:.3f} "
            f"trust_lb={portfolio['trust_lower_bound']:.3f} "
            f"latency_ms={portfolio['latency_ms']:.1f} "
            f"cost={portfolio['cost_per_query']:.3f}"
        )

    if payload["rejected_sources"]:
        lines.extend(["", "## Source Rejections"])
        for rejected in payload["rejected_sources"]:
            lines.append(f"- `{rejected['name']}`: {', '.join(rejected['reasons'])}")
    return "\n".join(lines)
