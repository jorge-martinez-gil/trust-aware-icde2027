"""Trust-aware query optimization primitives."""

from .catalog import SourceCatalog
from .certificates import TrustCertificate, certify_plan, certify_portfolio
from .evaluation import EvaluationReport, StudyResult, run_reproducible_study
from .feedback import FeedbackEvent, TrustLedger
from .models import (
    Capability,
    DataSource,
    QueryPlan,
    QueryRequest,
    PlanStep,
    RejectedSource,
    ScoreBreakdown,
    TrustEvidence,
)
from .optimizer import TrustAwareQueryOptimizer
from .pipeline import PipelinePlan, QueryStage, TrustAwarePipelinePlanner
from .policies import (
    BALANCED,
    COST_EFFICIENT,
    HIGH_ASSURANCE,
    LATENCY_CRITICAL,
    TRUST_ONLY,
    TrustPolicy,
    get_policy,
)
from .portfolio import (
    PortfolioBudget,
    PortfolioPlan,
    SourcePortfolio,
    TrustAwarePortfolioPlanner,
)
from .realdata import (
    RealDatasetReport,
    SourceEvaluation,
    WDBCRecord,
    evaluate_wdbc_sources,
    load_wdbc_dataset,
    run_wdbc_real_study,
)
from .stats import ConfidenceInterval, DistributionSummary
from .visualization import generate_all_figures

__all__ = [
    "BALANCED",
    "Capability",
    "COST_EFFICIENT",
    "DataSource",
    "EvaluationReport",
    "FeedbackEvent",
    "HIGH_ASSURANCE",
    "LATENCY_CRITICAL",
    "PlanStep",
    "PipelinePlan",
    "PortfolioBudget",
    "PortfolioPlan",
    "QueryPlan",
    "QueryRequest",
    "QueryStage",
    "RealDatasetReport",
    "RejectedSource",
    "ScoreBreakdown",
    "SourceCatalog",
    "SourcePortfolio",
    "SourceEvaluation",
    "StudyResult",
    "TRUST_ONLY",
    "TrustAwarePipelinePlanner",
    "TrustAwarePortfolioPlanner",
    "TrustEvidence",
    "TrustCertificate",
    "TrustPolicy",
    "TrustAwareQueryOptimizer",
    "WDBCRecord",
    "certify_plan",
    "certify_portfolio",
    "ConfidenceInterval",
    "DistributionSummary",
    "evaluate_wdbc_sources",
    "get_policy",
    "generate_all_figures",
    "load_wdbc_dataset",
    "run_reproducible_study",
    "run_wdbc_real_study",
    "TrustLedger",
]
