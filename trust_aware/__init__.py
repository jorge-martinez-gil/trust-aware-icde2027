"""Trust-aware query optimization primitives."""

from .adaptive import AdaptiveTrustManager, ExecutionFeedback
from .catalog import SourceCatalog
from .certificates import TrustCertificate, certify_plan, certify_portfolio
from .evaluation import EvaluationReport, StudyResult, run_reproducible_study
from .feedback import FeedbackEvent, TrustLedger
from .models import (
    Capability,
    DataSource,
    ExecutionStrategy,
    QueryPlan,
    QueryRequest,
    PlanStep,
    RejectedSource,
    ScoreBreakdown,
    TrustEvidence,
    compute_pareto_front,
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
from .scoring import BayesianUCBScorer, LinearWeightedScorer, ScoringStrategy, TOPSISScorer
from .stats import ConfidenceInterval, DistributionSummary
from .trust_graph import TrustEdge, TrustGraph
from .visualization import generate_all_figures

__all__ = [
    "AdaptiveTrustManager",
    "BALANCED",
    "BayesianUCBScorer",
    "Capability",
    "COST_EFFICIENT",
    "DataSource",
    "EvaluationReport",
    "ExecutionFeedback",
    "ExecutionStrategy",
    "FeedbackEvent",
    "HIGH_ASSURANCE",
    "LATENCY_CRITICAL",
    "LinearWeightedScorer",
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
    "ScoringStrategy",
    "SourceCatalog",
    "SourcePortfolio",
    "SourceEvaluation",
    "StudyResult",
    "TOPSISScorer",
    "TRUST_ONLY",
    "TrustAwarePipelinePlanner",
    "TrustAwarePortfolioPlanner",
    "TrustEdge",
    "TrustEvidence",
    "TrustGraph",
    "TrustCertificate",
    "TrustPolicy",
    "TrustAwareQueryOptimizer",
    "WDBCRecord",
    "certify_plan",
    "certify_portfolio",
    "compute_pareto_front",
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
