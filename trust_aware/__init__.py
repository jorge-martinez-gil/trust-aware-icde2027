"""Trust-aware query optimization for AI-native data systems.

Public API
----------
Models:
    TrustEvidence           Bayesian Beta-Binomial trust evidence
    DataSource              Data source with Bayesian-shrinkage trust
    QueryRequest            Query constraints and optimization weights
    QueryPlan               Ranked execution plan with Pareto front
    ExecutionStrategy       SINGLE / FALLBACK / ENSEMBLE enum
    ScoreBreakdown          Per-criterion score decomposition

Scoring strategies:
    ScoringStrategy         Abstract base class
    LinearWeightedScorer    Weighted additive utility model
    TOPSISScorer            TOPSIS multi-criteria ranking
    BayesianUCBScorer       UCB exploration-exploitation scorer

Optimizer:
    TrustAwareQueryOptimizer Main optimizer (supports all strategies)
    compute_pareto_front    Pareto dominance analysis utility

Trust graph:
    TrustEdge               Directed trust endorsement edge
    TrustGraph              PageRank-style trust propagation graph

Adaptive learning:
    ExecutionFeedback       Query execution outcome record
    AdaptiveTrustManager    Online Bayesian trust updater
"""

from .adaptive import AdaptiveTrustManager, ExecutionFeedback
from .models import (
    DataSource,
    ExecutionStrategy,
    QueryPlan,
    QueryRequest,
    ScoreBreakdown,
    TrustEvidence,
)
from .optimizer import TrustAwareQueryOptimizer, compute_pareto_front
from .scoring import BayesianUCBScorer, LinearWeightedScorer, ScoringStrategy, TOPSISScorer
from .trust_graph import TrustEdge, TrustGraph

__all__ = [
    # Models
    "TrustEvidence",
    "DataSource",
    "QueryRequest",
    "QueryPlan",
    "ExecutionStrategy",
    "ScoreBreakdown",
    # Scoring
    "ScoringStrategy",
    "LinearWeightedScorer",
    "TOPSISScorer",
    "BayesianUCBScorer",
    # Optimizer
    "TrustAwareQueryOptimizer",
    "compute_pareto_front",
    # Trust graph
    "TrustEdge",
    "TrustGraph",
    # Adaptive
    "ExecutionFeedback",
    "AdaptiveTrustManager",
]
