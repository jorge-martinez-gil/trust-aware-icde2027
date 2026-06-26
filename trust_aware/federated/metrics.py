"""Retrieval and routing metrics for the federated study."""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple


def ndcg_at_k(ranked: List[str], rel: Set[str], k: int = 10) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: List[str], rel: Set[str]) -> float:
    if not rel:
        return 0.0
    hits = 0
    ap = 0.0
    for i, d in enumerate(ranked):
        if d in rel:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(rel)


def reciprocal_rank_of(order: List[str], home: str) -> float:
    for i, s in enumerate(order):
        if s == home:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_budget(order: List[str], home: str, budget: int) -> float:
    """Fraction of relevant docs reachable = 1 if home selected within budget."""
    return 1.0 if home in order[:budget] else 0.0
