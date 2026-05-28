"""Graph-based trust propagation for indirect trust inference.

Scientific contribution:
    Models the trust network as a directed weighted graph and computes
    Personalized PageRank scores via power iteration.  This allows the
    optimizer to infer trust for sources that lack direct evidence by
    propagating trust through transitive relationships.

Reference:
    Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank
    citation ranking: Bringing order to the web. Stanford InfoLab.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import DataSource, _clamp_0_1


@dataclass
class TrustEdge:
    """A directed trust relationship: *source_name* endorses *target_name*."""

    source_name: str
    target_name: str
    weight: float


class TrustGraph:
    """Personalized PageRank trust propagation over a source graph.

    Usage::

        graph = TrustGraph(damping=0.85)
        graph.add_source(source_a)
        graph.add_source(source_b)
        graph.add_trust_edge("source_a", "source_b", weight=0.9)
        scores = graph.propagated_trust()
        augmented = graph.augmented_sources()

    Algorithm
    ---------
    Let G = (V, E) be the trust graph with |V| = n nodes.  Denote the
    damping factor as d ∈ (0, 1) and the teleportation vector as **v**
    (initialized from static trust scores).  Power iteration computes::

        r_{t+1}[j] = (1 - d) * v[j]
                   + d * Σ_{i→j} (r_t[i] * w_{ij} / Σ_k w_{ik})
                   + d * dangling_mass * v[j]

    Convergence is declared when ||r_{t+1} - r_t||_1 < tolerance.

    Parameters
    ----------
    damping : float
        Random-walk damping factor (default 0.85).  Higher values give
        greater weight to structural graph links vs. the prior.
    """

    def __init__(self, damping: float = 0.85) -> None:
        if not 0.0 < damping < 1.0:
            raise ValueError("damping must be in (0, 1)")
        self.damping = damping
        self._nodes: Dict[str, DataSource] = {}
        self._edges: List[TrustEdge] = []

    def add_source(self, source: DataSource) -> None:
        """Register a data source as a graph node."""
        self._nodes[source.name] = source

    def add_trust_edge(
        self, source_name: str, target_name: str, weight: float
    ) -> None:
        """Add a directed endorsement edge with the given weight ∈ [0, 1]."""
        self._edges.append(
            TrustEdge(source_name, target_name, _clamp_0_1(weight))
        )

    def propagated_trust(
        self,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
    ) -> Dict[str, float]:
        """Compute Personalized PageRank scores for all registered nodes.

        Returns
        -------
        dict
            Mapping ``source_name → propagated_trust`` in [0, 1].
        """
        if not self._nodes:
            return {}

        names = list(self._nodes.keys())
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}

        # Teleportation vector: proportional to static trust scores
        raw = [self._nodes[name].trust_score for name in names]
        total = sum(raw) or 1.0
        teleport = [s / total for s in raw]

        # Build adjacency list with normalized out-weights
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        out_weight: List[float] = [0.0] * n

        for edge in self._edges:
            si, ti = idx.get(edge.source_name), idx.get(edge.target_name)
            if si is None or ti is None:
                continue
            adj[si].append((ti, edge.weight))
            out_weight[si] += edge.weight

        for i in range(n):
            if out_weight[i] > 0:
                adj[i] = [(j, w / out_weight[i]) for j, w in adj[i]]

        # Power iteration
        rank = list(teleport)
        for _ in range(max_iterations):
            new_rank = [(1.0 - self.damping) * teleport[i] for i in range(n)]

            # Distribute dangling mass proportionally to teleport
            dangling = sum(rank[i] for i in range(n) if not adj[i])
            for i in range(n):
                new_rank[i] += self.damping * dangling * teleport[i]

            for i in range(n):
                for j, w in adj[i]:
                    new_rank[j] += self.damping * rank[i] * w

            delta = sum(abs(new_rank[i] - rank[i]) for i in range(n))
            rank = new_rank
            if delta < tolerance:
                break

        max_r = max(rank) or 1.0
        return {names[i]: _clamp_0_1(rank[i] / max_r) for i in range(n)}

    def augmented_sources(self, blend: float = 0.5) -> List[DataSource]:
        """Return sources with trust scores blended with propagated values.

        Parameters
        ----------
        blend : float
            Weight for propagated trust in [0, 1] (default 0.5).
            ``trust_blended = (1 - blend) * static + blend * propagated``
        """
        propagated = self.propagated_trust()
        result = []
        for name, source in self._nodes.items():
            prop = propagated.get(name, source.trust_score)
            blended = _clamp_0_1(
                (1.0 - blend) * source.trust_score + blend * prop
            )
            result.append(
                DataSource(
                    name=source.name,
                    trust_score=blended,
                    latency_ms=source.latency_ms,
                    cost_per_query=source.cost_per_query,
                    freshness_score=source.freshness_score,
                    supports_vector=source.supports_vector,
                    reliability=source.reliability,
                    evidence=source.evidence,
                    tags=source.tags,
                )
            )
        return result
