"""Online Bayesian adaptive trust management.

Scientific contribution:
    Implements an online Bayesian trust learning loop: as query execution
    feedback arrives, the Beta-Binomial posterior for each source is updated
    incrementally.  Temporal decay handles non-stationary reliability by
    shrinking observation counts toward the uninformative prior over time.

    Optionally, a quality threshold on execution feedback gates whether a
    nominally successful interaction is counted as a trust confirmation.

References:
    Thompson, W. R. (1933). On the likelihood that one unknown probability
        exceeds another. Biometrika, 25(3/4), 285-294.
    Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018).
        A tutorial on Thompson sampling. Foundations and Trends in Machine
        Learning, 11(1), 1-96.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import DataSource, TrustEvidence


@dataclass
class ExecutionFeedback:
    """Outcome of executing a query against a data source.

    Attributes:
        source_name:   Name of the source that served the query.
        success:       Whether the query completed without error.
        latency_ms:    Observed round-trip latency in milliseconds.
        quality_score: Optional result-quality estimate in [0, 1].
                       When provided, a successful interaction is only
                       counted as a trust confirmation if
                       quality_score ≥ quality_threshold (default 0.7).
    """

    source_name: str
    success: bool
    latency_ms: float
    quality_score: Optional[float] = None


class AdaptiveTrustManager:
    """Online Bayesian trust updater for data sources.

    Maintains a Beta-Binomial posterior for each registered source and
    updates it as :class:`ExecutionFeedback` events arrive.  Temporal
    decay is applied via :meth:`decay` to model non-stationary reliability.

    Example::

        manager = AdaptiveTrustManager(decay_factor=0.95)
        manager.register(source)
        manager.observe(ExecutionFeedback(source.name, success=True, latency_ms=45))
        updated_sources = manager.augment_all([source])

    Parameters
    ----------
    decay_factor : float
        Multiplicative factor applied to observation counts on each
        :meth:`decay` call (default 1.0 = no decay).  Values such as
        0.95 model gradual forgetting of past behavior.
    quality_threshold : float
        Minimum quality score (0–1) required for a successful execution
        to be counted as a trust confirmation (default 0.7).
    """

    def __init__(
        self,
        decay_factor: float = 1.0,
        quality_threshold: float = 0.7,
    ) -> None:
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1]")
        self.decay_factor = decay_factor
        self.quality_threshold = quality_threshold
        self._evidence: Dict[str, TrustEvidence] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, source: DataSource) -> None:
        """Register a source, seeding its evidence from the embedded value."""
        if source.name not in self._evidence:
            self._evidence[source.name] = source.evidence

    def register_all(self, sources: List[DataSource]) -> None:
        """Register multiple sources at once."""
        for s in sources:
            self.register(s)

    # ------------------------------------------------------------------
    # Evidence update
    # ------------------------------------------------------------------

    def observe(self, feedback: ExecutionFeedback) -> None:
        """Update the Bayesian posterior for the given source.

        A quality-filtered success criterion is applied: if a
        quality_score is provided it must meet quality_threshold.
        """
        ev = self._evidence.get(feedback.source_name, TrustEvidence())

        confirmed_success = feedback.success
        if feedback.quality_score is not None:
            confirmed_success = (
                feedback.success
                and feedback.quality_score >= self.quality_threshold
            )

        self._evidence[feedback.source_name] = ev.update(confirmed_success)

    # ------------------------------------------------------------------
    # Temporal decay
    # ------------------------------------------------------------------

    def decay(self) -> None:
        """Apply temporal decay to all stored evidence posteriors."""
        for name in list(self._evidence):
            self._evidence[name] = self._evidence[name].decay(self.decay_factor)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def augment(self, source: DataSource) -> DataSource:
        """Return a copy of source with current Bayesian evidence applied."""
        ev = self._evidence.get(source.name, source.evidence)
        return source.with_evidence(ev)

    def augment_all(self, sources: List[DataSource]) -> List[DataSource]:
        """Augment a list of sources with their current evidence."""
        return [self.augment(s) for s in sources]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def trust_summary(self) -> Dict[str, Dict[str, float]]:
        """Return posterior statistics for all tracked sources.

        Returns
        -------
        dict
            ``{source_name: {"mean", "std", "lcb", "ucb", "n"}}``
        """
        return {
            name: {
                "mean": ev.mean,
                "std": ev.std,
                "lcb": ev.lcb,
                "ucb": ev.ucb,
                "n": float(ev.n),
            }
            for name, ev in self._evidence.items()
        }
