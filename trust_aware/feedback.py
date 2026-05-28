from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .models import DataSource, TrustEvidence


@dataclass(frozen=True)
class FeedbackEvent:
    """Observed outcome used to update source trust evidence."""

    source_name: str
    dimension: str
    accepted: bool
    weight: float = 1.0


class TrustLedger:
    """Incremental trust updates for online AI-native systems."""

    def __init__(self, events: Iterable[FeedbackEvent] = ()) -> None:
        self._events = list(events)

    def append(self, event: FeedbackEvent) -> None:
        self._events.append(event)

    def extend(self, events: Iterable[FeedbackEvent]) -> None:
        self._events.extend(events)

    def events_for(self, source_name: str) -> tuple[FeedbackEvent, ...]:
        return tuple(event for event in self._events if event.source_name == source_name)

    def update_source(self, source: DataSource) -> DataSource:
        events = self.events_for(source.name)
        if not events:
            return source

        evidence = {item.dimension: item for item in source.trust_evidence}
        counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 1.0])
        for event in events:
            counts[event.dimension][0] += event.weight if event.accepted else 0.0
            counts[event.dimension][1] += event.weight
            counts[event.dimension][2] = max(counts[event.dimension][2], event.weight)

        updated = []
        for dimension, previous in evidence.items():
            positive_delta, total_delta, weight = counts.pop(dimension, [0.0, 0.0, 1.0])
            updated.append(
                replace(
                    previous,
                    positive=previous.positive + positive_delta,
                    total=previous.total + total_delta,
                    weight=max(previous.weight, weight),
                )
            )

        for dimension, (positive, total, weight) in counts.items():
            updated.append(
                TrustEvidence(
                    dimension=dimension,
                    positive=positive,
                    total=total,
                    weight=weight,
                )
            )

        return replace(source, trust_evidence=tuple(updated))

    def update_sources(self, sources: Iterable[DataSource]) -> tuple[DataSource, ...]:
        return tuple(self.update_source(source) for source in sources)

    def summary(self) -> Mapping[str, int]:
        accepted = sum(1 for event in self._events if event.accepted)
        rejected = len(self._events) - accepted
        return {
            "events": len(self._events),
            "accepted": accepted,
            "rejected": rejected,
        }
