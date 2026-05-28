from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Tuple

from .models import DataSource, normalize_capabilities


@dataclass
class SourceCatalog:
    """Small in-memory catalog for experiments and examples."""

    _sources: dict[str, DataSource] = field(default_factory=dict)

    def register(self, source: DataSource) -> None:
        if not source.name:
            raise ValueError("source name must be non-empty")
        self._sources[source.name] = source

    def extend(self, sources: Iterable[DataSource]) -> None:
        for source in sources:
            self.register(source)

    def get(self, name: str) -> DataSource:
        return self._sources[name]

    def match_capabilities(self, capabilities: Iterable[str]) -> Tuple[DataSource, ...]:
        required = normalize_capabilities(capabilities)
        return tuple(
            source
            for source in self._sources.values()
            if source.supports_all(required)
        )

    def __iter__(self) -> Iterator[DataSource]:
        return iter(self._sources.values())

    def __len__(self) -> int:
        return len(self._sources)
