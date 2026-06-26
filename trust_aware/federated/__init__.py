"""Federated trust-aware retrieval study.

This subpackage reframes trust-aware source selection as *resource selection /
query routing over a federation of heterogeneous real retrieval collections*.

Each collection (CACM, MED, NPL, CRANFIELD, CISI) is treated as an independent
queryable *source* with its own corpus, retrieval quality, latency, and cost.
The trust-aware optimizer must decide which source(s) to route each query to.

Unlike the dependency-free core package, this research extension uses numpy,
scipy and matplotlib. Install with ``pip install -e .[federated]``.

Public entry points:

- :func:`trust_aware.federated.testbed.load_testbed`
- :func:`trust_aware.federated.experiment.run_study`
"""

from .testbed import Collection, Testbed, load_testbed, build_testbed

__all__ = ["Collection", "Testbed", "load_testbed", "build_testbed"]
