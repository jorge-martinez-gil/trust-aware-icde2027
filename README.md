# Trust-Aware Query Optimization for AI-Native Data Systems

This repository provides a compact reference implementation for a **trust-aware query optimizer** suitable for AI-native data systems.

## What is included

- A small Python module that models query requests and data sources
- A trust-aware optimizer that ranks candidate data sources
- Focused unit tests validating source filtering and ranking behavior

## Quick start

```bash
python -m unittest discover -s tests -v
```

## Core idea

The optimizer combines trust, latency, freshness, and cost into a weighted score while enforcing hard constraints (for example, vector support, freshness minimum, and latency ceiling). Cost is normalized as `1 / (1 + cost_per_query)` so lower-cost sources contribute higher score while keeping the component bounded. This gives a practical baseline for experimenting with trust-aware query planning policies.