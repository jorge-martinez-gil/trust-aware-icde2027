"""Vectorised BM25 retrieval over each federated source + collection statistics.

BM25 weights are precomputed into a sparse (docs x vocab) matrix per source, so a
query is scored with a single sparse mat-vec -- fast enough to run the full
multi-setting, multi-seed study on CPU. Provides the df/cf statistics needed by
the CORI and ReDDE resource-selection baselines.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from .testbed import Collection, Testbed

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = set(
    """a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with we you your i he
    she his her them from has have had which who whom what when where how why all
    any can do does did been being more most other some can't cannot""".split()
)


def tokenize(text: str) -> List[str]:
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        if t in _STOPWORDS or len(t) < 2:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        toks.append(t)
    return toks


@dataclass
class SourceStats:
    name: str
    num_docs: int
    df: Dict[str, int]
    cf: Dict[str, int]
    total_terms: int
    avg_doc_len: float


class BM25Index:
    """Okapi BM25 with precomputed sparse weight matrix."""

    def __init__(self, collection: Collection, k1: float = 1.5, b: float = 0.75):
        self.name = collection.name
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = list(collection.docs.keys())
        self.N = len(self.doc_ids)

        vocab: Dict[str, int] = {}
        df: Dict[str, int] = defaultdict(int)
        cf: Dict[str, int] = defaultdict(int)
        doc_len = np.zeros(self.N, dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        tfs: List[float] = []
        total_terms = 0

        for d_idx, did in enumerate(self.doc_ids):
            toks = tokenize(collection.docs[did])
            doc_len[d_idx] = len(toks)
            total_terms += len(toks)
            for term, tf in Counter(toks).items():
                t_idx = vocab.setdefault(term, len(vocab))
                rows.append(d_idx)
                cols.append(t_idx)
                tfs.append(tf)
                df[term] += 1
                cf[term] += tf

        self.vocab = vocab
        self.avgdl = float(doc_len.mean()) if self.N else 0.0
        V = len(vocab)
        tfs = np.asarray(tfs, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        # idf per term (BM25 form).
        idf = np.zeros(V, dtype=np.float64)
        for term, t_idx in vocab.items():
            d = df[term]
            idf[t_idx] = math.log((self.N - d + 0.5) / (d + 0.5) + 1.0)

        # BM25 weight per (doc, term).
        if self.N:
            dl = doc_len[rows]
            denom = tfs + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            w = idf[cols] * (tfs * (self.k1 + 1)) / denom
            self.M = sparse.csc_matrix((w, (rows, cols)), shape=(self.N, V))
        else:
            self.M = sparse.csc_matrix((0, V))

        self.df = dict(df)
        self.stats = SourceStats(self.name, self.N, dict(df), dict(cf),
                                 total_terms, self.avgdl)

    def search(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        if self.N == 0:
            return []
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return []
        scores = np.asarray(self.M[:, cols].sum(axis=1)).ravel()
        if top_k < self.N:
            cand = np.argpartition(-scores, top_k)[:top_k]
            cand = cand[np.argsort(-scores[cand])]
        else:
            cand = np.argsort(-scores)
        return [(self.doc_ids[i], float(scores[i])) for i in cand if scores[i] > 0]


@dataclass
class Federation:
    indexes: Dict[str, BM25Index]
    stats: Dict[str, SourceStats]
    sample_doc_source: Dict[str, str] = field(default_factory=dict)
    sample_index: "BM25Index | None" = None
    sample_rate: float = 0.0
    latency_ms: Dict[str, float] = field(default_factory=dict)
    cost_per_query: Dict[str, float] = field(default_factory=dict)

    def source_names(self) -> List[str]:
        return list(self.indexes.keys())


def build_federation(testbed: Testbed, sample_rate: float = 0.30, seed: int = 7,
                     k1: float = 1.5, b: float = 0.75) -> Federation:
    rng = np.random.default_rng(seed)
    indexes: Dict[str, BM25Index] = {}
    stats: Dict[str, SourceStats] = {}
    for name, coll in testbed.collections.items():
        idx = BM25Index(coll, k1=k1, b=b)
        indexes[name] = idx
        stats[name] = idx.stats

    sample_docs: Dict[str, str] = {}
    sample_doc_source: Dict[str, str] = {}
    for name, coll in testbed.collections.items():
        ids = list(coll.docs.keys())
        n_sample = max(1, int(round(len(ids) * sample_rate)))
        chosen = rng.choice(len(ids), size=n_sample, replace=False)
        for j in chosen:
            did = ids[int(j)]
            key = f"{name}::{did}"
            sample_docs[key] = coll.docs[did]
            sample_doc_source[key] = name

    sample_index = BM25Index(
        Collection("__SAMPLE__", "sample", sample_docs, {}, {}), k1=k1, b=b)
    fed = Federation(indexes=indexes, stats=stats,
                     sample_doc_source=sample_doc_source,
                     sample_index=sample_index, sample_rate=sample_rate)
    assign_cost_profile(fed)
    return fed


def assign_cost_profile(fed: Federation) -> None:
    for name, st in fed.stats.items():
        fed.latency_ms[name] = round(20.0 + st.num_docs / 50.0, 2)
        fed.cost_per_query[name] = round(0.01 + st.num_docs / 10000.0, 4)
