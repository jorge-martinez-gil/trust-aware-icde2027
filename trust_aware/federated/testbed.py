"""Unified federated IR testbed: parse heterogeneous IR collections.

Six public-domain test collections spanning six domains are normalized into a
common ``Collection`` structure so the federation can be treated as a set of
queryable sources:

==========  ============================  ======  =======  ===========
Collection  Domain                        ~Docs   Queries  Format
==========  ============================  ======  =======  ===========
CACM        Computer science abstracts     3204     64     JSON
MED         Biomedicine (Medline)          1033     30     JSON
NPL         Electrical engineering        11429     93     JSON
CRAN        Aeronautics (Cranfield)        1400    225     TREC/XML
CISI        Library & information science  1460    112     SMART
SCIFACT     Scientific claim verification   5183    300     JSON (BEIR)
==========  ============================  ======  =======  ===========

The first five are classic collections assembled in the 1960s-1990s; SCIFACT
(Wadden et al., 2020) is a modern BEIR collection whose queries are natural
scientific claims rather than curated topic statements, and whose judgments are
far sparser (~1.1 relevant docs per query vs. ~14-41 for the classics). That
spread in query style, era, and judgment density is exactly the heterogeneity
a trust-aware selector has to cope with.

Raw provenance is recorded in ``data/federated/raw/PROVENANCE.md``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
RAW_DIR = os.path.join(_REPO, "data", "federated", "raw")
TESTBED_JSON = os.path.join(_REPO, "data", "federated", "testbed.json")

COLLECTION_DOMAINS = {
    "CACM": "Computer science",
    "MED": "Biomedicine",
    "NPL": "Electrical engineering",
    "CRAN": "Aeronautics",
    "CISI": "Library & information science",
    "SCIFACT": "Scientific claim verification",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Collection:
    """A single retrieval collection treated as one federated source."""

    name: str
    domain: str
    docs: Dict[str, str]                      # doc_id -> text
    queries: Dict[str, str]                   # query_id -> text
    qrels: Dict[str, Set[str]]                # query_id -> relevant doc_ids

    def num_docs(self) -> int:
        return len(self.docs)

    def num_queries(self) -> int:
        # Only queries that have at least one judged-relevant doc are usable.
        return len([q for q in self.queries if self.qrels.get(q)])

    def usable_query_ids(self) -> List[str]:
        return [q for q in self.queries if self.qrels.get(q)]

    def avg_doc_len(self) -> float:
        if not self.docs:
            return 0.0
        return sum(len(t.split()) for t in self.docs.values()) / len(self.docs)


@dataclass
class Testbed:
    """The full federation of collections."""

    collections: Dict[str, Collection] = field(default_factory=dict)

    def names(self) -> List[str]:
        return list(self.collections.keys())

    def total_docs(self) -> int:
        return sum(c.num_docs() for c in self.collections.values())

    def summary_rows(self) -> List[dict]:
        rows = []
        for name, c in self.collections.items():
            rows.append(
                {
                    "collection": name,
                    "domain": c.domain,
                    "docs": c.num_docs(),
                    "queries": c.num_queries(),
                    "avg_doc_len": round(c.avg_doc_len(), 1),
                    "avg_rel_per_query": round(
                        sum(len(v) for v in c.qrels.values())
                        / max(1, c.num_queries()),
                        1,
                    ),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Parsers (one per raw format)
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_jsonl_collection(name: str, folder: str, doc_file: str) -> Collection:
    """CACM / MED / NPL / SCIFACT: JSON-lines docs + queries.json + TREC qrels."""
    docs: Dict[str, str] = {}
    with open(os.path.join(folder, doc_file), encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs[str(obj["DOCID"])] = _norm(obj.get("TEXT", ""))

    queries: Dict[str, str] = {}
    with open(os.path.join(folder, "queries.json"), encoding="utf-8", errors="ignore") as fh:
        qobj = json.load(fh)
    for q in qobj["QUERIES"]:
        queries[str(q["QUERYID"])] = _norm(q["QUERY"])

    qrels = _parse_trec_qrels(os.path.join(folder, "qrels-treceval.txt"))
    return Collection(name, COLLECTION_DOMAINS[name], docs, queries, qrels)


def _parse_trec_qrels(path: str) -> Dict[str, Set[str]]:
    """Standard ``qid 0 docid rel`` lines; relevance > 0 counts as relevant."""
    qrels: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            try:
                relevant = float(rel) > 0
            except ValueError:
                continue
            if relevant:
                qrels.setdefault(str(qid), set()).add(str(docid))
    return qrels


def _parse_cran(folder: str) -> Collection:
    """CRANFIELD: pseudo-XML docs + queries, TREC qrels."""
    docs: Dict[str, str] = {}
    with open(os.path.join(folder, "cran.all.1400.xml"), encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    for m in re.finditer(r"<doc>(.*?)</doc>", content, flags=re.S):
        block = m.group(1)
        did = re.search(r"<docno>\s*(.*?)\s*</docno>", block, flags=re.S)
        txt = re.search(r"<text>(.*?)</text>", block, flags=re.S)
        title = re.search(r"<title>(.*?)</title>", block, flags=re.S)
        if not did:
            continue
        body = (title.group(1) if title else "") + " " + (txt.group(1) if txt else "")
        docs[str(int(did.group(1)))] = _norm(body)

    queries: Dict[str, str] = {}
    with open(os.path.join(folder, "cran.qry.xml"), encoding="utf-8", errors="ignore") as fh:
        qcontent = fh.read()
    # Cranfield queries carry sparse original ids (1..365) but the TREC qrels
    # renumber them sequentially 1..225 in file order; align to the qrels.
    seq = 0
    for m in re.finditer(r"<top>(.*?)</top>", qcontent, flags=re.S):
        block = m.group(1)
        title = re.search(r"<title>(.*?)</title>", block, flags=re.S)
        seq += 1
        queries[str(seq)] = _norm(title.group(1) if title else "")

    qrels = _parse_trec_qrels(os.path.join(folder, "cranqrel.trec.txt"))
    # Cranfield qrel query ids are 1..225 matching the sequential query order.
    return Collection("CRAN", COLLECTION_DOMAINS["CRAN"], docs, queries, qrels)


def _parse_smart(name: str, all_file: str, qry_file: str, rel_file: str,
                 folder: str) -> Collection:
    """CISI: classic SMART ``.I/.T/.W`` format + pairwise REL file."""

    def parse_smart_docs(path: str) -> Dict[str, str]:
        items: Dict[str, str] = {}
        with open(path, encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        # Split on ".I <id>" markers.
        chunks = re.split(r"\n?\.I ", content)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            first_nl = chunk.find("\n")
            if first_nl == -1:
                continue
            did = chunk[:first_nl].strip()
            body = chunk[first_nl + 1:]
            # Keep .T (title) and .W (abstract); drop .A/.X tags and their markers.
            body = re.sub(r"\.[A-Z]\b", " ", body)
            items[str(did)] = _norm(body)
        return items

    docs = parse_smart_docs(os.path.join(folder, all_file))
    queries = parse_smart_docs(os.path.join(folder, qry_file))

    qrels: Dict[str, Set[str]] = {}
    with open(os.path.join(folder, rel_file), encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            qid, docid = parts[0], parts[1]
            qrels.setdefault(str(qid), set()).add(str(docid))
    return Collection(name, COLLECTION_DOMAINS[name], docs, queries, qrels)


# ---------------------------------------------------------------------------
# Build + load
# ---------------------------------------------------------------------------


def build_testbed(raw_dir: str = RAW_DIR) -> Testbed:
    """Parse all raw collections into a normalized :class:`Testbed`."""
    tb = Testbed()
    tb.collections["CACM"] = _parse_jsonl_collection(
        "CACM", os.path.join(raw_dir, "cacm"), "cacm.json"
    )
    tb.collections["MED"] = _parse_jsonl_collection(
        "MED", os.path.join(raw_dir, "med"), "med.json"
    )
    tb.collections["NPL"] = _parse_jsonl_collection(
        "NPL", os.path.join(raw_dir, "npl"), "npl.json"
    )
    tb.collections["CRAN"] = _parse_cran(os.path.join(raw_dir, "cran"))
    tb.collections["CISI"] = _parse_smart(
        "CISI", "CISI.ALL", "CISI.QRY", "CISI.REL", os.path.join(raw_dir, "cisi")
    )
    # SciFact ships in BEIR layout; ``script/convert_scifact.py`` normalizes it
    # into the same JSON-lines shape as CACM/MED/NPL.
    tb.collections["SCIFACT"] = _parse_jsonl_collection(
        "SCIFACT", os.path.join(raw_dir, "scifact"), "scifact.json"
    )
    # Drop qrels that point to non-existent docs (keeps metrics honest).
    for c in tb.collections.values():
        for qid, rels in list(c.qrels.items()):
            c.qrels[qid] = {d for d in rels if d in c.docs}
    return tb


def save_testbed(tb: Testbed, path: str = TESTBED_JSON) -> None:
    payload = {
        "collections": {
            name: {
                "domain": c.domain,
                "docs": c.docs,
                "queries": c.queries,
                "qrels": {q: sorted(d) for q, d in c.qrels.items()},
            }
            for name, c in tb.collections.items()
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_testbed(path: str = TESTBED_JSON, rebuild: bool = False) -> Testbed:
    """Load the normalized testbed, building + caching it on first use."""
    if rebuild or not os.path.exists(path):
        tb = build_testbed()
        save_testbed(tb, path)
        return tb
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    tb = Testbed()
    for name, cd in payload["collections"].items():
        tb.collections[name] = Collection(
            name=name,
            domain=cd["domain"],
            docs=cd["docs"],
            queries=cd["queries"],
            qrels={q: set(d) for q, d in cd["qrels"].items()},
        )
    return tb


if __name__ == "__main__":
    tb = build_testbed()
    save_testbed(tb)
    print(f"Built federated testbed: {len(tb.collections)} collections, "
          f"{tb.total_docs()} docs total")
    for row in tb.summary_rows():
        print(row)
