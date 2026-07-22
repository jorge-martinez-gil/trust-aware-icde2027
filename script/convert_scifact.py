#!/usr/bin/env python3
"""Normalize the BEIR SciFact release into this repo's raw collection layout.

SciFact ships from BEIR as::

    scifact/corpus.jsonl        {"_id", "title", "text", "metadata"}
    scifact/queries.jsonl       {"_id", "text", "metadata"}
    scifact/qrels/test.tsv      query-id \t corpus-id \t score
    scifact/qrels/train.tsv

which this script rewrites into the same JSON-lines shape already used by the
CACM / MED / NPL collections, so ``testbed._parse_jsonl_collection`` can read it
without a SciFact-specific parser::

    data/federated/raw/scifact/scifact.json          {"DOCID", "TEXT"}
    data/federated/raw/scifact/queries.json          {"QUERIES": [...]}
    data/federated/raw/scifact/qrels-treceval.txt    qid 0 docid rel

Two decisions worth stating explicitly:

* **Test split only.** BEIR evaluates SciFact on the 300-query test split, so
  that is what we keep. Including the 809 train queries would inflate the query
  count with judgments the published baselines never score against.
* **Title + abstract concatenated** into one document text, matching how the
  CRANFIELD parser folds ``<title>`` into ``<text>``.

Usage::

    python script/convert_scifact.py /path/to/unzipped/scifact

The BEIR archive lives at
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_OUT = os.path.join(_REPO, "data", "federated", "raw", "scifact")

SPLIT = "test"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def convert(src: str, out: str) -> dict:
    qrels_path = os.path.join(src, "qrels", f"{SPLIT}.tsv")
    for path in (os.path.join(src, "corpus.jsonl"),
                 os.path.join(src, "queries.jsonl"),
                 qrels_path):
        if not os.path.exists(path):
            raise SystemExit(f"missing expected BEIR file: {path}")

    os.makedirs(out, exist_ok=True)

    qrels: dict = collections.defaultdict(dict)
    with open(qrels_path, encoding="utf-8") as fh:
        next(fh)  # header: query-id corpus-id score
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                qrels[parts[0]][parts[1]] = int(parts[2])

    doc_ids = set()
    with open(os.path.join(out, "scifact.json"), "w", encoding="utf-8") as w:
        for line in open(os.path.join(src, "corpus.jsonl"), encoding="utf-8"):
            obj = json.loads(line)
            did = str(obj["_id"])
            doc_ids.add(did)
            text = _norm((obj.get("title") or "") + " " + (obj.get("text") or ""))
            w.write(json.dumps({"DOCID": did, "TEXT": text}) + "\n")

    all_queries = {}
    for line in open(os.path.join(src, "queries.jsonl"), encoding="utf-8"):
        obj = json.loads(line)
        all_queries[str(obj["_id"])] = _norm(obj["text"])

    kept = [{"QUERYID": q, "QUERY": all_queries[q]}
            for q in sorted(qrels, key=int) if q in all_queries]
    with open(os.path.join(out, "queries.json"), "w", encoding="utf-8") as w:
        json.dump({"QUERIES": kept}, w, indent=1)

    pairs = 0
    with open(os.path.join(out, "qrels-treceval.txt"), "w", encoding="utf-8") as w:
        for q in sorted(qrels, key=int):
            for d, score in sorted(qrels[q].items(), key=lambda kv: int(kv[0])):
                w.write(f"{q} 0 {d} {score}\n")
                pairs += 1

    dangling = {d for v in qrels.values() for d in v} - doc_ids
    return {
        "docs": len(doc_ids),
        "queries": len(kept),
        "pairs": pairs,
        "dangling_qrel_docs": len(dangling),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="directory holding the unzipped BEIR scifact release")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output dir (default: {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    stats = convert(args.src, args.out)
    print(f"wrote {args.out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["dangling_qrel_docs"]:
        print("  warning: some qrel doc ids are absent from the corpus", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
