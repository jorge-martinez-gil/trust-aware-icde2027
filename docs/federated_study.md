# Federated Trust-Aware Retrieval: Experimental Study

This document describes the real-data experimental study that evaluates
trust-aware source selection as a *resource-selection / query-routing* problem
over a federation of heterogeneous retrieval collections. It is the empirical
core intended for the paper's evaluation section.

## 1. Problem

In an AI-native data system a query may be answerable by many *sources* — tables,
vector indexes, retrieval services, model-backed tools, or third-party mirrors.
A router must decide *which source(s)* to query under a cost/latency budget. We
argue that **trust must be a first-class routing objective**: content-based
resource selection is blind to *provenance*, so it cannot tell a genuine source
from an untrustworthy duplicate that returns plausible-but-unverifiable content
(a scraped mirror, a stale replica, or a hallucinated RAG store).

## 2. Federated testbed (real data)

We assemble six openly licensed IR test collections spanning six domains into one
federation; each collection is one source. The five classic collections are public
domain; SciFact's claims are CC BY 4.0 over S2ORC abstracts under ODC-By 1.0. Documents, queries, and relevance
judgements are real. Provenance, exact GitHub mirror commits, and the SciFact
release hash are recorded in `data/federated/raw/PROVENANCE.md`.

| Collection | Domain                        | Docs   | Queries | Rel./q | BM25 nDCG@10 |
|------------|-------------------------------|--------|---------|--------|--------------|
| CACM       | Computer science              | 3,204  | 52      | 14.2   | 0.48 |
| MED        | Biomedicine (Medline)         | 1,033  | 30      | 23.2   | 0.69 |
| NPL        | Electrical engineering        | 11,429 | 93      | 22.4   | 0.39 |
| CRAN       | Aeronautics (Cranfield)       | 1,400  | 225     | 7.2    | 0.38 |
| CISI       | Library & information science | 1,460  | 76      | 41.0   | 0.36 |
| SCIFACT    | Scientific claim verification | 5,183  | 300     | 1.1    | 0.67 |

Total: 23,709 documents, 776 judged queries. Per-collection BM25 nDCG@10 values
are consistent with the published literature, validating the retrieval harness.
Our SciFact BM25 scores 0.668, which lands next to the BM25 baseline of ~0.665
reported by BEIR — close agreement, though not a like-for-like reproduction:
we index a single concatenated title+abstract field with k1=1.5, b=0.75, while
BEIR's figure comes from Anserini's multi-field BM25 with k1=0.9, b=0.4.

The first five collections are classics assembled between the 1960s and 1990s.
SCIFACT (Wadden et al., EMNLP 2020) is deliberately different in kind: its
queries are natural scientific claims rather than curated topic statements, and
its judgements are roughly an order of magnitude sparser (~1.1 relevant docs per
query vs. 7-41). We use BEIR's 300-query test split so our per-collection numbers
stay comparable to published baselines.

**Caveat on absolute values.** Adding SciFact raises every method's absolute
nDCG@10 and MAP, including Oracle (0.409 -> 0.509). This is a property of the
judgement density, not an improvement in routing: with ~1.1 relevant documents
per claim, a single hit at rank 1 saturates nDCG@10. Comparisons *between*
methods on the same testbed remain the meaningful quantity; absolute values are
not comparable to the five-collection version of this table.

Each collection's relevant documents live only in its home collection, so
resource selection reduces to *routing each query to its home source(s)* — the
standard single-home federated-search evaluation.

## 3. Methods

All methods expose the same `rank_sources(query)` interface and are evaluated
under identical, fair, seeded random tie-breaking.

**Baselines (federated-IR literature).**
- *Random* — seeded random ordering (lower bound).
- *Search-All* — query every source (exhaustive; recall upper bound, max cost).
- *CORI* (Callan et al., 1995) — collection ranking from per-collection df / cw
  statistics with an INQUERY belief.
- *ReDDE* (Si & Callan, 2003) — relevant-document estimation via a centralized
  sample index (30% sample) scaled by collection size.
- *Oracle* — routes to the true home source first (routing upper bound).

**Proposed: Trust-Aware router.** Expected utility = relevance × calibrated
trust. The relevance signal is CORI; trust is a per-source beta-binomial
reliability with an *optimistic prior* (unseen sources are trusted) and a
posterior lower confidence bound. Trust is updated online from observed routing
outcomes (did the chosen source return a judged-relevant document?), so the
router *learns* which sources are reliable. At cold start trust is uniform and
the ranking reduces to CORI; as failures accumulate, unreliable sources are
multiplicatively suppressed.

**Online reputation ablations.**
- *CORI+Mean* receives the same feedback as Trust-Aware but multiplies CORI by
  the posterior mean, without an uncertainty penalty.
- *CORI+UCB* receives the same feedback but uses an optimistic upper confidence
  bound. These baselines test whether the gains come from calibrated lower-bound
  trust rather than from online feedback alone.

## 4. Adversary: untrusted "evil-twin" sources

To probe provenance-robustness we inject *evil-twin* sources: exact-content
duplicates of genuine collections, under fresh document ids that appear in **no**
relevance judgement. They model a scraped duplicate, an unverified mirror, or a
hallucinated store. Because the content is identical, *no content-based selector
can distinguish a twin from its original* — they must coin-flip. Only a method
that learns from outcomes can route around them. We sweep the number of injected
twins from 0 to 8.

We also run a sensitivity study at 6 untrusted sources with less stylized
variants: `corrupted-20` perturbs 20% of mirror tokens, and `partial-50` keeps a
50% mirror subset. Exact twins remain the hardest provenance-blindness case; the
variants mark the boundary where content statistics start to reveal degradation
and content-based routing can recover.

## 5. Protocol

Queries from all collections are pooled and streamed in a fixed, seeded order.
The Trust-Aware router is evaluated **prequentially** (scored on a query, then
updated from the observed outcome), so its trust at query *t* depends only on
queries < *t* — there is no train/test leakage. We report routing accuracy
(R@1), reciprocal rank (MRR), and end-to-end merged-retrieval quality (nDCG@10,
MAP) at cost budgets of 1–3 sources. All numbers are averaged over seeds
{7, 11, 13}. Confirmatory inference first averages seed replicates within each
query, then uses the 776 queries as independent clusters. We report two-sided
paired Wilcoxon signed-rank tests, 10,000-resample query-bootstrap confidence
intervals, matched-pairs rank-biserial effects, and Holm-Bonferroni correction
across the complete family of four comparisons (two metrics x two baselines).

## 6. Results

**Clean federation (0 twins).** Trust-Aware is competitive with the strongest
baseline but does not beat it — the expected "no free lunch" cost of the trust
machinery. CORI R@1 = 0.893, Trust-Aware R@1 = 0.822 ± 0.014. This is reported
transparently.

**Untrusted federation.** As twins are added, content-based selection collapses
while Trust-Aware stays robust:

| Method      | R@1 (0 twins) | R@1 (6 twins) | nDCG@10 (6 twins) |
|-------------|---------------|---------------|-------------------|
| ReDDE       | 0.785         | 0.405         | 0.204 |
| CORI        | 0.893         | 0.451         | 0.228 |
| **Trust-Aware** | **0.822** | **0.807**     | **0.421** |
| Oracle      | 1.000         | 1.000         | 0.509 |

At 6 twins, Trust-Aware improves nDCG@10 over CORI by +0.193
(95% CI [0.173, 0.214], p_Holm = 4.9e-59, paired rank-biserial = 0.809) and
over ReDDE by +0.217 (95% CI [0.196, 0.238], p_Holm = 2.6e-66,
paired rank-biserial = 0.841). Both are large paired effects. The full
robustness curves, online learning curve, and learned per-source trust are in
`figures/federated/`; exact inferential outputs are in
`results/federated/statistical_analysis.json`.

## 7. Reproducing

```bash
pip install -e .[federated]
python -m trust_aware --federated-study      # sweep + multiseed + variants + tables + figures
python -m trust_aware.federated.plots         # figures only (from cached JSON)
python -m trust_aware.federated.report        # LaTeX tables + RESULTS.md
python -m unittest tests.test_federated -v    # validation incl. headline claim
```

Results JSON is written to `results/federated/`. Figures are written as both PDF
and PNG to `figures/federated/` and `paper/figures/`; LaTeX tables are written
to `results/federated/tables/` and `paper/tables/`.

## 8. Honest limitations

- **Collections are modest in scale.** They are real and standard, and the
  federation now spans both classic collections and one modern BEIR collection
  (SciFact), but the whole testbed is still only 23,709 documents. Results
  should be confirmed on a large modern federation (e.g., the bigger BEIR
  shards such as FiQA, TREC-COVID, or MS MARCO) before strong external-validity
  claims. The design is dataset-agnostic and ports directly.
- **Single-home routing.** Each query's relevant documents reside in one
  collection. Multi-home queries (relevant evidence spread across sources) are a
  natural and important extension.
- **Lexical retrieval only.** BM25 is used within each source. Neural retrievers
  would change absolute numbers; the trust mechanism is orthogonal to the base
  retriever and should compose with it.
- **Evil-twin model.** Treating exact duplicates with invalid provenance as
  non-relevant is the standard pooling assumption (unjudged ⇒ non-relevant) and
  reflects the provenance/verifiability motivation. The artifact now includes
  corrupted and partial mirror sensitivity variants; these show that degraded
  mirrors can become visible to content-based routers. Stale replicas and
  multi-source partial overlap remain useful extensions.
- **Trust supervision.** The online feedback signal ("did the source return a
  judged-relevant document") is a proxy for real user/verification feedback.
