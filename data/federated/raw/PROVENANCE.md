# Federated IR Testbed — Data Provenance
Assembled 2026-06-02T07:18Z; SciFact added 2026-07-16. Openly licensed IR test
collections: five classic (1960s-1990s, public domain) plus one modern (BEIR
SciFact; see its licence terms below).

## Sources (GitHub mirrors, depth-1 clone commit hashes)
- CACM, MED, NPL  <- github.com/massimilianoviola/advanced-information-retrieval @ 5d5f282ce6e93043614e92c320b614beeb7b9cb6
- CRANFIELD       <- github.com/oussbenk/cranfield-trec-dataset @ 1208e6edfb6cb2527b2c44398d3d8fefd3249144
- CISI            <- github.com/Shakiba-Alipour/Information-Retrieval-on-CISI @ 7a23c34602e8949d795bb1835004546e00d1d1ba

## Sources (direct release download)
- SCIFACT         <- https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
                     BEIR v1.0.0 release of SciFact (Wadden et al., EMNLP 2020).
                     sha256 of scifact.zip: 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165
                     Claims/annotations: CC BY 4.0. Corpus abstracts are from
                     Semantic Scholar S2ORC: ODC-By 1.0 (attribution required).
                     Per github.com/allenai/scifact/blob/master/LICENSE.md.

### SciFact normalization
`script/convert_scifact.py` rewrites the BEIR layout into this repo's JSON-lines
collection format. Two choices are baked into that script:

- **Test split only** (300 judged queries). BEIR's published SciFact baselines
  score against the test split; the 809 train queries are excluded so our
  per-collection numbers stay comparable to that literature.
- **Title + abstract concatenated** into one document text, matching how the
  CRANFIELD parser folds `<title>` into `<text>`.

Result: 5,183 documents, 300 queries, 339 judgments (~1.1 relevant docs per
query).
