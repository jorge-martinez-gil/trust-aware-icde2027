# Cover Letter

Dear Editors of *Information Sciences*,

Please consider the manuscript "Trust as a First-Class Routing Objective:
Provenance-Robust Resource Selection for AI-Native Data Systems" for publication
as an original research article in *Information Sciences*.

AI-native data systems increasingly route queries across heterogeneous sources:
relational stores, vector indexes, retrieval services, model-backed tools, and
third-party or machine-generated mirrors. Existing resource-selection methods
primarily optimize content relevance, which makes them structurally blind to a
source's provenance. This manuscript formalizes that failure mode, proves a
content-only impossibility bound for content-identical impostor sources, and
introduces TrustRoute, a minimal expected-utility router that combines content
relevance with online beta-binomial trust calibration.

The contribution fits the journal's emphasis on intelligent systems, data
engineering, information and knowledge systems, and a balance of theory and
practice. The paper combines a formal problem statement and bound with a fully
reproducible empirical study over six real information-retrieval collections
(23,709 documents and 776 judged queries), spanning five classic corpora and the
modern BEIR SciFact collection. Under heavy contamination with eight evil-twin
sources, TrustRoute preserves 96% of its clean routing accuracy, while
content-only baselines retain roughly half. At the six-twin operating point,
TrustRoute improves nDCG@10 over CORI by 0.193 and over ReDDE by 0.217, with
large effects after Holm-corrected paired Wilcoxon tests. Additional experiments
show that the trust posterior recovers graded reliability, tracks non-stationary
drift, and is stable across a 4 x 3 hyper-parameter grid.

The manuscript is original, is not under consideration elsewhere, and all data,
baselines, statistics, figures, and tables are reproducible from the public
artifact at https://github.com/jorge-martinez-gil/trust-aware. The author
declares no known competing interests.

Sincerely,

Jorge Martinez-Gil  
Software Competence Center Hagenberg GmbH (SCCH), Austria  
jorge.martinez-gil@scch.at
