# Federated Trust-Aware Retrieval - Results

Generated from `results/federated/*.json` (seeds 7, 11, 13).

## Testbed

```
\begin{tabular}{llrrrr}
\toprule
Collection & Domain & Docs & Queries & Avg.\ len & Rel./q \\
\midrule
CACM & Computer science & 3204 & 52 & 53 & 14.2 \\
MED & Biomedicine & 1033 & 30 & 154 & 23.2 \\
NPL & Electrical engineering & 11429 & 93 & 42 & 22.4 \\
CRAN & Aeronautics & 1400 & 225 & 177 & 7.2 \\
CISI & Library \& information science & 1460 & 76 & 296 & 41.0 \\
\midrule
\textbf{Total} & 5 domains & 18526 & 476 & -- & -- \\
\bottomrule
\end{tabular}
```

## Main results (mean +/- std over 3 seeds)

```
\begin{tabular}{l cccc}
\toprule
Method & R@1 & nDCG@10 & MAP & MRR \\
\midrule
\multicolumn{5}{l}{\emph{(a) Clean federation (0 untrusted sources)}}\\
\midrule
Random & 0.190\,$\pm$\,0.014 & 0.079\,$\pm$\,0.002 & 0.054\,$\pm$\,0.003 & 0.447\,$\pm$\,0.009 \\
ReDDE & 0.696\,$\pm$\,0.013 & 0.284\,$\pm$\,0.006 & 0.190\,$\pm$\,0.003 & 0.836\,$\pm$\,0.007 \\
CORI & 0.874\,$\pm$\,0.000 & 0.363\,$\pm$\,0.000 & 0.249\,$\pm$\,0.000 & 0.933\,$\pm$\,0.000 \\
\textbf{Trust-Aware} & 0.832\,$\pm$\,0.022 & 0.339\,$\pm$\,0.010 & 0.237\,$\pm$\,0.006 & 0.900\,$\pm$\,0.015 \\
Oracle & 1.000\,$\pm$\,0.000 & 0.409\,$\pm$\,0.000 & 0.278\,$\pm$\,0.000 & 1.000\,$\pm$\,0.000 \\
\midrule
\multicolumn{5}{l}{\emph{(b) Untrusted federation (6 evil-twin sources)}}\\
\midrule
Random & 0.088\,$\pm$\,0.009 & 0.034\,$\pm$\,0.003 & 0.023\,$\pm$\,0.003 & 0.271\,$\pm$\,0.009 \\
ReDDE & 0.335\,$\pm$\,0.015 & 0.135\,$\pm$\,0.005 & 0.091\,$\pm$\,0.001 & 0.596\,$\pm$\,0.013 \\
CORI & 0.417\,$\pm$\,0.004 & 0.170\,$\pm$\,0.002 & 0.116\,$\pm$\,0.001 & 0.671\,$\pm$\,0.003 \\
\textbf{Trust-Aware} & 0.779\,$\pm$\,0.011 & 0.316\,$\pm$\,0.006 & 0.221\,$\pm$\,0.003 & 0.858\,$\pm$\,0.009 \\
Oracle & 1.000\,$\pm$\,0.000 & 0.409\,$\pm$\,0.000 & 0.278\,$\pm$\,0.000 & 1.000\,$\pm$\,0.000 \\
\bottomrule
\end{tabular}
```

## Confirmatory inference (query-clustered paired Wilcoxon, family-wise Holm correction)

```
\begin{tabular}{llrrrrl}
\toprule
Metric & vs.\ & $N_q$ & $\Delta$ & 95\% CI & $p_{\text{Holm}}$ & $r_{\mathrm{rb}}$ (effect) \\
\midrule
nDCG@10 & CORI & 476 & +0.146 & [+0.125, +0.168] & 2.1e-33 & +0.749 (large) \\
nDCG@10 & ReDDE & 476 & +0.181 & [+0.158, +0.205] & 7.0e-39 & +0.791 (large) \\
MAP & CORI & 476 & +0.105 & [+0.089, +0.122] & 2.9e-38 & +0.769 (large) \\
MAP & ReDDE & 476 & +0.130 & [+0.112, +0.149] & 2.1e-43 & +0.800 (large) \\
\bottomrule
\end{tabular}
```

## Online reputation baselines (6 evil twins)

```
\begin{tabular}{l cccc}
\toprule
Method & R@1 & nDCG@10 & MAP & cost \\
\midrule
CORI & 0.417\,$\pm$\,0.004 & 0.170\,$\pm$\,0.002 & 0.116\,$\pm$\,0.001 & 0.385 \\
CORI+Mean & 0.790\,$\pm$\,0.019 & 0.320\,$\pm$\,0.009 & 0.224\,$\pm$\,0.007 & 0.330 \\
CORI+UCB & \textbf{0.814\,$\pm$\,0.015} & \textbf{0.331\,$\pm$\,0.005} & \textbf{0.230\,$\pm$\,0.003} & 0.332 \\
Trust-Aware & 0.779\,$\pm$\,0.011 & 0.316\,$\pm$\,0.006 & 0.221\,$\pm$\,0.003 & \textbf{0.322} \\
\bottomrule
\end{tabular}
```

## Adversary variant sensitivity (6 untrusted sources)

```
\begin{tabular}{l ccc ccc}
\toprule
Variant & CORI R@1 & Trust R@1 & $\Delta$ & CORI nDCG & Trust nDCG & $\Delta$ \\
\midrule
exact & 0.417 & 0.779 & +0.363 & 0.170 & 0.316 & +0.146 \\
corrupted-20 & 0.885 & 0.785 & -0.100 & 0.366 & 0.317 & -0.049 \\
partial-50 & 0.878 & 0.798 & -0.080 & 0.363 & 0.327 & -0.036 \\
\bottomrule
\end{tabular}
```

Machine-readable details: `statistical_analysis.json`.
