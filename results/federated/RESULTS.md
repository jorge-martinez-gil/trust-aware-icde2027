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
SCIFACT & Scientific claim verification & 5183 & 300 & 215 & 1.1 \\
\midrule
\textbf{Total} & 6 domains & 23709 & 776 & -- & -- \\
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
Random & 0.176\,$\pm$\,0.003 & 0.088\,$\pm$\,0.004 & 0.071\,$\pm$\,0.003 & 0.416\,$\pm$\,0.001 \\
ReDDE & 0.785\,$\pm$\,0.005 & 0.403\,$\pm$\,0.001 & 0.331\,$\pm$\,0.001 & 0.884\,$\pm$\,0.003 \\
CORI & 0.893\,$\pm$\,0.000 & 0.461\,$\pm$\,0.000 & 0.379\,$\pm$\,0.000 & 0.941\,$\pm$\,0.000 \\
\textbf{Trust-Aware} & 0.822\,$\pm$\,0.014 & 0.430\,$\pm$\,0.004 & 0.365\,$\pm$\,0.001 & 0.893\,$\pm$\,0.009 \\
Oracle & 1.000\,$\pm$\,0.000 & 0.509\,$\pm$\,0.000 & 0.413\,$\pm$\,0.000 & 1.000\,$\pm$\,0.000 \\
\midrule
\multicolumn{5}{l}{\emph{(b) Untrusted federation (6 evil-twin sources)}}\\
\midrule
Random & 0.088\,$\pm$\,0.005 & 0.045\,$\pm$\,0.004 & 0.036\,$\pm$\,0.004 & 0.264\,$\pm$\,0.005 \\
ReDDE & 0.405\,$\pm$\,0.024 & 0.204\,$\pm$\,0.017 & 0.168\,$\pm$\,0.014 & 0.658\,$\pm$\,0.013 \\
CORI & 0.451\,$\pm$\,0.011 & 0.228\,$\pm$\,0.006 & 0.187\,$\pm$\,0.007 & 0.698\,$\pm$\,0.005 \\
\textbf{Trust-Aware} & 0.807\,$\pm$\,0.012 & 0.421\,$\pm$\,0.003 & 0.357\,$\pm$\,0.001 & 0.881\,$\pm$\,0.007 \\
Oracle & 1.000\,$\pm$\,0.000 & 0.509\,$\pm$\,0.000 & 0.413\,$\pm$\,0.000 & 1.000\,$\pm$\,0.000 \\
\bottomrule
\end{tabular}
```

## Confirmatory inference (query-clustered paired Wilcoxon, family-wise Holm correction)

```
\begin{tabular}{llrrrrl}
\toprule
Metric & vs.\ & $N_q$ & $\Delta$ & 95\% CI & $p_{\text{Holm}}$ & $r_{\mathrm{rb}}$ (effect) \\
\midrule
nDCG@10 & CORI & 776 & +0.193 & [+0.173, +0.214] & 4.9e-59 & +0.809 (large) \\
nDCG@10 & ReDDE & 776 & +0.217 & [+0.196, +0.238] & 2.6e-66 & +0.841 (large) \\
MAP & CORI & 776 & +0.170 & [+0.152, +0.190] & 1.5e-66 & +0.825 (large) \\
MAP & ReDDE & 776 & +0.190 & [+0.170, +0.209] & 1.5e-74 & +0.854 (large) \\
\bottomrule
\end{tabular}
```

## Online reputation baselines (6 evil twins)

```
\begin{tabular}{l cccc}
\toprule
Method & R@1 & nDCG@10 & MAP & cost \\
\midrule
CORI & 0.451\,$\pm$\,0.011 & 0.228\,$\pm$\,0.006 & 0.187\,$\pm$\,0.007 & 0.441 \\
CORI+Mean & 0.832\,$\pm$\,0.011 & 0.430\,$\pm$\,0.001 & 0.362\,$\pm$\,0.002 & 0.429 \\
CORI+UCB & \textbf{0.841\,$\pm$\,0.009} & \textbf{0.436\,$\pm$\,0.003} & \textbf{0.365\,$\pm$\,0.003} & \textbf{0.428} \\
Trust-Aware & 0.807\,$\pm$\,0.012 & 0.421\,$\pm$\,0.003 & 0.357\,$\pm$\,0.001 & 0.433 \\
\bottomrule
\end{tabular}
```

## Adversary variant sensitivity (6 untrusted sources)

```
\begin{tabular}{l ccc ccc}
\toprule
Variant & CORI R@1 & Trust R@1 & $\Delta$ & CORI nDCG & Trust nDCG & $\Delta$ \\
\midrule
exact & 0.451 & 0.807 & +0.356 & 0.228 & 0.421 & +0.193 \\
corrupted-20 & 0.890 & 0.810 & -0.079 & 0.458 & 0.423 & -0.034 \\
partial-50 & 0.860 & 0.802 & -0.059 & 0.438 & 0.420 & -0.018 \\
\bottomrule
\end{tabular}
```

Machine-readable details: `statistical_analysis.json`.
