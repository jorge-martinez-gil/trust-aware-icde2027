# Real Dataset: Wisconsin Diagnostic Breast Cancer

This directory contains the Wisconsin Diagnostic Breast Cancer dataset files:

- `wdbc.data`
- `wdbc.names`

The dataset is distributed by the UCI Machine Learning Repository as
`Breast Cancer Wisconsin (Diagnostic)`, DOI `10.24432/C5DW2B`, under CC BY 4.0.
It has 569 instances and 30 continuous features computed from digitized
fine-needle aspirate images of breast masses.

The artifact uses this dataset only for reproducible optimizer evaluation:
feature-family diagnostic rules are cross-validated, converted into calibrated
trust evidence, and then passed through the trust-aware optimizer.
