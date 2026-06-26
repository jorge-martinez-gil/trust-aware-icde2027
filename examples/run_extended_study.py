"""Reproduce the journal-revision extended studies end-to-end:
graded-reliability calibration (RQ8), concept drift (RQ9), hyper-parameter
sensitivity (RQ10), then regenerate the fresh sweep/budget tables, the new EXT
tables, and the new figures used by paper/main_vldbj.tex.

    python examples/run_extended_study.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
from trust_aware.federated.experiment_ext import run_all_ext
from trust_aware.federated import report_ext

if __name__ == "__main__":
    run_all_ext()
    report_ext.generate_all_ext()
    print("Done. See paper/tables/table_{calibration,hyperparam,sweep,budget}.tex "
          "and paper/figures/fig_{calibration,drift}.{pdf,png}.")
