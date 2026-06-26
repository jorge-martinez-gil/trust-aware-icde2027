"""Regenerate the freshly-computed sweep/budget tables and the new EXT tables
and figures (RQ8 calibration, RQ9 drift, RQ10 sensitivity) for the paper."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "..", "results", "federated")
PT = os.path.join(HERE, "..", "..", "paper", "tables")
PF = os.path.join(HERE, "..", "..", "paper", "figures")
SEEDS = (7, 11, 13)


def _aggs():
    return [json.load(open(os.path.join(RES, f"sweep_agg_{s}.json"))) for s in SEEDS]


def regenerate_sweep_budget():
    aggs = _aggs()
    M = lambda metric, c, m: float(np.mean([a[c][m][metric] for a in aggs]))
    ret = lambda metric, m: (M(metric, "8", m) / M(metric, "0", m)) if M(metric, "0", m) else float("nan")
    counts = ["0", "2", "4", "6", "8"]
    methods = ["Random", "ReDDE", "CORI", "Trust-Aware", "Oracle"]

    def block(metric):
        out = []
        for m in methods:
            name = "\\textbf{Trust-Aware}" if m == "Trust-Aware" else m
            cells = " & ".join(f"{M(metric, c, m):.3f}" for c in counts)
            out.append(f"{name} & {cells} & {ret(metric, m)*100:.0f}\\% \\\\")
        return "\n".join(out)

    sweep = (r"\begin{tabular}{l ccccc c}" "\n" r"\toprule" "\n"
             r"Method & $0$ & $2$ & $4$ & $6$ & $8$ & Retention \\" "\n" r"\midrule" "\n"
             r"\multicolumn{7}{l}{\emph{(a) Routing accuracy $R@1$}}\\" "\n" r"\midrule" "\n"
             + block("R@1") + "\n" r"\midrule" "\n"
             r"\multicolumn{7}{l}{\emph{(b) End-to-end nDCG@10 ($b{=}1$)}}\\" "\n" r"\midrule" "\n"
             + block("nDCG@10(b=1)") + "\n" r"\midrule" "\n"
             r"\multicolumn{7}{l}{\emph{(c) End-to-end MAP ($b{=}1$)}}\\" "\n" r"\midrule" "\n"
             + block("MAP(b=1)") + "\n" r"\bottomrule" "\n" r"\end{tabular}")
    open(os.path.join(PT, "table_sweep.tex"), "w").write(sweep + "\n")

    bud = [r"\begin{tabular}{l cc cc cc}", r"\toprule",
           r"& \multicolumn{2}{c}{$b{=}1$} & \multicolumn{2}{c}{$b{=}2$} & \multicolumn{2}{c}{$b{=}3$}\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Method & nDCG@10 & cost & nDCG@10 & cost & nDCG@10 & cost \\", r"\midrule"]
    for m in ["ReDDE", "CORI", "Trust-Aware"]:
        name = "\\textbf{Trust-Aware}" if m == "Trust-Aware" else m
        cells = [f"{M(f'nDCG@10(b={b})','6',m):.3f} & {M(f'cost(b={b})','6',m):.3f}" for b in (1, 2, 3)]
        bud.append(f"{name} & " + " & ".join(cells) + r" \\")
    bud += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PT, "table_budget.tex"), "w").write("\n".join(bud) + "\n")


def regenerate_ext_tables_and_figures():
    cal = json.load(open(os.path.join(RES, "ext_graded_calibration.json")))
    agg = cal["aggregate"]
    mirrors = sorted([r for r in agg.values() if r["kind"] == "mirror"], key=lambda r: r["true_reliability"])
    ct = [r"\begin{tabular}{l ccccc}", r"\toprule",
          r"Injected source & True $p$ & Realised & Learned $\hat\mu$ & Learned LCB $\tau$ & \#obs \\", r"\midrule"]
    for r in mirrors:
        ct.append(f"mirror & {r['true_reliability']:.2f} & {r['realised']:.3f} & "
                  f"{r['learned_mean']:.3f}\\,$\\pm$\\,{r['learned_mean_std']:.3f} & {r['learned_lower']:.3f} & {r['n_obs']:.0f} \\\\")
    ct.append(r"\midrule")
    gen = [r for r in agg.values() if r["kind"] == "genuine"]
    ct.append(f"genuine (mean of 5) & {np.mean([r['true_reliability'] for r in gen]):.3f} & -- & "
              f"{np.mean([r['learned_mean'] for r in gen]):.3f} & -- & -- \\\\")
    ct += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PT, "table_calibration.tex"), "w").write("\n".join(ct) + "\n")

    hp = json.load(open(os.path.join(RES, "ext_hyperparam.json")))["grid"]
    priors = [(1, 1), (4, 2), (8, 2), (20, 5)]; zs = [0.0, 1.0, 2.0]
    ht = [r"\begin{tabular}{l ccc}", r"\toprule",
          r"Prior $(\alpha_0,\beta_0)$ & $z=0$ & $z=1$ & $z=2$ \\", r"\midrule"]
    for pr in priors:
        cells = []
        for z in zs:
            v = hp[f"prior={pr[0]:.0f}/{pr[1]:.0f},z={z:g}"]
            cell = f"{v['nDCG@10']:.3f}"
            if pr == (8, 2) and z == 1.0:
                cell = f"\\textbf{{{v['nDCG@10']:.3f}}}"
            cells.append(cell)
        ht.append(f"$({pr[0]},{pr[1]})$ & " + " & ".join(cells) + r" \\")
    ht += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PT, "table_hyperparam.tex"), "w").write("\n".join(ht) + "\n")

    # fig_calibration
    xs = [r["true_reliability"] for r in mirrors]; ys = [r["learned_mean"] for r in mirrors]
    lcb = [r["learned_lower"] for r in mirrors]; es = [r["learned_mean_std"] for r in mirrors]
    gx = [r["true_reliability"] for r in gen]; gy = [r["learned_mean"] for r in gen]
    fig, ax = plt.subplots(figsize=(4.3, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    ax.errorbar(xs, ys, yerr=es, fmt="o", color="#1f5fa8", ms=7, capsize=3, label="mirror: learned $\\hat\\mu$")
    ax.scatter(xs, lcb, marker="v", color="#1f5fa8", alpha=0.55, label="mirror: routing LCB $\\tau$")
    ax.scatter(gx, gy, marker="s", color="#159957", s=55, label="genuine source")
    ax.set_xlabel("true source reliability $p$"); ax.set_ylabel("learned trust")
    ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.25)
    c = cal["calibration"]
    ax.set_title(f"Trust calibration to graded reliability\n(mirror MAE={c['mae_mirror']:.3f}, Pearson $r$={c['pearson_mirror']:.3f})", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(PF, "fig_calibration.pdf")); fig.savefig(os.path.join(PF, "fig_calibration.png"), dpi=150); plt.close(fig)

    # fig_drift
    dr = json.load(open(os.path.join(RES, "ext_drift.json"))); tr = dr["trajectory"]; sw = dr["regime_change_at_target_query"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    colors = {"stale->refreshed": "#159957", "fresh->poisoned": "#b03030"}
    for tw, lab in dr["drift"].items():
        pts = tr[tw]
        ax.plot([p["t"] for p in pts], [p["mean"] for p in pts], label=lab, color=colors.get(lab, "k"), lw=1.8)
    ax.axvline(sw, ls="--", color="gray", lw=1); ax.text(sw + 2, 0.05, "reliability\nregime change", fontsize=7, color="gray")
    ax.set_xlabel(f"query index over the {dr['target']} stream"); ax.set_ylabel("learned trust (posterior mean)")
    ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.25)
    ax.set_title("Online re-calibration under non-stationary reliability", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(PF, "fig_drift.pdf")); fig.savefig(os.path.join(PF, "fig_drift.png"), dpi=150); plt.close(fig)


def generate_all_ext():
    os.makedirs(PT, exist_ok=True); os.makedirs(PF, exist_ok=True)
    regenerate_sweep_budget()
    regenerate_ext_tables_and_figures()
    print("Wrote fresh sweep/budget + EXT tables and figures to paper/")


if __name__ == "__main__":
    generate_all_ext()
