"""Figure for the report: simulated Brier ALC of v1 / v2 / v3 (reads results/results_cell11.csv)."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root / "results" / "results_cell11.csv")
df["modeles_visibles"] = df["modeles_visibles"].astype(str)
names = {
    "lissage vers 0,5 (v1)": "v1  shrinkage toward 0.5",
    "IRT 1.0/1.5 (v2)": "v2  shared-label IRT (fixed scales)",
    "IRT EB depart 1.25/2.0": "v3  shared-label IRT (empirical Bayes)",
}
style = {names["lissage vers 0,5 (v1)"]: ("#6b6b6b", "s", ":"),
         names["IRT 1.0/1.5 (v2)"]: ("#1f77b4", "o", "--"),
         names["IRT EB depart 1.25/2.0"]: ("#d95f02", "^", "-")}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))

sub = df[df.modeles_visibles == "tous"]
bud = [0, 1, 3, 7, 15, 31]
for k, lab in names.items():
    r = sub[sub.modele == k].iloc[0]
    c, m, ls = style[lab]
    ax[0].plot(range(6), [r[f"B{b}"] for b in bud], color=c, marker=m, ls=ls, lw=1.5, ms=4.5, label=lab)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(bud)
ax[0].set_xlabel("labels acquired per pair"); ax[0].set_ylabel("Brier score")
ax[0].set_title("(a) all other subjects visible", fontsize=9)

order = ["10", "30", "100", "tous"]
for k, lab in names.items():
    c, m, ls = style[lab]
    ys = [df[(df.modeles_visibles == s) & (df.modele == k)].Brier_ALC.iloc[0] for s in order]
    ax[1].plot(range(4), ys, color=c, marker=m, ls=ls, lw=1.5, ms=4.5)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["10", "30", "100", "all"])
ax[1].set_xlabel("other subjects visible per benchmark"); ax[1].set_ylabel("Brier ALC")
ax[1].set_title("(b) Brier ALC vs. visibility", fontsize=9)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = root / "report" / "figures" / "results.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
