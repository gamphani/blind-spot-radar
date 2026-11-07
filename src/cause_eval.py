# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Evaluate the transparent cause rule + chance level for the ranking metric (20 fresh seeds). Run: python src/cause_eval.py. Writes out/cause_eval.json."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from generate_data import simulate
from model import cause_label, cause_rule, make_frame

aucs, accs3, accs2, chance, nact = [], [], [], [], []
for s in range(20):
    fr = make_frame(simulate(s))
    te = fr[fr.week.isin(range(31, 39))].copy()
    te["diff"] = np.nan
    for w, g in te.groupby("week"):
        te.loc[g.index, "diff"] = cause_rule(g).values
    act = te[te.active == 1]
    chance.append(np.mean([5 / n for n in act.groupby("week").size() if n >= 10]))
    nact.append(act.groupby("week").size().mean())
    hi = act[act.frac_next2 >= act.frac_next2.median()]
    y = (hi.gap_sys > hi.gap_care).astype(int)
    if y.nunique() == 2:
        aucs.append(roc_auc_score(y, hi["diff"]))
        accs2.append(((hi["diff"] > 0).astype(int) == y).mean())
        lab = hi["diff"].map(cause_label)
        # 3-way: 'both' counts as right when the two true gaps are within 0.15 of each other
        close = (hi.gap_sys - hi.gap_care).abs() < 0.15
        ok = ((lab == "reporting") & (y == 1)) | ((lab == "community") & (y == 0)) | ((lab == "both") & close)
        accs3.append(ok.mean())
r = dict(cause_rule_auc=float(np.mean(aucs)), cause_rule_auc_min=float(np.min(aucs)), cause_rule_auc_max=float(np.max(aucs)),
         cause_rule_acc2=float(np.mean(accs2)), cause_rule_acc3=float(np.mean(accs3)),
         chance_p5=float(np.mean(chance)), mean_active=float(np.mean(nact)), n=len(aucs))
print(json.dumps(r, indent=1))
out = Path(__file__).resolve().parents[1] / "out"
out.mkdir(exist_ok=True)
(out / "cause_eval.json").write_text(json.dumps(r, indent=1))
