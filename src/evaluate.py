# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Rolling-origin test, ablations and stress tests. Run: python src/evaluate.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from generate_data import simulate
from model import ALL_FEATS, REPORT_ONLY, SIGNAL_FEATS, BlindSpotModel, make_frame

TRAIN_MAX, CAL = 22, (23, 28)
TEST = list(range(31, 39))
K = 5


def split(fr):
    return (fr[fr.week <= TRAIN_MAX], fr[(fr.week >= CAL[0]) & (fr.week <= CAL[1])], fr[fr.week.isin(TEST)])


def rank_metrics(test, score_col, target="frac_next2", act_only=True):
    p, s = [], []
    for w, g in test.groupby("week"):
        if act_only:
            g = g[g.active == 1]
        if len(g) < 10:
            continue
        top_true = set(g.nlargest(K, target).zone)
        top_pred = set(g.nlargest(K, score_col).zone)
        p.append(len(top_true & top_pred) / K)
        s.append(spearmanr(g[score_col], g[target]).statistic)
    return float(np.mean(p)), float(np.nanmean(s))


def zscore_composite(te):
    """A fair 'traffic-light checklist' baseline: equal-weight average of the warning signals."""
    parts = []
    for c, bad_low in [("completeness_m4", True), ("contact_followup_m4", True), ("linked_share_m4", True),
                       ("device_coverage_m4", True), ("community_death_share_m4", False),
                       ("report_delay_days_m4", False), ("burial_refusals_m4", False)]:
        v = te.groupby("week")[c].transform(lambda x: (x - x.median()) / (x.std() or 1)).fillna(0)
        parts.append(-v if bad_low else v)
    return sum(parts) / len(parts)


def run_one(seed, **sim):
    df = simulate(seed, **sim)
    fr = make_frame(df)
    tr, ca, te = split(fr)
    res = {}
    full = BlindSpotModel(ALL_FEATS, seed, "gbm").fit(tr, ca)
    pf = full.predict(te)
    te = te.assign(full=pf.score.values, lo80=pf.lo80.values, hi80=pf.hi80.values, lo95=pf.lo95.values,
                   hi95=pf.hi95.values, burden=pf.burden.values, p_rep=pf.p_reporting.values)
    li = BlindSpotModel(SIGNAL_FEATS, seed, "linear").fit(tr, ca).predict(te)
    te = te.assign(linear=li.score.values, l_lo80=li.lo80.values, l_hi80=li.hi80.values, l_lo95=li.lo95.values,
                   l_hi95=li.hi95.values, l_p_rep=li.p_reporting.values)
    te["rep_only"] = BlindSpotModel(REPORT_ONLY, seed, "gbm").fit(tr, ca).predict(te).score.values
    te["b_followup"] = 1 - te.contact_followup_m4.fillna(te.contact_followup_m4.median())
    te["b_complete"] = 1 - te.completeness_m4
    te["b_checklist"] = zscore_composite(te)
    te["b_lowreported"] = -te.rep_sum4          # naive: "few reported cases = something is missed"
    for name, col in [("gbm_model", "full"), ("linear_model", "linear"), ("reported_only_model", "rep_only"), ("checklist_of_signals", "b_checklist"),
                      ("followup_only", "b_followup"), ("completeness_only", "b_complete"), ("fewest_reported", "b_lowreported")]:
        res[name] = rank_metrics(te, col)
    # burden (absolute hidden cases): does the model beat simply ranking by reported cases?
    res["burden_model"] = rank_metrics(te, "burden", "hidden_next2", act_only=False)
    res["burden_by_reported"] = rank_metrics(te.assign(r=te.rep_sum4), "r", "hidden_next2", act_only=False)
    act = te[te.active == 1]
    res["cov80"] = float(((act.frac_next2 >= act.l_lo80) & (act.frac_next2 <= act.l_hi80)).mean())
    res["cov95"] = float(((act.frac_next2 >= act.l_lo95) & (act.frac_next2 <= act.l_hi95)).mean())
    res["gbm_cov80"] = float(((act.frac_next2 >= act.lo80) & (act.frac_next2 <= act.hi80)).mean())
    res["gbm_cov95"] = float(((act.frac_next2 >= act.lo95) & (act.frac_next2 <= act.hi95)).mean())
    res["mae_model"] = float(np.mean(np.abs(act.linear - act.frac_next2)))
    res["mae_gbm"] = float(np.mean(np.abs(act.full - act.frac_next2)))
    res["mae_constant"] = float(np.mean(np.abs(tr[tr.active == 1].frac_next2.mean() - act.frac_next2)))
    hi = act[act.frac_next2 >= tr[tr.active == 1].frac_next2.median()]
    y = (hi.gap_sys > hi.gap_care).astype(int)
    if y.nunique() == 2:
        res["cause_auc"] = float(roc_auc_score(y, hi.l_p_rep))
        res["cause_acc"] = float(((hi.l_p_rep > 0.5).astype(int) == y).mean())
        res["cause_auc_gbm"] = float(roc_auc_score(y, hi.p_rep))
        rule_s = hi.contact_followup_m4.fillna(0.5).rank(pct=True) - hi.completeness_m4.rank(pct=True)
        res["cause_auc_rule"] = float(roc_auc_score(y, rule_s))
        rule = (hi.completeness_m4.rank(pct=True) < hi.contact_followup_m4.fillna(0.5).rank(pct=True)).astype(int)
        res["cause_acc_rule"] = float((rule == y).mean())
        res["cause_majority"] = float(max(y.mean(), 1 - y.mean()))
    return res, te


def aggregate(runs):
    keys = ["gbm_model", "linear_model", "reported_only_model", "checklist_of_signals", "followup_only", "completeness_only",
            "fewest_reported", "burden_model", "burden_by_reported"]
    out = {}
    for k in keys:
        a = np.array([r[k] for r in runs])
        out[k] = dict(p_at_5=float(a[:, 0].mean()), p_at_5_min=float(a[:, 0].min()), p_at_5_max=float(a[:, 0].max()),
                      spearman=float(a[:, 1].mean()), spearman_min=float(a[:, 1].min()), spearman_max=float(a[:, 1].max()))
    for k in ["cov80", "cov95", "gbm_cov80", "gbm_cov95", "mae_model", "mae_gbm", "mae_constant", "cause_auc", "cause_auc_gbm", "cause_auc_rule", "cause_acc", "cause_acc_rule", "cause_majority"]:
        v = [r[k] for r in runs if k in r]
        out[k] = float(np.mean(v)) if v else None
    out["n_runs"] = len(runs)
    return out


def main(n_seeds=20, first=0):
    root = Path(__file__).resolve().parents[1]
    result = {"main": aggregate([run_one(s)[0] for s in range(first, first + n_seeds)]),
              "noisy_signals_x3": aggregate([run_one(s, signal_noise=3.0)[0] for s in range(first, first + 10)]),
              "uninformative_signals": aggregate([run_one(s, informative=False)[0] for s in range(first, first + 10)])}
    (root / "out").mkdir(exist_ok=True)
    (root / "out" / "results.json").write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"== {k} (n={v['n_runs']})")
        for m in ["gbm_model", "linear_model", "reported_only_model", "checklist_of_signals", "followup_only", "completeness_only", "fewest_reported", "burden_model", "burden_by_reported"]:
            d = v[m]
            print(f"   {m:22s} P@5 {d['p_at_5']:.2f} [{d['p_at_5_min']:.2f}-{d['p_at_5_max']:.2f}]  spearman {d['spearman']:.2f}")
        print(f"   linear cov80 {v['cov80']:.2f} cov95 {v['cov95']:.2f} | gbm cov80 {v['gbm_cov80']:.2f} cov95 {v['gbm_cov95']:.2f} | "
              f"MAE linear {v['mae_model']:.3f} gbm {v['mae_gbm']:.3f} constant {v['mae_constant']:.3f}")
        print(f"   cause AUC linear {v['cause_auc']:.2f} gbm {v['cause_auc_gbm']:.2f} simple-rule {v['cause_auc_rule']:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
