# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Blind-spot model: predict hidden (unreported) cases per health zone for the next 2 weeks.

Inputs use ONLY information known at the forecast origin (week t). Target = true hidden cases
in weeks t+1..t+2 (known only in the simulation). Three parts:
  1. Hidden-burden regressor (gradient boosting, log scale) + conformal intervals.
  2. Cause classifier: is the dominant gap in REPORTING (system) or COMMUNITY (care-seeking)?
  3. Plain-language drivers: how far each signal group is from the typical zone.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

HORIZON = 2
SIGNALS = ["completeness", "report_delay_days", "contact_followup", "linked_share",
           "community_death_share", "burial_refusals", "security_incidents", "device_coverage", "online"]
GROUPS = {   # signal groups used in plain-language drivers and the two-cause split
    "reporting": ["completeness", "report_delay_days", "device_coverage", "online"],
    "community": ["contact_followup", "linked_share", "community_death_share", "burial_refusals"],
    "security": ["security_incidents"],
}


def make_frame(df):
    """One row per (zone, origin week) with past-only features and future-only targets."""
    df = df.sort_values(["zone_index", "week"]).reset_index(drop=True)
    grid = df.pivot(index="zone_index", columns="week", values="reported_cases")
    out = []
    for z, g in df.groupby("zone_index"):
        g = g.set_index("week")
        nb = [n for n in range(36) if abs(g.row.iloc[0] - df[df.zone_index == n].row.iloc[0]) +
              abs(g.col.iloc[0] - df[df.zone_index == n].col.iloc[0]) == 1]
        for t in range(5, 41 - HORIZON):
            rep = g.reported_cases
            f = dict(zone=g.zone.iloc[0], zone_index=z, province=g.province.iloc[0], row=int(g.row.iloc[0]),
                     col=int(g.col.iloc[0]), week=t, pop_rel=g.pop_rel.iloc[0],
                     rep_1=rep.loc[t], rep_2=rep.loc[t - 1], rep_3=rep.loc[t - 2], rep_4=rep.loc[t - 3],
                     rep_sum4=rep.loc[t - 3:t].sum(),
                     rep_growth=(rep.loc[t - 1:t].sum() + 1) / (rep.loc[t - 3:t - 2].sum() + 1),
                     nb_rep4=float(grid.loc[nb, t - 3:t].to_numpy().mean()))
            for s in SIGNALS:
                f[s] = g[s].loc[t]
                f[s + "_m4"] = g[s].loc[t - 3:t].mean()
            fut = g.loc[t + 1:t + HORIZON]
            f["hidden_next2"] = float(fut.true_hidden.sum())
            f["true_next2"] = float(fut.true_cases.sum())
            f["frac_next2"] = f["hidden_next2"] / f["true_next2"] if f["true_next2"] >= 5 else np.nan
            f["active"] = int(f["true_next2"] >= 5)
            f["gap_sys"] = float(fut.true_log_gap_sys.mean())
            f["gap_care"] = float(fut.true_log_gap_care.mean())
            out.append(f)
    return pd.DataFrame(out)


REPORT_ONLY = ["rep_1", "rep_2", "rep_3", "rep_4", "rep_sum4", "rep_growth", "nb_rep4", "pop_rel"]
SIGNAL_FEATS = [c for s in SIGNALS for c in (s, s + "_m4")]
ALL_FEATS = REPORT_ONLY + SIGNAL_FEATS


def reg(seed=0, **kw):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=15, random_state=seed, **kw)


def conformal_q(scores, alpha):
    n = len(scores)
    return float(np.quantile(scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))


def lin(model):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=False), StandardScaler(), model)


class BlindSpotModel:
    """Primary score = predicted SHARE of next-2-week cases that will go unreported (0-1).
    kind="linear": transparent weighted score (ridge) on the warning signals, symmetric conformal band.
    kind="gbm":    gradient boosting challenger, conformalised quantile band.
    Also: expected hidden cases (burden) and a reporting-led vs community-led cause probability."""

    def __init__(self, feats=ALL_FEATS, seed=0, kind="gbm"):
        self.feats, self.seed, self.kind = feats, seed, kind

    def fit(self, train, calib):
        a, ca = train[train.active == 1], calib[calib.active == 1]
        hf = a[a.frac_next2 >= a.frac_next2.median()]
        y_cause = (hf.gap_sys > hf.gap_care).astype(int)
        if self.kind == "linear":
            self.m = lin(Ridge(alpha=30)).fit(a[self.feats], a.frac_next2)
            res = np.abs(ca.frac_next2 - np.clip(self.m.predict(ca[self.feats]), 0, 1))
            self.c80, self.c95 = conformal_q(res, 0.20), conformal_q(res, 0.05)
            self.cause = lin(LogisticRegression(C=0.5, max_iter=500)).fit(hf[self.feats], y_cause)
        else:
            self.m = reg(self.seed).fit(a[self.feats], a.frac_next2)
            self.q = {q: reg(self.seed, loss="quantile", quantile=q).fit(a[self.feats], a.frac_next2) for q in (0.1, 0.9)}
            lo, hi = self.q[0.1].predict(ca[self.feats]), self.q[0.9].predict(ca[self.feats])
            sc = np.maximum(lo - ca.frac_next2, ca.frac_next2 - hi)
            self.c80, self.c95 = conformal_q(sc, 0.20), conformal_q(sc, 0.05)
            self.cause = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                                        min_samples_leaf=15, random_state=self.seed).fit(hf[self.feats], y_cause)
        self.burden = reg(self.seed).fit(train[self.feats], np.log1p(train.hidden_next2))
        return self

    def predict(self, X):
        c = lambda v: np.clip(v, 0, 1)
        mid = c(self.m.predict(X[self.feats]))
        if self.kind == "linear":
            lo80, hi80, lo95, hi95 = mid - self.c80, mid + self.c80, mid - self.c95, mid + self.c95
        else:
            lo, hi = self.q[0.1].predict(X[self.feats]), self.q[0.9].predict(X[self.feats])
            lo80, hi80, lo95, hi95 = lo - self.c80, hi + self.c80, lo - self.c95, hi + self.c95
        return pd.DataFrame(dict(
            score=mid, lo80=np.minimum(c(lo80), mid), hi80=np.maximum(c(hi80), mid),
            lo95=np.minimum(c(lo95), mid), hi95=np.maximum(c(hi95), mid),
            burden=np.expm1(self.burden.predict(X[self.feats])).clip(0),
            p_reporting=self.cause.predict_proba(X[self.feats])[:, 1]), index=X.index)


def drivers(row, ref):
    """Plain-language drivers: the signals furthest from the typical zone in the same week."""
    names = {"completeness": "reports received", "report_delay_days": "report delay",
             "contact_followup": "contacts followed up", "linked_share": "cases from known contacts",
             "community_death_share": "deaths in the community", "burial_refusals": "burial refusals",
             "security_incidents": "security incidents", "device_coverage": "community workers with a phone"}
    bad_if_low = {"completeness", "contact_followup", "linked_share", "device_coverage"}
    out = []
    for s, label in names.items():
        v, m = row[s + "_m4"], ref[s + "_m4"].median()
        if pd.isna(v) or pd.isna(m):
            continue
        sd = ref[s + "_m4"].std() or 1
        z = (v - m) / sd
        worse = (z < 0) if s in bad_if_low else (z > 0)
        if worse and abs(z) >= 0.75:
            out.append((abs(z), s, label, v, m))
    out.sort(reverse=True)
    return out[:3]


def _z(X, col):
    """Cross-zone z-score of a signal within the same week (missing -> 0, i.e. 'typical')."""
    v = X[col]
    sd = v.std()
    return ((v - v.median()) / (sd if sd and sd > 0 else 1)).fillna(0)


def cause_rule(X):
    """Transparent cause rule. Positive = the gap looks more like a REPORTING problem
    (late/incomplete reports, no devices, offline); negative = more like a COMMUNITY problem
    (contacts not followed, cases not linked, deaths at home, burial refusals)."""
    rep_bad = (-_z(X, "completeness_m4") + _z(X, "report_delay_days_m4")
               - _z(X, "device_coverage_m4") - _z(X, "online_m4")) / 4
    com_bad = (-_z(X, "contact_followup_m4") - _z(X, "linked_share_m4")
               + _z(X, "community_death_share_m4") + _z(X, "burial_refusals_m4")) / 4
    return rep_bad - com_bad


def cause_label(diff, band=0.3):
    return "reporting" if diff > band else "community" if diff < -band else "both"
