# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Build the JSON that feeds the prototype dashboard (seed 42, test origins only). Run: python src/build_dashboard_data.py
Writes out/dashboard_data.json and the standalone page out/blindspot.html."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import simulate
from model import SIGNAL_FEATS, BlindSpotModel, cause_label, cause_rule, drivers, make_frame

ORIGINS = list(range(31, 39))
ACTION = {
    "reporting": "Check reports are arriving: devices, connectivity and trained DHIS2 users.",
    "community": "Send community-engagement and safe-burial support; talk with local leaders.",
    "both": "Check reporting first, then add community engagement.",
}


def main(seed=42):
    root = Path(__file__).resolve().parents[1]
    df = simulate(seed)
    fr = make_frame(df)
    tr, ca = fr[fr.week <= 22], fr[(fr.week >= 23) & (fr.week <= 28)]
    m = BlindSpotModel(SIGNAL_FEATS, seed, "linear").fit(tr, ca)
    out = {"seed": seed, "origins": ORIGINS, "by_origin": {}, "history": {}}
    for z, g in df.groupby("zone"):
        g = g.sort_values("week")
        out["history"][z] = dict(reported=g.reported_cases.tolist(), completeness=g.completeness.round(2).tolist(),
                                 followup=[None if pd.isna(v) else round(v, 2) for v in g.contact_followup])
    for o in ORIGINS:
        te = fr[fr.week == o].copy()
        p = m.predict(te)
        te = te.assign(**{c: p[c].values for c in p.columns})
        te['diff'] = cause_rule(te).values
        rows = []
        for _, r in te.iterrows():
            active = (r.rep_sum4 >= 1) or (r.burden >= 3)
            cause = cause_label(r["diff"])
            drv = [dict(label=l, value=round(float(v), 2), typical=round(float(t), 2), key=k) for _, k, l, v, t in drivers(r, te)]
            rows.append(dict(
                zone=r.zone, province=r.province, row=int(r.row), col=int(r.col),
                score=round(float(r.score) * 100), lo80=round(float(r.lo80) * 100), hi80=round(float(r.hi80) * 100),
                lo95=round(float(r.lo95) * 100), hi95=round(float(r.hi95) * 100),
                burden=round(float(r.burden), 1), rep4=int(r.rep_sum4), 
                cause=cause, action=ACTION[cause], active=bool(active), drivers=drv,
                sig=dict(completeness=round(float(r.completeness_m4), 2), followup=None if pd.isna(r.contact_followup_m4) else round(float(r.contact_followup_m4), 2),
                         comm_deaths=None if pd.isna(r.community_death_share_m4) else round(float(r.community_death_share_m4), 2),
                         refusals=round(float(r.burial_refusals_m4), 1), delay=round(float(r.report_delay_days_m4), 1),
                         devices=round(float(r.device_coverage_m4), 2)),
                truth=None if pd.isna(r.frac_next2) else round(float(r.frac_next2) * 100)))
        out["by_origin"][str(o)] = rows
    (root / "out").mkdir(exist_ok=True)
    (root / "out" / "dashboard_data.json").write_text(json.dumps(out, separators=(",", ":")))
    # standalone dashboard page: the template with the data embedded
    template = (Path(__file__).resolve().parent / "dashboard_template.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", json.dumps(out, separators=(",", ":")))
    (root / "out" / "blindspot.html").write_text(page, encoding="utf-8")
    a = [r for r in out["by_origin"]["38"] if r["active"]]
    a.sort(key=lambda r: -r["score"])
    for r in a[:6]:
        print(r["zone"], r["score"], (r["lo80"], r["hi80"]), r["cause"], r["truth"], r["rep4"], [d["label"] for d in r["drivers"]])
    inside = [(t["truth"] is not None and t["lo80"] <= t["truth"] <= t["hi80"]) for o in out["by_origin"].values() for t in o if t["truth"] is not None]
    print("80% band held", round(np.mean(inside), 2), "n", len(inside), "| active at 38:", len(a))


if __name__ == "__main__":
    main()
