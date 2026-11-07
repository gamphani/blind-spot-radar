# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Synthetic weekly health-zone data for an Ebola-like outbreak. ALL NUMBERS ARE INVENTED.

Purpose: test whether a "blind-spot" workflow can work when the signals are informative, and
whether it fails safely when they are not. It says NOTHING about DRC.

Design (every rule below is an assumption, listed in docs/assumptions.md):
  * 36 health zones on a 6x6 grid in 3 provinces; 40 weekly time steps.
  * Latent per-zone traits drift over time: community trust, connectivity, trained DHIS2 users,
    insecurity. They drive BOTH how many cases are detected AND what the observable signals show.
  * True infections follow a spatial branching process. Low detection raises transmission
    (cases not isolated), so hidden transmission today predicts growth later.
  * Detection = system detection (connectivity, trained users, insecurity) x care detection
    (community trust). The split lets us test whether the tool can tell the two causes apart.
  * Observed signals are noisy functions of the latent traits. Reports arrive with a delay.
Run:  python src/generate_data.py [seed]   ->  data/synthetic/zone_weeks.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_SIDE, N_PROV, T = 6, 3, 40
SERIAL = np.array([0.35, 0.40, 0.25])   # weight on infections 1, 2, 3 weeks ago


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def simulate(seed=42, signal_noise=1.0, informative=True):
    """Return a zone-week DataFrame. signal_noise>1 blurs signals; informative=False cuts the
    link between latent traits and observable signals (a falsification test)."""
    rng = np.random.default_rng(seed)
    Z = N_SIDE * N_SIDE
    xy = np.array([(i // N_SIDE, i % N_SIDE) for i in range(Z)])
    prov = (xy[:, 0] // 2)                                   # 3 horizontal bands of 12 zones
    nbrs = [[j for j in range(Z) if abs(xy[i, 0] - xy[j, 0]) + abs(xy[i, 1] - xy[j, 1]) == 1]
            for i in range(Z)]
    pop = rng.lognormal(0, 0.45, Z); pop /= pop.mean()      # relative population size

    # ---- latent traits over time --------------------------------------------------------
    trust = np.zeros((Z, T)); conn = np.zeros((Z, T)); users = np.zeros((Z, T)); insec = np.zeros((Z, T))
    trust[:, 0] = rng.beta(4, 3, Z)
    users[:, 0] = rng.beta(2, 4, Z)
    insec_base = rng.beta(1.5, 5, Z)
    conn_week = rng.integers(3, 30, Z)                       # week a zone gets stable connectivity
    conn_week[rng.random(Z) < 0.35] = 99                     # a third never do within the window
    shock_zone = rng.choice(Z, 6, replace=False)
    shock_week = rng.integers(8, 30, 6)
    for t in range(T):
        if t:
            trust[:, t] = clip(trust[:, t - 1] + rng.normal(0, 0.02, Z), 0.05, 0.98)
            users[:, t] = clip(users[:, t - 1] + rng.normal(0.012, 0.01, Z), 0, 1)
            for z, w in zip(shock_zone, shock_week):         # an incident cuts trust, recovers slowly
                if t == w:
                    trust[z, t] = clip(trust[z, t] - 0.35, 0.05, 0.98)
                if w < t <= w + 10:
                    trust[z, t] = clip(trust[z, t] + 0.015, 0.05, 0.98)
        conn[:, t] = (t >= conn_week).astype(float)
        spike = np.zeros(Z)
        for z, w in zip(shock_zone, shock_week):
            if w - 1 <= t <= w + 3:
                spike[z] = 0.35
        insec[:, t] = clip(insec_base + spike + rng.normal(0, 0.03, Z), 0, 1)

    # ---- detection and true transmission ------------------------------------------------
    d_sys = clip(0.30 + 0.30 * conn + 0.35 * users - 0.25 * insec, 0.08, 0.95)
    d_care = clip(0.25 + 0.70 * trust, 0.10, 0.95)
    det = d_sys * d_care
    I = np.zeros((Z, T)); D = np.zeros((Z, T), dtype=int); deaths_true = np.zeros((Z, T))
    seeds = rng.choice(Z, 5, replace=False)
    I[seeds, 0] = rng.integers(3, 8, 5)
    r_base = 1.06 + 0.08 * rng.standard_normal(Z)
    for t in range(1, T):
        past = np.zeros(Z)
        for k, w in enumerate(SERIAL, start=1):
            if t - k >= 0:
                past += w * I[:, t - k]
        nb = np.array([np.mean(I[n, t - 1]) if n else 0 for n in nbrs])
        det_prev = det[:, t - 1]
        r_eff = r_base * (1 + 0.45 * (1 - trust[:, t])) * (1 - 0.50 * det_prev)
        lam = r_eff * past + 0.10 * nb + 0.05 * (rng.random(Z) < 0.02)
        I[:, t] = np.minimum(rng.poisson(clip(lam, 0, 400) * (0.5 + 0.5 * pop ** 0.3)), 300)
    for t in range(T):
        D[:, t] = rng.binomial(I[:, t].astype(int), det[:, t])
    hidden = I - D

    # ---- reporting delay: reports arrive 0-3 weeks after onset ---------------------------
    arrive = np.zeros((Z, T + 4))
    for z in range(Z):
        for t in range(T):
            p_fast = 0.35 + 0.35 * conn[z, t] + 0.2 * users[z, t]
            probs = np.array([p_fast, (1 - p_fast) * 0.55, (1 - p_fast) * 0.30, (1 - p_fast) * 0.15])
            arrive[z, t:t + 4] += rng.multinomial(D[z, t], probs / probs.sum())
    arrived = arrive[:, :T]

    # ---- observable signals --------------------------------------------------------------
    k = signal_noise
    if informative:
        t_sig, c_sig, u_sig, i_sig = trust, conn, users, insec
    else:                                                    # signals no longer track the truth
        t_sig = rng.permutation(trust.reshape(-1)).reshape(Z, T)
        c_sig, u_sig = conn, users
        i_sig = rng.permutation(insec.reshape(-1)).reshape(Z, T)
    completeness = clip(0.30 + 0.30 * c_sig + 0.35 * u_sig - 0.20 * i_sig + rng.normal(0, 0.07 * k, (Z, T)), 0.02, 1)
    delay_days = np.maximum(1, 9 - 4 * c_sig - 3 * u_sig + 2 * i_sig + rng.normal(0, 1.2 * k, (Z, T)))
    followup = clip(0.30 + 0.45 * t_sig + 0.15 * u_sig - 0.20 * i_sig + rng.normal(0, 0.09 * k, (Z, T)), 0, 1)
    linked = clip(0.08 + 0.55 * followup + rng.normal(0, 0.08 * k, (Z, T)), 0, 1)
    comm_death_share = clip(0.90 - 0.60 * t_sig + 0.10 * (1 - u_sig) + rng.normal(0, 0.10 * k, (Z, T)), 0, 1)
    refusals = rng.poisson(0.15 + 3.0 * (1 - t_sig) ** 2 * np.log1p(np.maximum(I, 1)) / 2.5)
    incidents = rng.poisson(0.05 + 1.6 * i_sig)
    devices = clip(0.35 + 0.35 * u_sig + 0.15 * c_sig + rng.normal(0, 0.05 * k, (Z, T)), 0.1, 1)

    rows = []
    for z in range(Z):
        for t in range(T):
            has_c = arrived[z, max(0, t - 1):t + 1].sum() > 0
            rows.append(dict(
                zone=f"{'ABC'[prov[z]]}-{z % 12 + 1:02d}", zone_index=z, province=f"Province {'ABC'[prov[z]]}",
                row=int(xy[z, 0]), col=int(xy[z, 1]), week=t + 1, pop_rel=round(float(pop[z]), 3),
                reported_cases=int(arrived[z, t]),
                completeness=round(float(completeness[z, t]), 3),
                report_delay_days=round(float(delay_days[z, t]), 2),
                contact_followup=round(float(followup[z, t]), 3) if has_c else np.nan,
                linked_share=round(float(linked[z, t]), 3) if has_c else np.nan,
                community_death_share=round(float(comm_death_share[z, t]), 3) if has_c else np.nan,
                burial_refusals=int(refusals[z, t]), security_incidents=int(incidents[z, t]),
                device_coverage=round(float(devices[z, t]), 3), online=int(c_sig[z, t]),
                # ---- ground truth, never available to the model as input ----
                true_cases=int(I[z, t]), true_detected=int(D[z, t]), true_hidden=int(hidden[z, t]),
                true_log_gap_sys=round(float(np.log(1 / d_sys[z, t])), 4),
                true_log_gap_care=round(float(np.log(1 / d_care[z, t])), 4),
            ))
    return pd.DataFrame(rows)


def main(seed=42):
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    df = simulate(seed)
    df.to_csv(out / "zone_weeks.csv", index=False)
    tot_true, tot_det = df.true_cases.sum(), df.true_detected.sum()
    print(f"seed {seed}: {df.zone.nunique()} zones x {df.week.nunique()} weeks; "
          f"overall detection {tot_det / tot_true:.2f}; total true cases {tot_true}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)
