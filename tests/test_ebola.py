# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import T, simulate  # noqa: E402
from model import BlindSpotModel, SIGNAL_FEATS, conformal_q, drivers, make_frame  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return simulate(7)


@pytest.fixture(scope="module")
def frame(df):
    return make_frame(df)


def test_shape_and_ranges(df):
    assert df.zone.nunique() == 36 and df.week.max() == T
    assert df.completeness.between(0, 1).all() and df.device_coverage.between(0, 1).all()
    assert (df.true_hidden >= 0).all() and (df.true_detected <= df.true_cases).all()


def test_reproducible():
    a, b = simulate(3), simulate(3)
    pd.testing.assert_frame_equal(a, b)


def test_detection_in_plausible_range(df):
    d = df.true_detected.sum() / df.true_cases.sum()
    assert 0.15 < d < 0.7


def test_no_future_leakage(df, frame):
    """Features at origin t must not change if we alter data after week t."""
    t0 = 20
    df2 = df.copy()
    late = df2.week > t0
    for c in ["reported_cases", "completeness", "contact_followup", "burial_refusals"]:
        df2.loc[late, c] = df2.loc[late, c].fillna(0) * 0 + 99
    f2 = make_frame(df2)
    cols = [c for c in frame.columns if c not in ("hidden_next2", "true_next2", "frac_next2", "active", "gap_sys", "gap_care")]
    a = frame[frame.week == t0][cols].reset_index(drop=True)
    b = f2[f2.week == t0][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_targets_are_future_only(frame):
    r = frame.iloc[0]
    assert r.week == 5 and r.hidden_next2 >= 0


def test_conformal_quantile_basic():
    s = np.arange(1, 101)
    assert conformal_q(s, 0.2) == pytest.approx(81, abs=1.5)


def test_model_runs_and_bands_ordered(frame):
    tr, ca, te = frame[frame.week <= 22], frame[(frame.week >= 23) & (frame.week <= 28)], frame[frame.week >= 31]
    m = BlindSpotModel(SIGNAL_FEATS, 0, "linear").fit(tr, ca)
    p = m.predict(te)
    assert (p.lo95 <= p.lo80 + 1e-9).all() and (p.lo80 <= p.score + 1e-9).all()
    assert (p.score <= p.hi80 + 1e-9).all() and (p.hi80 <= p.hi95 + 1e-9).all()
    assert p.score.between(0, 1).all() and p.p_reporting.between(0, 1).all()


def test_uninformative_signals_lose_their_edge():
    """Falsification: when signals stop tracking the truth, rank quality must fall."""
    from evaluate import run_one
    good = np.mean([run_one(s)[0]["linear_model"][1] for s in (1, 2, 3)])
    bad = np.mean([run_one(s, informative=False)[0]["linear_model"][1] for s in (1, 2, 3)])
    assert good > bad


def test_drivers_flag_worst_signals(frame):
    wk = frame[frame.week == 30]
    row = wk.iloc[0].copy()
    row["completeness_m4"] = wk.completeness_m4.min() - 0.5
    d = drivers(row, wk)
    assert any(x[1] == "completeness" for x in d)


def test_map_outlines():
    """Province outlines load, decode inside the DRC, and DHIS2 geometry converts round-trip."""
    from geo import dashboard_geo, decode_rings, encode_rings, zones_from_dhis2
    g = dashboard_geo()
    names = [n for n, _ in g["provinces"]["items"]]
    assert len(names) == 26 and "Ituri" in names and set(g["scope"]) <= set(names)
    for _, enc in g["provinces"]["items"]:
        for ring in decode_rings(enc, g["provinces"]["f"]):
            assert all(11.5 < x < 31.5 and -14 < y < 5.5 for x, y in ring)
    sq = [[[30.0, 1.0], [30.5, 1.0], [30.5, 1.5], [30.0, 1.5], [30.0, 1.0]]]
    rows = zones_from_dhis2({"organisationUnits": [{"id": "abc123", "name": "Test", "geometry": {"type": "Polygon", "coordinates": sq}}]})
    assert rows[0][0] == "abc123" and decode_rings(rows[0][1], 1000)[0] == [tuple(p) for p in sq[0]]
    assert decode_rings(encode_rings(sq, 100), 100)[0] == [tuple(p) for p in sq[0]]
