# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC)
"""Map outlines for the dashboard's real-map view.

Outlines are stored as compact text: one line per area, "<key>\t<rings>". Rings are separated by "|";
each ring is a list of "dx,dy" steps in integer units of 1/F degree (F = 100 for provinces, 1000 for zones),
where the first step is the absolute position. The dashboard decodes them in the browser.

Two sources:
  geo/drc_provinces.tsv   DRC provinces (26), shipped with the repo. ODbL 1.0, see geo/README.md.
  geo/zones.tsv           Health-zone outlines keyed by DHIS2 org unit id. NOT shipped: make it from your
                          own DHIS2 instance so the outlines and the data always match:

    python src/geo.py --from-dhis2 orgunits.json

  where orgunits.json is saved from
    <dhis2>/api/organisationUnits.json?level=<health-zone level>&fields=id,name,geometry&paging=false
"""
import argparse
import json
from pathlib import Path

GEO = Path(__file__).resolve().parents[1] / "geo"
FIELD_SITE = {"name": "Bunia", "lon": 30.25, "lat": 1.56}
SCOPE = ["Ituri"]  # provinces highlighted on the map; edit when the scope is confirmed
ATTRIBUTION = ("Boundaries: health provinces from the Référentiel Géographique Commun via OpenStreetMap RDC "
               "(HDX, ODbL 1.0), simplified. © OpenStreetMap contributors.")


def _read_tsv(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, enc = line.split("\t", 1)
            rows.append([key, enc])
    return rows


def _dp(pts, tol):
    """Douglas-Peucker line simplification (tolerance in degrees)."""
    if len(pts) < 4:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, z = stack.pop()
        (x1, y1), (x2, y2) = pts[a], pts[z]
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        best, bi = 0.0, -1
        for i in range(a + 1, z):
            x, y = pts[i]
            t = ((x - x1) * dx + (y - y1) * dy) / L if L else 0.0
            t = max(0.0, min(1.0, t))
            ex, ey = x1 + t * dx - x, y1 + t * dy - y
            d = ex * ex + ey * ey
            if d > best:
                best, bi = d, i
        if best > tol * tol:
            keep[bi] = True
            stack += [(a, bi), (bi, z)]
    return [p for p, k in zip(pts, keep) if k]


def encode_rings(rings, f):
    out = []
    for ring in rings:
        px = py = 0
        steps = []
        for x, y in ring:
            X, Y = round(x * f), round(y * f)
            steps.append(f"{X - px},{Y - py}")
            px, py = X, Y
        out.append(" ".join(steps))
    return "|".join(out)


def decode_rings(enc, f):
    rings = []
    for part in enc.split("|"):
        x = y = 0
        ring = []
        for s in part.split(" "):
            dx, dy = s.split(",")
            x, y = x + int(dx), y + int(dy)
            ring.append((x / f, y / f))
        rings.append(ring)
    return rings


def outer_rings(geometry):
    """Outer rings of a GeoJSON Polygon or MultiPolygon (holes are dropped)."""
    if not geometry:
        return []
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    return []


def zones_from_dhis2(orgunits, tol=0.005, f=1000):
    """[[org unit id, encoded outline], ...] from a DHIS2 organisationUnits export (id, name, geometry)."""
    items = orgunits.get("organisationUnits", orgunits if isinstance(orgunits, list) else [])
    rows = []
    for ou in items:
        rings = [r for r in (_dp([tuple(p) for p in ring], tol) for ring in outer_rings(ou.get("geometry"))) if len(r) >= 4]
        if rings:
            rows.append([ou["id"], encode_rings(rings, f)])
    return rows


def dashboard_geo():
    """The 'geo' block of dashboard_data.json. Zone outlines are included only if geo/zones.tsv exists."""
    geo = {"provinces": {"f": 100, "items": _read_tsv(GEO / "drc_provinces.tsv")},
           "zones": None, "scope": SCOPE, "site": FIELD_SITE, "attribution": ATTRIBUTION}
    zpath = GEO / "zones.tsv"
    if zpath.exists():
        geo["zones"] = {"f": 1000, "items": dict(_read_tsv(zpath))}
    return geo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make geo/zones.tsv from a DHIS2 organisation-unit export.")
    ap.add_argument("--from-dhis2", required=True, help="JSON saved from api/organisationUnits.json with fields=id,name,geometry")
    ap.add_argument("--tolerance", type=float, default=0.005, help="simplification in degrees (default 0.005, about 500 m)")
    a = ap.parse_args()
    rows = zones_from_dhis2(json.loads(Path(a.from_dhis2).read_text(encoding="utf-8")), a.tolerance)
    (GEO / "zones.tsv").write_text("\n".join(f"{k}\t{e}" for k, e in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} zone outlines to {GEO / 'zones.tsv'}")
