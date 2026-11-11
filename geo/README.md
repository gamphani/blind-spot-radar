# Map outlines

`drc_provinces.tsv`: the 26 health provinces (divisions provinciales de la santé) of the DRC, simplified to about 5 km for an overview map.

- Source: "Divisions provinciales de la santé (RDC)", OpenStreetMap RDC, on the Humanitarian Data Exchange (HDX), extract of 12 December 2021. Boundaries originate from the Référentiel Géographique Commun (RGC.cd).
- Licence: Open Database License (ODbL) 1.0. © OpenStreetMap contributors. This simplified file is a derived database and is shared under the same licence. It is **not** covered by the repository's BSD-3-Clause licence.
- Format: one province per line, `name<TAB>outline`. The outline is delta-encoded in steps of 0.01 degree; see `src/geo.py`.

`zones.tsv` (not shipped): health-zone outlines keyed by DHIS2 org unit id. Make it from the response DHIS2 so that outlines and data always match:

```bash
# save this from DHIS2 (replace <level> with the health-zone level, usually 3):
#   <dhis2>/api/organisationUnits.json?level=<level>&fields=id,name,geometry&paging=false
python src/geo.py --from-dhis2 orgunits.json
```

When `zones.tsv` exists and the data rows carry the DHIS2 id in `uid`, the dashboard's "DRC map" view shades each health zone by the kind and size of its gap. Without it, the view shows provinces only, with the provinces in scope highlighted. Zone estimates are never drawn on real boundaries while the data are synthetic.

An alternative source for zone outlines, if DHIS2 has no geometry: "Zones de santé (RDC)", OpenStreetMap RDC on HDX (ODbL), which already carries a `ref_dhis2` id for each of the 519 zones.
