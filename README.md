# Blind-Spot Radar

Finding the blind spots: an uncertainty-aware decision layer on DHIS2 to locate hidden Ebola transmission in the DRC response.

Africa CDC AES-PHI Fellowship capstone, Digital Health Division, Africa CDC Headquarters. Author: Dave Phiri.

## What it does

For each health zone, the tool estimates the **share of the next two weeks' cases that will probably go unreported**. It puts an **80% and a 95% range** around that estimate, says whether the gap looks like a **reporting problem**, a **community problem** or **both**, and suggests a **first step**.

It works on weekly counts per health zone (one row per zone per week). It does not use names or case lists.

> **NB: This version runs on a simulator, not on real data.** The simulator is a computer-made outbreak (36 zones, 40 weeks) in which the true number of cases and the true share missed are known, so the method can be tested. The dashboard page carries a "SYNTHETIC DATA" banner. Nothing here describes the DRC response. Testing on real Africa CDC DHIS2 data comes next.

## How it works, in order

| Step | What happens | Where in the code |
|---|---|---|
| Simulate (for now) | Make a synthetic outbreak with known truth | `src/generate_data.py` |
| Tidy and build signals | One row per zone per week, past information only, nine warning signals plus 4-week averages | `make_frame` in `src/model.py` |
| Score | Ridge regression (a weighted checklist) predicts the share unreported | `BlindSpotModel` in `src/model.py` |
| Ranges | Split conformal prediction gives 80% and 95% ranges | `BlindSpotModel`, `conformal_q` in `src/model.py` |
| Cause and first step | A readable rule compares reporting signals with community signals | `cause_rule`, `cause_label`, `drivers` in `src/model.py` |
| Dashboard | A standalone web page with overview, zone view and test view | `src/build_dashboard_data.py`, `src/dashboard_template.html` |
| Check honestly | Baselines, rolling-origin test, range coverage, falsification tests | `src/evaluate.py`, `src/cause_eval.py`, `tests/` |

The score is the one small learning part that managers see. Gradient boosting is also used behind the scenes (to decide whether a zone has enough activity to rate) and as a rival model in testing.

## Run it

You need Python 3.11 or later (tested on 3.12 and 3.13). The commands below work on Linux, macOS and WSL. On Windows, use `py -m venv .venv` and `.venv\Scripts\activate` instead.

```bash
# 1. Make an environment and install the packages (once)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Check that everything works (9 tests, about a minute)
python -m pytest -q

# 3. Build the dashboard and serve it on your computer
python run.py
```

Then open <http://localhost:8000> in your browser. Stop the server with Ctrl+C. If port 8000 is busy, use `python run.py --port 8080`. On WSL2 the Windows browser can open `localhost` directly. The server listens on your computer only.

Optional:

```bash
python src/build_dashboard_data.py   # build out/blindspot.html only, a static page you can open directly
python src/evaluate.py               # the full evaluation, 20 fresh runs, several minutes; writes out/results.json
python src/cause_eval.py             # check the cause label, about a minute
python src/generate_data.py          # write the simulated data to data/synthetic/zone_weeks.csv
```

Outputs go to `out/` and `data/`. Both are rebuilt by the commands above and are not committed.

## Results so far (simulator only, 20 fresh test runs)

| Method | Top-5 hit rate |
|---|---|
| Pick zones at random | 0.26 |
| Rank by reported cases alone | 0.36 |
| Equal-weight checklist of the nine signals | 0.62 |
| **Transparent score (this tool)** | **0.62** |
| Gradient boosting (complex challenger) | 0.53 |

The 80% range held the truth 85% of the time and the 95% range 96%. The reporting-versus-community label scored 0.97 (area under the curve), which is partly circular because the link between signals and causes is built into the simulator. Ranking by expected number of missed cases was no better than ranking by reported cases. These numbers come from the simulator and say nothing about the outbreak itself.

## Limits

- Not yet tested on real DRC data.
- The ranges are wide (about 40 points). Use the score to rank zones and prompt a check, not as an exact figure.
- The latest two weeks of reports are always incomplete.
- Zones with too little activity are not rated.

## Next

1. Replace the simulator with the Africa CDC DHIS2 weekly zone extract (aggregate counts only).
2. Define proxy outcomes for real data (for example, cases that arrive late for earlier weeks).
3. Prototype on real data by 30 November 2026, then deploy and refine on site.

## Data policy

This repository must never contain real data. `.gitignore` blocks data files (`data/`, `*.csv`, `*.xlsx` and similar). Only aggregate counts are used, and any real extract stays outside the repository.

## Licence

Licensing follows the [Digital Public Goods Standard](https://github.com/DPGAlliance/DPG-Standard/blob/main/standard.md): an OSI-approved licence for software, Creative Commons for content, and Open Data Commons for data.

| Part | Licence | File |
|---|---|---|
| Source code (`src/`, `tests/`) | BSD 3-Clause | [`LICENSE`](LICENSE) |
| Documentation (this README and `docs/`) | CC BY 4.0 | [`LICENSE-docs.md`](LICENSE-docs.md) |
| Data | None is committed. Any dataset released with this project would need an Open Data Commons licence. | n/a |

Copyright (c) 2026, Africa Centres for Disease Control and Prevention (Africa CDC).

> **NB:** Using these licences is one requirement of the Digital Public Goods Standard. This repository is not a registered Digital Public Good. Registration is a separate application that also covers privacy, documentation, interoperability and other indicators.
