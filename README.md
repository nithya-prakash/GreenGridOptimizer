# GreenGrid Optimizer

[![CI](https://github.com/nithya-prakash/GreenGridOptimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/nithya-prakash/GreenGridOptimizer/actions/workflows/ci.yml)

An end-to-end ML application that forecasts renewable energy generation (wind onshore, wind offshore, and solar) for Germany. It ingests real public data, trains and compares multiple forecasting approaches per source, explains its predictions with SHAP, serves real recursive multi-step forecasts through a FastAPI backend, and lets you ask questions about the forecast in plain language via an LLM chat feature — all containerized and deployable with `docker compose up`.

![GreenGrid Optimizer dashboard — changing the forecast horizon live and watching the 1–72h forecast recompute for wind onshore, wind offshore, and solar](docs/dashboard.gif)

## Problem Statement

Accurate short-horizon (24–72h) forecasts of renewable output are essential for grid balancing. For operators like TenneT, 50Hertz, Amprion, and TransnetBW in Germany, poor forecasts increase redispatch costs, reliance on fossil backup capacity, grid instability, and renewable curtailment. GreenGrid Optimizer addresses this by simulating a grid operator's forecasting system.

**The scale of the problem:** German grid operators spent €2.77 billion on grid congestion management (redispatch + curtailment) in 2024 across 30,304 GWh of measures, including 1,389 GWh of curtailed solar generation alone — up 97% year-over-year ([Bundesnetzagentur data, via Clean Energy Wire](https://www.cleanenergywire.org/news/germanys-needs-and-costs-grid-management-down-2024-network-agency)). Forecast accuracy isn't the only lever on that number, but it's one of the more tractable ones: redispatch and curtailment decisions are made under uncertainty about how much wind/solar will actually generate over the next 24–72h, and tighter forecasts reduce how conservatively operators have to plan around that uncertainty.

## Architecture

Data Source (SMARD generation + Open-Meteo multi-site weather) → Ingestion Pipeline → Feature Engineering → Model Training (Prophet vs. XGBoost, per source) → FastAPI Backend (recursive multi-step forecast, SHAP explainability, LLM chat) → Streamlit Dashboard

## What it actually does

- **Data**: hourly realised generation from SMARD (Bundesnetzagentur) and weather from Open-Meteo, refreshed every 6h by an `updater` service (retraining weekly), so forecasts start from the latest published hour; the dashboard shows how old the data is. Weather is averaged over 6 land sites across Germany (north for onshore wind, south for solar) plus separate North Sea / Baltic points for offshore wind (`configs/pipeline.yaml`). Models are trained on Open-Meteo's archived *forecast* weather, not observations, so training sees the same kind of imperfect weather inputs as serving.
- **Forecasting**: multi-step forecasts (1–72h ahead) for wind onshore, wind offshore, and solar (`src/models/recursive.py` — the same function the backtest evaluates).
  - **Wind** is recursive: each step predicts t+1 from current generation, lags/rolling stats and the forecast weather *for* t+1, then feeds its prediction back in.
  - **Solar** at 1h uses the 1h model (forced to 0 in deep night: no current output and no forecast sun); beyond 1h it uses a direct multi-horizon model (`src/models/solar_direct.py`): one model over all lead times that predicts solar at the target hour from that hour's forecast radiation, the same hour on the last observed day, and the current output level. Recursive solar compounded its own errors and lost to a "same as yesterday" baseline beyond ~6h.
- **Modeling**: Prophet (base + weather regressors) and XGBoost trained and compared per target on a holdout week; XGBoost wins on every target and is what's served — refit on all data, including the holdout week, once evaluated.
- **Explainability**: SHAP feature contributions for any target's latest prediction, browsable from the dashboard.
- **Chat**: ask questions about the current forecast in plain language ("why is wind onshore low tomorrow morning?") — answered by Claude grounded in the forecast/actuals/SHAP data the dashboard is already showing. Optional — the rest of the app works without an API key.
- **Validation**: walk-forward backtest that evaluates the forecast actually served (recursive, out to 72h) against persistence and seasonal-naive baselines, with hyperparameters re-tuned inside each fold — see Model Performance below. Ingestion also checks the data's physics (solar must be ~0 at night), not just its schema.

### Prediction intervals and battery dispatch

`/forecast` returns a 90% split-conformal interval per target and hour (`*_lower_mw`, `*_upper_mw`, `interval_confidence`). Half-widths are quantiles of the backtest's absolute errors per lead-time bucket and are written to `models/forecast_intervals.json` by `python -m src.evaluation.backtest`. Coverage is checked on a held-out fold (calibrated on earlier folds): wind onshore 91-100%, wind offshore 77-97%, solar 85-91% against a 90% target. Offshore and solar sometimes under-cover because the held-out week differs from the calibration weeks, and only one week is held out, so treat the intervals as approximate. Re-running the backtest on a refreshed dataset shifts the fold windows and therefore the numbers.

`/forecast/dispatch?power_mw=...&capacity_mwh=...` schedules a battery to firm the forecast total toward its mean, as a linear program (`src/optimization/dispatch.py`, `scipy.optimize.linprog`, energy-neutral over the horizon, with round-trip losses). It is firming of the *forecast*, not price arbitrage; real gains depend on forecast error.

## Model Performance

Walk-forward backtest (`src/evaluation/backtest.py`; full results in `models/backtest_results.json`, plots in `models/backtest_*.png`): 120 days of data (2026-05-29 to 2026-09-26), 4 expanding-window folds with a 1-week test window each, and all models (incl. XGBoost hyperparameters and the direct solar model) re-trained on each fold's training data only. Within each test week the served forecast is started from an origin every 7h (so every hour of the day is an origin once) and run 72h ahead — 86 forecast runs in total.

**Weather is lead-matched.** For a step L hours ahead, the backtest uses the weather forecast that was issued ~L//24 days before the target hour (Open-Meteo Previous Runs API), not the short-lead forecast the models train on. So a 60h-ahead step is scored with a 2-day-old weather forecast, as it would be live.

**Forecast MAE (MW) by lead time** — model ± std across the 4 folds / persistence (last value) / seasonal naive (same hour, last observed day):

| Lead time | wind_onshore (mean 10.6 GW) | wind_offshore (mean 2.5 GW) | solar (mean 14.9 GW) |
|---|---|---|---|
| 1h | 1342 ±537 / **1192** / 7951 | **316** ±39 / 328 / 1366 | **1056** ±265 / 3009 / 2651 |
| 2–6h | **2959** ±539 / 3526 / 7792 | **592** ±128 / 773 / 1411 | **1326** ±197 / 12031 / 2695 |
| 7–24h | **3966** ±970 / 6779 / 8045 | **794** ±67 / 1356 / 1437 | **1359** ±222 / 17101 / 2680 |
| 25–48h | **3783** ±559 / 9881 / 10855 | **857** ±69 / 1417 / 1483 | **1568** ±56 / 15555 / 3100 |
| 49–72h | **4093** ±593 / 10634 / 10694 | **863** ±113 / 1490 / 1547 | **1618** ±201 / 15637 / 2980 |

What this shows:
- **From 2h onwards, all three targets beat both baselines at every lead time**, by 1.7–2.6× against the better baseline at 25–72h.
- **At 1h, onshore wind loses to persistence** (1342 vs 1192 MW) and offshore only ties it. One hour ahead, "same as now" is very hard to beat. The value of the model is in the 2–72h range.
- **Solar needed a direct model.** A recursive solar forecast (feeding its own predictions back as lags) had 7–24h MAE ~3400 MW and lost to seasonal naive beyond ~6h; the direct multi-horizon model is at 1359.
- Lead-matched weather makes long leads honestly worse than scoring with short-lead weather would (solar 49–72h: 1618 here vs ~1520 with short-lead weather in an earlier run).
- The 1h-ahead numbers the models are trained on (MAE 1140 / 327 / 1103 MW, R² 0.96 / 0.86 / 0.98 across folds) say little about a 24–72h forecast.

**Two data bugs this evaluation surfaced (both fixed):**
1. Until 2026-09-26 the SMARD client used the wrong filter IDs: "wind_onshore" was actually photovoltaics, "wind_offshore" hard coal, and "solar" pumped storage. Every metric this README reported before that date was for the wrong sources.
2. Open-Meteo labels radiation/precipitation by the *end* of the hour they average over; SMARD labels an hour by its *start*. The join was therefore one hour out of step (solar vs. radiation correlation 0.93 misaligned, 0.98 aligned). Fixing it improved every solar lead except 1h: the 1h model now leans almost entirely on forecast radiation instead of on observed output, and its 1h error rose from ~840 to ~1060 MW. Blending in persistence at 1h is a possible follow-up.

## Folder Structure

- `configs/`: Non-secret pipeline parameters (holdout/backtest windows, forecast horizon, weather sites, chat model and limits) — see `configs/pipeline.yaml`. Secrets and deployment-specific values (API keys, auth token, CORS origins) stay in `.env`.
- `data/`: Raw, processed, and external datasets (git-ignored)
- `models/`: Trained model artifacts (`production/*.pkl` and `production/metadata.json` git-ignored; diagnostic/backtest/SHAP plots and `backtest_results.json` are tracked)
- `mlruns/`: MLflow tracking data (git-ignored)
- `notebooks/`: Jupyter notebooks for EDA and model comparison (run with `requirements-dev.txt` installed; both verified to execute against the current data)
- `src/`: Source code modules (ingestion, preprocessing, features, models, evaluation, explainability, api, jobs, utils)
- `frontend/`: Streamlit dashboard
- `locks/`, `scripts/lock.sh`: hash-pinned lockfiles per architecture and the script that regenerates them
- `tests/`: Pytest suite — 70 tests covering feature engineering (incl. next-hour weather), train/serve parity of the recursive forecaster, the forecast/explain/model-info endpoints, chat hardening (auth, rate limit, input limits, oversized context), backtest folds/origins/baselines, multi-site weather aggregation (incl. circular wind-direction averaging and the radiation timestamp relabelling), lead-matched backtest weather, the refresh job (failure keeps the previous state, no needless retrain), model hot-reload, forecast caching, per-client rate limits on the read endpoints (and the trusted-dashboard exemption), the SMARD filter-ID mapping, the direct solar model (no data after the origin except weather, training cut-off), data validation (incl. the solar day/night check), config loading, and the ENTSO-E XML parser
- `.github/workflows/`: CI — ruff lint; build the image and run the tests; check that no `.env`/data/`.git` is baked into the image; start the `docker compose` stack on a clean checkout (no `.env`) and check the API and dashboard come up

## Installation and Docker Instructions

Ensure you have Docker with Compose v2.24+ installed.

1. Clone the repository.
2. Optionally copy `.env.example` to `.env`. Everything in it is optional, and the stack starts without the file:
   - `ENTSOE_API_KEY` — enables real ENTSO-E data (see Limitations: implemented but unverified against a live key). Without it, ingestion uses SMARD, no key needed.
   - `ANTHROPIC_API_KEY` — enables the dashboard's chat feature. Without it, `/chat` returns a clear 503 and the rest of the app is unaffected.
   - `API_AUTH_TOKEN` — shared secret required as the `X-API-Key` header on `/chat`, which spends Anthropic credits. The dashboard sends it automatically. **Required for chat whenever `ANTHROPIC_API_KEY` is set** — chat stays disabled (503) without it, so the key is never exposed unauthenticated.
   - `CORS_ORIGINS` — browser origins allowed to call the API (default `http://localhost:8501`).
3. Run `docker compose up --build` to start the API (`:8000`), the Streamlit dashboard (`:8501`) and the `updater` service, all published on `127.0.0.1` only. The MLflow UI is a developer tool and starts only on request: `docker compose --profile dev-tools up mlflow` (`127.0.0.1:5000`). On start-up the updater ingests the last 120 days, builds features and (since no models exist yet) trains them; after that it refreshes the data every 6h and retrains weekly (`refresh:` in `configs/pipeline.yaml`). A failed refresh keeps the last good data and models.
4. To run the steps by hand instead (inside the `api` container, or locally with `PYTHONPATH=.` and `pip install -r requirements-dev.txt`):
   ```
   python -m src.jobs.refresh --once             # everything below, as the updater does it
   python -m src.ingestion.pipeline --days 120   # SMARD + Open-Meteo through today, incl. lead-day weather for the backtest
   python -m src.features.feature_engineering
   python -m src.models.trainer                  # Prophet vs XGBoost per target; saves XGBoost + metadata.json
   python -m src.evaluation.backtest             # optional: walk-forward backtest (~5 min)
   python -m src.explainability.shap_analysis    # optional: SHAP plots per target
   ```

Tests and lint: `docker build --build-arg INSTALL_DEV=true -t greengrid:dev . && docker run --rm greengrid:dev python -m pytest tests/` and `ruff check .`.

Dependencies: `requirements.txt` (and `requirements-dev.txt` for tests/lint/notebooks) lists the top-level packages; the image installs from full, hash-pinned lockfiles covering every transitive dependency, one per CPU architecture (`locks/`, x86_64 and aarch64). After changing a top-level pin, regenerate them with `scripts/lock.sh`.

## Security Notes

- `/chat` is the only endpoint that costs money. It fails closed: it requires a shared token (`API_AUTH_TOKEN`) whenever an Anthropic key is configured, and adds a per-client-IP rate limit (`chat.rate_limit_per_minute`), input-size limits (message, history length, roles), and a context-size cap that rejects oversized requests (413) instead of truncating them. The rate limit is in-process, so each API replica counts separately.
- The read endpoints (`/forecast`, `/historical`, `/metrics`, …) are public by design for a demo, but rate limited per client IP (`api:` in `configs/pipeline.yaml`; `/forecast`, which runs the model and calls Open-Meteo, has a tighter limit). `/forecast` results are cached and only recomputed when the data or models change, so repeated requests cost ~nothing. The dashboard presents `API_AUTH_TOKEN` and is exempt from these limits (all its users share one IP).
- All ports are published on `127.0.0.1` only. To serve the dashboard publicly, put it behind a reverse proxy with TLS rather than exposing the containers directly, set `API_AUTH_TOKEN`, and keep MLflow (unauthenticated, accepts writes) private.
- CORS allows only the origins in `CORS_ORIGINS`.
- `.dockerignore` keeps `.env`, `data/`, `mlruns/` and `.git` out of the image; CI checks this.

## Limitations and Next Steps

- The ENTSO-E client (`src/ingestion/entsoe_client.py`) implements the real Transparency Platform API (Actual Generation per Type, XML parsing) per ENTSO-E's public documentation, but no API key was available to test the live call — only the parsing logic is verified (`tests/test_entsoe_client.py`, against a schema-accurate fixture). SMARD remains the verified, real data path; ENTSO-E is attempted first and falls back to SMARD on any failure.
- No cloud deployment — runs locally via `docker compose` only.
- At 1h ahead, the models don't beat persistence for wind onshore (see Model Performance); a persistence blend for the first hours would help. The 1h solar model also got worse after the radiation-alignment fix, for the same reason.
- Lead-matched weather is matched to the day (the run issued ~L//24 days earlier), not to the exact model run and lead hour.
- Wind is still recursive; a direct multi-horizon wind model like the solar one is the natural next experiment.
- Weather is a weighted average over a handful of sites with hand-set weights, not capacity-weighted per grid cell.
