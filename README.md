# GreenGrid Optimizer

An end-to-end ML application that forecasts renewable energy generation (wind + solar) for Germany. It uses real public data sources, compares multiple forecasting approaches, explains its predictions, and serves results through an API and dashboard — all containerized and deployable.

## Problem Statement

Accurate short-horizon (24–72h) forecasts of renewable output are essential for grid balancing. For operators like TenneT, 50Hertz, Amprion, and TransnetBW in Germany, poor forecasts increase redispatch costs, reliance on fossil backup capacity, grid instability, and renewable curtailment. GreenGrid Optimizer addresses this by simulating a grid operator's forecasting system.

## Architecture
Data Source (ENTSO-E/SMARD & Open-Meteo) → Ingestion Pipeline → Feature Engineering → Model Training (Prophet/XGBoost) → FastAPI Backend → Streamlit Dashboard

## Folder Structure
- `configs/`: Configuration files
- `data/`: Raw, processed, and external datasets (git-ignored)
- `models/`: Trained model artifacts (git-ignored)
- `mlruns/`: MLflow tracking data (git-ignored)
- `notebooks/`: Jupyter notebooks for EDA and model comparison
- `src/`: Source code modules (ingestion, preprocessing, features, models, evaluation, explainability, api, utils)
- `frontend/`: Streamlit dashboard
- `tests/`: Pytest suite

## Installation and Docker Instructions
Ensure you have Docker and docker-compose installed.

1. Clone the repository
2. Copy `.env.example` to `.env` and provide your ENTSO-E API key.
3. Run `docker-compose up --build` to start the backend, frontend, and MLflow UI.

## Limitations and Next Steps
- Currently focusing on a v1 with single-step forecasting. Multi-step forecasting is a stretch goal.
- Relying on simple fallback (SMARD) if ENTSO-E data ingestion fails.
- (Will be updated as project evolves)
