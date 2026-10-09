# Power BI report

`python -m scripts.export_bi` writes five CSVs to `bi/` (committed, ~260 KB) from the processed dataset,
`models/backtest_results.json` and the MLflow DB. Power BI Desktop → Get data → Text/CSV → select all five.

| Table | Grain |
|---|---|
| `dim_target` | wind_onshore, wind_offshore, solar (+ mean actual MW) |
| `fact_generation_hourly` | one row per hour: generation (MW) per target + key weather variables |
| `fact_backtest_fold` | target × walk-forward fold: MAE, RMSE, R², train size, test window, tuned params |
| `fact_backtest_horizon` | target × lead-time bucket: recursive MAE, persistence and seasonal-naive MAE, skill scores |
| `fact_model_runs` | latest MLflow run per model × target × metric (Prophet vs XGBoost) |

Relationships: `dim_target[target]` 1→* each fact table's `target` column (for `fact_generation_hourly` unpivot the
three generation columns in Power Query first, or build three measures).

```DAX
Backtest MAE (MW) = AVERAGE(fact_backtest_fold[mae_mw])
Skill vs Persistence = AVERAGE(fact_backtest_horizon[skill_vs_persistence])
Capacity-Normalised MAE = DIVIDE([Backtest MAE (MW)], AVERAGE(dim_target[mean_actual_mw]))
```

Suggested pages: **Generation** (hourly lines by target with weather slicer), **Forecast quality** (MAE by horizon
vs persistence/seasonal naive; skill score), **Model comparison** (`fact_model_runs` matrix, Prophet vs XGBoost).

## Caveats
- Only ~4 months of hourly data (2026-05-29 onward) and 4 walk-forward folds: error bars are wide.
- MAPE is intentionally not exported (solar ≈ 0 at night makes it meaningless).
- `fact_model_runs` metrics are the single-holdout MLflow runs, not the walk-forward numbers; do not mix them in one visual.
- Backtest uses lead-matched archived weather forecasts, so the numbers reflect realistic forecast-weather error.
- The `.pbix` itself is not included; build it from the steps above.
