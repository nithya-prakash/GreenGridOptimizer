import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.logger import log
from src.utils.config import settings
from src.evaluation.metrics import calculate_metrics
from src.features.feature_engineering import GENERATION_COLUMNS

N_FOLDS = 4
TEST_SIZE = 168  # 1 week per fold, hourly data


def _fold_bounds(n_rows: int, n_folds: int, test_size: int):
    """Expanding-window fold boundaries: each fold's test window is the `test_size`
    hours immediately after the previous fold's, so later folds train on strictly
    more history. Returns (train_end, test_end) index pairs."""
    first_test_start = n_rows - n_folds * test_size
    for fold in range(n_folds):
        train_end = first_test_start + fold * test_size
        test_end = train_end + test_size
        yield train_end, test_end


def run_backtest(n_folds: int = N_FOLDS, test_size: int = TEST_SIZE):
    """
    Walk-forward (rolling-origin) backtest: trains on an expanding window and
    evaluates on the following `test_size`-hour block, repeated across `n_folds`
    time-separated folds, instead of the single last-week holdout trainer.py uses
    for the production run. This is what tells you whether accuracy is stable
    across different weeks or the single holdout number was a lucky/unlucky week.

    Reuses each target's already-tuned XGBoost hyperparameters (from the saved
    production model) rather than re-running RandomizedSearchCV per fold — the
    hyperparameters were already tuned once; this evaluates how that model
    class + feature set generalizes across time, not re-tuning.
    """
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        log.error("Features dataset not found. Run the ingestion + feature engineering pipeline first.")
        return None

    df = pd.read_parquet(features_path)
    n_rows = len(df)
    first_fold_train_size = n_rows - n_folds * test_size
    min_first_fold_train_size = test_size * 2  # at least 2 weeks of history to train the first fold
    if first_fold_train_size < min_first_fold_train_size:
        log.error(
            f"Not enough data for {n_folds} folds of {test_size}h each: only {first_fold_train_size} rows "
            f"would be left to train the first fold (need >= {min_first_fold_train_size}). "
            f"Reduce n_folds/test_size or ingest more history."
        )
        return None

    drop_cols = [c for c in df.columns if c.startswith('target_')]
    summary = {}

    for target in GENERATION_COLUMNS:
        model_path = settings.PRODUCTION_MODEL_DIR / f"model_{target}.pkl"
        if not model_path.exists():
            log.warning(f"No production model for {target} — skipping (run trainer.py first).")
            continue

        production_model = joblib.load(model_path)
        params = production_model.get_params()
        model_cls = type(production_model)

        fold_results = []
        for fold, (train_end, test_end) in enumerate(_fold_bounds(n_rows, n_folds, test_size)):
            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            X_train = train_df.drop(columns=drop_cols)
            X_test = test_df.drop(columns=drop_cols)
            y_train = train_df[f'target_{target}']
            y_test = test_df[f'target_{target}']

            model = model_cls(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = calculate_metrics(y_test.values, preds)

            fold_results.append({
                **metrics,
                "fold": fold,
                "train_size": len(train_df),
                "test_start": str(test_df.index.min()),
                "test_end": str(test_df.index.max()),
            })
            log.info(
                f"[{target}] fold {fold} ({test_df.index.min().date()} to {test_df.index.max().date()}, "
                f"train_size={len(train_df)}): MAE={metrics['MAE']:.1f} RMSE={metrics['RMSE']:.1f} R2={metrics['R2']:.3f}"
            )

        maes = [r["MAE"] for r in fold_results]
        rmses = [r["RMSE"] for r in fold_results]
        r2s = [r["R2"] for r in fold_results]
        summary[target] = {
            "folds": fold_results,
            "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
            "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
            "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        }
        log.info(
            f"{target} across {n_folds} folds: MAE {summary[target]['MAE_mean']:.1f} +/- {summary[target]['MAE_std']:.1f} MW, "
            f"R2 {summary[target]['R2_mean']:.3f} +/- {summary[target]['R2_std']:.3f}"
        )
        _plot_fold_metrics(target, fold_results)

    output_path = settings.MODELS_DIR / "backtest_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Backtest summary saved to {output_path}")
    return summary


def _plot_fold_metrics(target: str, fold_results: list):
    folds = [r["fold"] for r in fold_results]
    maes = [r["MAE"] for r in fold_results]
    r2s = [r["R2"] for r in fold_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(folds, maes, color="#4C72B0")
    ax1.axhline(np.mean(maes), color="r", linestyle="--", label=f"mean={np.mean(maes):.1f}")
    ax1.set_title(f"{target}: MAE per fold")
    ax1.set_xlabel("Fold (earlier -> later in time)")
    ax1.set_ylabel("MAE (MW)")
    ax1.legend()

    ax2.bar(folds, r2s, color="#55A868")
    ax2.axhline(np.mean(r2s), color="r", linestyle="--", label=f"mean={np.mean(r2s):.3f}")
    ax2.set_title(f"{target}: R² per fold")
    ax2.set_xlabel("Fold (earlier -> later in time)")
    ax2.set_ylabel("R²")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(settings.MODELS_DIR / f"backtest_{target}.png")
    plt.close()


if __name__ == "__main__":
    run_backtest()
