import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from src.utils.logger import log
from src.utils.config import settings
from src.features.feature_engineering import GENERATION_COLUMNS

# Prevent matplotlib from opening GUI windows
import matplotlib
matplotlib.use('Agg')

def run_shap_analysis(target: str):
    log.info(f"Starting SHAP analysis for {target}...")

    # Load the production model for this target
    model_path = settings.PRODUCTION_MODEL_DIR / f"model_{target}.pkl"
    if not model_path.exists():
        log.error(f"Production model for {target} not found.")
        return

    model = joblib.load(model_path)

    # Load dataset
    features_path = settings.PROCESSED_DATA_DIR / "features.parquet"
    if not features_path.exists():
        log.error("Features dataset not found.")
        return
        
    df = pd.read_parquet(features_path)
    
    # Extract test set
    test_size = 168
    if len(df) <= test_size:
        log.error("Not enough data for test split.")
        return
        
    test_df = df.iloc[-test_size:]
    
    # Prepare features
    drop_cols = [c for c in df.columns if c.startswith('target_')]
    X_test = test_df.drop(columns=drop_cols)
    
    # TreeExplainer is exact and fast for XGBoost; the model-agnostic Explainer
    # is only a fallback for non-tree models.
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    except Exception as e:
        log.warning(f"TreeExplainer failed, using Explainer (Kernel/Exact) fallback: {e}")
        # Use a small background dataset to speed up KernelExplainer if needed
        background = shap.sample(X_test, 50)
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X_test)
        
    log.info("SHAP values computed. Generating plots...")

    # 1. Summary Plot (Dot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(settings.MODELS_DIR / f"shap_summary_{target}.png")
    plt.close()

    # 2. Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(settings.MODELS_DIR / f"shap_bar_{target}.png")
    plt.close()
    
    # 3. Dependence Plot
    # Pick the most important feature to plot dependence
    if hasattr(shap_values, 'values'):
        vals = np.abs(shap_values.values).mean(0)
        feature_importance = pd.DataFrame(list(zip(X_test.columns, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
        top_feature = feature_importance.iloc[0]['col_name']
    else:
        top_feature = X_test.columns[0]
        
    try:
        # Dependence plot
        # For old shap versions, shap.dependence_plot might be needed. 
        # Using newer shap.plots.scatter
        if hasattr(shap.plots, 'scatter'):
            shap.plots.scatter(shap_values[:, top_feature], color=shap_values, show=False)
            plt.tight_layout()
            plt.savefig(settings.MODELS_DIR / f"shap_dependence_{target}.png")
            plt.close()
    except Exception as e:
        log.warning(f"Failed to generate dependence plot: {e}")

    # 4. Waterfall Plot for a single prediction (the very last hour)
    try:
        plt.figure(figsize=(10, 6))
        # Note: waterfall only works with single explanations
        shap.plots.waterfall(shap_values[-1], show=False)
        plt.tight_layout()
        plt.savefig(settings.MODELS_DIR / f"shap_waterfall_single_{target}.png")
        plt.close()
    except Exception as e:
        log.warning(f"Failed to generate waterfall plot: {e}")

    log.info(f"SHAP analysis for {target} completed successfully.")

if __name__ == "__main__":
    for target in GENERATION_COLUMNS:
        run_shap_analysis(target)
