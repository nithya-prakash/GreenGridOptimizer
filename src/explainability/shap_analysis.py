import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from src.utils.logger import log
from src.utils.config import settings

# Prevent matplotlib from opening GUI windows
import matplotlib
matplotlib.use('Agg')

def run_shap_analysis():
    log.info("Starting SHAP analysis...")
    
    # Load best model
    model_path = settings.PRODUCTION_MODEL_DIR / "model.pkl"
    if not model_path.exists():
        log.error("Production model not found.")
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
    
    # Depending on whether the model is HistGradientBoosting or actual XGBoost
    # TreeExplainer is best, but if we fell back to HistGradientBoosting, we might need ExactExplainer
    # Let's try TreeExplainer first
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
    plt.savefig(settings.MODELS_DIR / "shap_summary.png")
    plt.close()
    
    # 2. Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(settings.MODELS_DIR / "shap_bar.png")
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
            plt.savefig(settings.MODELS_DIR / "shap_dependence.png")
            plt.close()
    except Exception as e:
        log.warning(f"Failed to generate dependence plot: {e}")
        
    # 4. Waterfall Plot for a single prediction (the very last hour)
    try:
        plt.figure(figsize=(10, 6))
        # Note: waterfall only works with single explanations
        shap.plots.waterfall(shap_values[-1], show=False)
        plt.tight_layout()
        plt.savefig(settings.MODELS_DIR / "shap_waterfall_single.png")
        plt.close()
    except Exception as e:
        log.warning(f"Failed to generate waterfall plot: {e}")
        
    log.info("SHAP analysis completed successfully.")

if __name__ == "__main__":
    run_shap_analysis()
