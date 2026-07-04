import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, model_name: str, target_name: str, save_dir: Path):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.index, y_true.values, label="Actual", alpha=0.7)
    plt.plot(y_true.index, y_pred, label="Predicted", alpha=0.7, linestyle="--")
    plt.title(f"{model_name} Forecast vs Actual - {target_name}")
    plt.ylabel("Generation (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target_name}_predictions.png")
    plt.close()

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str, target_name: str, save_dir: Path):
    residuals = y_true.values - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f"{model_name} Residuals - {target_name}")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_{target_name}_residuals.png")
    plt.close()
