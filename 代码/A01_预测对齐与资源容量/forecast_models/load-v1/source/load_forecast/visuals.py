"""Headless loss and physical load plots; no metric bar charts."""
from pathlib import Path
import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def training_plots(output, history, index, truth, xgb_pred, baseline, origins):
    plt = _plt()
    directory = Path(output) / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot([r["round"] for r in history], [r["train_mse"] for r in history], label="Training MSE")
    ax.plot([r["round"] for r in history], [r["validation_mse"] for r in history], label="Validation MSE")
    ax.set(xlabel="Boosting round", ylabel="MSE (kW²)", title="XGBoost loss / synthetic data is not field validation")
    ax.legend()
    fig.savefig(directory / "training_loss.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    previous_end, plotted = -1, 0
    for i, origin in enumerate(origins):
        if origin < previous_end:
            continue
        times = index[origin:origin + 96]
        for values, name, color in ((truth, "Actual", "#263a4e"), (xgb_pred, "XGBoost", "#007f85"),
                                    (baseline, "Best baseline", "#de9338")):
            ax.plot(times, values[i], color=color, label=name if plotted == 0 else None, linewidth=1.2)
        previous_end, plotted = origin + 96, plotted + 1
        if plotted == 7:
            break
    ax.set(ylabel="Interval-average load (kW)", title="Held-out test: first 7 non-overlapping horizons")
    ax.legend()
    fig.savefig(directory / "test_comparison.png", dpi=150)
    plt.close(fig)
    return ["visualizations/training_loss.png", "visualizations/test_comparison.png"]


def forecast_plot(output, data, record, truth=None, filename="forecast.png"):
    plt = _plt()
    import pandas as pd
    directory = Path(output) / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    future = pd.to_datetime([item["timestamp"] for item in record["forecast_values"]])
    prediction = [item["value"] for item in record["forecast_values"]]
    bounds = record["prediction_interval"]["values"]
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    if truth is None:
        past = data.frame.loc[data.frame.index < future[0]].iloc[-192:]
        ax.plot(past.index, past[data.spec.target_col], color="#263a4e", label="History")
    else:
        ax.plot(future, truth, color="#263a4e", label="Actual")
    ax.plot(future, prediction, color="#007f85", label=record["selected_model"])
    ax.fill_between(future, [v["lower"] for v in bounds], [v["upper"] for v in bounds],
                    color="#007f85", alpha=.18, label="Empirical interval")
    ax.set(ylabel="Interval-average load (kW)", title=f"{data.spec.target_col} / candidate model")
    ax.legend()
    fig.savefig(directory / filename, dpi=150)
    plt.close(fig)
    return f"visualizations/{filename}"
