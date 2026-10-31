"""XGBoost training with visible, persisted physical-unit metrics."""
import csv
import json
import time
from pathlib import Path

import numpy as np


def metrics(truth, prediction, lower=None, upper=None):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics require aligned finite arrays")
    residual = truth - prediction
    mse = float(np.mean(residual ** 2))
    variance = float(np.mean((truth - truth.mean()) ** 2))
    result = {"rmse_kw": mse ** .5, "mae_kw": float(np.mean(abs(residual))),
              "r2": 1 - mse / variance if variance > 1e-12 else None,
              "nrmse": mse ** .5 / float(abs(truth).mean()) if abs(truth).mean() > 1e-12 else None,
              "point_count": int(truth.size)}
    if lower is not None:
        result["coverage"] = float(np.mean((truth >= lower) & (truth <= upper)))
    return result


def calibrate(truth, pred, nominal=.8):
    residual = abs(truth - pred)
    count = len(residual)
    if count < 20:
        raise ValueError("at least 20 calibration origins are required")
    rank = min(int(np.ceil((count + 1) * nominal)), count)
    quantiles = np.sort(residual, axis=0)[rank - 1]
    return {"method": "held_out_absolute_residual_per_lead", "nominal_coverage": nominal,
            "quantiles_kw": quantiles.tolist(), "origin_count": count,
            "order_statistic_rank": rank,
            "limitation": "correlated rolling windows and temporal drift; no guaranteed coverage"}


def interval(pred, calibration):
    q = np.asarray(calibration["quantiles_kw"], float)
    if pred.shape[-1] != len(q) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("invalid interval calibration")
    return np.maximum(0., pred - q), pred + q


def xgb_module():
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError("XGBoost is required; install requirements.txt. Training is never silently skipped.") from exc
    return xgboost


def train_xgboost(x_train, y_train, x_val, y_val, names, config, output, emit):
    xgb = xgb_module()
    dtrain = xgb.DMatrix(x_train, label=y_train.reshape(-1), feature_names=names, nthread=config.threads)
    dval = xgb.DMatrix(x_val, label=y_val.reshape(-1), feature_names=names, nthread=config.threads)
    history, started = [], time.perf_counter()
    fieldnames = ["round", "train_mse", "validation_mse", "train_rmse_kw", "train_mae_kw", "train_r2",
                  "validation_rmse_kw", "validation_mae_kw", "validation_r2", "elapsed_seconds"]

    class Recorder(xgb.callback.EarlyStopping):
        def after_iteration(self, model, epoch, evals_log):
            row = {"round": epoch + 1, "elapsed_seconds": round(time.perf_counter() - started, 3)}
            for dataset, prefix in (("train", "train"), ("validation", "validation")):
                for metric, suffix in (("rmse", "rmse_kw"), ("mae", "mae_kw"), ("r2", "r2")):
                    value = float(evals_log[dataset][metric][-1])
                    row[f"{prefix}_{suffix}"] = value if np.isfinite(value) else None
                row[f"{prefix}_mse"] = row[f"{prefix}_rmse_kw"] ** 2
            history.append(row)
            with (output / "training_history.csv").open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                if epoch == 0:
                    writer.writeheader()
                writer.writerow(row)
            def fmt(key):
                return "N/A" if row[key] is None else f"{row[key]:.4f}"
            emit(f"Round {epoch + 1:03d}/{config.rounds} | train RMSE={fmt('train_rmse_kw')} kW "
                 f"MAE={fmt('train_mae_kw')} R²={fmt('train_r2')} | validation "
                 f"RMSE={fmt('validation_rmse_kw')} kW MAE={fmt('validation_mae_kw')} "
                 f"R²={fmt('validation_r2')} | {row['elapsed_seconds']:.1f}s")
            # One callback guarantees logging precedes the stopping decision.
            return super().after_iteration(model, epoch, evals_log)

    def r2(pred, data):
        truth = data.get_label()
        variance = float(np.sum((truth.astype(float) - float(truth.mean())) ** 2))
        return "r2", 1 - float(np.sum((truth.astype(float) - pred) ** 2)) / variance if variance > 1e-12 else float("nan")

    model = xgb.train(
        {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": config.max_depth,
         "eta": config.learning_rate, "subsample": 1., "colsample_bytree": 1.,
         "seed": config.seed, "nthread": config.threads, "eval_metric": ["mae", "rmse"]},
        dtrain, num_boost_round=config.rounds, evals=[(dtrain, "train"), (dval, "validation")],
        custom_metric=r2, verbose_eval=False,
        callbacks=[Recorder(rounds=config.patience, metric_name="rmse",
                            data_name="validation", maximize=False, save_best=True)],
    )
    model.save_model(output / "xgboost.json")
    (output / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    return model, {"status": "completed", "actual_rounds": len(history),
                   "best_round": model.best_iteration + 1, "retained_trees": model.num_boosted_rounds(),
                   "early_stopped": len(history) < config.rounds, "xgboost_version": xgb.__version__,
                   "elapsed_seconds": time.perf_counter() - started, "model_file": "xgboost.json"}


def predict_xgboost(model, x, names, threads=4):
    xgb = xgb_module()
    return np.maximum(0., model.predict(xgb.DMatrix(x, feature_names=names, nthread=threads))).reshape(-1, 96)
