"""Shared-lead nonlinear residual model on the same causal physical inputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .metrics import regression_metrics


@dataclass(frozen=True)
class PhysicsTreeConfig:
    iterations: int = 200
    learning_rate: float = .05
    max_leaf_nodes: int = 31
    l2_regularization: float = 10.
    log_every: int = 50
    seed: int = 42


class PhysicsTreeForecaster:
    def __init__(self, config=PhysicsTreeConfig()):
        if config.iterations < 1 or config.log_every < 1 or config.learning_rate <= 0 or config.max_leaf_nodes < 2 or config.l2_regularization < 0:
            raise ValueError("invalid physics tree configuration")
        self.config = config

    @staticmethod
    def features(history, future, history_names, future_names):
        history, future = np.asarray(history, dtype=np.float32), np.asarray(future, dtype=np.float32)
        if history.ndim != 3 or future.ndim != 3 or history.shape[1] != 288 or future.shape[1] != 96 or len(history) != len(future):
            raise ValueError("tree input must be 288 historical and 96 future points")
        if history.shape[-1] != len(history_names) or future.shape[-1] != len(future_names):
            raise ValueError("tree input names must match feature axes")
        names = list(history_names)
        selected = [names.index(k) for k in (names[0], "SWGDN", "temperature_c", "cloud_cover",
                     "power_clear_ratio", "ghi_clear_ratio", "power_ratio_delta", "cloud_delta")]
        h = history[..., selected]
        context = np.concatenate([h[:, -1], h[:, -4:].mean(axis=1), h[:, -24:].mean(axis=1), h[:, -96:].mean(axis=1),
                                  h[:, -4:].std(axis=1)], axis=1)
        repeated = np.broadcast_to(context[:, None], (len(history), 96, context.shape[-1]))
        ratio = history[..., names.index("power_clear_ratio")]
        slots = ratio.reshape(len(history), 3, 96).mean(axis=1)
        leads = np.broadcast_to(np.arange(96, dtype=np.float32)[None, :, None]/96, (len(history), 96, 1))
        features = np.concatenate([future, leads, repeated, ratio[:, -96:, None], slots[..., None]], axis=-1)
        if not np.isfinite(features).all():
            raise ValueError("non-finite tree features")
        baseline = future[..., list(future_names).index("clear_pv_kw")]
        day = future[..., list(future_names).index("is_daylight")] > .5
        return features, baseline, day

    def fit(self, train, validation, *, capacity_kw, output_dir=None, progress=None):
        self.history_names, self.future_names = tuple(train.history_feature_names), tuple(train.feature_names)
        self.capacity_kw = float(capacity_kw)
        self.window_count = train.size
        x, base, day = self.features(train.gru_history, train.future_features, self.history_names, self.future_names)
        vx, vb, vd = self.features(validation.gru_history, validation.future_features, self.history_names, self.future_names)
        tx, ty = x[day], np.asarray(train.target-base, dtype=np.float64)[day]
        validation_x = vx[vd]
        if not len(ty):
            raise ValueError("tree fitting requires daytime labels")
        self.daylight_label_count = len(ty)
        self.model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=self.config.learning_rate,
            max_iter=min(self.config.log_every, self.config.iterations), max_leaf_nodes=self.config.max_leaf_nodes,
            l2_regularization=self.config.l2_regularization, early_stopping=False, warm_start=True,
            random_state=self.config.seed)
        rows = []
        directory = Path(output_dir) if output_dir else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        stages = [*range(self.config.log_every, self.config.iterations, self.config.log_every), self.config.iterations]
        for stage in stages:
            self.model.set_params(max_iter=stage)
            self.model.fit(tx, ty)
            pred = np.clip(self.model.predict(tx)+base[day], 0, self.capacity_kw)
            val_pred = np.clip(self.model.predict(validation_x)+vb[vd], 0, self.capacity_kw)
            tr, va = regression_metrics(train.target[day], pred), regression_metrics(validation.target[vd], val_pred)
            rows.append({"iteration": stage, "training_daylight": tr, "validation_daylight": va})
            if progress:
                progress(f"Tree {stage}/{self.config.iterations} | 全部{train.size}窗口/{len(ty)}白天标签 | "
                    f"训练RMSE={tr['rmse']:.3f} 验证RMSE={va['rmse']:.3f} kW 验证R²={va['r2']:.4f}")
            if directory:
                (directory/"training_history.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        self.training_summary = {"status": "completed", "training_windows_used": train.size,
            "daylight_labels_used": len(ty), "all_available_windows_used": True, "iterations": self.config.iterations,
            "internal_random_validation": False, "loss_target": "kW residual from deterministic clear-sky power",
            "reported_metrics": "reconstructed physical kW, daytime", "night_points": "explicit zero power, no learned night component"}
        if directory:
            (directory/"training_summary.json").write_text(json.dumps(self.training_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def predict(self, history, future):
        if not hasattr(self, "model"):
            raise ValueError("physics tree has not been fitted")
        x, base, day = self.features(history, future, self.history_names, self.future_names)
        result = np.zeros(base.shape, dtype=float)
        if day.any():
            result[day] = np.clip(self.model.predict(x[day])+base[day], 0, self.capacity_kw)
        return result
