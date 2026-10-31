"""Pool errors using label moments; never average RMSE or R2 across paths."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
pv = out.parents[1]
sys.path.insert(0, str(pv))
from pv_forecast.weather import solar_elevation


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def moments(frame, station):
    """Exact float32 target contract, all overlapping forecast origins."""
    index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp))
    truth = np.lib.stride_tricks.sliding_window_view(frame.pv_power_kw.to_numpy(dtype=np.float32), 96)[288:]
    day = np.lib.stride_tricks.sliding_window_view(
        np.asarray(solar_elevation(index, station["latitude_deg"], station["longitude_deg"]) > 0), 96)[288:]
    month = np.lib.stride_tricks.sliding_window_view(index.month.to_numpy(), 96)[288:]
    states = np.lib.stride_tricks.sliding_window_view(frame.weather_type.to_numpy(), 96)[288:]
    groups = {}

    def add(key, mask):
        value = truth[mask].astype(np.float64)
        groups[key] = {"count": value.size, "sum": float(value.sum()), "square": float(np.sum(value**2))}

    add("all_day", np.ones(truth.shape, dtype=bool))
    add("daylight", day)
    add("night", ~day)
    for m in range(1, 13):
        add(f"month_{m:02d}", day & (month == m))
    for state in ("clear", "partly_cloudy", "overcast", "rain"):
        add("weather_"+state, day & (states == state))
    for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24)):
        mask = day.copy()
        mask[:, :lo*4] = False
        mask[:, hi*4:] = False
        add(f"lead_{lo:02d}_{hi:02d}h", mask)
    assert truth.shape == (34657, 96)
    return groups


def pool(items, truth_moments):
    """Reconstruct SSE/SAE from full precision reports and exact label sums."""
    assert len(items) == len(truth_moments)
    assert all(a["count"] == b["count"] for a, b in zip(items, truth_moments))
    n = sum(m["count"] for m in truth_moments)
    total = sum(m["sum"] for m in truth_moments)
    square = sum(m["square"] for m in truth_moments)
    sse = sum(item["count"]*item["rmse"]**2 for item in items if item["count"])
    sae = sum(item["count"]*item["mae"] for item in items if item["count"])
    variance = square-total**2/n if n else 0
    rmse = float(np.sqrt(sse/n)) if n else None
    result = {"count": n, "rmse": rmse, "mae": sae/n if n else None,
        "r2": 1-sse/variance if variance > 1e-12 else None,
        "nrmse": rmse/(total/n) if total > 1e-12 else None}
    for optional in ("mean_error_kw", "coverage", "mean_interval_width_kw"):
        if all(item.get(optional) is not None for item in items if item["count"]):
            result[optional] = sum(item["count"]*item[optional] for item in items if item["count"])/n if n else None
    # Verify per-path R2 from target sums as an independent aggregation check.
    for item, m in zip(items, truth_moments):
        sst = m["square"]-m["sum"]**2/m["count"] if m["count"] else 0
        if sst > 1e-12:
            np.testing.assert_allclose(item["r2"], 1-item["rmse"]**2*m["count"]/sst, atol=3e-8, rtol=0)
    return result


folder = out/"frozen_three_paths"
completion, protocol = read(folder/"completion.json"), read(folder/"protocol.json")
assert completion["status"] == "completed" and completion["evaluated_windows"] == 103971
reports, label_moments = [], []
for seed in protocol["seeds"]:
    seed_dir = folder/str(seed)
    config = read(seed_dir/"simulation_config.json")
    data_path = seed_dir/config["output"]["data_filename"]
    report = read(seed_dir/"backtest.json")
    assert report["evaluated_windows"] == 34657 and report["skipped_origins"] == 0
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == report["source_sha256"]
    reports.append(report)
    label_moments.append(moments(pd.read_csv(data_path), protocol["base_generator_config"]["station"]))
models = {}
for name in reports[0]["models"]:
    values = [r["models"][name] for r in reports]
    overall = {group: pool([v["overall"][group] for v in values], [m[group] for m in label_moments])
        for group in ("all_day", "daylight", "night")}
    weather = {state: pool([v["daylight_weather"][state] for v in values], [m["weather_"+state] for m in label_moments])
        for state in ("clear", "partly_cloudy", "overcast", "rain")}
    months = {f"{month:02d}": pool([v["months"][f"{month:02d}"]["daylight"] for v in values],
        [m[f"month_{month:02d}"] for m in label_moments]) for month in range(1, 13)}
    leads = {lead: pool([v["error_diagnostics"]["daylight_by_lead"][lead] for v in values],
        [m["lead_"+lead] for m in label_moments]) for lead in values[0]["error_diagnostics"]["daylight_by_lead"]}
    models[name] = {"overall": overall, "daylight_weather": weather, "months": months,
        "daylight_by_lead": leads, "worst_month_daylight_rmse": max(v["rmse"] for v in months.values()),
        "per_path": {str(seed): v["overall"]["daylight"] for seed, v in zip(protocol["seeds"], values)}}
oracle = pool([r["ideal_future_weather_diagnostic"]["error_diagnostics"]["daylight"] for r in reports],
              [m["daylight"] for m in label_moments])
control = models["20261009_point_correction_01"]
proposal = protocol["proposed_primary_bundle"]
proposed = models[proposal]
conditions = {"each_path_daylight_RMSE_and_MAE_lower": all(
    proposed["per_path"][str(seed)]["rmse"] < control["per_path"][str(seed)]["rmse"] and
    proposed["per_path"][str(seed)]["mae"] < control["per_path"][str(seed)]["mae"] for seed in protocol["seeds"]),
    "pooled_clear_and_partly_cloudy_RMSE_not_worse": all(proposed["daylight_weather"][state]["rmse"] <=
        control["daylight_weather"][state]["rmse"] for state in ("clear", "partly_cloudy")),
    "pooled_0_1h_and_1_3h_RMSE_MAE_not_worse": all(proposed["daylight_by_lead"][lead][metric] <=
        control["daylight_by_lead"][lead][metric] for lead in ("00_01h", "01_03h") for metric in ("rmse", "mae")),
    "worst_target_month_RMSE_not_worse": proposed["worst_month_daylight_rmse"] <= control["worst_month_daylight_rmse"]}
accepted = proposal != "20261009_point_correction_01" and all(conditions.values())
result = {"status": "completed", "models": models, "oracle_diagnostic_daylight": oracle,
    "evaluated_windows": 103971, "scored_labels_per_model": completion["scored_labels_per_model"],
    "aggregation": "SSE/SAE pooled by count; R2 rebuilt from exact repeated float32 target sums and squares",
    "rolling_selected_candidate": protocol["rolling_selected_candidate"], "proposed_primary_bundle": proposal,
    "adoption_conditions": conditions, "stable_point_accuracy_gain": accepted,
    "recommended_bundle": proposal if accepted else "20261009_point_correction_01",
    "oracle_eligible_for_adoption": False,
    "limitations": protocol["limitations"],
    "no_tuning_after_viewing_paths": True}
write(folder/"aggregate.json", result)
write(folder/"target_moments.json", {str(seed): m for seed, m in zip(protocol["seeds"], label_moments)})
print(json.dumps({"models": {name: v["overall"]["daylight"] for name, v in models.items()},
    "conditions": conditions, "recommendation": result["recommended_bundle"], "oracle": oracle}, indent=2))
