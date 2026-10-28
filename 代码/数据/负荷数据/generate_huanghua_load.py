"""Generate auditable interval-average load data, independently of load models.

Uses only numpy/pandas/stdlib. Never imports XGBoost or load_forecast, never
trains a model, and opens the existing photovoltaic data only for reading.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import heapq
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCALE = 1_000_000
COMPONENTS = ["base_load_kw", "charging_load_kw", "total_load_kw"]
COLUMNS = ["timestamp", *COMPONENTS, "station_id", "is_simulated", "temperature_c", "is_weekend"]


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_bytes(frame, config):
    return frame.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode(config["output"]["encoding"])


def index_for_year(config):
    y = config["year"]
    return pd.date_range(f"{y}-01-01", f"{y+1}-01-01", freq="15min", inclusive="left", tz=config["timezone"])


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(cfg["schema_version"] == "huanghua-load-simulation-1", "Unsupported schema version")
    require(cfg["year"] == 2025 and cfg["interval_minutes"] == 15, "This delivery requires year 2025 and 15-minute intervals")
    require(cfg["timezone"] == "Asia/Shanghai" and cfg["is_simulated"] is True, "Invalid time zone/simulation flag")
    require(cfg["output"]["float_decimals"] == 6, "Fixed-point contract requires 6 decimals")
    require(cfg["output"]["encoding"] == "utf-8-sig", "CSV contract requires UTF-8 BOM")
    ch = cfg["charging"]
    require(np.isclose(sum(ch["phase_time_fractions"]), 1), "Phase time fractions must sum to one")
    require(len(ch["phase_time_fractions"]) == len(ch["phase_power_fractions"]), "Phase lengths differ")
    require(all(0 < f <= 1 for f in ch["phase_power_fractions"]), "Invalid phase power fraction")
    require(np.isclose(sum(s["probability"] for s in ch["session_types"]), 1), "Type probabilities must sum to one")
    for s in ch["session_types"]:
        require(np.isclose(sum(s["peak_power_probabilities"]), 1), "Invalid power probabilities")
        require(max(s["peak_power_choices_kw"]) <= cfg["reference_resources"]["standard_gun_power_kw"], "Peak exceeds scenario gun rating")
    return cfg


def load_temperature(config, source_path):
    require(digest(source_path) == config["temperature_source"]["sha256"], "PV source hash changed; review provenance before regenerating")
    source = pd.read_csv(source_path, encoding="utf-8-sig", dtype={"is_simulated": str})
    index = pd.DatetimeIndex(pd.to_datetime(source["timestamp"], utc=True)).tz_convert(config["timezone"])
    require(index.equals(index_for_year(config)), "PV time grid differs from required 2025 grid")
    require(source["station_id"].eq(config["station_id"]).all(), "PV station id mismatch")
    require(source["is_simulated"].str.lower().eq("true").all(), "PV source must be explicitly synthetic")
    temperature = source[config["temperature_source"]["column"]].to_numpy(dtype=float)
    require(np.isfinite(temperature).all(), "PV temperature contains missing/non-finite values")
    return temperature


def ar_noise(rng, size, coefficient, standard_deviation):
    values = np.empty(size)
    value = rng.normal(0, standard_deviation)
    innovation_std = standard_deviation * np.sqrt(1 - coefficient ** 2)
    for i in range(size):
        value = coefficient * value + rng.normal(0, innovation_std)
        values[i] = value
    return values


def base_profile(config, temperature, rng):
    index = index_for_year(config)
    b = config["base_load"]
    # Average the assumed intraday profile over 15 one-minute midpoints. The
    # resulting value is an interval-average model parameter, not a point sample.
    hours = np.asarray(index.hour + index.minute / 60)
    hours_in_interval = hours[:, None] + (np.arange(15) + 0.5) / 60
    reference = np.interp(hours_in_interval, b["daily_anchor_hours"], b["daily_anchor_kw"]).mean(axis=1)
    weekdays = np.asarray(b["weekday_factors_mon_to_sun"])[index.dayofweek]
    # Interpolate seasonal factors at month midpoints to avoid month-end steps.
    anchors = pd.DatetimeIndex([pd.Timestamp(year=y, month=m, day=15, tz=index.tz)
                               for y in range(2024, 2027) for m in range(1, 13)])
    season = np.interp(index.asi8.astype(float), anchors.asi8.astype(float),
                       np.tile(b["monthly_factors_jan_to_dec"], 3))
    daily = ar_noise(rng, 366, b["daily_factor_ar"], b["daily_factor_log_std"])
    day_factor = np.exp(np.interp(np.arange(len(index)) / 96, np.arange(366), daily)
                        - b["daily_factor_log_std"] ** 2 / 2)
    hvac_demand = (b["cooling_kw_per_c"] * np.maximum(temperature - b["cooling_threshold_c"], 0)
                   + b["heating_kw_per_c"] * np.maximum(b["heating_threshold_c"] - temperature, 0))
    hvac_mean = np.empty(len(index))
    response = np.exp(-15 / b["hvac_response_minutes"])
    previous = float(hvac_demand[0])
    for i, demand in enumerate(hvac_demand):
        current = response * previous + (1 - response) * demand
        hvac_mean[i] = (previous + current) / 2
        previous = current
    noise = ar_noise(rng, len(index), b["interval_noise_ar"], b["interval_noise_std_kw"])
    result = np.maximum(reference * weekdays * season * day_factor + hvac_mean + noise,
                        b["minimum_kw_simulated"])
    return np.rint(result * SCALE).astype(np.int64)


def phases(duration_minutes, config):
    fractions = config["charging"]["phase_time_fractions"]
    edges = np.r_[0, np.rint(np.cumsum(fractions) * duration_minutes).astype(int)]
    edges[-1] = duration_minutes
    require(np.all(np.diff(edges) > 0), "Charging phases must have positive duration")
    return edges


def charging_profile(config, rng):
    index = index_for_year(config)
    first, last = index[0], index[-1] + pd.Timedelta(minutes=15)
    c = config["charging"]
    epoch = first - pd.Timedelta(days=c["warmup_days"])
    days = pd.date_range(epoch, last, freq="D", inclusive="left")
    daily_anomaly = ar_noise(rng, len(days), c["daily_intensity_ar"], c["daily_intensity_log_std"])
    arrivals = []
    kinds = c["session_types"]
    type_probs = [s["probability"] for s in kinds]
    curve_mean = np.dot(c["phase_time_fractions"], c["phase_power_fractions"])
    for d, day in enumerate(days):
        intensity = (c["mean_sessions_per_weekday"] * c["weekday_factors_mon_to_sun"][day.dayofweek]
                     * c["monthly_factors_jan_to_dec"][day.month - 1]
                     * np.exp(daily_anomaly[d] - c["daily_intensity_log_std"] ** 2 / 2))
        for _ in range(rng.poisson(intensity)):
            spec = kinds[int(rng.choice(len(kinds), p=type_probs))]
            minute = int(np.clip(np.rint(rng.normal(spec["arrival_hour_mean"], spec["arrival_hour_std"]) * 60), 0, 1439))
            arrival = int((day - epoch).total_seconds() // 60) + minute
            peak = float(rng.choice(spec["peak_power_choices_kw"], p=spec["peak_power_probabilities"]))
            target = round(float(np.clip(rng.normal(spec["energy_mean_kwh"], spec["energy_std_kwh"]),
                                         spec["energy_min_kwh"], spec["energy_max_kwh"])), 6)
            duration = int(np.ceil(target / peak / curve_mean * 60))
            # Increase by whole minutes until the discrete-phase delivery
            # covers requested energy; retain both requested and served energy.
            while np.dot(np.diff(phases(duration, config)), c["phase_power_fractions"]) * peak / 60 < target:
                duration += 1
            arrivals.append((arrival, len(arrivals), spec["name"], peak, target, duration))
    arrivals.sort(key=lambda item: (item[0], item[1]))
    available_ports = [(0, p) for p in range(1, config["reference_resources"]["charging_ports"] + 1)]
    heapq.heapify(available_ports)
    start_minute = int((first - epoch).total_seconds() // 60)
    end_minute = int((last - epoch).total_seconds() // 60)
    # Includes the Jan 1 carry-out tail for truthful full-session accounting.
    minute_watts = np.zeros(end_minute + 1440, dtype=np.int64)
    occupied = np.zeros_like(minute_watts, dtype=np.int16)
    sessions = []
    for number, (arrival, _, kind, peak, target, duration) in enumerate(arrivals, start=1):
        free_at, port = heapq.heappop(available_ports)
        start = max(arrival, free_at)
        end = start + duration
        require(end <= len(minute_watts), "Simulation queue exceeds allocated carry-out day")
        heapq.heappush(available_ports, (end, port))
        edges = phases(duration, config)
        served = in_year = 0.0
        for left, right, fraction in zip(edges[:-1], edges[1:], c["phase_power_fractions"]):
            a, z = start + int(left), start + int(right)
            watts = int(round(peak * fraction * 1000))
            minute_watts[a:z] += watts
            served += (z - a) * watts / 60000
            overlap = max(0, min(z, end_minute) - max(a, start_minute))
            in_year += overlap * watts / 60000
        occupied[start:end] += 1
        if end > start_minute and start < end_minute:
            sessions.append({
                "session_id": f"HH-SIM-{config['year']}-{number:06d}",
                "station_id": config["station_id"], "is_simulated": "true",
                "session_type": kind, "port_id": f"SIM-G-{port:03d}",
                "arrival_at": (epoch + pd.Timedelta(minutes=arrival)).isoformat(),
                "start_at": (epoch + pd.Timedelta(minutes=start)).isoformat(),
                "end_at": (epoch + pd.Timedelta(minutes=end)).isoformat(),
                "duration_minutes": duration, "queue_minutes": start - arrival,
                "peak_power_kw": peak, "requested_energy_kwh": target,
                "charging_energy_kwh": round(served, 6), "energy_in_2025_kwh": round(in_year, 6),
            })
    watts_in_year = minute_watts[start_minute:end_minute]
    # Sum energy first, then convert to interval-average power. Never sample
    # only the start/end power of a session to build the main table.
    charging_micro_kw = np.rint(watts_in_year.reshape(-1, 15).sum(axis=1) / 15 * 1000).astype(np.int64)
    evidence = {
        "session_count_intersecting_2025": len(sessions),
        "maximum_simultaneous_sessions": int(occupied[start_minute:end_minute].max()),
        "peak_one_minute_charging_kw": float(watts_in_year.max() / 1000),
        "one_minute_charging_energy_kwh": float(watts_in_year.sum() / 60000),
    }
    return charging_micro_kw, pd.DataFrame(sessions), evidence


def simulate(config, temperature):
    base_rng, charging_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(config["seed"]).spawn(2)]
    index = index_for_year(config)
    base = base_profile(config, temperature, base_rng)
    charging, sessions, evidence = charging_profile(config, charging_rng)
    # Fixed-point summation enforces exact decimal conservation in exported CSV.
    total = base + charging
    frame = pd.DataFrame({
        "timestamp": [t.isoformat() for t in index],
        "base_load_kw": base / SCALE, "charging_load_kw": charging / SCALE,
        "total_load_kw": total / SCALE, "station_id": config["station_id"],
        "is_simulated": "true", "temperature_c": temperature,
        "is_weekend": (index.dayofweek >= 5).astype(int),
    }, columns=COLUMNS)
    return frame, sessions, evidence


def monthly_summary(frame, config):
    index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).tz_convert(config["timezone"])
    rows = []
    for month in range(1, 13):
        part = frame.loc[index.month == month]
        row = {"month": f"2025-{month:02d}", "station_id": config["station_id"],
               "is_simulated": "true", "sample_count": len(part)}
        for component in COMPONENTS:
            prefix = component.removesuffix("_kw")
            micro = np.rint(part[component].to_numpy() * SCALE).astype(np.int64)
            row[prefix + "_energy_kwh"] = float(micro.sum() / SCALE / 4)
            row[prefix + "_mean_kw"] = float(part[component].mean())
            row[prefix + "_peak_kw"] = float(part[component].max())
        row["mean_temperature_c"] = float(part.temperature_c.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_frame(frame, config, temperature):
    require(list(frame.columns) == COLUMNS, "Main table columns differ from contract")
    expected = index_for_year(config)
    require(len(frame) == 35040, "Expected exactly 35040 rows")
    require(not frame.isna().any().any(), "Missing values")
    index = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).tz_convert(config["timezone"])
    require(index.equals(expected), "Incorrect coverage/order/uniqueness/frequency")
    require(frame.timestamp.str.endswith("+08:00").all(), "Timestamp must carry +08:00")
    require(np.isfinite(frame[COMPONENTS + ["temperature_c"]].to_numpy(dtype=float)).all(), "Non-finite numerical value")
    require(frame[COMPONENTS].ge(0).all().all(), "Negative component power")
    require(frame.station_id.eq(config["station_id"]).all(), "Station id mismatch")
    require(frame.is_simulated.astype(str).str.lower().eq("true").all(), "Missing simulation flags")
    require(np.array_equal(frame.is_weekend.to_numpy(), (expected.dayofweek >= 5).astype(int)), "Weekend labels mismatch")
    require(np.array_equal(frame.temperature_c.to_numpy(), temperature), "Temperature must match PV source exactly")
    micro = {k: np.rint(frame[k].to_numpy() * SCALE).astype(np.int64) for k in COMPONENTS}
    require(np.array_equal(micro["total_load_kw"], micro["base_load_kw"] + micro["charging_load_kw"]), "Load components do not conserve")
    # This is a scenario plausibility check, not a certified grid limit.
    require(frame.total_load_kw.max() < config["reference_resources"]["converter_power_kw"], "Generated gross demand exceeds scenario reference")
    for component in COMPONENTS:
        days = frame[component].to_numpy().reshape(365, 96)
        require(np.all(np.any(np.abs(np.diff(days, axis=0)) > 1e-6, axis=1)), f"Repeated adjacent days: {component}")
    statistics = {}
    for component in COMPONENTS:
        values = frame[component]
        statistics[component] = {
            "min_kw": float(values.min()), "max_kw": float(values.max()),
            "mean_kw": float(values.mean()), "std_kw": float(values.std(ddof=0)),
            "annual_energy_kwh": float(micro[component].sum() / SCALE / 4),
            "weekday_mean_kw": float(values.loc[frame.is_weekend == 0].mean()),
            "weekend_mean_kw": float(values.loc[frame.is_weekend == 1].mean()),
        }
    return {"rows": 35040, "days": 365, "start": frame.timestamp.iloc[0], "end": frame.timestamp.iloc[-1],
            "interval_minutes": 15, "missing_values": 0, "duplicate_timestamps": 0,
            "non_finite_values": 0, "negative_powers": 0, "fixed_point_conservation_max_error_kw": 0,
            "temperature_exactly_matches_pv": True, "component_statistics": statistics}


def audit_sessions(sessions, frame, config):
    """Reconstruct every charging interval from the exported session ledger."""
    index = index_for_year(config)
    year_start = index[0]
    length_minutes = len(index) * 15
    watts = np.zeros(length_minutes, dtype=np.int64)
    occupied = np.zeros(length_minutes, dtype=np.int16)
    require(sessions.session_id.is_unique, "Duplicate session id")
    require(sessions.is_simulated.astype(str).str.lower().eq("true").all(), "Session simulation flag missing")
    require(sessions.station_id.eq(config["station_id"]).all(), "Session station mismatch")
    port_last_end = {}
    for s in sessions.itertuples(index=False):
        arrival, start, end = (pd.Timestamp(getattr(s, k)) for k in ("arrival_at", "start_at", "end_at"))
        require(arrival <= start < end, "Invalid session time ordering")
        duration = int((end - start).total_seconds() / 60)
        require(duration == s.duration_minutes and int((start - arrival).total_seconds() / 60) == s.queue_minutes,
                "Session duration/queue mismatch")
        if s.port_id in port_last_end:
            require(start >= port_last_end[s.port_id], "Overlapping sessions on same port")
        port_last_end[s.port_id] = end
        require(0 < s.peak_power_kw <= config["reference_resources"]["standard_gun_power_kw"], "Session power outside scenario rating")
        relative_start = int((start - year_start).total_seconds() // 60)
        edges = phases(duration, config)
        full_energy = selected_energy = 0.0
        for a, z, multiplier in zip(edges[:-1], edges[1:], config["charging"]["phase_power_fractions"]):
            phase_watts = int(round(s.peak_power_kw * multiplier * 1000))
            full_energy += int(z - a) * phase_watts / 60000
            left, right = max(0, relative_start + int(a)), min(length_minutes, relative_start + int(z))
            if left < right:
                watts[left:right] += phase_watts
                selected_energy += (right - left) * phase_watts / 60000
        require(abs(full_energy - s.charging_energy_kwh) < 1e-5, "Full-session energy mismatch")
        require(abs(selected_energy - s.energy_in_2025_kwh) < 1e-5, "Year-clipped session energy mismatch")
        require(s.charging_energy_kwh + 1e-5 >= s.requested_energy_kwh, "Session underserves requested energy")
        occupied[max(0, relative_start):min(length_minutes, relative_start + duration)] += 1
    reconstructed = np.rint(watts.reshape(-1, 15).sum(axis=1) / 15 * 1000).astype(np.int64)
    actual = np.rint(frame.charging_load_kw.to_numpy() * SCALE).astype(np.int64)
    require(np.array_equal(reconstructed, actual), "Charging table differs from session reconstruction")
    require(occupied.max() <= config["reference_resources"]["charging_ports"], "Too many simultaneous sessions")
    require(watts.max() / 1000 < config["reference_resources"]["converter_power_kw"], "One-minute charging exceeds reference scale")
    tolerance = len(sessions) * 0.5e-6 + len(frame) * 0.25 * 0.5e-6 + 1e-6
    interval_energy = float(actual.sum() / SCALE / 4)
    require(abs(sessions.energy_in_2025_kwh.sum() - interval_energy) <= tolerance, "Session and interval energy totals disagree")
    first = pd.Timestamp("2025-01-01", tz=config["timezone"])
    final = pd.Timestamp("2026-01-01", tz=config["timezone"])
    starts = pd.to_datetime(sessions.start_at, utc=True)
    ends = pd.to_datetime(sessions.end_at, utc=True)
    return {"status": "passed", "session_count": len(sessions), "ports_used": len(port_last_end),
            "max_simultaneous_sessions": int(occupied.max()), "max_queue_minutes": int(sessions.queue_minutes.max()),
            "peak_one_minute_charging_kw": float(watts.max() / 1000),
            "carry_in_sessions_from_2024": int((starts < first).sum()),
            "carry_out_sessions_to_2026": int((ends > final).sum()),
            "all_35040_charging_intervals_reconstructed": True,
            "session_energy_within_2025_kwh": float(sessions.energy_in_2025_kwh.sum()),
            "energy_rounding_tolerance_kwh": tolerance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "simulation_config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--temperature-source", type=Path, help="Override file location; configured hash is still enforced")
    parser.add_argument("--validate-only", action="store_true", help="Keep CSVs unchanged; refresh report")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source = (args.temperature_source or args.config.parent / config["temperature_source"]["path_relative_to_config"]).resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {k: output / v for k, v in config["output"].items() if k in {"main", "monthly", "sessions", "report"}}
    report = {"dataset_id": config["dataset_id"], "is_simulated": True, "status": "failed"}
    try:
        source_hash_before = digest(source)
        temperature = load_temperature(config, source)
        frame, sessions, evidence = simulate(config, temperature)
        validate_frame(frame, config, temperature)
        monthly = monthly_summary(frame, config)
        expected = {"main": csv_bytes(frame, config), "monthly": csv_bytes(monthly, config), "sessions": csv_bytes(sessions, config)}
        repeated, repeated_sessions, _ = simulate(config, temperature)
        require(expected["main"] == csv_bytes(repeated, config), "Main data is not reproducible")
        require(expected["sessions"] == csv_bytes(repeated_sessions, config), "Session ledger is not reproducible")
        require(expected["monthly"] == csv_bytes(monthly_summary(repeated, config), config), "Monthly summary is not reproducible")
        if not args.validate_only:
            for key, content in expected.items():
                paths[key].write_bytes(content)
            config_copy = output / "simulation_config.json"
            if config_copy.resolve() != args.config.resolve():
                copy = dict(config)
                copy["temperature_source"] = dict(config["temperature_source"], path_relative_to_config=str(source))
                write_json(config_copy, copy)
        for key, content in expected.items():
            require(paths[key].read_bytes() == content, f"Existing {key} differs from deterministic regeneration")
        loaded = pd.read_csv(paths["main"], encoding="utf-8-sig", dtype={"is_simulated": str})
        facts = validate_frame(loaded, config, temperature)
        # Verify the actual exported decimal strings, without binary rounding tolerance.
        exact = pd.read_csv(paths["main"], encoding="utf-8-sig", dtype=str)
        for base, charging, total in zip(exact.base_load_kw, exact.charging_load_kw, exact.total_load_kw):
            require(Decimal(base) + Decimal(charging) == Decimal(total), "CSV decimal component conservation failed")
        saved_sessions = pd.read_csv(paths["sessions"], encoding="utf-8-sig", dtype={"is_simulated": str})
        session_checks = audit_sessions(saved_sessions, loaded, config)
        saved_monthly = pd.read_csv(paths["monthly"], encoding="utf-8-sig")
        require(len(saved_monthly) == 12 and saved_monthly.sample_count.sum() == 35040, "Monthly coverage invalid")
        require(csv_bytes(monthly_summary(loaded, config), config) == paths["monthly"].read_bytes(), "Monthly statistics inconsistent with exported main data")
        for component in COMPONENTS:
            energy_key = component.removesuffix("_kw") + "_energy_kwh"
            error = abs(saved_monthly[energy_key].sum() - facts["component_statistics"][component]["annual_energy_kwh"])
            require(error <= 1e-5, f"Monthly/annual energy mismatch: {component}")
        require(digest(source) == source_hash_before, "PV input changed during validation")
        report.update({
            "status": "passed", "contract": config["contract"], "checks": facts,
            "exact_csv_decimal_conservation": True, "monthly_energy_tolerance_kwh": 1e-5,
            "session_audit": session_checks, "generation_session_statistics": evidence,
            "reproducibility": {"seed": config["seed"], "all_three_csvs_repeat_byte_identical": True,
                                "exported_csvs_match_regeneration": True},
            "temperature_source": {"path": str(source), "sha256_before": source_hash_before,
                                   "sha256_after": digest(source), "unchanged": True,
                                   "row_by_row_temperature_match": True, "is_simulated": True},
            "data_hashes_sha256": {paths[k].name: digest(paths[k]) for k in expected},
            "generator_sha256": digest(__file__), "config_sha256": digest(args.config),
            "source_evidence": [{**s, "sha256": digest(s["path"]) if Path(s["path"]).is_file() else None}
                                for s in config["sources"]],
            "scope": {"written_directory": str(output), "load_model_imported": False, "model_training_performed": False,
                      "model_compatibility": "只验证交接字段契约；不导入正在开发的XGBoost模型，不宣称模型联调通过"},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "limitations": config["limitations"],
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_json(paths["report"], report)
        print(json.dumps(report, ensure_ascii=True))
        return 1
    write_json(paths["report"], report)
    print(json.dumps({"status": report["status"], "rows": 35040, "main_file": str(paths["main"]),
                      "statistics": facts["component_statistics"], "sessions": session_checks["session_count"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
