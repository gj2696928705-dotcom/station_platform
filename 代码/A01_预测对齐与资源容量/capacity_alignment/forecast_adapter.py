"""Read-only candidate inference in an independent numerical interpreter.

Deployment manifests are administrator-created local files, never HTTP inputs.
Each run has a fresh directory, bounded logs/output, fixed hashes, and no training.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from .forecast_capacity import BASE, PV, fail, number
from .common import digest, parse_time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, limit=2_000_000):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limit:
        fail("FORECAST_OUTPUT_INVALID", "预测文件缺失或超过大小限制")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        fail("FORECAST_OUTPUT_INVALID", "预测文件不是完整有效的 UTF-8 JSON")


def verify_deployment(path):
    path = Path(path).resolve()
    deployment = read_json(path)
    if deployment.get("schema_version") != "candidate-deployment-1" or deployment.get("simulation") is not True:
        fail("MODEL_MANIFEST_INVALID", "需要固定版本的模拟模型部署清单")
    if deployment["manifest_hash"] != digest({k: v for k, v in deployment.items() if k != "manifest_hash"}):
        fail("MODEL_HASH_MISMATCH", "部署清单内容已变化")
    for name, expected in deployment["files"].items():
        target = (path.parent/name).resolve()
        if not target.is_relative_to(path.parent) or not target.is_file() or sha(target) != expected:
            fail("MODEL_HASH_MISMATCH", f"模型包或源文件哈希不匹配: {name}")
    return deployment


def normalize(raw, deployment, as_of, input_hash, source_ref):
    """Preserve raw JSON; only an outer envelope carries projection semantics."""
    kind = deployment["kind"]
    clock = parse_time(as_of)
    if raw["asset_scope"] != deployment["station_id"]:
        fail("STATION_MISMATCH", "模型输出的资产范围与登记模拟站点不符")
    expected_type = "station_load" if kind == "load" else "pv_power"
    if raw["forecast_type"] != expected_type or raw["model_version"] != deployment["model_version"]:
        fail("MODEL_VERSION_MISMATCH", "输出模型版本或预测类型不匹配")
    envelope = {"record_id": raw["forecast_id"], "station_id": deployment["station_id"],
                "meter_boundary": deployment["meter_boundary"], "simulation": True,
                "issued_at": as_of, "available_at": as_of, "status": raw["status"],
                "target_semantics": BASE if kind == "load" else PV,
                "unit": "kW", "value_semantics": "interval_average_power", "values": [],
                "model_version": raw["model_version"], "bundle_hash": deployment["manifest_hash"],
                "method_hash": digest(deployment["files"]), "input_hash": input_hash, "source_ref": source_ref,
                "raw_record": raw, "publication_status": "candidate", "retrospective_candidate": True,
                "real_dispatch_eligible": False,
                "limitations": ["候选模型仅作离线计算；as_of 是反事实回放时钟，并非原始记录的生成/真实发布时间"]}
    if raw["status"] != "available":
        envelope.update(status="unavailable", reason=raw.get("reason", "candidate_not_available"), expires_at=as_of)
        return envelope
    if kind == "load":
        if raw.get("target_semantics") != BASE or raw.get("as_of") is None or parse_time(raw["as_of"]) != clock or raw["interval_minutes"] != 15:
            fail("TARGET_SEMANTICS_MISMATCH", "负荷候选须提供同一 as_of、基础负荷口径及 15 分钟序列")
        count, minutes = 96, 15
    else:
        if raw["confidence"]["sampling_layout"] != "day13_07_19" or raw["interval_minutes"] != 60 or deployment["value_semantics"] != "hourly_mean_start_label":
            fail("TARGET_SEMANTICS_MISMATCH", "day13 须有已登记的小时均值起点标签口径")
        count, minutes = 13, 60
        envelope["sampling_layout"] = "day13_07_19"
    points = raw["forecast_values"]
    bounds = raw["prediction_interval"]["values"]
    if len(points) != count or len(bounds) != count:
        fail("FORECAST_OUTPUT_INVALID", "原生点数或区间点数不符")
    first = parse_time(raw["forecast_start_at"])
    for index, (point, bound) in enumerate(zip(points, bounds)):
        left = parse_time(point["timestamp"])
        if left != first+timedelta(minutes=minutes*index) or point["timestamp"] != bound["timestamp"] or point["unit"] != "kW" or bound["unit"] != "kW":
            fail("FORECAST_OUTPUT_INVALID", "原生时间网格或单位不一致")
        if not number(bound["lower"]) <= number(point["value"]) <= number(bound["upper"]):
            fail("FORECAST_OUTPUT_INVALID", "原生区间未包络点值")
        envelope["values"].append({"start": left.isoformat(), "end": (left+timedelta(minutes=minutes)).isoformat(), "value": point["value"]})
    expected_end = first+timedelta(minutes=minutes*(count if kind == "load" else count-1))
    if parse_time(raw["forecast_end_at"]) != expected_end:
        fail("FORECAST_OUTPUT_INVALID", "原生终点标签不一致")
    envelope["expires_at"] = envelope["values"][-1]["end"]
    if kind == "pv":
        local = first.astimezone(parse_time("2000-01-01T00:00:00+08:00").tzinfo)
        if (local.hour, local.minute, local.second, local.microsecond) != (7, 0, 0, 0):
            fail("FORECAST_OUTPUT_INVALID", "day13 必须从上海时区 07:00 开始")
        issue = raw["confidence"].get("weather_forecast_issued_at")
        envelope["weather_issued_at"] = issue
        if issue is None or parse_time(issue) > min(clock, first):
            envelope.update(status="unavailable", reason="no_complete_weather_batch_visible_by_as_of_and_D07")
    return envelope


def run_candidate(manifest_path, history, as_of, output_root, weather=None, timeout=120, max_bytes=2_000_000):
    deployment = verify_deployment(manifest_path)
    clock = parse_time(as_of)
    run_dir = Path(output_root).resolve()/uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    history = Path(history).resolve()
    with history.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "station_id" in (reader.fieldnames or []):
            if any(row["station_id"] != deployment["station_id"] for row in reader):
                fail("STATION_MISMATCH", "历史数据站点与部署登记不匹配")
    job = {"kind": deployment["kind"], "station_id": deployment["station_id"], "as_of": as_of,
           "history": str(history), "bundle": str(Path(manifest_path).resolve().parent/"bundle"),
           "output": str(run_dir/"forecast.json"), "forecast_date": clock.astimezone(parse_time("2000-01-01T00:00:00+08:00").tzinfo).date().isoformat()}
    hashes = {"history": sha(history)}
    if weather is not None:
        weather = Path(weather).resolve()
        # Select a single latest eligible batch before calling code that otherwise
        # selects by D07 alone. Never mix batches or fall back to an earlier one.
        cutoff = min(clock, parse_time(job["forecast_date"]+"T07:00:00+08:00"))
        with weather.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        eligible = [r for r in rows if parse_time(r["issued_at"]) <= cutoff]
        if not eligible:
            fail("LATE_WEATHER", "无 as_of 与 D07 之前的天气批次")
        issue = max(parse_time(r["issued_at"]) for r in eligible)
        selected = [r for r in eligible if parse_time(r["issued_at"]) == issue]
        expected = {parse_time(job["forecast_date"]+f"T{hour:02}:00:00+08:00") for hour in range(7, 20)}
        if len(selected) != 13 or {parse_time(r["timestamp"]) for r in selected} != expected:
            fail("WEATHER_BATCH_INCOMPLETE", "最新合格天气批次不是完整 13 点")
        prepared = run_dir/"weather.csv"
        with prepared.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        hashes["weather"] = sha(weather)
        hashes["selected_weather"] = sha(prepared)
        job["weather"] = str(prepared)
    job_path = run_dir/"job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="2")
    env["PYTHONPATH"] = str(Path(manifest_path).resolve().parent/"source")
    command = [deployment["python"], "-X", "utf8", str(Path(__file__).with_name("forecast_worker.py")), str(job_path)]
    try:
        process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        # Drain bounded chunks so a noisy model cannot fill memory or deadlock pipes.
        import threading
        logs, exceeded = {}, threading.Event()
        def drain(stream, name):
            chunks, total = [], 0
            while True:
                chunk = stream.read(8192)
                if not chunk: break
                total += len(chunk)
                if total > max_bytes:
                    exceeded.set()
                    process.kill()
                    break
                chunks.append(chunk)
            logs[name] = b"".join(chunks)
        threads = [threading.Thread(target=drain, args=(pipe, name)) for pipe, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))]
        for thread in threads: thread.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            fail("FORECAST_TIMEOUT", "独立数值进程超时")
        finally:
            for thread in threads: thread.join()
            for name, value in logs.items(): (run_dir/(name+".log")).write_bytes(value)
        if exceeded.is_set(): fail("FORECAST_OUTPUT_INVALID", "独立进程日志超过大小限制")
        if process.returncode != 0: fail("FORECAST_PROCESS_FAILED", f"独立数值进程失败: {process.returncode}")
        raw = read_json(run_dir/"forecast.json", max_bytes)
        result = normalize(raw, deployment, as_of, digest(hashes), str(run_dir/"forecast.json"))
        result["process"] = {"returncode": process.returncode, "run_dir": str(run_dir), "input_hashes": hashes}
        (run_dir/"normalized.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        # Recheck immutable deployment after inference (including any accidental write).
        verify_deployment(manifest_path)
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        fail("FORECAST_OUTPUT_INVALID", str(error))
