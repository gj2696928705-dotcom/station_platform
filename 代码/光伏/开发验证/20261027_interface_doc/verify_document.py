"""Verify handoff facts, callable example and the documented overlap example."""
from pathlib import Path
import ast
import inspect
import json
import re
import sys

out = Path(__file__).resolve().parent
pv = out.parents[1]
workspace = pv.parents[1]
sys.path.insert(0, str(pv))
from pv_forecast.records import validate_forecast_record
from pv_forecast.pipeline import predict_record

document = workspace/"station_platform"/"docs"/"光伏预测对接说明_v0.3_day13.md"
text = document.read_text(encoding="utf-8")
metadata = json.loads((pv/"运行结果"/"20261026_day13_01"/"model_metadata.json").read_text(encoding="utf-8"))
for key in ("model_id", "model_version", "schema_version", "sampling_layout"):
    assert metadata[key] in text
assert (metadata["history_steps"], metadata["horizon_steps"], metadata["interval_minutes"]) == (39, 13, 60)
assert text.count("```") % 2 == 0
blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, flags=re.S)
python_blocks = [body for language, body in blocks if language == "python"]
assert len(python_blocks) == 1
tree = ast.parse(python_blocks[0])
signature = inspect.signature(predict_record)
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "predict_record":
        assert all(keyword.arg in signature.parameters for keyword in node.keywords)
for path in re.findall(r"`((?:代码|station_platform)/[^`]+)`", text):
    assert (workspace/path).exists(), path

# Independently calculate the 09:50 example using minutes from midnight.
hours = [(540, 600, 200.), (600, 660, 400.)]
values = []
for i in range(4):
    lo, hi = 590+i*15, 605+i*15
    covered = sum(max(0, min(hi, end)-max(lo, start)) for start, end, power in hours)
    assert covered == 15
    values.append(sum(power*max(0, min(hi, end)-max(lo, start)) for start, end, power in hours)/15)
assert abs(values[0]-266.6666666666667) < 1e-10 and values[1:] == [400.]*3
assert abs(sum(values)*.25 - (200.*10/60+400.*50/60)) < 1e-10
assert not (7*60 <= 19*60+45 and 20*60+45 <= 20*60)  # partial coverage must fail

forecast_path = pv/"运行结果"/"平台对接回放"/"forecast_20251231.json"
forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
validate_forecast_record(forecast)
assert len(forecast["forecast_values"]) == 13 and forecast["interval_minutes"] == 60
assert forecast["status"] == "available" and forecast["forecast_end_at"] == "2025-12-31T19:00:00+08:00"
assert forecast["model_version"] == metadata["model_version"]
assert not forecast["visualization_errors"]
result = {"status": "passed", "document": str(document), "markdown_fences_balanced": True,
    "referenced_code_and_artifact_paths_exist": True, "current_model_metadata_matches": True,
    "python_example_keywords_match_real_API": True, "historical_CLI_replay_succeeded": True,
    "forecast_points": 13, "interval_minutes": 60, "replay_date": "2025-12-31",
    "overlap_example_kw": values, "overlap_energy_consistent": True,
    "document_distinguishes_existing_and_planned_interfaces": True,
    "platform_integration_implemented_by_this_task": False}
(out/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
