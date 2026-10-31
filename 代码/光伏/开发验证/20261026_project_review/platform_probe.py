"""Exercise actual project resource functions using isolated synthetic state."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import importlib.util
import json
import sys

out = Path(__file__).resolve().parent
workspace = out.parents[3]
sys.path.insert(0, str(workspace))
from station_platform.defaults import DEFAULT_DEMAND
from station_platform.domain import capability, request_from_site, resources, schedule, evaluation
from station_platform.legacy.demand import MockStationAdapter
from station_platform.util import now, stamp

current = now()
state = {"id": "synthetic-project-fit-review", "station_id": "mock-station-001", "revision": 1,
    "simulation": True, "quality": "B", "measured_at": stamp(current),
    "expires_at": stamp(current+timedelta(minutes=30)), "site": MockStationAdapter().site}
request = request_from_site(deepcopy(DEFAULT_DEMAND), state)
resource = resources(request)
service = schedule(request)
result = evaluation(request, resource, service)
candidate = json.loads((workspace/"代码"/"光伏"/"运行结果"/"20261026_day13_01"/"forecast.json").read_text(encoding="utf-8"))
injected = deepcopy(request)
injected["station_state_snapshot"]["forecasts"] = [candidate]
injected_resource = resources(injected)
assert resource["forecast_status"] == injected_resource["forecast_status"] == "not_used"
assert resource["charging_headroom_kw"] == injected_resource["charging_headroom_kw"]
cap = capability(state)
assert cap["interval_minutes"] == 15 and len(cap["charging_headroom_kw"]) == 4
assert request["poc_context"]["interval_minutes"] == 15
probe = {"status": "passed", "simulation_only": True, "platform_python": sys.executable,
    "platform_dependencies": {name: importlib.util.find_spec(name) is not None
        for name in ("fastapi", "numpy", "pandas", "sklearn", "tensorflow")},
    "current_platform_interval_minutes": 15, "current_platform_slots": 4,
    "current_platform_duration_minutes": cap["duration_minutes"],
    "current_platform_forecast_status": resource["forecast_status"],
    "injected_record_does_not_change_resources": True,
    "resource_headroom_kw": resource["charging_headroom_kw"],
    "selected_candidate": result["selected_candidate"]["candidate_id"],
    "new_forecast_interval_minutes": candidate["interval_minutes"],
    "new_forecast_points": len(candidate["forecast_values"]),
    "new_forecast_scope": candidate["asset_scope"], "platform_scope": state["station_id"],
    "candidate_published_at": candidate["model_published_at"],
    "runtime_weather_policy": candidate["confidence"]["runtime_weather_input_policy"],
    "real_platform_configuration_or_execution_changed": False}
(out/"platform_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(probe, ensure_ascii=False, indent=2))
