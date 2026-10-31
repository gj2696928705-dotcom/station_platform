"""Evidence-based verification and compatibility conclusion for this project."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

out = Path(__file__).resolve().parent
pv = out.parents[1]
workspace = pv.parents[1]
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inference = read(out/"inference_verification.json")
probe = read(out/"platform_probe.json")
assert inference["status"] == probe["status"] == "passed"
assert all(sha(pv/name) == value for name, value in inference["source_sha256"].items())
log = (out/"pv_tests_after_fix.log").read_text(encoding="utf-8")
match = re.search(r"Ran (\d+) tests in ([\d.]+)s\s+OK", log)
assert match and int(match[1]) == 81
xml = ET.parse(out/"platform_full_tests.xml").getroot()
suites = list(xml.iter("testsuite"))
platform_tests = sum(int(s.attrib["tests"]) for s in suites)
assert platform_tests == 43 and all(int(s.attrib.get(key, 0)) == 0 for s in suites for key in ("failures", "errors", "skipped"))
assert probe["injected_record_does_not_change_resources"]
assert probe["current_platform_forecast_status"] == "not_used"
assert probe["current_platform_interval_minutes"] == 15 and probe["current_platform_slots"] == 4
assert probe["new_forecast_interval_minutes"] == 60 and probe["new_forecast_points"] == 13
bundle = pv/"运行结果"/"20261026_day13_01"
assert all(sha(bundle/name) == value for name, value in inference["bundle_hashes_unchanged"].items())
previous = read(pv/"开发验证"/"20261026_day13"/"delivery_verification.json")
assert inference["original_sha256"] == previous["original_data_sha256"]
for filename, digest in previous["model_hashes"].items():
    assert sha(bundle/filename) == digest
metrics = inference["fresh_metrics"]
gaps = [
    {"item": "consumer", "status": "not_integrated", "evidence": "actual resources(request) returns not_used and ignores injected forecast"},
    {"item": "time_grid", "status": "incompatible_for_direct_substitution", "model": "13 hourly points 07:00–19:00", "platform": "rolling next hour, four 15-minute intervals"},
    {"item": "load_alignment", "status": "adapter_required", "evidence": "load_forecast/data.py: INTERVAL=15min,HORIZON=96"},
    {"item": "runtime_dependencies", "status": "separate_inference_runtime_required", "evidence": probe["platform_dependencies"]},
    {"item": "publication_and_scope", "status": "candidate_only", "published_at": probe["candidate_published_at"], "forecast_scope": probe["new_forecast_scope"], "platform_scope": probe["platform_scope"]},
    {"item": "future_weather", "status": "offline_conditional_input", "policy": probe["runtime_weather_policy"]},
    {"item": "deployment", "status": "model_not_in_platform_repository", "evidence": "root .gitignore publishes station_platform only"},
]
report = {"status": "verification_completed", "independent_model_matches_requested_day13_contract": True,
    "direct_current_platform_integration_ready": False, "platform_consumer_uses_new_model": False,
    "pv_regression": {"tests": 81, "seconds": float(match[2]), "status": "passed"},
    "platform_regression": {"tests": 43, "status": "passed", "isolated_local_mock_only": True},
    "fresh_holdout": {"days": 55, "points": 715, "metrics": metrics, "weather_condition": "known simulated target-day weather"},
    "bugs_fixed": ["forecast end time consistency", "point/interval length consistency", "point/interval timestamp alignment", "nonfinite or reversed interval rejection"],
    "compatibility_gaps": gaps,
    "existing_model_data_and_calibration_preserved": True, "platform_business_logic_changed": False,
    "reference_sources_sha256": {name: sha(workspace/name) for name in (
        "station_platform/domain.py", "station_platform/defaults.py", "station_platform/legacy/demand.py",
        "station_platform/requirements-lock.txt", "代码/负荷/load_forecast/data.py", ".gitignore")},
    "pv_source_sha256": inference["source_sha256"],
    "local_socket_sandbox_issue": "sandbox socketpair blocked asyncio event loop; same complete platform suite passed with local loopback permitted",
    "recommendation": "keep independent candidate; implement explicit read-only consumer/time/scope/publication adapter before enabling platform use"}
(out/"project_compatibility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
note = f"""# 光伏复验与当前项目适配检查（2026-10-26）

结论：**独立光伏组件验证通过，满足当天07:00–19:00、60分钟、前三天39点功率、当天13×4辐照/降雨的要求；尚不能直接作为当前场站平台的资源输入，平台并未接入该预测。**

## 本轮实际执行结果

| 检查 | 结果与范围 |
| --- | --- |
| 修改前独立回归 | 75项通过（378.439秒），不是修复后最终证据 |
| 修复后完整光伏回归 | 81项通过，{match[2]}秒；含新增6项记录边界测试 |
| 当前平台完整回归 | 43项通过，无跳过/失败；使用平台专用虚拟环境、隔离临时数据、本机模拟模型 |
| 已存模型重新加载并做留出推理 | 55天、715点，复现RMSE {metrics['rmse']:.6f} kW、MAE {metrics['mae']:.6f} kW、R² {metrics['r2']:.6f} |
| 全年小时数据重新核算 | 功率/辐照小时均值与rain小时累计全量一致；统一UTC核对时间戳，避免固定+08与Asia/Shanghai时区名称差异误判 |
| 新预测记录 | 严格13点、07:00–19:00、60分钟、kW，与已存预测点值/区间一致 |
| 权重与数据 | 未重新训练、未重校准；新包全部已存JSON/权重/缩放器及原数据哈希保持不变 |

上述精度仍以目标日模拟气象已知为条件；数据为此前查看过的2025模拟路径，不是真实可获得气象预报下的验收分数，不与旧15分钟/无未来气象分数直接比较。

## 发现并修复的问题

复核实际预测记录时，原校验器错误接受三种记录：结束时间改为23:00但点值只到19:00；区间少一个点；区间日期与点值日期不一致。修复位于 `pv_forecast/records.py`，现在检查点值与区间长度、共同时间网格和覆盖终点，同时拒绝非有限值、负值及上下界倒置。

修复前的可复现结果为 `record_boundary_before.json`，三项均为incorrectly accepted。修复后使用真实新模型输出重新构造同样反例，全部拒绝，见 `inference_verification.json`。新增 `tests/test_record_alignment.py` 同时覆盖小时与旧15分钟记录，以及unavailable空记录。该修复不修改网络或预测精度。

## 与当前项目逐项对照

| 对照项 | 当前事实 | 适配判断 |
| --- | --- | --- |
| 光伏组件自身 | 默认39×1历史功率、13×4未来SWGDN/DHI/DNI/rain、13点kW输出 | 符合最新用户要求 |
| 资源调用链 | `station_platform/domain.py:resources`及capability明确返回forecast_status=not_used；O2工具仍说明使用模拟资源 | 未接入，不能声称能源Agent已调用模型 |
| 真实代码注入反例 | 向实际request快照放入新预测，再调用resources，仍not_used，charging_headroom_kw不变 | 仅传递JSON不会启用预测，已有服务没有消费逻辑 |
| 时间粒度与窗口 | 平台未来1小时固定4个15分钟时段；新PV固定当天13个小时点 | 不能把13点数组直接替换4点资源序列；需显式时间适配 |
| 负荷模型 | `代码/负荷/load_forecast/data.py`仍为15分钟、24小时96点 | 未来能源平衡需统一时间轴和基础负荷口径 |
| 运行依赖 | 平台虚拟环境有FastAPI，未安装numpy/pandas/sklearn/TensorFlow；光伏在独立数值环境运行 | 当前不支持平台直接import并加载GRU，应单独提供只读推理边界 |
| 站点与发布 | 新例子的asset_scope=configured_pv_assets，model_published_at=null；平台是mock-station-001模拟站 | 候选/演示数据，须绑定同站点及模型发布/回退记录 |
| 气象输入 | 当前性能实验采用同步模拟气象；尚无实际未来预报接口归档证据 | 可做离线验证，不能声称已达到在线预测适配 |
| 部署 | 根.gitignore仅发布station_platform，光伏代码和模型不在该仓库发布范围 | 新电脑安装平台不会自动获得光伏候选及其依赖 |

平台探针实际执行了request_from_site、resources、schedule、evaluation和capability，使用隔离模拟状态；没有只凭字符串搜索得出结论。结果见 `platform_probe.json`。平台角色、实际配置和执行链路未改动，原有用户工作区中的平台修改保留。

## 平台接入前需要确定的内容

1. 新增只读预测消费者，显式记录使用了哪个预测、哪个站点和版本，并处理不可用/降级，不能继续只填not_used后宣称已经接入。
2. 保持用户要求的13点模型，用明确的适配策略对齐平台15分钟计划与负荷；小时平均功率可用于小时能量核算，但不能把细分复制当作已有15分钟精度。未覆盖的时间、固定日窗与滚动窗口交界必须拒绝或采用经验证的保守依据，不能擅自补零。
3. 提供与计划起点时效相符的天气预报，绑定站点/资产容量，确认小时均值与时间标签语义，保留可追溯发布和回退。
4. 在独立推理环境部署候选及依赖，再对接口、过期数据、部分覆盖、网格换算和失败回退做联调。当前独立模型通过，不代表这些尚不存在的适配已通过。

## 测试环境与复现

本轮平台最初在沙箱内阻塞于Python asyncio的本机socketpair初始化，堆栈在 `platform_native_tests.log`；这是本机套接字限制，不是光伏或模型API响应失败。停止该次测试后，允许本机回环连接，以相同平台虚拟环境运行完整43项，140.13秒全部通过，见 `platform_full_tests.log` 与XML。只调用测试替身，未访问外部模型API或已有平台配置。

主要证据：`pv_tests_after_fix.log`、`platform_full_tests.xml`、`inference_verification.json`、`platform_probe.json`、`project_compatibility.json`。小时包说明已同步更新至 `../../README.md`。
"""
(out/"适配验证记录.md").write_text(note, encoding="utf-8")
print(json.dumps({"status": "completed", "pv_tests": 81, "platform_tests": 43,
    "direct_platform_ready": False, "note": str(out/"适配验证记录.md")}, ensure_ascii=False))
