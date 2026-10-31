# A03 独立可行性与安全复核

开发日期：2026-10-29。源码、测试、运行入口和证据均在本目录，Python 包名 `station_review`。这是独立的**离线模拟计划最终复核算法**；没有接入网页发布链，也不生成真实设备指令或安全认证结论。

## 已实现

复核器从冻结计划、冻结原约束和最新站端状态重新计算，不读取优化器自报的 `feasible`、`verification.passed` 或角色文字。生产代码仅依赖 Python 标准库，不导入 `station_joint`、OR-Tools、预测模型或平台模块。

- 独立重建逐车、逐端口功率，检查服务电量、到离站窗口、端口独占、连接器和占用；当前与 A02 一致，每车在整个窗口使用一个端口和一个连续连接区间。
- 从最新实测储能电量重新递推 SOC，检查效率、PCS、备用、同段充放和终态；不沿用计划自己报告的递推结果作为起点。
- 复算 PV 上限、并网净功率、禁止反送和最新任务预算；同时检查计划声明的并网曲线与独立计算值，避免伪造曲线掩盖越限。
- 检查最新状态的时刻、质量、站点/任务/设备映射、保护、接管、冻结、失联和降额；所有冻结来源哈希都须与站端当前引用一致。
- 从实际执行回执重建连续前缀，核对实际功率、电量、SOC 和连接映射；剩余服务按“总承诺减实际交付”检查，拒绝旧完整计划重发、前缀修改、回执冲突和取消未确认的替代计划。
- 多版本执行归属链允许保留先前版本合法完成的步骤；旧版本超过其确认停止边界的回执拒绝，不能回滚当前版本。
- 物理校验与有版本的治理校验分别留痕。只有同一冻结计划指纹、同一最新状态哈希的两项结果均为布尔 `true`，才返回 `eligible_for_offline_simulation=true`。

缺失、未知、非有限数、错误长度/时区、未来才可见、过期或不支持的契约均失败关闭。失败不改写计划、不修复功率、不沿用旧指纹自动放行。

## 快速运行

要求 Python 3.10+；算法没有第三方运行依赖。以下命令在本目录执行。

```powershell
# 完整的显式模拟手算示例；包含输入与复核结果
python -X utf8 -m station_review demo --output 开发验证/hand_demo.json

# 输入是一个 JSON 请求，见下文和 开发验证/requests/
python -X utf8 -m station_review review --request 开发验证/requests/latest_pv_drop.json --output 开发验证/recheck.json

# pytest 是开发依赖；预期全部通过
python -X utf8 -m pytest tests -q --basetemp 开发验证/new-test-temp --junitxml 开发验证/tests.xml

# 生成/重放端到端证据，并核对原数据与 A02 归档前后哈希不变
python -X utf8 开发验证/collect_evidence.py
```

`review` 退出码：通过 `0`，正常拒绝 `2`，JSON/文件输入错误 `1`。正常拒绝仍输出结构化 JSON。`archive-a02` 因存在无计划场景也可能返回 `2`，不能仅凭非零退出码认定算法崩溃。

```python
from station_review import freeze_plan, review_plan

# constraints、plan 原样来自优化器共享的冻结输入与输出；下列其他输入
# 必须由可信站端/任务授权域提供，不能由优化器或页面自报。
frozen = freeze_plan(
    constraints, plan,
    producer="A02", producer_version="a02-joint-v1", publisher="O1-SIM",
    plan_version=1, dependencies=dependencies,
    issued_at=issued_at, valid_until=valid_until,
)
result = review_plan(frozen, latest_state, governance_policy, as_of=business_clock)
```

Python 对象 `FrozenPlan` 只保存不可变 JSON 字节；修改 `to_dict()` 返回值不会改变原冻结计划。冻结构造本身不等于通过复核。JSON 请求的四个必填字段为 `frozen_plan`、`latest_state`、`governance_policy`、`as_of`。发布时可额外提供先前审查的 `expected_state_hash`，任意状态变化都会阻断并要求重新审查/授权。

## 冻结计划契约

`freeze_plan` 绑定以下内容，外层 `fingerprint` 对除自身外的全部字段计算规范 JSON SHA-256：

| 字段 | 内容 |
| --- | --- |
| `constraints`, `constraints_hash` | 优化器使用的原始任务、数据、原约束、来源和版本，完整原样冻结 |
| `plan`, `plan_fingerprint` | 优化器原计划及其规范指纹；`plan.input_hash` 必须等于 `constraints_hash` |
| `producer`, `producer_version`, `publisher`, `plan_version` | 生产者算法/版本、模拟发布主体、计划版本 |
| `issued_at`, `valid_until` | 计划生效来源时刻与半开有效期 |
| `dependencies` | 负荷预测、PV 预测、既有承诺的冻结引用 |
| `execute_from_slot`, `executed_prefix_hash` | 下一未执行区间及实际执行前缀指纹 |

兼容 A02 `a02-joint-v1` 的 4×15 分钟计划，保留原 `input_hash` 和原计划指纹，外层另绑定发布元数据。通用接口为 `a03-constraints-v1` / `a03-plan-v1`，沿用 A02 的字段、AC 物理口径和逐车连接模型，扩展为 1—96 个 15 分钟区间。A01/A04/A05/A07 后续生产者需使用该接口并通过治理白名单；本次实际兼容验证只覆盖已有 A02 产物，没有伪造尚不存在的其他算法适配结果。

每个依赖有 `ref,kind,sha256,status,issued_at,available_at,valid_until,window_start,window_end`；`kind` 为 `forecast_load/forecast_pv/commitments`，`status=usable`。预测还需 `native_grid=15min_interval_average`，拒绝把瞬时 PV 采样当区间平均预测。同一来源清单可分别供三类依赖引用，重复的 `(ref,kind)` 拒绝。`sha256` 必须与优化器 `source_refs` 和最新状态一致；时网格必须覆盖待执行后缀，已执行前缀不要求新预测回补历史。默认没有“预测过期后继续使用”的降级规则。

## 最新状态契约

完整可运行样本见 [正常请求](开发验证/requests/normal_hand_plan.json) 和 [部分执行请求](开发验证/requests/partial_actual_prefix.json)。不能将示例假设作为缺失实时输入的默认值。

| 字段组 | 必填内容 |
| --- | --- |
| 身份/时钟 | `schema_version=a03-state-v1,state_id,revision,station_id,task_id,task_version,meter_boundary,measured_at,available_at,valid_until,quality` |
| 模式/语义 | `is_simulated=true,power_semantics=ac_dispatch_snapshot` |
| 最新量测 | `measurement.quality,base_load_kw,pv_available_kw,net_import_kw,storage_energy_kwh,storage_soc`；SOC 为 0—1 比例 |
| 站级安全 | `online,protection_active,manual_takeover,frozen`，必须显式布尔 |
| 最新能力 | `grid_import_limit_kw,pv_max_power_kw,new_charge_limit_kw`；后者为完整区间曲线，显式 `null` 仅表示没有新增预算限制 |
| 设备 | `ports` 精确覆盖冻结端口，逐项有 `asset_id,connector,max_power_kw,occupied_slots,online,fault,protection_active,manual_takeover,frozen`；`storage` 有相同安全标记和 `storage_id,max_charge_kw,max_discharge_kw` |
| 来源 | `dependency_hashes` 精确覆盖全部冻结 `source_refs`，不允许静默替换台账/储能参数等来源 |
| 执行账本 | `execution.accepted_plan_version,accepted_plan_fingerprint,accepted_plan_history,cancel_status,stopped_at,stopped_after_slot,receipts` |

所有时间必须明确使用 `+08:00`；`measured_at ≤ available_at ≤ as_of < valid_until`，量测年龄最多 60 秒，过期边界即拒绝。这个 60 秒是本算法固定的模拟规则，不是经实站认证的保护周期。站级保护/接管/冻结/失联阻断全部计划；设备级故障等阻断该设备的待执行使用，最新降额与原约束取更严上限，不重写历史执行边界。

实时与未来的使用边界明确区分：下一段使用最新基础负荷复算并网，且必须与计划相符；后续段使用冻结的基础负荷预测。所有待执行段的 PV 利用均不得超过冻结预测、最新 PV 可用快照与额定能力三者最小值。这是明确的保守快照帽，可能拒绝部分可行计划，不代表对未来出力的保证。储能从最新实测电量递推；任何新事件都应由调用方重新取最新状态再审查。报告有效期取计划、量测年龄、状态、依赖和授权的最早失效时刻，不是可脱离最新状态独立复用的设备令牌。

并网按 AC 有功边界计算：

```text
P_new[t] = sum_vehicle P_vehicle[t]
P_grid[t] = P_base[t] + P_existing[t] + P_new[t] + Pc[t] - Pd[t] - P_pv_used[t]
0 <= P_grid[t] <= min(frozen_grid_limit[t], latest_grid_limit)
E[t+1] = E[t] + eta_c * Pc[t] * 0.25 - Pd[t] * 0.25 / eta_d
max(min_soc, reserve_soc) * capacity <= E[t] <= max_soc * capacity
```

单位为 kW、kWh，容差固定为绝对 `1e-5`，不按场站容量比例放大。车辆电量与 A02 一致为 AC 等效口径；不额外制造未知 DC 电池转换损耗。当前不支持反送、跨窗未定义的末端目标或中途更换车辆端口；相关功能必须扩展有版本契约后再验收。

## 执行前缀与多版本归属

初始执行账本显式为 `accepted_plan_version=0,accepted_plan_fingerprint=null,accepted_plan_history=[],receipts=[]`。已接受版本的归属链每项有 `plan_version,plan_fingerprint,execute_from_slot,stop_after_slot`；版本严格递增、区间连续，只有最后一个活动版本可使用 `stop_after_slot=null`。取消确认后，最后一项停止边界必须与 `stopped_after_slot` 一致。

回执必填身份为 `event_id,station_id,task_id,task_version,plan_version,plan_fingerprint,slot,measured_at,available_at,quality`；可带 `received_at`。`actual` 必须完整给出：

- `vehicle_energy_kwh`：所有任务车辆的实际电量，包含显式零；`connected_vehicle_port_ids`：该段实际连接的车辆与端口，即使充电为零也要记录连接。
- `existing_by_port_energy_kwh`：全部端口的既有承诺实际电量。
- `base_energy_kwh,pv_used_energy_kwh,grid_import_energy_kwh,storage_charge_energy_kwh,storage_discharge_energy_kwh,storage_energy_before_kwh,storage_energy_after_kwh`。

回执必须属于归属链中对应版本及其拥有的区间。相同事件 ID 的业务内容相同则幂等；内容冲突拒绝。冻结时及复核时都剔除纯传输字段 `received_at`，按 slot 排序后计算前缀哈希。`freeze_plan(...,executed_prefix=...)` 可直接接收有传输字段和精确重复项的回执。

复核器逐段复算回执 AC 能量平衡和储能动态，再核对计划历史曲线及连接映射与实绩一致。实际完成步骤只能连续从 0 开始；旧完整计划的 `execute_from_slot=0` 不能与非空账本同时通过。替代计划必须有更大版本及确认停止边界；仅发出取消请求不释放旧计划占用。下一未执行段已经开始时直接拒绝，不自动把区间截短或改变电量。

## 治理与结果

规则固定为 `a03-offline-governance-v1`。治理策略必须包含 `policy_id,rule_version,simulation_only,authorized_publishers,allowed_producers,allowed_sources,allow_assumptions,approval`。`approval` 必须明确 `released=true,simulation_only=true`，绑定同一 `frozen_fingerprint,plan_fingerprint,state_hash,state_id,state_revision,task_id,task_version,publisher`，并有有效的 `approved_at,valid_until`。旧状态授权不能用于新状态，即使数值能力恰好相同。

结果保留 `passed`、两个独立检查、逐条 `code/field/actual/expected`、任务/计划版本、双指纹、状态哈希及引用/时刻、策略哈希和重建指标。类型、缺值或不支持契约的违规用 `actual/expected` 说明期望结构。两层中任一缺失或异常都会阻断，不存在页面/角色文字覆盖入口。

**信任边界**：SHA-256 证明绑定与完整性，不证明来源身份。原约束、最新状态、执行账本和治理策略必须由可信数据/授权域提供；不允许把优化器自建策略、LLM 文字或页面字段当作这些输入。`examples.synthetic_state/synthetic_policy` 仅构造明确命名的离线测试假设，不能用于生产认证。真实授权签名、并发状态存储的原子发布、潮流/电压/无功、保护定值、秒级动态和真实运行审批需要外部系统与证据。本模块所有路径都保持 `device_dispatch_authorized=false,certified_safety_service=false`。

## 已有数据的验证方式

```powershell
# 严格模式：没有最新站端状态和授权，全部拒绝
python -X utf8 -m station_review archive-a02 --archive ../场站光储充联合优化/开发验证/replay_demo.json --data-root ../数据 --output 开发验证/a02_strict.json

# 显式模拟快照/授权：验证归档计划的独立数值与接口兼容
python -X utf8 -m station_review archive-a02 --archive ../场站光储充联合优化/开发验证/replay_demo.json --data-root ../数据 --demo-assumptions --output 开发验证/a02_compatibility.json

# D02 原流程桩计划缺少复核契约，在适配阶段拒绝
python -X utf8 -m station_review audit-d02 --data-root ../数据 --output 开发验证/d02_strict_audit.json
```

9 份已有 A02 计划在显式模拟快照和授权下通过；另 3 个场景原本没有优化器计划，继续拒绝。严格模式 0 份放行。没有把 D01 真值或 D02 `execution_truth` 回填成当时实时量测。读取事后真值文件的 SHA-256 只用于核对源包完整性，不读取其数值作为复核特征。

D02 的 22 个场景、152 条事件是非优化流程夹具。审计核对事件和来源哈希，在每次 `plan_sent` 时仅列出已可见事件；因缺少完整设备映射、承诺曲线、最新状态和治理授权，所有尝试均在适配阶段阻断。这个审计不宣称已实现 A05 事件归约器，也不将夹具原有的“接受”事件解释为 A03 放行。

详细验收与限制见 [验收记录](开发验证/验收记录.md)；可直接重跑的输入、逐条结果和来源/代码哈希见 `开发验证/requests/`、`results/`、`simulation_e2e.jsonl` 和 `verification_manifest.json`。
