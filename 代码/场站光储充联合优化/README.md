# A02 场站光储充联合优化 v1

开发日期：2026-10-29；算法 `a02-joint-v1`。实现位于 `station_joint/`，入口为 Python `solve(input)` 和 `python -m station_joint`。仅支持单站、4×15 分钟离线模拟，输入绑定站点、AC 计量边界、任务版本、可见时刻及来源 SHA-256。

## 实现与边界

算法源码、测试、说明和验收证据统一位于 `代码/场站光储充联合优化/`，与总平台目录独立。

本轮完成联合数值算法、D01 只读适配、独立物理复算、离线模拟双门槛、反例测试及运行证据。依赖 OR-Tools 9.15.6755 / SCIP 10.0.0，见 `requirements.txt`；测试另见 `requirements-dev.txt`。可复用已安装相同依赖的 Python 环境。采用连续功率与二进制连接/充放模式的混合整数线性规划；接口依据 [OR-Tools 官方混合整数规划文档](https://developers.google.com/optimization/mip/mip_example)。

新算法通过独立入口显式调用。现有网页/Agent 的 `domain.schedule/evaluation/check_local`、O1 发布和站端执行仍使用原有 idle 流程。黄花站与 `mock-station-001` 没有被悄然映射为同一站点。A01 的实时冻结资源链、A03 的最新量测/保护/授权/已执行前缀复核尚未接入本算法；本轮的离线复算不能称为完整 A03 或真实安全认证。接口不产生设备命令，也不直接生成旧格式可发布计划。

## 硬约束与服务口径

每段功率都是 AC 侧区间平均 kW，电量为 kWh，`Δt=0.25h`。禁止反送：

```text
P_new[t] = sum_vehicle P_vehicle[t]
P_existing[t] = sum_port P_existing_port[t]
P_grid[t] = P_base[t] + P_existing[t] + P_new[t] + Pc[t] - Pd[t] - P_pv_used[t]
0 <= P_grid[t] <= grid_import_limit[t]
0 <= P_pv_used[t] <= P_pv_available[t]
E[t+1] = E[t] + eta_c * Pc[t] * 0.25 - Pd[t] * 0.25 / eta_d
capacity * max(min_soc, reserve_soc) <= E[t] <= capacity * max_soc
E[0] = initial_energy_kwh
E[4] = terminal_energy_kwh  （显式硬等式）
```

PCS 每段充、放功率分别受限，用二进制变量保证同段互斥，不以净额掩盖同充同放。共输出 5 个储能边界能量，分别对应窗口起点及四段终点。当前参数无自放电、爬坡、温度、老化容量演化或网络潮流模型。

每车必须在到离站窗口内完成**精确的 AC 等效请求电量**，不减配、不延期、不超充。每车全窗口选择一个兼容且在允许列表中的端口，仅允许一次连续连接；连接期间可逐段调功率，也可为零。`connected_slots` 表示连接占用，而非仅用正功率推断占用；两车不可同时占同一端口。既有预约占用即使功率为零，也禁止新增车辆使用。车/桩上限与新充电预算始终是硬约束。

这扩展了旧版“单桩、连续时段、近均匀充电”的候选集合，因此新模型最优不等于旧版三种候选中的最优。车辆电量按现有台账的无充电损耗 AC 等效口径，不解释成真实电池入电量；真实 DC 转换需另扩输入。

原有基础/既有负荷超限时，允许联合储能在其真实能量边界内消除冲突；若无法消除即判不可行。没有使用 `max(0, grid)` 截掉反送或掩盖功率不平衡。

## 目标与求解状态

先满足全部物理/服务硬约束，再最小化下式，不用罚项交换承诺违约：

```text
purchase_cost = sum(grid_kw[t] * tariff[t] * 0.25)
throughput_cost = throughput_cost_rate * sum((Pc[t] + Pd[t]) * 0.25)
curtailment_cost = curtailment_cost_rate * sum((PV_available[t] - PV_used[t]) * 0.25)
objective = purchase_cost + throughput_cost + curtailment_cost
```

两个软目标系数均须显式输入，并纳入输入哈希。弃光罚项是模拟偏好，不是实际结算收入。价格可为负；互斥、SOC、备用及终态仍有效。峰值购电、等待时间和储能损耗没有额外独立优化优先级；不能从“目标下降”推断这些指标均改善。同价时可存在不同充电分布的等价最优解。

SCIP 单线程、固定随机种子、零配置相对/绝对目标间隙，数值可行容差 `1e-9`。求解秒数显式限制为 0.001～60 秒（默认 10 秒）；该限时只约束后端求解，不包含输入校验、模型构建与独立复算。状态严格取求解器返回值：

| 状态 | 意义 | 是否有计划 |
| --- | --- | --- |
| `OPTIMAL` | 该有限模型在求解器数值容差内证明最优 | 经独立复算后有 |
| `FEASIBLE` | 找到可行候选，未证明最优 | 经独立复算后有 |
| `UNKNOWN` | 终止时无可行 incumbent，未知是否存在解 | 无 |
| `INFEASIBLE` | 证明当前模型无解 | 无 |
| `INVALID_INPUT` | 缺项、未知承诺、错误时网格或非有限数等 | 无 |
| `ERROR` | 求解器异常或独立复算不通过 | 无 |

记录后端版本、实际耗时、时限、变量/约束数、搜索节点、目标界、绝对/相对间隙。绝对间隙为 `incumbent - lower_bound`，相对间隙分母为 `max(1 CNY, abs(incumbent))`；无有效界时为 null。不可行时给出窗口能量、端口容量、备用或终态的充分冲突线索；复杂组合冲突只报告完整模型无解，不声称得到最小冲突集。最优证明只适用于本输入和模型，无真实设备最优性含义。

## 输入与输出

完整可运行示例见 `开发验证/hand_input.json`；校验器为 `station_joint/contract.py::validate_input`。所有资源曲线必须明确给 4 项，不能用缺值、NaN、标量或默认零替代未知。业务时间统一 `+08:00`，原生 15 分钟网格；不支持 13 点 PV 直接扩全天、非整刻窗口或跨窗承诺重规划。

| 输入 | 关键要求 |
| --- | --- |
| 身份/来源 | schema、任务 ID/版本、站点、`meter_boundary=station_ac`、as_of、窗口、来源 SHA-256、假设清单 |
| 资源 | 基础负荷、PV 可用、并网限额、费率、可空的新充电预算 |
| 端口 | 独立端口 ID、连接器、逐段可用功率（故障/降额由显式曲线表达）、既有占用槽 |
| 既有充电 | `commitment_status=complete`、覆盖每个端口的功率数组，与既有占用一致；不混入基础负荷 |
| 车辆 | 唯一 ID、窗口槽、精确请求电量、车上限、连接器、允许端口列表 |
| 储能 | 容量、初态、效率、SOC/备用、逐段 PCS 上限、窗口末端硬等式 |
| 目标 | 吞吐/弃光系数显式给出，可为零 |

计划给出 `assignments`（每车端口、连接槽和功率）、`charge_by_vehicle_kw`、`charge_by_port_kw`、`pv_used_kw`、`storage_charge_kw`、`storage_discharge_kw`、`storage_energy_kwh`、`net_import_kw`。按端口输出的是**新增**充电；既有充电仍引用冻结输入，避免重复计数。每份计划哈希同时绑定完整输入哈希及全部曲线/分配。目标分解、交付电量、峰值、弃光、吞吐、转换损耗和功率残差由独立复算得出。

`verify_plan` 不导入求解器，以分车分配重新累计总功率，复算全套公式，校验聚合曲线和哈希；数值门槛为绝对 `1e-5 kW/kWh/CNY`。可传入同任务窗口的新完整模拟输入 `live_input` 复核降额、PV 下降或初态变化，分别记录原始/复核输入哈希；不自动修正曲线。该参数不是在线状态新鲜度服务。

`simulation_gate` 在同一计划/输入/复核状态哈希下重新运行数值校验，并独立执行版本化离线治理检查；含假设时要求调用者显式接受假设。只有双通过才返回 `eligible_for_offline_simulation=true`。`device_dispatch_authorized` 固定 false。没有对接 O1 发布 API，不绕过旧站端储能 idle 拒绝规则。

## D01 回放适配与未补齐的数据

`LinkedReplay(data_root).load(scenario_id)` 核对来源哈希、站点、可见时间、完整网格、预测版本、静态有效期、预约撤销、费率及储能身份。只用决策视图，不读取 `scenario_truth.csv`、全年真值或未来历史会话来构造计划；PV 使用 D01 已转换的区间平均基线，不使用原瞬时采样。所消费文件哈希固定在计划输入内。

默认 12 个场景均不生成计划：前 9 个缺少既有预约的逐段功率/占用资料；S10 承诺 unknown；S11/S12 预测不可用。预约的承诺电量不能推出功率曲线，不能拿未来到站事件或已发生会话补齐。

开发回放可显式传 `demo_assumptions=True`，生成**额外反事实测试条件**：每个可见预约占其端口四段，固定功率 `min(120 kW, 承诺kWh / 1h)`；这些值与真实/预测功率无关，未交付的预约剩余电量不作跨窗服务承诺。吞吐费假设 0.02 CNY/kWh、弃光罚项 0.001 CNY/kWh。每项写进输入假设和哈希；即使显式演示，也绝不把 unknown 承诺或不可用预测补零。

D01 储能午夜 60% 目标在全部有效样本的窗口之外。本适配器单独假设窗口末端恢复本窗口初始能量，保留原午夜目标为 `not_verified_outside_window`，不声称午夜可达。`source_site_dispatch_ready=false` 继续保留。算法调用者可在完整输入中选择其他显式末端电量；与 idle 比较时必须确保相同末端条件，否则不计算费用收益。

## 运行与复现

在 `代码/场站光储充联合优化` 目录执行。`python` 应指向已安装依赖的解释器，输出写到原数据目录以外：

```powershell
# 独立环境首次安装（已有依赖时无需重复安装）
python -m pip install -r requirements-dev.txt

# 完整输入与同输入的 idle 对照
python -X utf8 运行.py --input 开发验证/hand_input.json --output 开发验证/hand_result.json

# 严格模式：缺数据即报告不可用
python -X utf8 -m station_joint --data-root '../数据' --output 开发验证/replay_strict.json

# 显式假设的算法开发回放
python -X utf8 -m station_joint --data-root '../数据' --demo-assumptions --time-limit 5 --output 开发验证/replay_demo.json

# 使用新的临时目录，避免覆盖已有测试记录
python -X utf8 -m pytest tests -q --basetemp 开发验证/new-test-temp
```

项目内还可临时复用 `../../总平台/.venv/Scripts/python.exe` 替换上述 `python`；这只借用依赖环境，不从总平台加载算法源码。

目录结构：

```text
代码/场站光储充联合优化/
├── station_joint/       算法、输入契约、独立复算和回放适配
├── tests/               算法测试
├── 开发验证/            输入示例、运行结果、对照、JUnit、验收记录和哈希清单
├── 运行.py              独立运行入口
├── pyproject.toml       包与测试配置
├── requirements.txt     算法依赖
├── requirements-dev.txt 测试依赖
└── README.md            本说明
```

本轮的结果与限制见 `开发验证/验收记录.md`。后续在线集成必须让 A01、优化器、评价与 A03 使用同一冻结上下文，并补齐实时状态、跨窗剩余承诺、终态接续、执行幂等与角色权限，再扩站端储能执行。24 小时日前优化须另扩预测覆盖和输入/求解规模。
