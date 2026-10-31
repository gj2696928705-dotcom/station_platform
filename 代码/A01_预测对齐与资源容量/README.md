# A01：预测对齐与逐时段资源容量

本目录是 A01 的算法交付目录，源码、测试、固定候选模型包、说明和运行证据均在 `代码/A01_预测对齐与资源容量`。`总平台` 保留现有业务流程及薄调用接口 `station_platform/forecast_api.py`，通过路径加载本目录；不在总平台内保存 A01 算法副本。

实现范围是**模拟环境的点预测新增充电容量**。它不代表真实设备能力、正式发布预测或认证调度服务。储能保持 `idle`。

## 文件入口

| 文件 / 目录 | 用途 |
| --- | --- |
| `capacity_alignment/forecast_capacity.py` | 独立数值核心：原生区间校验、重叠加权、设备及并网容量、冻结哈希；只依赖 Python 标准库 |
| `capacity_alignment/forecast_runtime.py` | 将冻结上下文挂接到任务请求；统一资源、搜索和评价的能量口径 |
| `capacity_alignment/forecast_adapter.py` | 模型包完整性检查、独立进程、天气批次筛选、原始记录外层适配 |
| `capacity_alignment/forecast_worker.py` | 独立数值解释器中的只读推理入口，只调用 `predict_record` |
| `capacity_alignment/forecast_replay.py` | 读取 D01 决策输入；不读取 `scenario_truth.csv`、全年真值或未来会话 |
| `package_forecast_models.py` | 将指定候选代码、权重、元数据和依赖锁复制到新目录，拒绝覆盖 |
| `verify_a01_models.py` | 双候选模型→冻结→资源→编排→评价的回放实例 |
| `verify_a01_evidence.py` | 不导入算法模块的独立 Decimal 复算 |
| `tests/` | 算法、模型进程故障及 HTTP 集成测试 |
| `forecast_models/` | 固定候选代码 / 权重 / 原依赖说明 / 独立运行环境清单 |
| `evidence/` | 回放 JSON、原始推理、测试 XML 和独立复算结果 |

## 时间、功率及容量口径

所有时间携带时区，业务和原生区间均为 `[start,end)`。以真实业务起点生成 15 分钟段，不强制起点落在整刻。目标段 `[s,e)` 的区间平均功率为：

```text
P_slot = Σ(P_native × overlap_seconds) / (e-s).seconds
```

要求每段完整覆盖，没有插值、补零或外推。原生区间不能重叠。区间内常值是计算假设，不代表新的秒级预测精度。

输入必须显式声明 `unit=kW,value_semantics=interval_average_power`；若提供电量，必须声明 `unit=kWh,value_semantics=interval_energy`，按原生区间时长换算平均功率。瞬时功率和口径不明的数据不能进入这一接口。

```text
P_existing = P_base_forecast + P_committed_charge
grid_new_limit = grid_import_limit + P_pv_available - P_existing
device_new_limit = Σ(本段未占用、未冻结、可用或降额端口的当前功率上限)
capacity = min(device_new_limit, grid_new_limit)
```

`grid_new_limit<0` 时输出 `EXISTING_GRID_CONFLICT`，保留负的诊断余量，整个窗口不可用。有效窗口内余量恰好为零是已知的零容量，区别于缺失数据的 `null`。

既有充电与基础负荷分别输入。基础预测必须是 `base_load_excluding_charging_pv_storage`，禁止将总负荷或净并网功率当成基础负荷再次叠加充电。PV 为 AC 可用参考出力；评价按站内吸收量限发，禁止反送。

每个既有会话必须给出完整同窗功率曲线，包括明确的零功率段。只有预约电量时状态为 `unknown`，不能推断逐段充电功率。清单明确为空时使用 `known_empty`。会话 ID 重复、与本次新增 ID 重复、端口不存在、同一端口既有会话重叠或既有功率超过设备上限均拒绝。端口采用独占会话约束，只要本段与非零既有充电或占用区间重叠，该端口不再分配新会话；这比按剩余 kW 分配更保守。

## 数值核心接口

```python
from capacity_alignment.forecast_capacity import freeze, verify

context = freeze(inputs)
verify(context)  # 检查保存后的上下文哈希
```

`inputs` 的完整可执行样例见 `tests/test_forecast_capacity.py:sample` 和 `evidence/replay/S01_sunny.json` 中的 `request.forecast_context.context.inputs`。

| 字段 | 必需内容 |
| --- | --- |
| 身份与窗口 | `station_id,meter_boundary,task_id,task_version,as_of,start,end,interval_minutes=15` |
| `execution_mode` | `simulation` 或 `offline_replay`；仅后者以回放时钟评价历史任务 |
| `binding` | 同一站点、边界、模拟标记、`evidence_ref` 和可见 / 失效时间 |
| `state` | 当前量测来源、质量、保护、接管、边界已知状态、并网购电限额、逐端口状态及占用区间 |
| `load_forecast,pv_forecast` | 记录 ID、同站点 / 边界、语义、单位、状态、模型 / 方法版本、输入及方法哈希、来源、时间和原生 `values` |
| `values[]` | `start,end,value`；可选 `unit` 须与序列一致；基线历史来源可提供 `source_observation_end` |
| `commitments` | `known/known_empty/unknown`、完整覆盖范围、来源、可见 / 失效时间及 `sessions` |
| `sessions[]` | 唯一会话 ID、端口 ID、`committed_charge_ac` 语义、可见时间和显式同窗曲线 |

绑定、预测、承诺和设备必须属于同一站点及同一公共 AC 边界，并明确标记模拟。所有输入要求 `issued_at≤available_at≤as_of<expires_at`。基础负荷历史观测必须在预测发布时已经结束。预测覆盖时长和数据有效期分别检查。

输出 `context` 保存完整原始输入、每段对齐分量、既有负荷、两个新增上限、最终容量、源记录索引和重叠秒数、设备逐项贡献、失效时间、`input_hash` 及 `context_hash`。一个输入或时段失败时，整个窗口 `status=unavailable`，所有段的 `capacity_kw=null`，错误保存在 `issues` 中。

哈希用于完整性、版本一致性及复算，不是数字签名或输入真实性认证。数值核心最多计算 96 段；当前平台编排入口仍是 1 小时 4 段。

## 平台调用契约

HTTP 结构化任务可额外提供：

```json
{
  "forecast": {
    "mode": "simulation_use",
    "inputs": {
      "execution_mode": "simulation",
      "meter_boundary": "经核对的公共AC边界",
      "binding": {},
      "load_forecast": {},
      "pv_forecast": {},
      "commitments": {}
    }
  }
}
```

以上空对象只是结构示意，不能生成容量；完整字段见样例。HTTP 不接受模型解释器、权重路径或执行命令。只读模型推理由受信本地调用产生外层记录，再作为模拟任务输入。

任务创建时，程序以实际任务 ID、版本、`as_of`、业务窗口、量测、设备和并网边界覆盖输入中的对应字段；不允许通过预测输入抬高设备或并网限额。量测来源的实际失效时间参与冻结，不能以一小时业务窗口替代 20 秒的源量测有效期。保留量测 ID、站点修订、质量和原始到期时间。

| 模式 | 行为 |
| --- | --- |
| `disabled` | 默认。使用原模拟序列，`forecast_status=not_used`，不推理、不加载算法包；总平台单独部署时原流程仍可运行 |
| `shadow` | 校验、计算并保存预测诊断，原能量快照和方案不变；`forecast_status=not_used`；观察哈希置于 `forecast_diagnostics` |
| `simulation_use` | 显式启用。成功时仅将 `基础预测+既有充电` 写入冻结的 `base_load_kw`，同时替换 PV 和既有占用；资源、任务内能力查询、搜索、评价共享 `frozen_context_hash`。不可用时直接标记 `BLOCKED`，不能依赖模型决定是否继续 |

资源工具、编排和评价会检查请求及上下文完整性；评价还拒绝不同任务版本的资源 / 搜索引用。在线模拟模式的冻结数据到期后，资源返回 `usable=false` 和空容量，搜索 / 评价拒绝消费。离线回放使用保存的 `as_of`，不会因执行机器墙钟在 2026 年而把 2025 回放误判为在线数据。

上下文随任务 SQLite 记录持久化，中心重启后恢复同一请求和已登记计划，不重查预测、不重复发布。新输入只能建立新任务版本，历史中保留旧请求。站端继续使用最新量测、设备及并网限额独立复核；预测通过不代表现场必然通过。

任务详情展示使用模式、失效时间、原生预测表、逐段容量、错误原因及冻结引用。新建数据库的默认 O2 职责已匹配新契约；已有数据库的用户职责文档不自动覆盖，启用前可按本契约更新其已发布正文。

`总平台/station_platform/forecast_api.py` 默认定位本目录；部署到其他目录可配置 `A01_ALGORITHM_ROOT`。回放实例默认定位本工作区总平台；可用 `A01_PLATFORM_ROOT` 指定其源码根目录。核心模块 `forecast_capacity` 本身不导入平台或数值模型库。

## 候选模型、小时边界与独立环境

`forecast_models/load-v1` 固定原负荷 `20260928_232203_101506`；`forecast_models/pv-day13-v1` 固定原光伏 `20261026_day13_01`。打包保留代码、必要权重、原说明和依赖文件，新增逐文件 SHA-256、`runtime.json` 和实际环境的 `requirements-lock.txt`，没有训练、重新校准或选择“最新目录”。原模型与数据文件均未修改。

数值工作进程通过 `deployment.json` 中独立解释器运行，本机已实测 `D:/Anaconda3/anaconda/python.exe`。平台不需要在请求进程内导入 TensorFlow / XGBoost。部署到其他机器需安装清单依赖并重新生成清单，不能将本机解释器路径视为可移植运行时。TensorFlow 原项目 requirements 将 GRU 作为可选项，本交付锁定运行环境中已实际存在 TensorFlow。

每次推理创建独立目录，显式传 `as_of` 或目标自然日，检查退出码、JSON、原生时间网格、单位、区间包络、版本、站点与包哈希；超时、错误、缺失 / 损坏输出、日志或输出过大均拒绝，不读取旧输出。`degraded` 默认不消费。

负荷保留 96 个 15 分钟区间。光伏 day13 保留 07:00—19:00 的 13 个原始小时标签；原 JSON 的 `forecast_end_at=19:00` 不改写。依据已登记的“小时均值、起点标签”口径，包装层 coverage 为 `[07:00,20:00)`。06:50 或 19:07:30 开始且跨出覆盖边界的窗口不可用，未覆盖时间绝不补成零发电。

光伏天气只选择 `issued_at≤min(as_of,D日07:00)` 的最新合格批次，并要求同批完整 13 点。缺点不混批、不退选早批。没有合格归档的同步气象推理只能保存原始观察记录，包装状态为不可用。适配后的 `issued_at/as_of` 是明确反事实的离线回放时钟；原始生成墙钟始终保留在 `raw_record`，不证明 2025 年当时已有该 2026 候选模型。

## 运行与独立复算

在本目录运行 PowerShell；以下平台解释器仅用于算法、测试和启动工作进程，模型推理由部署清单中的独立解释器执行：

```powershell
# 53 项算法、独立进程和 HTTP 测试
& '../../总平台/.venv/Scripts/python.exe' -X utf8 -m pytest tests -q --basetemp evidence/tmp-check

# D01：显式假设充电曲线
& '../../总平台/.venv/Scripts/python.exe' -X utf8 -m capacity_alignment.forecast_replay --data-dir '../数据' --output evidence/replay --commitment-policy rated_from_as_of
& '../../总平台/.venv/Scripts/python.exe' -X utf8 verify_a01_evidence.py evidence/replay --output evidence/independent_replay_verification.json

# 不提供曲线假设：不得把预约电量变成已知功率
& '../../总平台/.venv/Scripts/python.exe' -X utf8 -m capacity_alignment.forecast_replay --data-dir '../数据' --output evidence/replay_without_commitment_policy

# 固定双候选模型 + 明示合成天气 / 空承诺场景
& '../../总平台/.venv/Scripts/python.exe' -X utf8 verify_a01_models.py --data-dir '../数据' --model-dir forecast_models --output evidence/candidate_smoke
& '../../总平台/.venv/Scripts/python.exe' -X utf8 verify_a01_evidence.py evidence/candidate_smoke/result.json --output evidence/independent_model_verification.json
```

重新打包时必须指定**新目标目录**：

```powershell
& '../../总平台/.venv/Scripts/python.exe' -X utf8 package_forecast_models.py --kind load --code-dir '../负荷' --bundle-dir '../负荷/运行结果/20260928_232203_101506' --python 'D:/Anaconda3/anaconda/python.exe' --output forecast_models/load-new
& '../../总平台/.venv/Scripts/python.exe' -X utf8 package_forecast_models.py --kind pv --code-dir '../光伏' --bundle-dir '../光伏/运行结果/20261026_day13_01' --python 'D:/Anaconda3/anaconda/python.exe' --output forecast_models/pv-new
```

## 数据回放的实际含义

D01 原始预约仅有电量，本实现默认全部保持未知。`rated_from_as_of` 是显式的反事实规则：只取决策时可见且未撤销的预约，从 `as_of` 起按当前端口上限充电，直到该承诺电量完成；保留窗口后的剩余电量，不宣称一小时履行了全部旧承诺。它不读取未来到站事件或未来历史会话。清单本身未知的 S10 即使指定该规则仍阻断。

指定该规则后，S01—S09 有可复算容量并形成方案，S10 / S11 / S12 分别因承诺未知 / PV 缺口 / 无合格天气而阻断。S03、S04 的 40 / 70 辆和 S05 的零新增车辆通过显式离线入口处理；交互入口保持 1～12 辆。D01 请求的指定端口约束在搜索和独立分桩检查中均保留。

`verify_a01_models.py` 单独生成数学函数定义的合成天气预报批次，并声明同窗既有会话为空。该场景证明双候选推理与平台消费链路可运行；不证明存在真实 2025 年预报归档或空闲站点。验收结果见 [验收记录.md](验收记录.md)。
