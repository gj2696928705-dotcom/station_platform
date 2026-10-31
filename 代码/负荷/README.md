# 负荷预测独立模型包

本模块完成路线中的第一项：**负荷预测 → 资源与储能能力计算 → 光储充协同优化 → 日内滚动重算**。参考光伏包的离线训练、评估、保存与只读预测结构，模型使用 XGBoost。当前是独立候选算法，未接入能源 Agent，也不生成设备命令。

## 负荷口径

默认预测 `base_load_kw`（站用基础负荷），**排除车辆充电、光伏和储能充放电**。后续资源计算应将基础负荷预测、已有充电会话、新充电计划分别处理。若显式选择 `--target total_load_kw`，输出包含历史充电负荷，只能作为总负荷参考，不能再次叠加充电计划。

最小 CSV / Excel 输入：

```csv
timestamp,base_load_kw,station_id,is_simulated
2025-01-01T00:00:00+08:00,110.0,HH-Huanghua-CSS,true
```

`timestamp` 为 15 分钟区间起点，kW 为该区间的**平均功率**，区间结束后才能作为历史输入。区间电量 = kW × 0.25 h；不把瞬时功率自动当平均功率。其他列名用 `--timestamp-col` / `--source-target-col` 显式映射。区间 kWh 必须指定 `--input-kind interval_energy --energy-interval-minutes 15 --source-target-col 实际能量列名`。支持 `.xlsx` / `.xls`（分别需要 openpyxl / xlrd）。

数据必须按时间递增、时间戳唯一、落在 15 分钟网格、站点一致、非负有限。缺失观测/缺行保留 NaN，不插值、不后向填充；训练剔除跨缺口窗口，推理的最近 7 天存在缺口时返回 `unavailable`。重复、乱序、负数、无穷、文本数值错误、其他采样频率均明确拒绝。三种负荷分量同时出现时检查总量守恒。`is_simulated` 必须全为 true 或全为 false；false 只标记 `measured_unverified`，不代表数据已认证。

模拟数据由指定会话单独生成，契约见 `模拟数据任务交接.md`。默认路径：`../数据/负荷数据/黄花站_2025年_15分钟负荷模拟数据.csv`。数据不存在会报错，不会偷偷合成训练数据。

## 模型设计与证据口径

- 最近 7 天（672 点）历史，直接预测未来 24 小时（96 点）。使用一个共享的 XGBoost 模型，单行代表一个“预测起点 × 提前步长”；并行给出 96 个结果，无递归误差累积。
- 30 个特征：提前步长、前日/前周目标同刻值、过去 7 日目标同刻均值/标准差、起点之前的若干滞后值、1 小时/1 天/7 天统计量、目标时刻的日内/周内/年内周期与周末标记。
- 不使用未来实测天气、未来真实负荷、整个数据集拟合的缩放器。周末只代表星期六/日，尚未加入中国节假日及调休；未来气象预报也未接入。
- 全数据按时间顺序 70% 训练 / 15% 验证 / 15% 测试。验证期再按时间均分：前半用于早停及选模，后半仅用于区间校准。每个窗口所有目标点均位于本分区；上下文可来自以前分区。
- 默认使用**所有整点预测起点**（`--stride 4`，4 个 15 分钟步长），并非每个 15 分钟起点。每个起点都有完整 96 个训练目标，不随机截取窗口；`--stride 1` 可切换全部 15 分钟起点。该计算量设置记录在配置及分区报告中。推理始终支持任意 15 分钟起点。
- XGBoost 默认 hist 树、深度 6、学习率 0.05、最多 500 轮、验证 RMSE 连续 30 轮未改善早停、恢复最佳树数、4 CPU 线程、seed 42。按验证 RMSE 与持久性、前日同刻、前周同刻、近 7 日同刻均值比较。基线胜出时也保留已训练的 XGBoost。
- 用独立校准期逐提前步长绝对残差的有限样本顺序统计量构造名义 80% 经验区间，预测和下界截到 0。滚动窗口相关、分布漂移下不保证 80% 覆盖，实际覆盖率在留出测试报告中记录。
- 完整测试按所有有效 stride 起点评估，同一目标时刻在不同起点下重复计分；`evaluation.json` 只评估末日 96 点。测试数据不用于训练、早停、选模或校准。
- 每轮日志记录训练/验证 RMSE(kW)、MAE(kW)、R²、MSE(kW²)、耗时；常数标签 R² 为 null / N/A。图片仅为 loss 和负荷对比，不生成指标曲线或柱状图。

## 安装与完整运行

在本目录执行：

```powershell
python -m pip install -r requirements.txt
python 运行.py
```

`运行.py` 是和光伏包 `运行.py` 对应的独立验证入口，不需要先调用 CLI。它会依次完成数据检查、XGBoost 训练、验证集选模、区间校准、留出测试、未来 24 小时预测和末日回测。终端与 `training.log` 每轮显示训练/验证 RMSE、MAE、R²、耗时；结束时再次显示选中模型的测试 RMSE、MAE、R² 和区间覆盖率。运行结果目录中的 `visualizations/` 会生成四张 PNG：`training_loss.png`、`test_comparison.png`、`forecast.png`、`last_day_backtest.png`。

```powershell
# 默认完整验证（全年数据、最多500轮，验证集早停）
python 运行.py

# 快速自验证：仍走完整流程，但减少轮数和预测起点数量；结果不代表正式完整训练
python 运行.py --rounds 15 --patience 5 --stride 24 --output-dir .\运行结果\self_verify
```

脚本返回码为 0 表示预测、末日回测和图片生成均成功；返回码 2 表示数据、模型或评估失败。`verification.json` 保存最终状态，`failure.json` 保存训练异常原因。模型即使被基线胜出，XGBoost 权重仍会保存并单独报告指标。

本机已知解释器为 `D:\Anaconda3\anaconda\python.exe`。Windows/Linux 默认使用官方 CPU 包 `xgboost-cpu`（导入名仍为 `xgboost`）。本机可隔离安装到本模块 `.deps` 目录：`python -m pip install --target .deps --no-deps 'xgboost-cpu>=3.0,<4'`，前提是解释器已有 numpy/scipy。模块只读取本地依赖，不会运行时联网安装。

```powershell
# 调试参数会明确写入配置；不等同于默认完整训练
python 运行.py --rounds 30 --stride 24
# 训练总负荷参考模型，使用独立结果目录
python 运行.py --target total_load_kw
```

每次默认创建 `运行结果/<时间戳>/`；拒绝写入非空模型目录，以保留已有证据。

| 文件 | 内容 |
| --- | --- |
| `xgboost.json` | 最佳轮数模型，XGBoost 原生格式 |
| `training.log`、`training_history.csv/.json` | 逐轮物理单位指标和 MSE，每轮 CSV 落盘 |
| `training_config.json`、`data_split.json` | 完整参数、输入口径、分区范围/窗口数 |
| `training_summary.json` | 实际轮数、最佳轮数、保留树数、耗时、版本 |
| `model_metadata.json` | 模型/特征/站点/单位/模拟来源、源数据与模型 SHA-256、候选状态 |
| `interval_calibration.json` | 与选中模型一致的独立校准残差区间 |
| `metrics.json` | 全部模型验证和测试指标，选中模型、XGBoost 独立成绩 |
| `forecast.json`、`evaluation.json` | 数据结束后未来 24 小时预测和末日回测 |
| `visualizations/*.png` | loss、测试比较、历史与预测衔接、末日回测 |
| `verification.json` / `failure.json` | 完整入口验收 / 训练失败原因 |

## 只读推理与回放

```powershell
python -m load_forecast.cli train --input data.csv --output-dir model_bundle
python -m load_forecast.cli predict --input recent.csv --bundle-dir model_bundle --output forecast.json
python -m load_forecast.cli evaluate --input labelled.csv --bundle-dir model_bundle --output evaluation.json
python -m load_forecast.cli predict --input recent.csv --bundle-dir model_bundle --output replay.json --as-of 2026-01-01T00:07:00+08:00
```

CLI 从模型包读取站点、目标与单位，避免推理时偷偷换口径。`--forecast-start` 指定任意 15 分钟起点；始终只读取该起点之前**已经完成**的区间，即使输入文件中还有未来标签。`--as-of` 是显式回放时钟，要求预测起点为其所在 15 分钟边界，且历史恰好覆盖至边界；过期输入不外推填空。预测起点不得早于模型训练、选模、校准数据截止时间。

预测记录的 `forecast_end_at` 是末区间的**排他终点**。生成时间与目标时间分别记录，默认 `execution_mode=offline_replay`、`publication_status=candidate`、`decision_eligible=false`；2025 年数据输出的 2026-01-01 预测不是当前实时预测。缺模型、错误站点/单位、长缺口、来源冲突、校准身份不符都输出明确 `unavailable`，不暗中换模型继续使用不匹配区间。CLI 对不可用/失败返回退出码 2。

在线推理不重新训练。未来滚动流程可重复传入最新 7 天历史调用 `predict_record`，但资源边界、优化、自动定时重算和模型发布/监控尚未实现。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试包含真实 XGBoost 训练和权重重载、两种选模路径、测试标签扰动不改变权重/校准、未来标签不影响预测、跨分区窗口、非零点起点对齐、缺口/重复/负数/单位/站点/时效、常数 R² 与图像失败处理。测试中的小型合成数据仅用于验证软件行为。

## 2026-09-28 实际运行记录

21 项测试全部通过。黄花站全年模拟数据完整运行保存在 `运行结果/20260928_232203_101506/`，默认参数未限制训练数据。5,941 个整点训练起点形成 570,336 个“起点×提前步长”样本；选模/校准各 634 个起点，留出测试 1,291 个起点。第 180 轮早停，保留第 150 轮，验证选中 XGBoost。

| 独立测试指标 | XGBoost | 验证选出的最佳基线（前周同刻） |
| --- | ---: | ---: |
| RMSE (kW) | 25.7098 | 38.3800 |
| MAE (kW) | 21.2282 | 30.6771 |
| R² | 0.96592 | 0.92405 |
| 名义 80% 区间实际覆盖率 | 88.85% | 93.54% |

默认完整运行、原生权重重载及 CLI `--as-of 2026-01-01T00:07:00+08:00` 只读回放通过。已检查 loss、完整测试参考、预测与末日回测四张 PNG。结果仅说明合成场景上的性能，不等同真实场站预测精度；模型保留候选状态。

## 2026-10-27 复验与平台适配检查

已按根目录交接文档重新验证。发现并修复当前Windows中文路径下XGBoost权重重载编码错误，改用Python读取原始字节再加载，保留SHA-256校验；同时加强模型包和输出记录的单位、时间、口径、区间及损坏结构校验。25项测试通过。

全年默认配置完整重新训练，180轮早停、最佳150轮，权重哈希与上述原包完全相同；完整测试RMSE25.7098 kW、MAE21.2282 kW、R²0.965920复现。独立校准分位数也逐项复现。补充全部5,161个15分钟起点，RMSE25.7174 kW，前1小时RMSE25.5595 kW。仍然只是已经查看过的2025模拟数据复验，没有新增真实站点精度证据。

平台环境启动独立数值进程调用原模型的14项隔离检查通过，验证业务窗口重叠、既有充电叠加、站点显式仿真绑定、资源/编排/评价/本地校验一致及故障阻断。**正式平台仍未接入**；`forecast_status=not_used`。模型的base排除充电，平台legacy的base约定含既有负荷，不能直接替换字段。

详见[验证报告](开发验证/20261027_review/验证报告.md)、[算法数值证据](开发验证/20261027_review/evidence/algorithm_verification.json)和[平台接入约定](../../station_platform/docs/负荷预测适配验证与接入约定_v0.1.md)。原数据与历史模型包未修改，新训练和运行证据均在新目录保存。
