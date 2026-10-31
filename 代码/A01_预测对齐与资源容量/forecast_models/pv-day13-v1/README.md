# 光伏当天小时预测

当前默认流程（2026-10-26）参考 `参考代码文档/光伏/主代码.py`：用前三天的功率，以及当天的辐照和降雨，预测当天 **07:00–19:00（含两端）**。算法仍独立于场站平台。

| 项目 | 当前约定 |
| --- | --- |
| 时间颗粒度 | 60分钟 |
| 每天行数 / 输出点数 | 13（7、8、…、19时） |
| 历史输入 | 前三个连续自然日各13点功率，共39×1，单位kW |
| 未来输入 | 当天13×4：`SWGDN、DHI、DNI、rain` |
| 目标 | 当天13点功率，输出kW |
| 每个样本 | 一天；N个完整连续日产生N−3个样本 |

辐照沿用参考代码的三个分量：SWGDN（水平总辐照）、DHI（散射）、DNI（直射），单位W/m²；rain为对应小时的累计降雨mm。历史分支不使用气象、PREV、日历或太阳位置。未来分支不使用功率标签、温度、云量或额外特征。

## 数据格式

CSV或Excel包含以下列；也自动识别参考工作簿的 `time` / `power` 列名。其他列名可通过CLI的 `--source-timestamp-column` / `--source-target-column` 指定。

```text
timestamp,pv_power_kw,SWGDN,DHI,DNI,rain
2025-01-01T07:00:00+08:00,...,...,...,...,...
...
2025-01-01T19:00:00+08:00,...,...,...,...,...
2025-01-02T07:00:00+08:00,...,...,...,...,...
```

每13行必须对应同一个自然日的07:00–19:00整点。先按时间排序，再校验日期、小时、重复点、有限值和非负值。缺行、缺日、重复时间或不足13行的尾日会报错；不会删行后重新拼组、截断尾部、补零或跨夜插值。训练可保留完整24小时的小时表，系统只选07:00–19:00；15分钟表必须先显式转换。

### 从原模拟数据转换为小时表

在 `代码/光伏` 目录运行：

```powershell
python -m pv_forecast.cli prepare-day13 `
  --input "..\数据\光伏数据\黄花站_2025年_15分钟光伏模拟数据.csv" `
  --output "..\数据\光伏数据\新的小时数据.csv" `
  --source-interval-minutes 15
```

功率和辐照取 `[小时起点, 下一小时)` 的平均值；降雨累加为mm/小时区间，不能取均值。时间标签为区间起点，19:00行对应19:00–20:00的小时平均功率。若来源已是小时数据，无需转换。转换器拒绝不完整小时，并拒绝覆盖现有输出。

已生成 `代码/数据/光伏数据/黄花站_2025年_07-19_小时数据.csv`：365天、4,745行。原始15分钟数据和原模拟器未修改。

## 训练

```powershell
python 运行.py
```

默认使用上述全年小时模拟数据，实际训练双输入GRU，最多50轮，带学习率调度、早停和最低验证损失权重保存，所有训练日均参与，不提供限样本捷径。输出保存在新的 `运行结果/day13_<时间>/`。默认末日预测是历史回测示例，并非计算机当天的实况预报。

自定义输入及训练参数：

```powershell
python 运行.py --input "合并后的小时数据.xlsx" --epochs 50 --units 32 --batch-size 32
```

也可通过统一CLI训练，默认采样模式已经是 `day13_07_19`：

```powershell
python -m pv_forecast.cli train `
  --input "..\数据\光伏数据\黄花站_2025年_07-19_小时数据.csv" `
  --output-dir "运行结果\新的当天小时模型"
```

按自然日70%/15%/15%划分训练、验证、测试；验证再按时间分为前半早停/故障基线选择、后半独立区间校准。365天示例为252个训练样本、27个早停样本、28个校准样本、55个测试样本。第一个样本从第4天开始；验证和测试的历史可取分区之前的三个自然日，目标13点始终在同一天，不跨分区。训练、历史、未来三个缩放器仅拟合训练数据。GRU权重不会依据测试指标替换或调参。另保留13点基线及其独立区间作为缺气象或缺权重时的故障回退。

Python接口：默认 `train_model(source, output_dir)` 使用新流程；自定义参数请传 `pv_forecast.day13.Day13TrainingConfig`。显式旧 `TrainingConfig` 对象保留历史连续15分钟实验行为，避免重解释旧权重。

### 当天气象的可获得性

参考代码把目标日同一表里的 `SWGDN/DHI/DNI/rain` 作为已提供的未来输入。默认保持这一离线用法，元数据和预测记录明确标注 `provided_target_day_weather`。这些输入若来自事后实测，所得精度只是“当日气象已知”条件下的回顾结果，不能当作实际气象预报精度。

实际预测可在训练和预测时提供已经发布的小时天气预报归档：

```powershell
python -m pv_forecast.cli train --input "小时功率数据.csv" `
  --future-weather-input "小时预报归档.csv" --output-dir "运行结果\新的预报模型"
```

归档需含 `issued_at,timestamp,SWGDN,DHI,DNI,rain`，每个发布时间是一套完整的13小时预报。系统只取目标日07:00前（含07:00）发布的最新一套，不拼接不同批次，不使用07:00之后才发布的预报。用预报归档训练的包在预测/评价时也要求归档，不能悄悄改用同步实测。默认离线包可接收同样格式的已发布预报，但其训练仍基于离线气象，应另验证预报误差的影响。

## 预测当天07:00–19:00

```powershell
python -m pv_forecast.cli predict `
  --input "最近三天功率和当天小时气象.csv" `
  --bundle-dir "运行结果\新的当天小时模型" `
  --output "运行结果\当天预测.json"
```

省略日期时使用配置时区的计算机当天，固定输出当天07:00、08:00、…、19:00，不从历史最后一行向后滑动。历史输入必须包含当天之前三个自然日的39点功率。也可用 `--forecast-start "YYYY-MM-DD"` 或该日07:00指定目标日，其他小时起点会被拒绝。

当日气象可以与历史功率放在同一表中，也可使用 `--future-weather-input "当日气象.csv"` 单独提供；单独气象表不需要功率列。当日及以后的功率标签始终不进入模型。若使用预报模式，单独文件必须有发布时间。

缺少三天完整功率时返回 `unavailable`，不伪造输出；主模型或未来气象缺失时返回 `degraded`，输出明确标注13点基线及其自己的区间。正式全量回测要求主模型可用。新预测图片标题为“当天07:00–19:00光伏功率预测”。

## 评价与产物

```powershell
python -m pv_forecast.cli evaluate --input "小时数据.csv" `
  --bundle-dir "运行结果\新的当天小时模型" --output "运行结果\小时评价.json"

python -m pv_forecast.cli backtest --input "小时数据.csv" `
  --bundle-dir "运行结果\新的当天小时模型" --output "运行结果\按天回测.json"
```

评价按完整目标日统计，每天13点，只用前三天历史；报告整体、逐日、逐小时RMSE、MAE、R²。RMSE只对MSE开方一次，修正参考代码中重复开方的问题。传入整个训练源文件的评价会包含训练日，不能称为独立留出测试；独立留出分数见训练产物 `metrics.json` 中的 `test`。旧15分钟分数与新小时/已知气象分数不能直接比较。

主要产物：`model_metadata.json`、`gru.keras`、`scalers.joblib`、`training_config.json`、`data_split.json`、`training_history.csv/json`、`training_summary.json`、`interval_calibration.json`、`metrics.json`、`test_predictions.csv`、预测及训练图片。演示入口还会保存 `forecast.json` 和 `evaluation.json`。所有输出仍为candidate，未作实测场站验收。

## 当前项目的适配状态

当前独立光伏组件已适配上述13点需求，但**尚未接入 `station_platform` 的能源资源调用链**。平台的 `resource_capacity` / `resources` 仍返回 `forecast_status=not_used`，使用未来1小时、4个15分钟时段的模拟资源；负荷模型仍输出24小时、96个15分钟点。把13点预测记录放进快照，也不会自动改变平台的资源计算。

因此，新光伏输出不能直接替换平台的四时段 `pv_available_kw` 数组。后续接入需要显式确定时间窗口、小时均值与15分钟量的换算/保守策略、与负荷的时间对齐以及缺少覆盖时的处理；不能直接复制数组或把缺失时段补零。还需绑定实际站点范围，验证可提前获得的天气，建立候选模型发布和回退记录。当前候选的 `asset_scope=configured_pv_assets`、`model_published_at=null` 属于独立演示状态。

平台专用环境没有安装numpy/pandas/scikit-learn/TensorFlow。建议后续通过独立推理环境的只读接口接入，保持平台既有依赖边界；本轮未修改平台角色、真实配置、调度资源或执行链路。实际验证与适配结论见 `开发验证/20261026_project_review/`。

## 历史15分钟实验

旧包和旧数据均保留，`predict/evaluate/backtest` 根据包的采样模式自动分派。重现旧模型训练时必须显式指定 `--sampling-layout continuous_15min`；`correct-point/recalibrate/compare-physics` 属于旧实验接口，不用于新13点包。历史方法和结果见 `开发验证/20261014_physics_01/source_before_physics`、`开发验证/20261014_physics_01/开发记录.md` 及 `开发验证/20261026_day13/source_before/README.md`，以对应阶段文件为准。

## 测试

```powershell
python -m unittest discover -s tests -v
```

新测试覆盖39/13点形状、自然日对齐、缺日/尾行/重复点拒绝、小时降雨单位、只用历史功率、目标功率隔离、预报发布时间、实际GRU训练重载、kW还原、缺输入/权重回退、默认CLI及按天回测。完整验证记录见 `开发验证/20261026_day13/`。

