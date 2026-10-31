"""Write final evidence-based note and update only the PV handoff/README."""
from pathlib import Path
import json
import shutil

out = Path(__file__).resolve().parent
pv = out.parents[1]
workspace = pv.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


audit = read(out/"evidence_audit.json")
aggregate = read(out/"frozen_three_paths"/"aggregate.json")
comparison_dir = Path(read(out/"active_comparison.json")["path"]) if (out/"active_comparison.json").exists() else pv/"运行结果"/"20261014_physics_comparison_01"
comparison = read(comparison_dir/"comparison_report.json")
diagnosis = read(out/"control_error_diagnosis.json")
assert audit["status"] == "passed" and comparison["status"] == "completed"
assert read(out/"candidate_cli_verification.json")["status"] == "passed"
labels = {"20261009_point_correction_01": "现有GRU＋Ridge", "enhanced_gru": "增强GRU",
    "physics_tree": "物理残差树", "clear_persistence": "晴空比例持久性"}
control = aggregate["models"]["20261009_point_correction_01"]
recommendation = aggregate["recommended_bundle"]
proposed = aggregate["proposed_primary_bundle"]
accepted = aggregate["stable_point_accuracy_gain"]
outcome = (f"预先按滚动验证选中的{labels[proposed]}通过三条新轨迹的预设稳定性检查，可作为新的模拟候选。"
    if accepted else "本轮对照未满足预设的稳定点预测收益条件，继续保留现有GRU＋Ridge候选。")
notes = ["# 光伏预测诊断与物理特征对照开发记录（2026-10-15）", "", outcome,
    "", "已实现细分诊断、独立理想气象重建、晴空物理特征/基线、梯度提升树和增强GRU对照；完成三段×三配置GRU和一次最终GRU的全部窗口训练，以及先冻结后生成的三条全年模拟轨迹。所有模型仍为candidate，结果只适用于当前已知模拟器。", "",
    "## 已验证的误差位置", "", "先用已经查看过的2026年 seed=20261010 重新诊断现有控制：", "",
    "| 白天提前量 | RMSE (kW) | MAE (kW) | R² | 均值偏差 (kW) |",
    "| --- | ---: | ---: | ---: | ---: |"]
current_d = diagnosis["models"]["20261009_point_correction_01"]["error_diagnostics"]
for lead, score in current_d["daylight_by_lead"].items():
    notes.append(f"| {lead} | {score['rmse']:.2f} | {score['mae']:.2f} | {score['r2']:.4f} | {score['mean_error_kw']:.2f} |")
peaks = current_d["peaks"]
notes += ["", f"峰值平均偏差{peaks['mean_peak_error_kw']:.2f} kW，峰值MAE{peaks['peak_mae_kw']:.2f} kW，峰时MAE{peaks['timing_mae_minutes']:.2f}分钟。白天归一化曲线形状RMSE{current_d['normalized_daylight_shape']['rmse']:.4f}，15分钟变化率RMSE{current_d['daylight_ramps_kw_per_15min']['rmse']:.2f} kW/15分钟；天气切换时变化率RMSE{current_d['weather_transition_ramps_kw_per_15min']['rmse']:.2f}。", "",
    "天气分组的均值偏差定义为预测减真值。短提前量较好，长提前量晴天低估、阴雨高估；不能仅由全天R²判断发电时段效果。所有提前量按起点后半开区间计分，0–1小时为步0–3，12–24小时为步48–95。未来天气类型只用于事后评分。", "",
    "## 理想未来气象诊断", ""]
ideal = diagnosis["ideal_future_weather_diagnostic"]["error_diagnostics"]["daylight"]
notes += [f"同一已查看路径，用目标期同步SWGDN、DHI、DNI和温度独立进行物理重建，白天RMSE{ideal['rmse']:.3f} kW、MAE{ideal['mae']:.3f} kW、R²{ideal['r2']:.6f}。", "",
    "这支持未来天气信息可能具有很高价值，但不能证明可获得的预报也能达到该分数，更不能断言当前网络表达能力足够。物理参数来自同一个已知模拟配置，重建式与模拟器物理结构接近，功率噪声很小；因此这一结果带有模拟结构优势。太阳几何采用独立UTC计算，与生成器本地时间近似存在微小差别。它不是实际预测、不是严格数学上限，不进入模型候选、权重拟合、校准或采用决定；加载诊断包为预测模型会失败。", "",
    "## 实现与实验约束", "",
    "- 历史仍为288个15分钟点，未来仍为96点、kW。原4维历史功率/辐照/温度/云量增加9个派生特征；原6维未来日历/太阳特征只增加晴空功率与辐照。所有观测均早于起点，不新增实际未来气象输入。",
    "- 晴空功率使用配置倾角/方位、Haurwitz辐照、倾斜面换算、NOCT温升和温度系数/损耗，参考环境温度20°C。归一化分母下限为50 kW与20 W/m²，日出/夜间的非零传感器值仍得到有限特征。手算水平1000 W/m²案例3938 kW、温度增加10°C后3762 kW、容量截断5000 kW均验证通过。",
    "- 树模型学习晴空功率的kW残差，共享提前量，全部白天训练标签参与，固定200轮、LR0.05、31叶、L2=10，不启用随机内部验证。",
    "- GRU完整保留32固定LR、32调度LR、16调度LR三组配置；三段×三配置各重新拟合训练缩放器、初始化网络、记录每轮、调度/早停/保存实际最低验证损失权重。批量回测1024只影响内存，不限制样本。",
    "- 原GRU＋Ridge控制复用九组完整历史训练证据中的已选GRU权重和固定alpha=0.1；三段控制结果与旧记录数值对照通过。原包、原数据、生成器及配置哈希不变，旧源码已快照并对照原冻结协议。", "",
    "## 三段滚动验证", "",
    "| 候选 | 合并白天RMSE (kW) | MAE (kW) | R² | 最差段RMSE (kW) |",
    "| --- | ---: | ---: | ---: | ---: |"]
for name, values in comparison["aggregate"].items():
    s = values["daylight"]
    notes.append(f"| {name} | {s['rmse']:.2f} | {s['mae']:.2f} | {s['r2']:.4f} | {values['worst_fold_daylight_rmse']:.2f} |")
notes += ["", "5月调度增强GRU虽改善合并指标，0–1小时RMSE却从控制203.12升至432.62 kW，1–3小时也退化，收益主要在6小时之后。基于这一开发分区反例，在任何新轨迹生成前补充了0–1/1–3小时不退化的采用条件；未改变候选权重、训练或参数。"]
notes += ["", "本轮比较的是完整候选：控制含Ridge点修正，增强GRU直接输出并进行物理边界处理，因而不是只改变物理特征的单因素实验。短时退化可能与缺少点修正层有关，当前证据不能把它全部归因于物理特征或GRU表达能力。下一轮可验证在增强GRU保留相同修正层，或按开发分区结果组合短时/长时模型；应另留新的冻结评估轨迹。"]
notes += ["", f"滚动按合并白天RMSE、再MAE选中 `{comparison['selected_candidate']}`；最佳增强GRU配置为 `{comparison['best_enhanced_gru']}`。全部误差按标签累计，不简单平均各段RMSE/R²。5/7/9月的训练窗口为11137/16993/22945，验证为2881/2881/2785。", "",
    "树模型5月出现训练误差下降而验证误差上升的反例；固定200轮验证RMSE1258.59 kW，高于控制672.51 kW，说明这次树特征/参数组合不能有效跨月份泛化，不能把它外推为所有树模型均无效。没有根据最终新轨迹调参。", "",
    "最终全部24145个训练窗口拟合树与最佳增强GRU。验证段前半2533窗口用于最终GRU早停及故障基线选择；后半2533窗口只校准主模型/独立故障基线区间，两段互不重叠，跨中间边界95个窗口排除。选择与校准冻结后才构建原2025年5161个测试窗口，它们仍是已查看的回顾性证据。", "",
    "| GRU拟合 | 全部训练窗口 | 实际轮数 | 最佳轮次 |",
    "| --- | ---: | ---: | ---: |"]
for fit in audit["full_GRU_fits"]:
    notes.append(f"| {fit['path'].split('/experiments/')[-1] if '/experiments/' in fit['path'] else '最终增强GRU'} | {fit['training_windows']} | {fit['epochs_completed']} | {fit['best_epoch']} |")
notes += ["", "## 三条先冻结后生成的全年轨迹", "",
    "协议在生成前冻结全部代码、四个比较包、校准与采用规则；保持同一模拟器及站点，只更换2026年随机种子2026101401/2026101402/2026101403。每条35040行、34657个完整起点，均无排除；三条合计103971窗口，每个模型9981216个重叠目标标签。重复生成逐字节一致，期间没有重新拟合、校准、选模或改参数。", "",
    "| 冻结模型 | 合并白天RMSE (kW) | MAE (kW) | R² | 最差月RMSE (kW) |",
    "| --- | ---: | ---: | ---: | ---: |"]
for name, values in aggregate["models"].items():
    s = values["overall"]["daylight"]
    notes.append(f"| {labels[name]} | {s['rmse']:.2f} | {s['mae']:.2f} | {s['r2']:.4f} | {values['worst_month_daylight_rmse']:.2f} |")
notes += ["", "| 种子 | 控制白天RMSE / MAE (kW) | 预选候选白天RMSE / MAE (kW) |", "| --- | ---: | ---: |"]
for seed, score in control["per_path"].items():
    other = aggregate["models"][proposed]["per_path"][seed]
    notes.append(f"| {seed} | {score['rmse']:.2f} / {score['mae']:.2f} | {other['rmse']:.2f} / {other['mae']:.2f} |")
notes += ["", "| 模型 | 晴天RMSE | 多云RMSE | 阴天RMSE | 雨天RMSE |", "| --- | ---: | ---: | ---: | ---: |"]
for name, values in aggregate["models"].items():
    notes.append("| "+labels[name]+" | "+" | ".join(f"{values['daylight_weather'][state]['rmse']:.2f}" for state in ("clear", "partly_cloudy", "overcast", "rain"))+" |")
notes += ["", "| 白天提前量 | 控制RMSE / MAE (kW) | 增强GRU RMSE / MAE (kW) | 控制R² | 增强GRU R² |",
    "| --- | ---: | ---: | ---: | ---: |"]
for lead, score in control["daylight_by_lead"].items():
    candidate_score = aggregate["models"]["enhanced_gru"]["daylight_by_lead"][lead]
    notes.append(f"| {lead} | {score['rmse']:.2f} / {score['mae']:.2f} | {candidate_score['rmse']:.2f} / {candidate_score['mae']:.2f} | {score['r2']:.4f} | {candidate_score['r2']:.4f} |")
notes += ["", "增强GRU在三条路径各自的全天发电时段RMSE、MAE均改善；合并白天RMSE下降约1.10%，MAE下降约2.81%，但0–1小时RMSE从207.31增至211.72 kW（约2.13%），因此不能称为无局部退化的稳定收益。", "",
    "采用只允许预先滚动选中的候选，并检查每条路径白天RMSE/MAE均下降、合并晴天/多云RMSE不退化、最差目标月RMSE不退化，以及0–1/1–3小时RMSE和MAE不退化；不能查看最终分数后改选其他候选。检查结果：", ""]
for condition, satisfied in aggregate["adoption_conditions"].items():
    notes.append(f"- `{condition}`：{'通过' if satisfied else '未通过'}。")
notes += ["", outcome, "",
    "合并结果用完整精度的RMSE恢复SSE、MAE恢复SAE，并按实际重复float64标签累计sum/square重算R²；逐路径R²与独立标签统计对照通过。原冻结汇总脚本误用float32核对标签，天气分组R²出现3.41×10⁻⁸差异并触发原3×10⁻⁸容差检查。保留原脚本字节与全部回测报告，另用 `aggregate_float64_correction.py` 按真实回测双精度标签核对；276组R²最大差异1.07×10⁻¹⁴，无需放宽容差。只修复独立汇总统计，不改原预测、模型、数据或采用条件，证据为 `aggregation_float64_verification.json`。置信区间覆盖和宽度另存在原回测报告中，不算点预测收益。三个新路径现在已查看，下一次开发须另留冻结后的新轨迹。", "",
    "## 下一阶段参考门槛与未知信息", "",
    "先以本轮控制为基准，要求白天RMSE与MAE均降低，晴/多云与最差月份不退化；重点改善12–24小时，并维持0–1小时短时效果。以下数值用于开发参考，不能在同三条路径上继续调参后声称独立通过：", "",
    "| 控制白天提前量 | RMSE (kW) | MAE (kW) | R² |", "| --- | ---: | ---: | ---: |"]
for lead, score in control["daylight_by_lead"].items():
    notes.append(f"| {lead} | {score['rmse']:.2f} | {score['mae']:.2f} | {score['r2']:.4f} |")
notes += ["", "当前证据尚不能保证保持现有输入就达到白天R²0.8或0.9。新增物理特征、具体树配置与三组GRU的对照可以检验这些候选，却不足以证明所有模型的最佳可能精度。理想气象重建支持天气信息瓶颈假设，实际收益还取决于真正可获得的预报误差。若后续允许增加预报，应使用 issued_at≤起点 的历史预报归档重新进行因果对照；本轮主实验保持原输入。", "",
    "尚无实测场站、未核验模拟参数到实站的适用性，重叠样本不独立，无生产精度或覆盖率保证。本轮没有修改场站平台、Agent、负荷算法或设备执行链路。", "",
    "路线参考的[IEA PVPS第四版太阳资源手册](https://iea-pvps.org/key-topics/best-practices-handbook-for-the-collection-and-use-of-solar-resource-data-for-solar-energy-applications-fourth-edition/)强调太阳辐射数据质量、时空变化及太阳能预测；该材料不提供本项目具体模型的收益保证，采用判断只依据上述可复现结果。", "",
    "## 验证与复现", "",
    f"完整回归61项通过，耗时{audit['full_regression']['seconds']:.3f}秒。另有手算物理边界核验、三个真实完整训练候选的predict/evaluate CLI、kW/夜间边界、保存重载、起点后非法未来观测/标签隔离、故障时各自基线及区间回退，以及12张功率PNG和10张GRU loss PNG核验。", "",
    "- `development_protocol.json` / `development_completion.json`：训练前冻结协议、全量完成。",
    "- `control_error_diagnosis.json`：细分提前量、天气、峰值/形状/变化率与独立理想气象。",
    "- `../../运行结果/20261015_physics_comparison_recovery_01/comparison_report.json`：完整滚动、最终训练/校准、回顾测试。",
    "- `frozen_three_paths/{protocol,completion,aggregate,target_moments}.json` 及各seed的 `backtest.json`：三路径冻结和完整统计。",
    "- `unit_tests.log`、`physical_boundary_verification.json`、`candidate_cli_verification.json`、`evidence_audit.json`：可执行核验结果。",
    "- `candidate_cli/`、`candidate_png_review.png`、`training_loss_review.png`：真实CLI记录与图片审核。", "",
    "从光伏目录运行 `diagnose` 与 `compare-physics`，命令见README；`开发验证/20261014_physics_01/run_full_comparison.py` 使用新的固定证据目录，不允许覆盖或盲目重跑。三路径生成入口 `run_frozen_cohort.py` 必须先完成比较与冻结，它同样拒绝覆写已查看数据。当前沙箱使用工作区临时目录，未减少训练样本。", ""]
(out/"开发记录.md").write_text("\n".join(notes), encoding="utf-8")
if (out/"recovery_completion.json").exists():
    recovery_note = "\n## 中断恢复记录\n\n实验于2026-10-14启动，2026-10-15核查时原训练进程已退出。原目录和第9组的20轮部分检查点全部保留；在新的恢复目录复用8组完成GRU和3组完成树模型，仅从头完整训练中断的第9组，并执行原定最终全量拟合。没有将部分训练当作完成，也未改变样本量、候选参数或未来信息规则。恢复协议、原产物哈希和完成记录见 `recovery_protocol.json`、`recovery_completion.json`。原单轮耗时日志含长时间停顿，不应作为纯计算性能数据。\n"
    with (out/"开发记录.md").open("a", encoding="utf-8") as handle:
        handle.write(recovery_note)
reference = {"status": "development_reference_only", "current_control": control["overall"]["daylight"],
    "current_control_daylight_leads": control["daylight_by_lead"],
    "next_phase_rule": "lower daylight RMSE/MAE; clear/partly-cloudy and worst month not worse; improve 12-24h and preserve 0-1h; validate on newly frozen paths",
    "universal_R2_target": None, "reuse_of_viewed_paths_as_unseen": False, "production_acceptance": False}
write(out/"next_phase_reference.json", reference)
short_table = ["| 模型 | 白天RMSE (kW) | MAE (kW) | R² |", "| --- | ---: | ---: | ---: |"]
for name, values in aggregate["models"].items():
    s = values["overall"]["daylight"]
    short_table.append(f"| {labels[name]} | {s['rmse']:.2f} | {s['mae']:.2f} | {s['r2']:.4f} |")
summary = "\n".join(["2026-10-15完整实验已结束。"+outcome, "", *short_table, "",
    "三条2026年seed=2026101401/2026101402/2026101403，在代码/模型/校准冻结后生成；每条34657个起点全部计分，合计103971窗口、每模型9981216个重叠标签，无最终数据调参。理想气象仅作诊断，扩宽区间不计作点精度收益。完整分提前量/天气/月度与采用条件见 `开发验证/20261014_physics_01/开发记录.md`、`frozen_three_paths/aggregate.json`；证据审计为 `evidence_audit.json`。原数据、模拟器、旧GRU＋Ridge与原九组权重哈希保持不变。"])
readme_path = pv/"README.md"
readme = readme_path.read_text(encoding="utf-8")
assert readme.count("## 测试\n") == 1
readme = readme.replace("## 测试\n", summary+"\n\n## 测试\n")
readme = readme.replace("点预测修正后的最新完整回归53项", "点预测修正阶段完整回归53项")
readme_path.write_text(readme, encoding="utf-8")
handoff_path = workspace/"项目交接文档.md"
backup = out/"handoff_before_final_results.md"
assert not backup.exists(), "preserve original handoff"
shutil.copy2(handoff_path, backup)
handoff = handoff_path.read_text(encoding="utf-8")
old_start = handoff.index("更新：")
old_end = handoff.index("## 1.")
handoff = handoff[:old_start]+"更新：光伏算法2026-10-15。根目录：`D:\\源网荷储\\codex\\2026.10.09\\场站智能体（4）`；当前无Git仓库。其他模块沿用原交接，平台最新状态见根README与station_platform/README。\n\n本次光伏完成细分误差/理想气象诊断、物理基线、树及增强GRU完整对照；三段×三配置与最终GRU均使用全部窗口，61项回归通过。"+outcome+" 原数据、旧模型和模拟器保留，尚未真实场站验收。最新完整证据见 `代码/光伏/开发验证/20261014_physics_01/开发记录.md` 与 `evidence_audit.json`。\n\n"+handoff[old_end:]
assert handoff.count("## 4. 正在开发的功能") == 1
handoff = handoff.replace("## 4. 正在开发的功能", "2026-10-15完成新一轮物理特征对照。"+outcome+" 完成三个滚动分区的九次增强GRU及一次最终全量GRU、四次完整树拟合、独立校准和三条先冻结后生成的全年新随机模拟轨迹（每条34657个起点、无排除）。比较包见 `代码/光伏/运行结果/20261015_physics_comparison_recovery_01/bundles/`；完整天气/月度/提前量数值见新开发记录，候选仍未发布。\n\n## 4. 正在开发的功能")
handoff = handoff.replace("最新修正在另一条冻结新模拟轨迹上改善整体与逐月白天RMSE，仍有晴/多云退化和条件区间欠覆盖。", "点修正的历史收益及晴/多云反例保留；2026-10-15新增诊断与物理模型对照已完成，采用结论以本轮冻结三路径报告为准。")
handoff = handoff.replace("recalibration,backtesting,point_correction,point_experiments,records,cli", "recalibration,backtesting,point_correction,point_experiments,physics,physics_tree,physics_experiments,diagnostics,records,cli")
handoff = handoff.replace("| `代码/数据/{光伏数据,负荷数据}/`", "| 光伏物理对照目录与 `开发验证/20261014_physics_01/` | 61项回归、十次完整GRU拟合、四次树拟合、物理手算边界、真实候选CLI/回退/PNG、三个冻结新全年路径及合并误差、推荐结论 |\n| `代码/数据/{光伏数据,负荷数据}/`")
handoff = handoff.replace("本轮独立光伏算法开发交付见 `代码/光伏/开发验证/20261009_point_correction/开发记录.md`。", "点修正历史交付见 `代码/光伏/开发验证/20261009_point_correction/开发记录.md`；2026-10-15新诊断与全量对照交付见 `代码/光伏/开发验证/20261014_physics_01/开发记录.md`，后续门槛参考 `next_phase_reference.json`。")
handoff_path.write_text(handoff, encoding="utf-8")
write(out/"completion_verification.json", {"status": "passed", "implemented_user_plan": True,
    "diagnostics_complete": True, "same_inputs_physical_baseline_and_nonlinear_comparison_complete": True,
    "complete_GRU_training_preserved": True, "three_paths_frozen_before_generation_and_fully_evaluated": True,
    "old_model_compatibility_reload_causality_units_fallback_verified": True,
    "readme_handoff_and_development_record_updated": True,
    "stable_point_gain": accepted, "recommended_bundle": recommendation,
    "next_phase_reference_ref": "next_phase_reference.json", "real_site_acceptance": False})
print(json.dumps({"status": "completed", "note": str(out/"开发记录.md"), "outcome": outcome}, ensure_ascii=False, indent=2))
