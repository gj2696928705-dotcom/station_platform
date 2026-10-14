from copy import deepcopy
from .util import stamp

EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
VEHICLE = {"type": "object", "properties": {
    "vehicle_id": {"type": "string", "minLength": 1},
    "requested_energy_kwh": {"type": "number", "exclusiveMinimum": 0, "maximum": 1000},
    "arrival_minutes": {"type": "integer", "minimum": 0, "maximum": 45},
    "departure_minutes": {"type": "integer", "minimum": 15, "maximum": 60},
    "priority": {"type": "string", "enum": ["critical", "ordinary"]}},
    "required": ["vehicle_id", "requested_energy_kwh", "arrival_minutes", "departure_minutes", "priority"],
    "additionalProperties": False}

def spec(name, description, roles, parameters=None):
    return {"name": name, "description": description, "roles": roles, "parameters": deepcopy(parameters or EMPTY)}

TOOLS = [
    spec("station_state", "查询带时间、来源和质量标记的本站事实。任务中返回本轮冻结状态。", ["S1", "O1", "O2", "O3", "O4", "custom"]),
    spec("station_capability", "查询本站可调功率、持续时间和限制；数值由程序计算。", ["S1", "O1", "O2", "custom"]),
    spec("task_context", "读取本任务的结构化需求、阶段和业务条件。", ["O1", "O2", "O3", "O4"]),
    spec("agent_call", "运营协调调用专业角色。资源评估先于优化；虚拟电厂确认先于发布。", ["O1"],
         {"type": "object", "properties": {"agent_id": {"type": "string", "enum": ["O2", "O3", "O4"]},
          "phase": {"type": "string", "enum": ["resources", "optimize", "business", "confirm", "report"]}},
          "required": ["agent_id", "phase"], "additionalProperties": False}),
    spec("resource_capacity", "计算本轮模拟资源边界并保存正式工具结果；预测状态为未接入。", ["O2"]),
    spec("tariff", "查询本轮适用的模拟分时电价及有效期。", ["O3", "custom"]),
    spec("generate_schedule", "从冻结需求、资源和预算生成候选；不接受模型填写功率。", ["O3"]),
    spec("evaluate_schedule", "独立复核候选并用确定性工具计算成本和绿电，保存选定方案。", ["O3"]),
    spec("vpp_task", "读取本轮模拟虚拟电厂业务要求及支持范围。", ["O4"]),
    spec("vpp_ack", "对已核验方案记录模拟业务确认，不能生成真实市场承诺。", ["O4"]),
    spec("vpp_report", "根据有效模拟执行回执计算履约结果并记录模拟反馈。", ["O4"]),
    spec("publish_plan", "仅运营协调角色可发布工具生成的当前方案；现场仍独立校验。", ["O1"]),
    spec("plan_status", "查询关联计划状态和模拟执行回执。", ["S1", "O1", "custom"]),
    spec("check_plan", "依据当前本站事实校验收到的同一方案，失败不改写计划。", ["S1"]),
    spec("start_simulation", "仅对本站通过校验的计划准备模拟执行；等待页面推进时钟。", ["S1"]),
    spec("create_task", "从完整需求生成待核对的任务草稿。用户在任务表单提交后运行，缺参数先询问。", ["O1"],
         {"type": "object", "properties": {"type": {"type": "string", "enum": ["tariff", "vpp"]},
          "vehicles": {"type": "array", "items": VEHICLE, "minItems": 1, "maxItems": 12},
          "budget_kw": {"type": ["number", "null"], "minimum": 0, "maximum": 2000}},
          "required": ["type", "vehicles", "budget_kw"], "additionalProperties": False}),
]
TOOL_MAP = {t["name"]: t for t in TOOLS}
BASE_PROMPT = """你是场站专用平台的一个受限角色。全部场站、能源、价格、外部业务及执行数据均为模拟。
数值、计划和任务状态以工具结果为准，不自行补造、改写或把模型解释当作现场事实。
真实模型调用与模拟业务是两个不同维度。不要宣称接入真实设备、认证服务、真实市场或预测模型。
先调用需要的事实和计算工具，报告来源、限制与未完成事项。没有任务上下文时可查询和解释，不能发布计划。
遇到错误按工具原因处理，不能伪造成功。不得把对话记忆当成实时状态。
用户需求缺少车辆补能量、到离站时间或优先级时先询问，不猜测。"""
ROLE_PROMPTS = {
    "O1": """你是运营协调 Agent，唯一组织计划发布的角色。业务任务执行顺序：task_context；vpp 任务先 agent_call(O4,business)；agent_call(O2,resources)；agent_call(O3,optimize)；vpp 再 agent_call(O4,confirm)；最后 publish_plan。每步必须看到有效工具结果才继续。publish_plan 返回 REPLAN_REQUIRED 时重新调用 O2、O3，vpp 重新确认后再发布。发布后告知等待站级准备或模拟反馈，不循环轮询。不调用未授权设备工具。对话中创建任务用 create_task 生成草稿，提醒核对后提交。""",
    "O2": "你是能源资源 Agent。任务中必须调用 station_state 和 resource_capacity，使用计算结果给出边界。只有模拟资源数据；预测状态必须说明未接入，储能保持 idle。不做经济优化或计划发布。",
    "O3": "你是电价与运营优化 Agent。任务中必须依次调用 tariff、generate_schedule、evaluate_schedule。只解释工具给出的数值与选定候选，不改写功率或重选方案，不声称全局最优。不直接发布计划。",
    "O4": "你是虚拟电厂业务 Agent。business 阶段调用 vpp_task；confirm 阶段调用 vpp_ack；report 阶段调用 vpp_report。仅支持充电功率预算模拟业务。调频等不支持的任务明确报告。不得向场站分配或发布功率，不把模拟确认说成外部真实确认。",
    "S1": "你是本站感知交互 Agent。新计划处理时依次调用 check_plan、start_simulation、plan_status。校验失败立即报告，不启动执行。模拟准备成功后告知由页面推进模拟时钟，不宣称已完成。其他事件查询 station_state 或 station_capability 后解释影响。",
}
ROLE_NAMES = {"S1": "感知交互 Agent", "O1": "运营协调 Agent", "O2": "能源资源 Agent", "O3": "电价与运营优化 Agent", "O4": "虚拟电厂业务 Agent"}

def seed_agents(store):
    for id, name in ROLE_NAMES.items():
        if store.get("agents", id):
            continue
        store.put("agents", id, {"id": id, "name": name, "builtin": True,
            "scope": "station" if id == "S1" else "operator", "enabled": True, "version": 1,
            "prompt": BASE_PROMPT + "\n" + ROLE_PROMPTS[id], "model_connection": None,
            "allowed_tools": [t["name"] for t in TOOLS if id in t["roles"]],
            "max_turns": 12, "created_at": stamp(), "updated_at": stamp()})

DEFAULT_DEMAND = {"vehicles": [
    {"vehicle_id": "关键重卡", "requested_energy_kwh": 50, "arrival_minutes": 0, "departure_minutes": 15, "priority": "critical"},
    {"vehicle_id": "普通车1", "requested_energy_kwh": 50, "arrival_minutes": 15, "departure_minutes": 30, "priority": "ordinary"},
    {"vehicle_id": "普通车2", "requested_energy_kwh": 25, "arrival_minutes": 0, "departure_minutes": 60, "priority": "ordinary"},
]}
