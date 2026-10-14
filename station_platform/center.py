import asyncio
import json
import re
import secrets
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .agent import AgentEngine
from .defaults import BASE_PROMPT, DEFAULT_DEMAND, ROLE_NAMES, TOOLS, TOOL_MAP, seed_agents
from .domain import capability, evaluation, request_from_site, resources, schedule, validate_payload
from .documents import AgentDocuments
from .http_support import install_errors, local_guard, require_node
from .models import Connections, ModelClient
from .security import SecretStore
from .storage import Store
from .util import ACTIVE, TERMINAL, ROOT, STATION_ID, PlatformError, canonical, digest, identifier, now, parse_time, public, redact, stamp

class CenterService:
    def __init__(self, data_dir, node_url, token, port=8780, vault=None, interval=2):
        self.store = Store(Path(data_dir)/"center.sqlite3")
        self.node_url, self.token, self.port, self.interval = node_url.rstrip("/"), token, port, interval
        self.vault = vault or SecretStore()
        self.connections = Connections(self.store, self.vault)
        self.model_client = ModelClient(self.vault)
        self.engine = AgentEngine(self.store, self.model_turn)
        self.browser_token = secrets.token_urlsafe(32)
        self.jobs, self.session_locks = {}, {}
        seed_agents(self.store)
        self.documents = AgentDocuments(self.store)
        self.documents.initialize_all()
        if not self.store.get("settings", "platform"):
            self.store.put("settings", "platform", {"default_connection": None, "task_timeout_seconds": 1800})

    async def start(self):
        self.client = httpx.AsyncClient(timeout=15, trust_env=False, headers={"Authorization": "Bearer "+self.token})
        for task in self.store.all("tasks"):
            if task["status"] in ACTIVE and task.get("plan_id"):
                task.update({"status": "WAITING_FEEDBACK", "message": "中心重启，恢复观察已发布计划；未重复发送"})
                self.store.put("tasks", task["id"], task)
            elif task["status"] in ACTIVE:
                task.update({"status": "INTERRUPTED", "message": "中心重启，任务中断，未自动重新执行"})
                self.store.put("tasks", task["id"], task)
        self.watch_job = asyncio.create_task(self.watch_loop())

    async def stop(self):
        self.watch_job.cancel()
        jobs = list(self.jobs.values())
        for job in jobs: job.cancel()
        await asyncio.gather(self.watch_job, *jobs, return_exceptions=True)
        await self.client.aclose()

    def station(self):
        record = self.store.get("nodes", STATION_ID)
        if not record: return {"station_id": STATION_ID, "online": False, "state": None, "simulation": True}
        record["online"] = (now()-parse_time(record["last_seen"])).total_seconds() <= 15
        return public(record)

    async def node_request(self, method, path, data=None):
        try:
            response = await self.client.request(method, self.node_url+path, json=data,
                timeout=self.store.get("settings", "platform")["task_timeout_seconds"] if path == "/internal/chat" else 15)
            if response.status_code >= 300:
                body = response.json()
                raise PlatformError(body.get("code", "NODE_ERROR"), body.get("message", "站端接口失败"), response.status_code)
            return response.json()
        except (httpx.HTTPError, ValueError):
            raise PlatformError("NODE_UNREACHABLE", "无法确认站端状态，请检查独立站级服务", 503) from None

    async def live_state(self):
        if not self.station()["online"]:
            raise PlatformError("NODE_OFFLINE", "本站心跳已离线，不能用过期能力规划", 409)
        return await self.node_request("GET", "/internal/state")

    async def model_turn(self, agent, messages, tools):
        connection = self.connections.get(agent["connection_ref"])
        result = await self.model_client.turn(connection, messages, tools)
        return redact(result, (self.vault.decrypt(connection["credential_cipher"]), self.token))

    def snapshot_for_node(self, agent, owner):
        ref = identifier("node-agent")
        self.store.put("node_snapshots", ref, {"id": ref, "owner": owner, "agent": deepcopy(agent)})
        return {**agent, "gateway_ref": ref}

    async def run_agent(self, agent, message, owner, phase="chat", task_id=None, history=None, expected_version=None):
        connection = self.connections.get(agent["connection_ref"])
        secret = self.vault.decrypt(connection["credential_cipher"])
        async def tool(name, args):
            return await self.tool(agent, name, args, owner, task_id, phase, expected_version)
        return await self.engine.run(agent, message, tool, owner, phase, history, secrets=(secret, self.token))

    def current_task(self, id, expected_version=None):
        task = self.store.get("tasks", id)
        if not task: raise PlatformError("TASK_CONTEXT_REQUIRED", "需要有效业务任务", 409)
        if expected_version is not None and task["version"] != expected_version:
            raise PlatformError("TASK_VERSION_CHANGED", "任务已换用新的输入版本", 409)
        if task["status"] not in ACTIVE:
            raise PlatformError("TASK_NOT_ACTIVE", "任务已结束或中断", 409)
        return task

    async def create_task(self, payload, submission_key):
        payload = validate_payload(payload)
        if not isinstance(submission_key, str) or not 1 <= len(submission_key) <= 200:
            raise PlatformError("SUBMISSION_KEY_REQUIRED", "任务提交需要唯一编号")
        fingerprint = digest(payload)
        prior = self.store.get("submission_keys", submission_key)
        if prior:
            if prior["fingerprint"] != fingerprint: raise PlatformError("IDEMPOTENCY_CONFLICT", "提交编号重复且内容变化", 409)
            return self.store.get("tasks", prior["task_id"])
        configs = {id: self.connections.freeze(self.store.get("agents", id)) for id in ROLE_NAMES}
        if any(not configs[id]["enabled"] for id in (["O1", "O2", "O3", "S1"]+(["O4"] if payload["type"] == "vpp" else []))):
            raise PlatformError("AGENT_DISABLED", "本流程所需角色已停用", 409)
        state = await self.live_state()
        request = request_from_site(payload["demand"], state)
        id = identifier("task")
        task = {"id": id, "station_id": STATION_ID, "type": payload["type"], "payload": payload,
                "submission_key": submission_key, "submission_fingerprint": fingerprint, "status": "QUEUED", "version": 1,
                "created_at": stamp(), "deadline": stamp(now()+timedelta(seconds=self.store.get("settings", "platform")["task_timeout_seconds"])),
                "request": request, "configs": configs, "simulation": True, "message": "已建立任务，等待模型与工具运行",
                "replans": 0, "artifacts": {}, "history": []}
        task, created = self.store.create_task(task)
        if created:
            self.store.event(id, "task_created", {"type": task["type"], "source_state": state["id"], "simulation": True,
                                                  "role_versions": {k: v["version"] for k, v in configs.items()}})
            self.jobs[id] = asyncio.create_task(self.run_task(id, task["version"]))
        return task

    async def run_task(self, id, version):
        try:
            task = self.current_task(id, version)
            task.update({"status": "RUNNING", "message": "运营协调正在组织专业工具"})
            self.store.put("tasks", id, task)
            remaining = max(0, (parse_time(task["deadline"])-now()).total_seconds())
            async with asyncio.timeout(remaining):
                await self.run_agent(task["configs"]["O1"], "执行已确认业务任务。任务编号："+id+"，输入版本："+str(version)+"。按职责调用工具；发布成功后结束本轮，等待站级执行反馈。",
                                     id, "workflow", id, expected_version=version)
            current = self.store.get("tasks", id)
            if current["version"] == version and current["status"] == "RUNNING":
                raise PlatformError("WORKFLOW_INCOMPLETE", "运营协调未完成规定工具链与计划发布", 409)
        except asyncio.CancelledError:
            current = self.store.get("tasks", id)
            if current and current["version"] == version and current["status"] in {"QUEUED", "RUNNING"}:
                current.update({"status": "INTERRUPTED", "message": "任务运行中断，未自动重跑"})
                self.store.put("tasks", id, current)
            raise
        except (PlatformError, TimeoutError) as error:
            current = self.store.get("tasks", id)
            if current and current["version"] == version and current["status"] in {"RUNNING", "QUEUED"}:
                blocked = getattr(error, "code", "") in {"SERVICE_REQUIREMENT_UNMET", "SERVICE_SEARCH_LIMIT", "CAPACITY_LIMIT", "BUDGET_EXCEEDED", "LOCAL_RESTRICTED", "STATE_EXPIRED"}
                current.update({"status": "BLOCKED" if blocked else "FAILED", "code": getattr(error, "code", "TASK_TIMEOUT"),
                                "message": getattr(error, "message", "整项任务超时，未继续执行")})
                self.store.put("tasks", id, current)
                self.store.event(id, "task_failed", {"code": current["code"], "message": current["message"]})
        except Exception:
            current = self.store.get("tasks", id)
            if current and current["version"] == version and current["status"] in {"RUNNING", "QUEUED"}:
                current.update({"status": "FAILED", "code": "INTERNAL_ERROR", "message": "运行异常，请查看工具记录；未伪造完成"})
                self.store.put("tasks", id, current)
                self.store.event(id, "task_failed", {"code": "INTERNAL_ERROR"})

    def artifact(self, task, name, value):
        task["artifacts"][name] = value
        self.store.put("tasks", task["id"], task)
        return value

    async def tool(self, agent, name, args, owner, task_id, phase, version):
        task = self.current_task(task_id, version) if task_id else None
        if name == "station_state":
            return deepcopy(task["request"]["station_state_snapshot"]) if task else await self.live_state()
        if name == "station_capability": return capability(await self.live_state())
        if name == "tariff":
            return task["request"]["station_state_snapshot"]["economic_context"] if task else (await self.live_state())["site"]["rates_cny_per_kwh"]
        if name == "plan_status":
            return task.get("receipt", {"plan_id": task.get("plan_id"), "status": task["status"]}) if task else [public(t) for t in self.store.all("tasks")[:10]]
        if name == "create_task":
            if agent["id"] != "O1": raise PlatformError("TOOL_PERMISSION_DENIED", "只有运营协调可创建业务草稿", 403)
            payload = validate_payload({"type": args["type"], "demand": {"vehicles": args["vehicles"]}, "budget_kw": args["budget_kw"]})
            id = "draft-"+digest({"session": owner, "payload": payload})[:32]
            draft = {"id": id, "session_id": owner, "payload": payload, "created_at": stamp(), "status": "DRAFT"}
            self.store.put("drafts", id, draft)
            return {"draft_id": id, "status": "DRAFT", "payload": payload, "message": "关键参数已生成草稿，请在任务表单核对后提交"}
        if not task: raise PlatformError("TASK_CONTEXT_REQUIRED", "该业务工具需要结构化任务，可通过运营协调或任务表单建立", 409)
        if name == "task_context":
            return {"task_id": task["id"], "version": task["version"], "payload": task["payload"], "status": task["status"],
                    "completed_stages": list(task["artifacts"]), "simulation": True}
        if name == "agent_call":
            if agent["id"] != "O1": raise PlatformError("TOOL_PERMISSION_DENIED", "只有运营协调可组织专业角色", 403)
            child, child_phase = args["agent_id"], args["phase"]
            if (child, child_phase) not in {("O2", "resources"), ("O3", "optimize"), ("O4", "business"), ("O4", "confirm"), ("O4", "report")}:
                raise PlatformError("WORKFLOW_INVALID", "角色和阶段不匹配", 409)
            if child == "O3" and "resource" not in task["artifacts"]:
                raise PlatformError("RESOURCE_REQUIRED", "先由能源资源工具形成边界", 409)
            if child == "O4" and task["type"] != "vpp": raise PlatformError("BUSINESS_CONTEXT_REQUIRED", "当前不是虚拟电厂任务", 409)
            if child_phase == "confirm" and "evaluation" not in task["artifacts"]:
                raise PlatformError("PLAN_REQUIRED", "先形成工具计算的推荐方案", 409)
            await self.run_agent(task["configs"][child], "本轮任务编号 "+task_id+"，阶段："+child_phase+"。依角色规定调用专业工具，不能只解释后结束。",
                                 owner, child_phase, task_id, expected_version=version)
            latest = self.current_task(task_id, version)
            key = {"resources": "resource", "optimize": "evaluation", "business": "business", "confirm": "business_confirmation", "report": "business_report"}[child_phase]
            if key not in latest["artifacts"]:
                raise PlatformError("SPECIALIST_INCOMPLETE", "专业角色没有实际完成所需工具调用", 409)
            return latest["artifacts"][key]
        if name == "resource_capacity":
            return self.artifact(task, "resource", resources(task["request"]))
        if name == "generate_schedule":
            if "resource" not in task["artifacts"]: raise PlatformError("RESOURCE_REQUIRED", "先形成能源边界", 409)
            return self.artifact(task, "service", schedule(task["request"], task["payload"].get("budget_kw") if task["type"] == "vpp" else None))
        if name == "evaluate_schedule":
            if not {"resource", "service"}.issubset(task["artifacts"]): raise PlatformError("SERVICE_REQUIRED", "先生成候选与能源边界", 409)
            return self.artifact(task, "evaluation", evaluation(task["request"], task["artifacts"]["resource"], task["artifacts"]["service"],
                                 task["payload"].get("budget_kw") if task["type"] == "vpp" else None))
        if name == "vpp_task":
            if task["type"] != "vpp": raise PlatformError("BUSINESS_CONTEXT_REQUIRED", "没有关联虚拟电厂任务", 409)
            return self.artifact(task, "business", {"simulation": True, "business_type": "charging_budget", "task_id": task_id,
                "budget_kw": task["payload"]["budget_kw"], "duration_minutes": 60, "real_market_connected": False})
        if name == "vpp_ack":
            if task["type"] != "vpp" or not {"business", "evaluation"}.issubset(task["artifacts"]):
                raise PlatformError("BUSINESS_PLAN_REQUIRED", "需要业务条件及工具核验的方案", 409)
            return self.artifact(task, "business_confirmation", {"simulation": True, "status": "SIMULATED_CONFIRMED",
                "candidate_ref": task["artifacts"]["evaluation"]["selected_candidate"]["candidate_id"], "at": stamp()})
        if name == "vpp_report":
            receipt = task.get("receipt", {})
            if receipt.get("status") != "COMPLETED" or len(receipt.get("steps", [])) != 4:
                raise PlatformError("EXECUTION_EVIDENCE_REQUIRED", "缺少四个时段的有效模拟回执", 409)
            peak = max(s["charging_kw"] for s in receipt["steps"])
            return self.artifact(task, "business_report", {"simulation": True, "real_settlement": False,
                "peak_charging_kw": peak, "budget_kw": task["payload"]["budget_kw"],
                "budget_met": peak <= task["payload"]["budget_kw"]+1e-6, "receipt_ref": receipt["id"], "at": stamp()})
        if name == "publish_plan":
            if agent["id"] != "O1": raise PlatformError("PUBLISH_DENIED", "发布权限仅属于运营协调", 403)
            if task.get("plan_id"): return {"status": task["status"], "plan_id": task["plan_id"], "duplicate": True}
            if "evaluation" not in task["artifacts"]: raise PlatformError("EVALUATION_REQUIRED", "须先完成确定性评价", 409)
            if task["type"] == "vpp" and "business_confirmation" not in task["artifacts"]:
                raise PlatformError("BUSINESS_CONFIRMATION_REQUIRED", "模拟业务尚未确认", 409)
            if not self.station()["online"]: raise PlatformError("NODE_OFFLINE", "场站已离线，不能发布", 409)
            selected = task["artifacts"]["evaluation"]["selected_candidate"]
            plan = {"id": task_id+"-v"+str(version), "task_id": task_id, "version": version, "station_id": STATION_ID,
                    "request": task["request"], "selected_candidate": selected,
                    "budget_kw": task["payload"].get("budget_kw") if task["type"] == "vpp" else None, "simulation": True}
            plan["fingerprint"] = digest({k: plan[k] for k in ["task_id", "version", "station_id", "request", "selected_candidate", "budget_kw"]})
            plan["station_agent"] = self.snapshot_for_node(task["configs"]["S1"], task_id)
            # Persist publication identity before I/O so uncertain sends are only observed, not replayed.
            task.update({"plan_id": plan["id"], "plan": plan, "status": "DISPATCHING", "message": "计划已登记，正在请求站端确认"})
            self.store.put("tasks", task_id, task)
            self.store.event(task_id, "plan_dispatching", {"plan_id": plan["id"], "version": version, "fingerprint": plan["fingerprint"]})
            try:
                receipt = await self.node_request("POST", "/internal/plans", plan)
            except PlatformError as error:
                latest = self.store.get("tasks", task_id)
                if latest["version"] != version or latest["status"] != "DISPATCHING":
                    return {"status": latest["status"], "plan_id": plan["id"], "message": "发布期间任务状态发生变化"}
                task.update({"status": "WAITING_FEEDBACK", "message": "发送结果尚未确认，保留发布编号并观察站端；未重复发送", "dispatch_error": error.code})
                self.store.put("tasks", task_id, task)
                return {"status": "WAITING_FEEDBACK", "plan_id": plan["id"], "uncertain": True}
            latest = self.store.get("tasks", task_id)
            if latest["version"] != version or latest["status"] != "DISPATCHING":
                return {"status": latest["status"], "plan_id": plan["id"], "message": "迟到回执不覆盖取消或新的输入版本"}
            task.update({"receipt": receipt, "status": "WAITING_FEEDBACK", "message": "已收到站端回执，等待站级准备或模拟执行"})
            self.store.put("tasks", task_id, task)
            return {"status": "WAITING_FEEDBACK", "plan_id": plan["id"], "station_status": receipt["status"], "simulation": True}
        raise PlatformError("TOOL_UNKNOWN", "工具未注册", 403)

    async def finalize(self, id, version):
        try:
            task = self.current_task(id, version)
            if task["type"] == "vpp":
                await self.run_agent(task["configs"]["O4"], "report 阶段：根据已完成的模拟执行回执调用 vpp_report，不能仅口头宣称完成。",
                                     id, "report", id, expected_version=version)
                task = self.current_task(id, version)
                if "business_report" not in task["artifacts"]:
                    raise PlatformError("REPORT_INCOMPLETE", "业务角色未调用履约反馈工具", 409)
            task.update({"status": "COMPLETED", "completed_at": stamp(), "message": "模拟任务与反馈完成；不代表真实设备或市场执行"})
            self.store.put("tasks", id, task)
            self.store.event(id, "task_completed", {"receipt_ref": task["receipt"]["id"], "simulation": True})
        except PlatformError as error:
            task = self.store.get("tasks", id)
            if task and task["version"] == version and task["status"] == "FINALIZING":
                task.update({"status": "FAILED", "code": error.code, "message": "模拟执行已完成，但业务反馈失败："+error.message})
                self.store.put("tasks", id, task)

    async def replan(self, task, receipt):
        if task["replans"] >= 2 or receipt.get("steps"):
            task.update({"status": "BLOCKED", "message": "达到重算上限或已有部分执行，保留回执并停止自动重跑", "receipt": receipt})
            self.store.put("tasks", task["id"], task)
            return
        state = await self.live_state()
        request = request_from_site(task["payload"]["demand"], state)
        task["history"].append({"version": task["version"], "plan_id": task.get("plan_id"), "receipt": receipt, "artifacts": task["artifacts"]})
        old = self.jobs.get(task["id"])
        if old and not old.done(): old.cancel()
        task.update({"status": "QUEUED", "version": task["version"]+1, "replans": task["replans"]+1,
                     "request": request, "artifacts": {}, "message": "未执行计划条件变化，使用新快照重算"})
        for key in ["plan_id", "plan", "receipt"]: task.pop(key, None)
        self.store.put("tasks", task["id"], task)
        self.store.event(task["id"], "replan_started", {"version": task["version"], "source_state": state["id"]})
        self.jobs[task["id"]] = asyncio.create_task(self.run_task(task["id"], task["version"]))

    async def observe(self, task):
        if task["status"] not in {"WAITING_FEEDBACK", "CANCEL_REQUESTED"}: return
        if now() >= parse_time(task["deadline"]):
            await self.cancel(task["id"], timed_out=True)
            return
        if not self.station()["online"]: return
        try:
            receipt = await self.node_request("GET", "/internal/plans/"+task["plan_id"])
        except PlatformError as error:
            if error.status == 404:
                task.update({"status": "BLOCKED", "message": "站端未找到已登记计划，发送结果未确认；需明确重试，不自动再发"})
                self.store.put("tasks", task["id"], task)
            return
        task["receipt"] = receipt
        if receipt["status"] == "COMPLETED":
            task.update({"status": "FINALIZING", "message": "模拟执行完成，正在整理反馈"})
            self.store.put("tasks", task["id"], task)
            self.jobs[task["id"]] = asyncio.create_task(self.finalize(task["id"], task["version"]))
        elif receipt["status"] in {"REJECTED", "BLOCKED"}:
            try: await self.replan(task, receipt)
            except PlatformError as error:
                task.update({"status": "BLOCKED", "code": error.code, "message": error.message})
                self.store.put("tasks", task["id"], task)
        elif receipt["status"] in {"FAILED", "INTERRUPTED", "CANCELLED"}:
            task.update({"status": receipt["status"], "message": receipt.get("message", "站端处理中断")})
            self.store.put("tasks", task["id"], task)
        else:
            self.store.put("tasks", task["id"], task)

    async def watch_loop(self):
        while True:
            for task in self.store.all("tasks"):
                try: await self.observe(task)
                except (PlatformError, KeyError, ValueError): pass
            await asyncio.sleep(self.interval)

    async def cancel(self, id, timed_out=False):
        task = self.store.get("tasks", id)
        if not task: raise PlatformError("NOT_FOUND", "任务不存在", 404)
        if task["status"] in TERMINAL: return task
        if task.get("plan_id"):
            task.update({"status": "CANCEL_REQUESTED", "message": "正在确认站端停止剩余模拟"})
            self.store.put("tasks", id, task)
            try:
                receipt = await self.node_request("POST", "/internal/plans/"+task["plan_id"]+"/cancel", {})
            except PlatformError:
                return task
            task["receipt"] = receipt
            if receipt["status"] == "COMPLETED":
                task.update({"status": "FINALIZING", "message": "取消到达前模拟已完成，保留实际完成状态"})
                self.store.put("tasks", id, task)
                self.jobs[id] = asyncio.create_task(self.finalize(id, task["version"]))
                return task
        task.update({"status": "CANCELLED", "message": "任务超时，剩余模拟已取消" if timed_out else "任务已取消，既有执行证据保留"})
        self.store.put("tasks", id, task)
        job = self.jobs.get(id)
        if job and not job.done(): job.cancel()
        self.store.event(id, "task_cancelled", {"timed_out": timed_out})
        return task

def create_center(data_dir, node_url, token, port=8780, vault=None, interval=2):
    service = CenterService(data_dir, node_url, token, port, vault, interval)
    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        yield
        await service.stop()
    app = FastAPI(title="场站 Agent 专用部署平台", lifespan=lifespan)
    app.state.service = service
    install_errors(app)
    local_guard(app, port, service.browser_token)
    app.mount("/static", StaticFiles(directory=ROOT/"static"), name="static")
    @app.get("/")
    async def page(): return FileResponse(ROOT/"static"/"index.html")
    @app.get("/health")
    async def health(): return {"service": "center", "version": "0.1.0", "hermes_dependency": False}
    @app.get("/api/bootstrap")
    async def bootstrap():
        return {"token": service.browser_token, "default_demand": DEFAULT_DEMAND,
                "settings": service.store.get("settings", "platform"), "simulation": True}
    @app.get("/api/overview")
    async def overview():
        connections = service.store.all("connections")
        return {"station": service.station(), "agents": service.store.all("agents"),
                "models": [service.connections.safe(c) for c in connections],
                "tasks": public(service.store.all("tasks")[:30]), "settings": service.store.get("settings", "platform"),
                "alerts": service.store.events("station", limit=50)}
    @app.get("/api/models")
    async def models(): return [service.connections.safe(c) for c in service.store.all("connections")]
    @app.post("/api/models")
    async def save_model(request: Request): return service.connections.save(await request.json())
    @app.post("/api/models/{id}/test")
    async def test_model(id: str): return await service.connections.test(id)
    @app.post("/api/settings")
    async def settings(request: Request):
        data = await request.json()
        current = service.store.get("settings", "platform")
        if "default_connection" in data:
            record = service.store.get("connections", data["default_connection"])
            if not record or not record.get("verified"): raise PlatformError("MODEL_NOT_VERIFIED", "默认连接需要先通过测试", 409)
            current["default_connection"] = record["id"]
        if "task_timeout_seconds" in data:
            timeout = data["task_timeout_seconds"]
            if type(timeout) is not int or not 60 <= timeout <= 3600: raise PlatformError("TIMEOUT_INVALID", "任务超时范围为 60～3600 秒")
            current["task_timeout_seconds"] = timeout
        return service.store.put("settings", "platform", current)
    @app.get("/api/agents")
    async def agents(): return service.store.all("agents")
    @app.get("/api/tools")
    async def tools(): return TOOLS
    @app.get("/api/agents/{id}/documents")
    async def documents(id: str): return {"documents": [service.documents.get(id)]}
    @app.put("/api/agents/{id}/documents/{document_id}/draft")
    async def document_draft(id: str, document_id: str, request: Request):
        return service.documents.save_draft(id, await request.json(), document_id)
    @app.post("/api/agents/{id}/documents/{document_id}/publish")
    async def document_publish(id: str, document_id: str, request: Request):
        return service.documents.publish(id, await request.json(), document_id)
    async def save_agent(body, id=None):
        if not isinstance(body, dict): raise PlatformError("AGENT_INVALID", "角色配置需要为对象")
        with service.store.transaction() as db:
            old = service.store.get("agents", id) if id else None
            if id and not old: raise PlatformError("NOT_FOUND", "角色不存在", 404)
            if old:
                if type(body.get("expected_version")) is not int or body["expected_version"] != old["version"]:
                    raise PlatformError("AGENT_CONFIG_CONFLICT", "角色配置已更新或缺少版本信息，请重新打开配置", 409)
                if "prompt" in body and body["prompt"] != old["prompt"]:
                    raise PlatformError("DOCUMENT_PUBLISH_REQUIRED", "已有角色的运行职责请在文档页保存草稿并发布", 409)
            role = id if old and old["builtin"] else "custom"
            names = body.get("allowed_tools", (old or {}).get("allowed_tools", ["station_state"]))
            if not isinstance(names, list) or any(not isinstance(n, str) for n in names) or len(names)!=len(set(names)) or any(n not in TOOL_MAP or role not in TOOL_MAP[n]["roles"] for n in names):
                raise PlatformError("TOOL_PERMISSION_DENIED", "只能选择该角色可用的已注册工具", 403)
            name = body.get("name", (old or {}).get("name", ""))
            prompt = old["prompt"] if old else body.get("prompt", "")
            turns = body.get("max_turns", (old or {}).get("max_turns", 12))
            if not isinstance(name, str) or not name.strip() or len(name)>100 or not isinstance(prompt, str) or not prompt.strip() or len(prompt)>24000 or type(turns) is not int or not 1<=turns<=30:
                raise PlatformError("AGENT_INVALID", "请填写有效职责、名称和 1～30 轮交互限制")
            connection_id = body.get("model_connection", (old or {}).get("model_connection")) or None
            if connection_id and (not isinstance(connection_id, str) or not service.store.get("connections", connection_id)): raise PlatformError("MODEL_NOT_FOUND", "模型连接不存在")
            id = id or identifier("agent")
            record = {"id": id, "name": name.strip(), "prompt": prompt, "allowed_tools": names, "max_turns": turns,
                "builtin": bool(old and old["builtin"]), "scope": (old or {}).get("scope", "operator"),
                "enabled": bool(body.get("enabled", (old or {}).get("enabled", True))), "model_connection": connection_id,
                "version": (old or {}).get("version", 0)+1, "created_at": (old or {}).get("created_at", stamp()), "updated_at": stamp()}
            service.documents.write(db, "agents", id, record)
            if not old: service.documents.initialize(db, record, created=True)
        return record
    @app.post("/api/agents")
    async def add_agent(request: Request): return await save_agent(await request.json())
    @app.put("/api/agents/{id}")
    async def update_agent(id: str, request: Request): return await save_agent(await request.json(), id)
    @app.get("/api/station")
    async def station(): return service.station()
    @app.post("/api/station/scenario")
    async def scenario(request: Request): return await service.node_request("POST", "/internal/scenario", await request.json())
    @app.get("/api/tasks")
    async def tasks(): return public(service.store.all("tasks"))
    @app.post("/api/tasks")
    async def create_task(request: Request):
        body = await request.json()
        key = body.pop("submission_key", None)
        return public(await service.create_task(body, key))
    @app.get("/api/tasks/{id}")
    async def get_task(id: str):
        task = service.store.get("tasks", id)
        if not task: raise PlatformError("NOT_FOUND", "任务不存在", 404)
        return public(task)
    @app.get("/api/tasks/{id}/events")
    async def task_events(id: str, after: int=0): return service.store.events(id, after)
    @app.get("/api/tasks/{id}/stream")
    async def stream(id: str, request: Request, after: int=0):
        async def events():
            cursor = after
            while not await request.is_disconnected():
                for record in service.store.events(id, cursor):
                    cursor = record["id"]
                    yield "id: "+str(cursor)+"\ndata: "+canonical(record)+"\n\n"
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
    @app.post("/api/tasks/{id}/cancel")
    async def cancel(id: str): return public(await service.cancel(id))
    @app.post("/api/tasks/{id}/advance")
    async def advance(id: str, request: Request):
        task = service.store.get("tasks", id)
        if not task or not task.get("plan_id"): raise PlatformError("PLAN_NOT_READY", "任务尚未发布计划", 409)
        body = await request.json()
        receipt = await service.node_request("GET", "/internal/plans/"+task["plan_id"])
        indices = range(len(receipt["steps"]), 4) if body.get("all") else [body.get("step", len(receipt["steps"]))]
        for step in indices:
            receipt = await service.node_request("POST", "/internal/plans/"+task["plan_id"]+"/advance", {"step": step})
            if receipt["status"] != "SIMULATING": break
        return public(receipt)
    @app.get("/api/sessions")
    async def sessions(): return [{k:v for k,v in s.items() if k not in {"messages", "agent_snapshot"}} for s in service.store.all("sessions")]
    @app.post("/api/sessions")
    async def session(request: Request):
        agent_id = (await request.json()).get("agent_id", "O1")
        agent = service.store.get("agents", agent_id)
        if not agent or not agent["enabled"]: raise PlatformError("AGENT_NOT_AVAILABLE", "角色不存在或已停用", 409)
        agent = service.connections.freeze(agent)
        id = identifier("session")
        if agent_id == "S1": agent = service.snapshot_for_node(agent, id)
        record = {"id": id, "agent_id": agent_id, "agent_snapshot": agent, "messages": [], "display": [], "created_at": stamp()}
        service.store.put("sessions", id, record)
        return public({"id": id, "agent_id": agent_id})
    @app.get("/api/sessions/{id}")
    async def get_session(id: str):
        record = service.store.get("sessions", id)
        if not record: raise PlatformError("NOT_FOUND", "会话不存在", 404)
        return {"id": id, "agent_id": record["agent_id"], "display": record["display"],
                "events": service.store.events(id), "drafts": [d for d in service.store.all("drafts") if d["session_id"] == id]}
    @app.post("/api/sessions/{id}/messages")
    async def message(id: str, request: Request):
        text = (await request.json()).get("message", "")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 8000: raise PlatformError("MESSAGE_INVALID", "消息需要为 1～8000 字")
        lock = service.session_locks.setdefault(id, asyncio.Lock())
        if lock.locked(): raise PlatformError("SESSION_BUSY", "会话正在处理上一条消息", 409)
        async with lock:
            record = service.store.get("sessions", id)
            if not record: raise PlatformError("NOT_FOUND", "会话不存在", 404)
            agent = record["agent_snapshot"]
            record["display"].append({"role": "user", "text": redact(text), "at": stamp()})
            service.store.put("sessions", id, record)
            if agent["id"] == "S1":
                result = await service.node_request("POST", "/internal/chat", {"session_id": id, "agent": agent, "message": text})
            else:
                result = await service.run_agent(agent, text, id, history=record["messages"])
                record["messages"] = result["messages"]
            record["display"].append({"role": "assistant", "text": result["text"], "at": stamp()})
            service.store.put("sessions", id, record)
            return {"text": result["text"]}
    @app.post("/internal/station/heartbeat")
    async def heartbeat(request: Request):
        require_node(request, token)
        body = await request.json()
        if body.get("station_id") != STATION_ID or body.get("state", {}).get("station_id") != STATION_ID:
            raise PlatformError("STATION_INVALID", "本站身份不匹配", 403)
        service.store.put("nodes", STATION_ID, {"station_id": STATION_ID, "last_seen": stamp(), "state": body["state"], "simulation": True})
        cursor = service.store.get("settings", "node_event_cursor", {"after": 0})["after"]
        for event in body.get("events", []):
            if event["id"] > cursor:
                service.store.event(event["owner"], "station_"+event["kind"], {**event["data"], "node_event_id": event["id"], "node_at": event["at"]}, secrets=(token,))
                cursor = event["id"]
        service.store.put("settings", "node_event_cursor", {"after": cursor})
        return {"ok": True, "event_cursor": cursor}
    @app.post("/internal/station/agent-snapshot")
    async def node_agent_snapshot(request: Request):
        require_node(request, token)
        body = await request.json()
        agent = service.connections.freeze(service.store.get("agents", "S1"))
        return service.snapshot_for_node(agent, body.get("owner", "station"))
    @app.post("/internal/model/turn")
    async def gateway(request: Request):
        require_node(request, token)
        body = await request.json()
        snapshot = service.store.get("node_snapshots", body.get("snapshot_ref"))
        if not snapshot or snapshot["agent"]["id"] != "S1": raise PlatformError("NODE_ROLE_DENIED", "节点只能调用其绑定的站级角色", 403)
        agent = snapshot["agent"]
        tools = [TOOL_MAP[n] for n in agent["allowed_tools"] if "S1" in TOOL_MAP[n]["roles"]]
        return await service.model_turn(agent, body["messages"], tools)
    return app
