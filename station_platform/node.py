import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import httpx
from fastapi import FastAPI, Request
from .agent import AgentEngine
from .defaults import DEFAULT_DEMAND
from .domain import capability, check_local
from .http_support import install_errors, local_guard, require_node
from .legacy.demand import MockStationAdapter
from .storage import Store
from .util import PlatformError, STATION_ID, canonical, identifier, now, public, stamp

PLAN_END = {"COMPLETED", "REJECTED", "FAILED", "BLOCKED", "CANCELLED", "INTERRUPTED"}

class NodeService:
    def __init__(self, data_dir, center_url, token, port=8781, interval=5):
        self.store = Store(Path(data_dir)/"node.sqlite3")
        self.center_url, self.token, self.port, self.interval = center_url.rstrip("/"), token, port, interval
        self.jobs, self.chat_locks = {}, {}
        self.client = None
        if not self.store.get("settings", "site"):
            self.store.put("settings", "site", {"site": MockStationAdapter().site, "revision": 1,
                "scenario": "normal", "heartbeat_paused": False, "stale": False, "quality": "B"})
        self.engine = AgentEngine(self.store, self.model_turn)

    def state(self):
        config = self.store.get("settings", "site")
        at = now()-timedelta(minutes=5) if config["stale"] else now()
        return {"id": identifier("state"), "station_id": STATION_ID, "revision": config["revision"],
                "measured_at": stamp(at), "expires_at": stamp(at+timedelta(seconds=20)),
                "quality": config["quality"], "simulation": True, "source": "station_platform_simulator",
                "site": config["site"], "scenario": config["scenario"]}

    async def start(self):
        self.client = httpx.AsyncClient(timeout=10, trust_env=False, headers={"Authorization": "Bearer "+self.token})
        for plan in self.store.all("plans"):
            if plan["status"] not in PLAN_END:
                plan.update({"status": "INTERRUPTED", "message": "站端重启，保留回执并等待处理，未重新执行"})
                self.store.put("plans", plan["id"], plan)
                self.store.event(plan["task_id"], "node_interrupted", {"plan_id": plan["id"], "steps": plan["steps"]})
        self.heartbeat_job = asyncio.create_task(self.heartbeat_loop())

    async def stop(self):
        self.heartbeat_job.cancel()
        jobs = list(self.jobs.values())
        for job in jobs: job.cancel()
        await asyncio.gather(self.heartbeat_job, *jobs, return_exceptions=True)
        await self.client.aclose()

    async def post_center(self, path, data):
        response = await self.client.post(self.center_url+path, json=data, timeout=610 if path == "/internal/model/turn" else 10)
        if response.status_code >= 300:
            body = response.json()
            raise PlatformError(body.get("code", "CENTER_ERROR"), body.get("message", "中心接口失败"), response.status_code)
        return response.json()

    async def heartbeat(self):
        config = self.store.get("settings", "site")
        if config["heartbeat_paused"]: return
        cursor = self.store.get("settings", "event_cursor", {"after": 0})["after"]
        events = self.store.global_events(cursor)
        reply = await self.post_center("/internal/station/heartbeat", {"station_id": STATION_ID, "state": self.state(),
            "plans": public(self.store.all("plans")), "events": events, "port": self.port})
        if events:
            self.store.put("settings", "event_cursor", {"after": max(cursor, reply.get("event_cursor", cursor))})

    async def heartbeat_loop(self):
        while True:
            try:
                await self.heartbeat()
            except (httpx.HTTPError, PlatformError, ValueError):
                pass
            await asyncio.sleep(self.interval)

    async def model_turn(self, agent, messages, tools):
        return await self.post_center("/internal/model/turn", {"snapshot_ref": agent["gateway_ref"], "messages": messages})

    def receive(self, incoming):
        id = incoming.get("id")
        if not isinstance(id, str) or len(id)>150:
            raise PlatformError("PLAN_INVALID", "计划标识无效")
        if self.store.get("cancelled_plan_ids", id):
            raise PlatformError("PLAN_CANCELLED", "该计划编号已取消，迟到的发送不能启动执行", 409)
        old = self.store.get("plans", id)
        if old:
            if old["fingerprint"] != incoming.get("fingerprint"):
                raise PlatformError("PLAN_ID_CONFLICT", "相同计划编号的内容发生变化", 409)
            return old
        if any(p["status"] not in PLAN_END for p in self.store.all("plans")):
            raise PlatformError("NODE_BUSY", "本站已有待处理或执行中的计划", 409)
        try:
            check = check_local(incoming, self.state())
            agent = incoming["station_agent"]
            if agent["id"] != "S1" or not agent.get("builtin") or not agent.get("gateway_ref"):
                raise PlatformError("AGENT_INVALID", "计划未绑定合法站级角色")
        except (KeyError, ValueError, TypeError):
            raise PlatformError("PLAN_INVALID", "计划缺少所需的来源和字段") from None
        plan = {**deepcopy(incoming), "status": "RECEIVED" if check["passed"] else "REJECTED", "check": check,
                "received_at": stamp(), "steps": [], "message": "等待站级 Agent 校验与准备" if check["passed"] else "现场条件不满足"}
        self.store.put("plans", id, plan)
        self.store.event(plan["task_id"], "node_plan_received", {"plan_id": id, "status": plan["status"], "check": check})
        if check["passed"]:
            self.jobs[id] = asyncio.create_task(self.handle_plan(id))
        return plan

    async def handle_plan(self, id):
        plan = self.store.get("plans", id)
        async def tool(name, args):
            return await self.local_tool(name, args, plan_id=id)
        try:
            await self.engine.run(plan["station_agent"], "处理新计划 "+id+"：依次校验、准备模拟执行、查询状态。实际反馈以工具为准。", tool, plan["task_id"], phase="plan")
            latest = self.store.get("plans", id)
            if latest["status"] in {"RECEIVED", "CHECKED"}:
                raise PlatformError("STATION_WORKFLOW_INCOMPLETE", "站级角色未调用所需工具，不能开始模拟", 409)
        except asyncio.CancelledError:
            latest = self.store.get("plans", id)
            if latest["status"] not in PLAN_END:
                latest.update({"status": "INTERRUPTED", "message": "站级运行中断，未重新执行"})
                self.store.put("plans", id, latest)
            raise
        except PlatformError as error:
            latest = self.store.get("plans", id)
            if latest["status"] not in PLAN_END:
                latest.update({"status": "FAILED", "code": error.code, "message": error.message})
                self.store.put("plans", id, latest)
        finally:
            try: await self.heartbeat()
            except (httpx.HTTPError, PlatformError, ValueError): pass

    async def local_tool(self, name, args, plan_id=None):
        if name == "station_state": return self.state()
        if name == "station_capability": return capability(self.state())
        if not plan_id:
            if name == "plan_status": return public(self.store.all("plans")[:10])
            raise PlatformError("PLAN_CONTEXT_REQUIRED", "该工具需要运营商发布的关联计划", 409)
        plan = self.store.get("plans", plan_id)
        if name == "plan_status": return public(plan)
        if plan["status"] in PLAN_END:
            raise PlatformError("PLAN_NOT_ACTIVE", "计划已结束或中断，不能重新执行", 409)
        if name == "check_plan":
            check = check_local(plan, self.state())
            plan.update({"check": check, "status": "CHECKED" if check["passed"] else "REJECTED"})
            self.store.put("plans", plan_id, plan)
            return check
        if name == "start_simulation":
            if plan["status"] == "SIMULATING": return {"status": "SIMULATING", "duplicate": True, "simulation": True}
            if plan["status"] != "CHECKED" or not plan["check"]["passed"]:
                raise PlatformError("LOCAL_CHECK_REQUIRED", "须先对同一计划完成本地校验", 409)
            check = check_local(plan, self.state())
            if not check["passed"]:
                plan.update({"status": "REJECTED", "check": check})
                self.store.put("plans", plan_id, plan)
                raise PlatformError("LOCAL_CHECK_FAILED", "执行前条件变化，计划已拒绝", 409)
            plan.update({"status": "SIMULATING", "started_at": stamp(), "message": "已准备模拟，等待页面推进业务时钟"})
            self.store.put("plans", plan_id, plan)
            return {"status": "SIMULATING", "simulation": True, "plan_id": plan_id, "step_count": 0}
        raise PlatformError("TOOL_PERMISSION_DENIED", "本站不支持该工具", 403)

    def advance(self, id, step):
        plan = self.store.get("plans", id)
        if not plan: raise PlatformError("NOT_FOUND", "计划不存在", 404)
        if type(step) is not int or not 0 <= step <= 3:
            raise PlatformError("STEP_INVALID", "时段需要为 0～3")
        if step < len(plan["steps"]): return plan
        if plan["status"] != "SIMULATING" or step != len(plan["steps"]):
            raise PlatformError("STEP_CONFLICT", "计划未准备好或时段顺序不一致", 409)
        state = self.state()
        check = check_local(plan, state)
        if not check["passed"]:
            plan.update({"status": "BLOCKED", "check": check, "message": "条件变化，停止剩余模拟；已执行回执保留，部分执行不自动重跑"})
            self.store.put("plans", id, plan)
            return plan
        powers = plan["selected_candidate"]["charger_power_kw"]
        charging = sum(p[step] for p in powers.values())
        energy = state["site"]["energy"]
        grid = max(0, charging+energy["base_load_kw"][step]-energy["pv_available_kw"][step])
        from .util import parse_time
        business_at = parse_time(plan["request"]["station_state_snapshot"]["generated_at"])+timedelta(minutes=15*(step+1))
        record = {"slot": step, "wall_at": stamp(), "business_at": stamp(business_at), "state_ref": state["id"],
                  "charging_kw": charging, "grid_import_kwh": grid*.25,
                  "purchase_cost_cny": grid*.25*state["site"]["rates_cny_per_kwh"][step],
                  "delivered_kwh": {a["vehicle_id"]: a["power_kw"][step]*.25 for a in plan["selected_candidate"]["vehicle_assignments"]},
                  "simulation": True}
        plan["steps"].append(record)
        plan["status"] = "COMPLETED" if len(plan["steps"]) == 4 else "SIMULATING"
        plan["message"] = "四个业务时段模拟完成" if plan["status"] == "COMPLETED" else "模拟时段已推进"
        if plan["status"] == "COMPLETED": plan["completed_at"] = stamp()
        self.store.put("plans", id, plan)
        self.store.event(plan["task_id"], "simulation_step", {"plan_id": id, "step": record, "status": plan["status"]})
        return plan

    def scenario(self, name):
        if name not in {"normal", "capacity_drop", "charger_fault", "protection", "heartbeat_pause", "stale_data", "bad_quality", "new_charger", "price_change"}:
            raise PlatformError("SCENARIO_INVALID", "模拟场景无效")
        old = self.store.get("settings", "site")
        site = MockStationAdapter().site
        if name == "capacity_drop": site["energy"]["grid_import_limit_kw"] = 50
        if name == "charger_fault": site["equipment"][0]["availability"] = "unavailable"
        if name == "protection": site["safety"]["protection_active"] = True
        if name == "new_charger": site["equipment"].append({**deepcopy(site["equipment"][0]), "asset_id": "charger-4"})
        if name == "price_change": site["rates_cny_per_kwh"] = [1.2, 1.2, .4, .4]
        self.store.put("settings", "site", {"site": site, "revision": old["revision"]+1, "scenario": name,
            "heartbeat_paused": name == "heartbeat_pause", "stale": name == "stale_data", "quality": "C" if name == "bad_quality" else "B"})
        self.store.event("station", "station_exception", {"station_id": STATION_ID, "scenario": name, "revision": old["revision"]+1})
        state = self.state()
        for plan in self.store.all("plans"):
            if plan["status"] not in PLAN_END:
                check = check_local(plan, state)
                if not check["passed"]:
                    plan.update({"status": "BLOCKED", "check": check, "message": "事件导致现场条件变化，剩余模拟暂停，回执保留"})
                    self.store.put("plans", plan["id"], plan)
        if self.client and name != "heartbeat_pause":
            self.jobs["event"] = asyncio.create_task(self.handle_event(state))
        return self.state()

    async def handle_event(self, state):
        try:
            agent = await self.post_center("/internal/station/agent-snapshot", {"owner": "station"})
            await self.engine.run(agent, "重要本站事件："+state["scenario"]+"。调用本站事实或能力工具，说明影响，不能发布或复供。", self.local_tool, "station", "event")
        except (PlatformError, httpx.HTTPError) as error:
            self.store.event("station", "event_agent_unavailable", {"code": getattr(error, "code", "CENTER_UNREACHABLE"),
                "message": "事件已由程序记录；模型不可用，未生成替代回答"})

    async def chat(self, body):
        id = body["session_id"]
        lock = self.chat_locks.setdefault(id, asyncio.Lock())
        if lock.locked(): raise PlatformError("SESSION_BUSY", "本站会话正在处理上一条消息", 409)
        async with lock:
            session = self.store.get("sessions", id)
            if not session:
                agent = body["agent"]
                if agent["id"] != "S1": raise PlatformError("AGENT_INVALID", "站端只部署 S1", 403)
                session = {"id": id, "agent": agent, "messages": []}
            result = await self.engine.run(session["agent"], body["message"], self.local_tool, id, history=session["messages"])
            session["messages"] = result["messages"]
            self.store.put("sessions", id, session)
            await self.heartbeat()
            return {"text": result["text"]}

def create_node(data_dir, center_url, token, port=8781, interval=5):
    service = NodeService(data_dir, center_url, token, port, interval)
    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        yield
        await service.stop()
    app = FastAPI(title="场站感知节点", lifespan=lifespan)
    app.state.service = service
    install_errors(app)
    local_guard(app, port)
    @app.get("/health")
    async def health(): return {"service": "station", "station_id": STATION_ID, "simulation": True}
    @app.get("/internal/state")
    async def state(request: Request):
        require_node(request, token)
        return service.state()
    @app.get("/internal/plans/{id}")
    async def plan(id: str, request: Request):
        require_node(request, token)
        result = service.store.get("plans", id) or service.store.get("cancelled_plan_ids", id)
        if not result: raise PlatformError("NOT_FOUND", "计划不存在", 404)
        return public(result)
    @app.post("/internal/plans")
    async def receive(request: Request):
        require_node(request, token)
        return public(service.receive(await request.json()))
    @app.post("/internal/plans/{id}/advance")
    async def advance(id: str, request: Request):
        require_node(request, token)
        body = await request.json()
        result = service.advance(id, body.get("step"))
        try: await service.heartbeat()
        except (httpx.HTTPError, PlatformError): pass
        return public(result)
    @app.post("/internal/plans/{id}/cancel")
    async def cancel(id: str, request: Request):
        require_node(request, token)
        plan = service.store.get("plans", id)
        if not plan:
            result = {"id": id, "status": "CANCELLED", "steps": [], "simulation": True,
                      "message": "取消编号已记录，迟到的发送不会执行"}
            service.store.put("cancelled_plan_ids", id, result)
            return result
        if plan["status"] == "COMPLETED": return public(plan)
        plan.update({"status": "CANCELLED", "message": "剩余模拟已取消，已有回执保留"})
        service.store.put("plans", id, plan)
        if id in service.jobs: service.jobs[id].cancel()
        return public(plan)
    @app.post("/internal/scenario")
    async def scenario(request: Request):
        require_node(request, token)
        result = service.scenario((await request.json()).get("name"))
        try: await service.heartbeat()
        except (httpx.HTTPError, PlatformError): pass
        return result
    @app.post("/internal/chat")
    async def chat(request: Request):
        require_node(request, token)
        return await service.chat(await request.json())
    return app
