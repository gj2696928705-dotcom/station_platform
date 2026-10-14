"""Explicit LOCAL TEST SUBSTITUTE, serving real HTTP for protocol integration QA."""
import asyncio
import json
from ..defaults import DEFAULT_DEMAND
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

def create_mock():
    app = FastAPI()
    app.state.mode = "normal"
    app.state.requests = []
    @app.get("/health")
    async def health(): return {"service": "TEST_MODEL_SUBSTITUTE", "real_model": False}
    async def answer(request, protocol):
        body = await request.json()
        app.state.requests.append({"protocol": protocol, "tool_count": len(body.get("tools", [])),
            "system_prompt": body.get("instructions", "") if protocol == "responses" else "\n".join(m.get("content", "") for m in body.get("messages", []) if m.get("role") == "system")})
        mode = app.state.mode
        if mode == "http_error": return JSONResponse({"error": "test injected provider failure"}, status_code=401)
        if mode == "delay": await asyncio.sleep(.3)
        if mode == "malformed": return {"unexpected": "format"}
        raw = body.get("input", []) if protocol == "responses" else body.get("messages", [])
        calls = []
        user = ""
        results = {}
        if protocol == "responses":
            for entry in raw:
                if entry.get("role") == "user": user = entry.get("content", "")
                if entry.get("type") == "function_call":
                    calls.append((entry["name"], json.loads(entry["arguments"])))
                if entry.get("type") == "function_call_output":
                    results[entry["call_id"]] = json.loads(entry["output"])
            tools = [t["name"] for t in body.get("tools", [])]
        else:
            for entry in raw:
                if entry.get("role") == "user": user = entry.get("content", "")
                for call in entry.get("tool_calls", []):
                    calls.append((call["function"]["name"], json.loads(call["function"]["arguments"])))
                if entry.get("role") == "tool": results[entry["tool_call_id"]] = json.loads(entry["content"])
            tools = [t["function"]["name"] for t in body.get("tools", [])]
        names = [c[0] for c in calls]
        next_call = None
        text = "测试替身：已读取工具结果，具体业务状态以程序记录为准。"
        if not tools: text = "OK"
        elif mode == "unauthorized": next_call = ("publish_plan", {})
        elif mode == "bad_args": next_call = (tools[0], {"unregistered_argument": True})
        elif mode == "draft" and "create_task" in tools:
            if "create_task" not in names:
                next_call = ("create_task", {"type": "tariff", "vehicles": DEFAULT_DEMAND["vehicles"], "budget_kw": None})
        elif "connection_probe" in tools:
            if "connection_probe" not in names: next_call = ("connection_probe", {"value": 7})
        elif "执行已确认业务任务" in user and "agent_call" in tools:
            context = next((r for r in results.values() if isinstance(r, dict) and "payload" in r), {})
            vpp = context.get("payload", {}).get("type") == "vpp"
            sequence = [("task_context", {})]
            if vpp: sequence.append(("agent_call", {"agent_id": "O4", "phase": "business"}))
            sequence.extend([("agent_call", {"agent_id": "O2", "phase": "resources"}), ("agent_call", {"agent_id": "O3", "phase": "optimize"})])
            if vpp: sequence.append(("agent_call", {"agent_id": "O4", "phase": "confirm"}))
            sequence.append(("publish_plan", {}))
            if len(calls)<len(sequence): next_call = sequence[len(calls)]
        elif "阶段：resources" in user:
            sequence = [("station_state", {}), ("resource_capacity", {})]
            if len(calls)<len(sequence): next_call = sequence[len(calls)]
        elif "阶段：optimize" in user:
            sequence = [("tariff", {}), ("generate_schedule", {}), ("evaluate_schedule", {})]
            if len(calls)<len(sequence): next_call = sequence[len(calls)]
        elif "阶段：business" in user:
            if "vpp_task" not in names: next_call = ("vpp_task", {})
        elif "阶段：confirm" in user:
            if "vpp_ack" not in names: next_call = ("vpp_ack", {})
        elif "report 阶段" in user:
            if "vpp_report" not in names: next_call = ("vpp_report", {})
        elif "处理新计划" in user:
            sequence = [("check_plan", {}), ("start_simulation", {}), ("plan_status", {})]
            if len(calls)<len(sequence): next_call = sequence[len(calls)]
        elif "station_state" in tools and "station_state" not in names:
            next_call = ("station_state", {})
        if mode == "leak": text = request.headers.get("authorization", "").removeprefix("Bearer ")
        if protocol == "responses":
            output = [{"type": "function_call", "id": "fc_test_"+str(len(raw)), "call_id": "test_call_"+str(len(raw)), "name": next_call[0], "arguments": json.dumps(next_call[1])}] if next_call else [
                {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}]
            return {"id": "response_TEST_SUBSTITUTE", "status": "completed", "output": output, "usage": {"input_tokens": 1, "output_tokens": 1}}
        message = {"role": "assistant", "content": None if next_call else text}
        if next_call: message["tool_calls"] = [{"id": "test_call_"+str(len(raw)), "type": "function", "function": {"name": next_call[0], "arguments": json.dumps(next_call[1])}}]
        return {"choices": [{"message": message, "finish_reason": "tool_calls" if next_call else "stop"}], "usage": {"prompt_tokens": 1}}
    @app.post("/v1/responses")
    async def responses(request: Request): return await answer(request, "responses")
    @app.post("/v1/chat/completions")
    async def completions(request: Request): return await answer(request, "chat_completions")
    return app
