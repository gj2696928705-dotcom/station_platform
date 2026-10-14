import asyncio
import json
from jsonschema import Draft202012Validator
from .defaults import TOOL_MAP
from .models import assistant_message
from .util import PlatformError, canonical, digest, redact

class AgentEngine:
    def __init__(self, store, request_model):
        self.store, self.request_model = store, request_model

    async def run(self, agent, user_message, handle_tool, owner, phase="chat", history=None, secrets=()):
        if not agent.get("enabled"):
            raise PlatformError("AGENT_DISABLED", "该角色已停用", 409)
        role = agent["id"] if agent.get("builtin") else "custom"
        names = agent["allowed_tools"]
        if any(n not in TOOL_MAP or role not in TOOL_MAP[n]["roles"] for n in names):
            raise PlatformError("TOOL_PERMISSION_DENIED", "角色工具配置越权", 403)
        tools = [TOOL_MAP[n] for n in names]
        messages = list(history or [])
        if not messages:
            messages.append({"role": "system", "content": agent["prompt"] + "\n平台角色编号："+agent["id"]})
        messages.append({"role": "user", "content": user_message})
        self.store.event(owner, "agent_started", {"agent_id": agent["id"], "version": agent["version"], "phase": phase})
        # A repeated call identifier cannot run an effect twice within this turn.
        seen = {}
        try:
            for round in range(agent.get("max_turns", 12)):
                turn = await self.request_model(agent, messages, tools)
                if not turn["text"] and not turn["tool_calls"]:
                    raise PlatformError("MODEL_EMPTY", "模型没有返回文本或工具调用", 502)
                messages.append(assistant_message(turn))
                self.store.event(owner, "model_turn", {"agent_id": agent["id"], "round": round+1, "usage": turn.get("usage", {}),
                                                      "tool_names": [c["name"] for c in turn["tool_calls"]]})
                if not turn["tool_calls"]:
                    text = redact(turn["text"], secrets)
                    self.store.event(owner, "agent_completed", {"agent_id": agent["id"], "text": text})
                    return {"text": text, "messages": redact(messages, secrets)}
                for call in turn["tool_calls"]:
                    name = call["name"]
                    if name not in names:
                        raise PlatformError("TOOL_PERMISSION_DENIED", "模型请求了未授权工具", 403)
                    try:
                        args = json.loads(call["arguments"])
                        canonical(args)
                        if list(Draft202012Validator(TOOL_MAP[name]["parameters"]).iter_errors(args)):
                            raise ValueError()
                    except (ValueError, TypeError):
                        raise PlatformError("TOOL_ARGS_INVALID", "工具参数不符合约定", 400) from None
                    fingerprint = digest({"name": name, "arguments": args})
                    if call["id"] in seen:
                        if seen[call["id"]][0] != fingerprint:
                            raise PlatformError("TOOL_CALL_CONFLICT", "重复工具调用编号改变了参数", 409)
                        result = seen[call["id"]][1]
                    else:
                        self.store.event(owner, "tool_started", {"agent_id": agent["id"], "name": name, "arguments": args}, secrets)
                        try:
                            result = await handle_tool(name, args)
                        except PlatformError as error:
                            result = {"ok": False, "code": error.code, "message": error.message}
                            if error.code in {"SERVICE_REQUIREMENT_UNMET", "SERVICE_SEARCH_LIMIT", "CAPACITY_LIMIT", "BUDGET_EXCEEDED",
                                              "STATE_EXPIRED", "LOCAL_RESTRICTED", "MODEL_TIMEOUT", "MODEL_HTTP_ERROR",
                                              "MODEL_CONNECTION_FAILED", "MODEL_PROTOCOL_INVALID", "MODEL_INCOMPLETE", "TASK_NOT_ACTIVE", "TASK_VERSION_CHANGED"}:
                                self.store.event(owner, "tool_completed", {"agent_id": agent["id"], "name": name, "result": result}, secrets)
                                raise
                        seen[call["id"]] = (fingerprint, result)
                        self.store.event(owner, "tool_completed", {"agent_id": agent["id"], "name": name, "result": result}, secrets)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": canonical(result)})
            raise PlatformError("AGENT_TURN_LIMIT", "角色达到工具交互上限，未完成任务", 409)
        except asyncio.CancelledError:
            self.store.event(owner, "agent_cancelled", {"agent_id": agent["id"]})
            raise
        except PlatformError as error:
            self.store.event(owner, "agent_failed", {"agent_id": agent["id"], "code": error.code, "message": error.message})
            raise
