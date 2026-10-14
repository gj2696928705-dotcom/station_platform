"""Explicit standard API adapters. API errors never become simulated answers."""
import json
from urllib.parse import urlparse
import httpx
from .util import PlatformError, identifier, public, stamp

class Connections:
    def __init__(self, store, vault):
        self.store, self.vault = store, vault

    def save(self, data):
        id = data.get("id") or identifier("model")
        current = self.store.get("connections", id)
        protocol = data.get("protocol", "responses")
        url = data.get("base_url", "").strip().rstrip("/")
        parsed = urlparse(url)
        if protocol not in {"responses", "chat_completions"} or parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise PlatformError("MODEL_CONFIG_INVALID", "需要有效的模型基础地址和支持的 API 协议")
        name, model = data.get("name", "").strip(), data.get("model", "").strip()
        if not name or not model or len(name)>100 or len(model)>200:
            raise PlatformError("MODEL_CONFIG_INVALID", "请填写连接名称和模型名称")
        credential = data.get("api_key", "")
        cipher = self.vault.encrypt(credential) if credential else (current or {}).get("credential_cipher")
        if not cipher:
            raise PlatformError("MODEL_KEY_REQUIRED", "请在页面填写模型凭据；不会读取 Hermes 凭据")
        timeout = data.get("timeout_seconds", 120)
        if type(timeout) not in {int, float} or not 5 <= timeout <= 600:
            raise PlatformError("MODEL_CONFIG_INVALID", "模型请求超时范围为 5～600 秒")
        version = (current or {}).get("version", 0)+1
        record = {"id": id, "name": name, "model": model, "base_url": url, "protocol": protocol,
                  "version": version, "ref": f"{id}@{version}", "credential_cipher": cipher,
                  "timeout_seconds": timeout, "verified": False, "updated_at": stamp()}
        self.store.put("connection_versions", record["ref"], record)
        self.store.put("connections", id, record)
        return self.safe(record)

    def safe(self, record):
        return {**public(record), "has_key": bool(record.get("credential_cipher"))}

    def get(self, ref):
        record = self.store.get("connection_versions", ref)
        if not record:
            raise PlatformError("MODEL_NOT_CONFIGURED", "模型连接尚未配置", 409)
        return record

    def freeze(self, agent):
        settings = self.store.get("settings", "platform", {})
        id = agent.get("model_connection") or settings.get("default_connection")
        connection = self.store.get("connections", id) if id else None
        if not connection or not connection.get("verified"):
            raise PlatformError("MODEL_NOT_VERIFIED", "请先配置模型并通过文本与工具调用测试", 409)
        return {**agent, "connection_ref": connection["ref"]}

    async def test(self, id):
        record = self.store.get("connections", id)
        if not record: raise PlatformError("NOT_FOUND", "连接不存在", 404)
        client = ModelClient(self.vault)
        probe = {"name": "connection_probe", "description": "验证本地工具调用，参数 value 必须为 7。",
                 "parameters": {"type": "object", "properties": {"value": {"type": "integer", "const": 7}},
                                "required": ["value"], "additionalProperties": False}}
        try:
            first = await client.turn(record, [{"role": "user", "content": "连接测试，请回复 OK。"}], [])
            if not first["text"]: raise PlatformError("MODEL_TEXT_FAILED", "接口未返回有效文本", 502)
            messages = [{"role": "user", "content": "请调用 connection_probe，value 为 7，然后读取工具结果。"}]
            turn = await client.turn(record, messages, [probe], force_tool=True)
            call = next((c for c in turn["tool_calls"] if c["name"] == "connection_probe"), None)
            if not call or json.loads(call["arguments"]) != {"value": 7}:
                raise PlatformError("MODEL_TOOLS_FAILED", "接口未按要求返回函数调用", 502)
            messages.append(assistant_message(turn))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": '{"value":7,"source":"local_probe"}'})
            final = await client.turn(record, messages, [probe])
            if not final["text"]:
                raise PlatformError("MODEL_TOOLS_FAILED", "工具结果返回后没有有效回复", 502)
            record.update({"verified": True, "verification": {"passed": True, "text": True, "tools": True, "at": stamp()}})
        except (PlatformError, ValueError, TypeError, KeyError) as error:
            record.update({"verified": False, "verification": {"passed": False, "at": stamp(),
                "code": getattr(error, "code", "MODEL_PROTOCOL_INVALID"),
                "message": getattr(error, "message", "模型工具调用格式无效")}})
        self.store.put("connection_versions", record["ref"], record)
        with self.store.lock:
            if self.store.get("connections", id)["version"] == record["version"]:
                self.store.put("connections", id, record)
        return self.safe(record)

def assistant_message(turn):
    message = {"role": "assistant", "content": turn["text"], "tool_calls": turn["tool_calls"],
               "response_items": turn.get("response_items", [])}
    # DeepSeek thinking-mode context must survive both tool results and new user turns.
    # This provider field is internal conversation state, not user-facing output.
    if turn.get("reasoning_content") is not None:
        message["reasoning_content"] = turn["reasoning_content"]
    return message

class ModelClient:
    def __init__(self, vault):
        self.vault = vault

    async def turn(self, connection, messages, tools, force_tool=False):
        protocol = connection["protocol"]
        deepseek_chat = protocol == "chat_completions" and urlparse(connection["base_url"]).hostname == "api.deepseek.com"
        if protocol == "responses":
            inputs = []
            for message in messages:
                if message["role"] == "system": continue
                if message["role"] == "tool":
                    inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]})
                elif message["role"] == "assistant" and message.get("response_items"):
                    inputs.extend(message["response_items"])
                else:
                    if message.get("content"):
                        inputs.append({"role": message["role"], "content": message["content"]})
                    for call in message.get("tool_calls", []):
                        inputs.append({"type": "function_call", "call_id": call["id"], "name": call["name"], "arguments": call["arguments"]})
            payload = {"model": connection["model"], "instructions": "\n".join(m["content"] for m in messages if m["role"] == "system"),
                       "input": inputs, "store": False, "include": ["reasoning.encrypted_content"]}
            if tools:
                payload["tools"] = [{"type": "function", "name": t["name"], "description": t["description"], "parameters": t["parameters"], "strict": False} for t in tools]
            endpoint = "/responses"
        else:
            converted = []
            for m in messages:
                row = {"role": m["role"], "content": m.get("content", "")}
                if m["role"] == "tool": row["tool_call_id"] = m["tool_call_id"]
                if m.get("tool_calls"):
                    row["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]]
                if m["role"] == "assistant" and m.get("reasoning_content") is not None:
                    row["reasoning_content"] = m["reasoning_content"]
                converted.append(row)
            payload = {"model": connection["model"], "messages": converted}
            if deepseek_chat:
                payload["thinking"] = {"type": "enabled"}
            if tools:
                payload["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]
            endpoint = "/chat/completions"
        # DeepSeek thinking mode rejects required/named tool choices. The probe
        # instead requests the call in its prompt and still validates its output.
        if force_tool: payload["tool_choice"] = "auto" if deepseek_chat else "required"
        key = self.vault.decrypt(connection["credential_cipher"])
        try:
            async with httpx.AsyncClient(timeout=connection["timeout_seconds"], trust_env=False, follow_redirects=False) as client:
                response = await client.post(connection["base_url"]+endpoint, json=payload,
                                             headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"})
            if response.status_code >= 300:
                raise PlatformError("MODEL_HTTP_ERROR", f"模型接口返回 HTTP {response.status_code}；请核对地址、模型、凭据及协议", 502)
            body = response.json()
            if protocol == "responses":
                if body.get("status") in {"failed", "incomplete", "cancelled"}:
                    raise PlatformError("MODEL_INCOMPLETE", "模型响应未正常完成", 502)
                items = body["output"]
                text = "".join(c.get("text", "") for item in items if item.get("type") == "message"
                               for c in item.get("content", []) if c.get("type") == "output_text")
                calls = [{"id": i["call_id"], "name": i["name"], "arguments": i["arguments"]} for i in items if i.get("type") == "function_call"]
                return {"text": text, "tool_calls": calls, "response_items": items, "usage": body.get("usage", {})}
            choice = body["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter", "insufficient_system_resource", "aborted"}:
                raise PlatformError("MODEL_INCOMPLETE", "模型响应未正常完成", 502)
            message = choice["message"]
            if message.get("refusal"):
                raise PlatformError("MODEL_REFUSED", "模型拒绝本次请求", 502)
            calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in message.get("tool_calls", [])]
            turn = {"text": message.get("content") or "", "tool_calls": calls, "response_items": [], "usage": body.get("usage", {})}
            if message.get("reasoning_content") is not None:
                if not isinstance(message["reasoning_content"], str):
                    raise PlatformError("MODEL_PROTOCOL_INVALID", "模型返回格式不符合选择的 API 协议", 502)
                turn["reasoning_content"] = message["reasoning_content"]
            return turn
        except httpx.TimeoutException:
            raise PlatformError("MODEL_TIMEOUT", "模型请求超时；没有使用模拟响应替代", 504) from None
        except httpx.HTTPError:
            raise PlatformError("MODEL_CONNECTION_FAILED", "无法连接模型接口；请检查网络和地址", 502) from None
        except (ValueError, KeyError, TypeError, IndexError):
            raise PlatformError("MODEL_PROTOCOL_INVALID", "模型返回格式不符合选择的 API 协议", 502) from None
