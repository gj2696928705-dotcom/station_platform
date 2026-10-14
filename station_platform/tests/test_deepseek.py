"""Adapter regressions. MockTransport is test-only, never real-model evidence."""
import asyncio
import json

import httpx
import pytest

from station_platform.models import ModelClient, assistant_message
from station_platform.util import PlatformError


class TestVault:
    def decrypt(self, cipher):
        return "throwaway-adapter-test-key"


def transport_client(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr("station_platform.models.httpx.AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def connection(url="https://api.deepseek.com"):
    return {"protocol": "chat_completions", "base_url": url, "model": "deepseek-v4-pro",
            "credential_cipher": "test-only-cipher", "timeout_seconds": 120}


def test_thinking_context_survives_tool_results_and_next_user_turn(monkeypatch):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["thinking"] == {"type": "enabled"}
        assert request.url.path == "/chat/completions"
        if len(requests) == 1:
            assert body["tool_choice"] == "auto"
            message = {"role": "assistant", "content": None, "reasoning_content": "test-provider-context-1",
                       "tool_calls": [{"id": "call_test", "type": "function",
                                       "function": {"name": "connection_probe", "arguments": '{"value":7}'}}]}
            finish = "tool_calls"
        else:
            assistants = [item for item in body["messages"] if item["role"] == "assistant"]
            assert assistants[0]["reasoning_content"] == "test-provider-context-1"
            assert assistants[0]["tool_calls"][0]["id"] == "call_test"
            assert any(item.get("tool_call_id") == "call_test" for item in body["messages"])
            if len(requests) == 3:
                assert assistants[1]["reasoning_content"] == "test-provider-context-2"
                assert not assistants[1].get("tool_calls")
            message = {"role": "assistant", "content": "tool result read", "reasoning_content": "test-provider-context-2"}
            finish = "stop"
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}]})

    transport_client(monkeypatch, handler)

    async def run():
        client = ModelClient(TestVault())
        tools = [{"name": "connection_probe", "description": "test", "parameters": {"type": "object"}}]
        messages = [{"role": "user", "content": "test tool"}]
        turn = await client.turn(connection(), messages, tools, force_tool=True)
        messages += [assistant_message(turn), {"role": "tool", "tool_call_id": "call_test", "content": '{"value":7}'}]
        turn = await client.turn(connection(), messages, tools)
        messages += [assistant_message(turn), {"role": "user", "content": "next test user turn"}]
        await client.turn(connection(), messages, tools)

    asyncio.run(run())
    assert len(requests) == 3


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "https://api.deepseek.com.example.test"])
def test_deepseek_options_do_not_change_other_providers(monkeypatch, url):
    def handler(request):
        body = json.loads(request.content)
        assert "thinking" not in body
        assert body["tool_choice"] == "required"
        return httpx.Response(200, json={"choices": [{"message": {"content": "test"}, "finish_reason": "stop"}]})
    transport_client(monkeypatch, handler)
    asyncio.run(ModelClient(TestVault()).turn(connection(url), [{"role": "user", "content": "test"}], [], force_tool=True))


def test_incomplete_response_is_not_success(monkeypatch):
    transport_client(monkeypatch, lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": "truncated test"}, "finish_reason": "length"}]}))
    with pytest.raises(PlatformError) as error:
        asyncio.run(ModelClient(TestVault()).turn(connection(), [{"role": "user", "content": "test"}], []))
    assert error.value.code == "MODEL_INCOMPLETE"
