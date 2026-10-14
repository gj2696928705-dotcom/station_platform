import asyncio
import json
import time
from copy import deepcopy
from pathlib import Path
import httpx
import pytest
from station_platform.defaults import DEFAULT_DEMAND
from station_platform.domain import capability, request_from_site, resources, schedule, evaluation, check_local
from station_platform.models import ModelClient
from station_platform.security import SecretStore
from station_platform.storage import Store
from station_platform.util import PlatformError, digest, identifier

def submit(stack, type="tariff", budget=250, key=None, demand=None):
    payload = {"type": type, "demand": demand or deepcopy(DEFAULT_DEMAND), "submission_key": key or identifier("submit")}
    if type == "vpp": payload["budget_kw"] = budget
    response = stack.client.post("/api/tasks", json=payload)
    assert response.status_code == 200, response.text
    return response.json()

@pytest.mark.parametrize("stack", ["responses", "chat_completions"], indirect=True)
def test_two_api_protocols_tariff_closed_loop(stack):
    task = submit(stack)
    pending = stack.wait_plan(task["id"])
    assert pending["artifacts"]["resource"]["forecast_status"] == "not_used"
    assert pending["artifacts"]["evaluation"]["objective_evaluation"]["purchase_cost"]["value"] >= 0
    reply = stack.client.post(f"/api/tasks/{task['id']}/advance", json={"all": True})
    assert reply.status_code == 200 and reply.json()["status"] == "COMPLETED", reply.text
    result = stack.wait(lambda: stack.task(task["id"]) if stack.task(task["id"])["status"] == "COMPLETED" else None)
    assert len(result["receipt"]["steps"]) == 4
    totals = {v["vehicle_id"]: sum(s["delivered_kwh"][v["vehicle_id"]] for s in result["receipt"]["steps"]) for v in DEFAULT_DEMAND["vehicles"]}
    assert totals == {v["vehicle_id"]:v["requested_energy_kwh"] for v in DEFAULT_DEMAND["vehicles"]}
    events = stack.client.get(f"/api/tasks/{task['id']}/events").json()
    assert {"O1","O2","O3","S1"} <= {e["data"].get("agent_id") for e in events}
    assert all(step["simulation"] and step["business_at"] != step["wall_at"] for step in result["receipt"]["steps"])

def test_vpp_all_five_roles_and_budget(stack):
    task = submit(stack, "vpp", 250)
    stack.wait_plan(task["id"])
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"all":True}).raise_for_status()
    result = stack.wait(lambda: stack.task(task["id"]) if stack.task(task["id"])["status"] == "COMPLETED" else None)
    assert result["artifacts"]["business_report"]["budget_met"]
    assert result["artifacts"]["business_report"]["real_settlement"] is False
    events = stack.client.get(f"/api/tasks/{task['id']}/events").json()
    assert {"S1","O1","O2","O3","O4"} <= {e["data"].get("agent_id") for e in events}
    assert max(s["charging_kw"] for s in result["receipt"]["steps"]) <= 250

def test_submission_and_execution_idempotency(stack):
    key = identifier("same")
    task = submit(stack, key=key)
    assert submit(stack, key=key)["id"] == task["id"]
    pending = stack.wait_plan(task["id"])
    with httpx.Client(headers={"Authorization":"Bearer "+stack.token}, trust_env=False) as client:
        repeat = client.post(stack.urls["node"]+"/internal/plans", json=pending["plan"])
        assert repeat.status_code == 200 and repeat.json()["status"] == "SIMULATING"
        for _ in range(2):
            assert client.post(stack.urls["node"]+"/internal/plans/"+pending["plan_id"]+"/advance", json={"step":0}).status_code == 200
        assert len(client.get(stack.urls["node"]+"/internal/plans/"+pending["plan_id"]).json()["steps"]) == 1
    other = stack.client.post("/api/tasks", json={"type":"vpp","demand":DEFAULT_DEMAND,"budget_kw":250,"submission_key":identifier("other")})
    assert other.status_code == 409

def test_budget_unmet_is_blocked_not_faked(stack):
    task = submit(stack, "vpp", 100)
    result = stack.wait(lambda: stack.task(task["id"]) if stack.task(task["id"])["status"] in {"BLOCKED","FAILED"} else None)
    assert result["status"] == "BLOCKED"
    assert "plan_id" not in result
    assert result["code"] == "SERVICE_REQUIREMENT_UNMET"

def test_role_and_model_tool_permissions(stack):
    response = stack.client.post("/api/agents", json={"name":"新角色","prompt":"查询状态","allowed_tools":["publish_plan"]})
    assert response.status_code == 403
    agent = stack.client.post("/api/agents", json={"name":"状态助手","prompt":"先调用 station_state","allowed_tools":["station_state"]}).json()
    session = stack.client.post("/api/sessions", json={"agent_id":agent["id"]}).json()
    stack.mock.state.mode = "unauthorized"
    response = stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询状态"})
    assert response.status_code == 403
    assert not stack.client.get("/api/tasks").json()

def test_invalid_tool_arguments_fail_closed(stack):
    stack.mock.state.mode = "bad_args"
    session = stack.client.post("/api/sessions", json={"agent_id":"O2"}).json()
    response = stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询状态"})
    assert response.status_code == 400 and response.json()["code"] == "TOOL_ARGS_INVALID"

def test_provider_failure_never_falls_back(stack):
    stack.mock.state.mode = "http_error"
    session = stack.client.post("/api/sessions", json={"agent_id":"O1"}).json()
    response = stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询状态"})
    assert response.status_code == 502 and response.json()["code"] == "MODEL_HTTP_ERROR"
    assert len(stack.client.get(f"/api/sessions/{session['id']}").json()["display"]) == 1

def test_models_and_logs_hide_credentials(stack):
    assert stack.secret not in stack.client.get("/api/models").text
    assert "credential_cipher" not in stack.client.get("/api/models").text
    stack.mock.state.mode = "leak"
    session = stack.client.post("/api/sessions", json={"agent_id":"O1"}).json()
    response = stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询状态"})
    assert response.status_code == 200 and stack.secret not in response.text
    assert stack.secret not in stack.client.get(f"/api/sessions/{session['id']}").text
    assert stack.secret.encode() not in (stack.path/"center.sqlite3").read_bytes()

def test_expired_data_and_node_auth(stack):
    stack.client.post("/api/station/scenario", json={"name":"stale_data"}).raise_for_status()
    response = stack.client.post("/api/tasks", json={"type":"tariff","demand":DEFAULT_DEMAND,"submission_key":identifier("expired")})
    assert response.status_code == 409 and response.json()["code"] == "STATE_EXPIRED"
    assert httpx.get(stack.urls["node"]+"/internal/state", trust_env=False).status_code == 401
    assert stack.client.post("/internal/model/turn", json={"snapshot_ref":"bogus","messages":[]}).status_code == 401

def test_partial_execution_is_preserved_after_capacity_drop(stack):
    task = submit(stack)
    stack.wait_plan(task["id"])
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"step":0}).raise_for_status()
    stack.client.post("/api/station/scenario", json={"name":"capacity_drop"}).raise_for_status()
    result = stack.wait(lambda: stack.task(task["id"]) if stack.task(task["id"])["status"] == "BLOCKED" else None)
    assert len(result["receipt"]["steps"]) == 1
    assert result["version"] == 1
    assert result["receipt"]["steps"][0]["delivered_kwh"]["关键重卡"] == 50

def test_cancellation_keeps_evidence(stack):
    task = submit(stack)
    stack.wait_plan(task["id"])
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"step":0}).raise_for_status()
    reply = stack.client.post(f"/api/tasks/{task['id']}/cancel", json={}).json()
    assert reply["status"] == "CANCELLED" and len(reply["receipt"]["steps"]) == 1
    assert stack.client.post(f"/api/tasks/{task['id']}/advance", json={"step":1}).status_code == 409

def test_deterministic_hash_and_live_gate(stack):
    task = submit(stack)
    pending = stack.wait_plan(task["id"])
    bad = deepcopy(pending["plan"])
    bad["selected_candidate"]["vehicle_assignments"][0]["power_kw"][0] += 1
    assert check_local(bad, stack.node.state.service.state())["passed"] is False
    before = stack.node.state.service.state()
    assert capability(before)["usable"]
    stack.client.post("/api/station/scenario", json={"name":"protection"}).raise_for_status()
    assert not capability(stack.node.state.service.state())["usable"]

def test_node_role_gateway_rejects_unknown_snapshot(stack):
    response = stack.client.post("/internal/model/turn", headers={"Authorization":"Bearer "+stack.token}, json={"snapshot_ref":"not-a-registered-agent","messages":[]})
    assert response.status_code == 403

def test_dpapi_roundtrip_and_timeout(stack):
    vault = SecretStore()
    encrypted = vault.encrypt("a throwaway key")
    assert vault.decrypt(encrypted) == "a throwaway key"
    record = stack.center.state.service.connections.get(stack.connection["ref"])
    record["timeout_seconds"] = .03
    stack.mock.state.mode = "delay"
    with pytest.raises(PlatformError) as error:
        asyncio.run(ModelClient(vault).turn(record, [{"role":"user","content":"test"}], []))
    assert error.value.code == "MODEL_TIMEOUT"

def test_durable_tasks_and_sessions(stack):
    task = submit(stack)
    pending = stack.wait_plan(task["id"])
    session = stack.client.post("/api/sessions", json={"agent_id":"O1"}).json()
    stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询状态"}).raise_for_status()
    reopened = Store(stack.path/"center.sqlite3")
    assert reopened.get("tasks", task["id"])["plan_id"] == pending["plan_id"]
    assert len(reopened.get("sessions", session["id"])["display"]) == 2
    reopened.close()

def test_cancel_tombstone_blocks_late_publication(stack):
    task = submit(stack)
    pending = stack.wait_plan(task["id"])
    plan = deepcopy(pending["plan"])
    plan["id"] = identifier("late-plan")
    with httpx.Client(headers={"Authorization":"Bearer "+stack.token}, trust_env=False) as client:
        cancelled = client.post(stack.urls["node"]+"/internal/plans/"+plan["id"]+"/cancel", json={})
        assert cancelled.status_code == 200
        assert client.post(stack.urls["node"]+"/internal/plans", json=plan).status_code == 409

def test_center_restart_observes_existing_plan_without_republishing(stack):
    from station_platform.center import create_center
    from .conftest import Server
    task = submit(stack)
    pending = stack.wait_plan(task["id"])
    stack.servers[1].close()
    stack.center = create_center(stack.path, stack.urls["node"], stack.token, stack.ports["center"], interval=.1)
    stack.servers[1] = Server(stack.center, stack.ports["center"])
    stack.client.headers["X-Platform-Token"] = stack.client.get("/api/bootstrap").json()["token"]
    result = stack.wait_plan(task["id"])
    assert result["plan_id"] == pending["plan_id"]
    assert len(stack.node.state.service.store.all("plans")) == 1
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"all":True}).raise_for_status()
    stack.wait(lambda: stack.task(task["id"])["status"] == "COMPLETED")

def test_node_restart_retains_partial_receipts_without_replay(stack):
    from station_platform.node import create_node
    from .conftest import Server
    task = submit(stack)
    stack.wait_plan(task["id"])
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"step":0}).raise_for_status()
    stack.servers[2].close()
    stack.node = create_node(stack.path, stack.urls["center"], stack.token, stack.ports["node"], interval=.1)
    stack.servers[2] = Server(stack.node, stack.ports["node"])
    result = stack.wait(lambda: stack.task(task["id"]) if stack.task(task["id"])["status"] == "INTERRUPTED" else None)
    assert len(result["receipt"]["steps"]) == 1
    assert stack.client.post(f"/api/tasks/{task['id']}/advance", json={"step":1}).status_code == 409

def test_model_config_version_is_frozen_for_active_task(stack):
    task = submit(stack)
    pending = stack.wait_plan(task["id"])
    updated = stack.client.post("/api/models", json={"id":stack.connection["id"],"name":"修改连接","model":"DIFFERENT_TEST_MODEL",
        "base_url":stack.urls["mock"]+"/v1","protocol":"chat_completions","api_key":""}).json()
    assert updated["version"] == 2 and not updated["verified"]
    assert pending["configs"]["O1"]["connection_ref"] == stack.connection["ref"]
    stack.client.post(f"/api/tasks/{task['id']}/advance", json={"all":True}).raise_for_status()
    stack.wait(lambda: stack.task(task["id"])["status"] == "COMPLETED")

def test_chat_creates_reviewable_draft_before_task_execution(stack):
    stack.mock.state.mode = "draft"
    session = stack.client.post("/api/sessions", json={"agent_id":"O1"}).json()
    reply = stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"根据完整的三辆车测试需求生成任务草稿。"})
    assert reply.status_code == 200
    record = stack.client.get(f"/api/sessions/{session['id']}").json()
    assert len(record["drafts"]) == 1
    assert record["drafts"][0]["payload"]["demand"] == DEFAULT_DEMAND
    assert stack.client.get("/api/tasks").json() == []

def test_custom_agent_reads_authorized_tool_facts(stack):
    agent = stack.client.post("/api/agents", json={"name":"本站状态解释助手","prompt":"调用 station_state 后解释依据。","allowed_tools":["station_state"]}).json()
    session = stack.client.post("/api/sessions", json={"agent_id":agent["id"]}).json()
    assert stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查看本站状态。"}).status_code == 200
    record = stack.client.get(f"/api/sessions/{session['id']}").json()
    result = next(e["data"]["result"] for e in record["events"] if e["kind"] == "tool_completed")
    assert result["source"] == "station_platform_simulator"
    assert result["station_id"] == "mock-station-001"

def test_unexecuted_plan_recomputes_with_fresh_snapshot(stack):
    task = submit(stack)
    original = stack.wait_plan(task["id"])
    stack.client.post("/api/station/scenario", json={"name":"capacity_drop"}).raise_for_status()
    result = stack.wait(lambda: stack.task(task["id"]) if stack.task(task["id"])["status"] == "BLOCKED" else None)
    assert result["version"] == 2
    assert result["request"]["station_state_snapshot"]["snapshot_id"] != original["request"]["station_state_snapshot"]["snapshot_id"]
    assert result["history"][0]["receipt"]["steps"] == []
    assert result["code"] == "SERVICE_REQUIREMENT_UNMET"

def test_actual_heartbeat_loss_blocks_new_tasks(stack):
    stack.client.post("/api/station/scenario", json={"name":"heartbeat_pause"}).raise_for_status()
    stack.wait(lambda: not stack.client.get("/api/station").json()["online"], timeout=18)
    response = stack.client.post("/api/tasks", json={"type":"tariff","demand":DEFAULT_DEMAND,"submission_key":identifier("offline")})
    assert response.status_code == 409 and response.json()["code"] == "NODE_OFFLINE"
