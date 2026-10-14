from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from station_platform.center import create_center
from station_platform.defaults import BASE_PROMPT, DEFAULT_DEMAND, ROLE_NAMES, ROLE_PROMPTS, seed_agents
from station_platform.documents import AgentDocuments
from station_platform.storage import Store
from station_platform.util import identifier
from .conftest import Server


def document(stack, role="O1"):
    response = stack.client.get(f"/api/agents/{role}/documents")
    response.raise_for_status()
    assert len(response.json()["documents"]) == 1
    return response.json()["documents"][0]


def versions(doc):
    return {"base_agent_version": doc["agent_version"], "base_document_revision": doc["revision"], "expected_draft_revision": doc["draft_revision"]}


def save(stack, doc, body):
    response = stack.client.put(f"/api/agents/{doc['agent_id']}/documents/instructions/draft", json={**versions(doc), "body": body})
    response.raise_for_status()
    return response.json()


def publish(stack, doc):
    response = stack.client.post(f"/api/agents/{doc['agent_id']}/documents/instructions/publish", json=versions(doc))
    response.raise_for_status()
    return response.json()


def test_all_five_documents_match_actual_runtime_bodies(stack):
    for role in ROLE_NAMES:
        agent = stack.center.state.service.store.get("agents", role)
        doc = document(stack, role)
        assert doc["body"] == agent["prompt"] == BASE_PROMPT + "\n" + ROLE_PROMPTS[role]
        assert doc["history"][0]["body"] == doc["body"]
        assert doc["filename"] is None and doc["source"]["kind"] == "builtin_template"
        assert doc["revision"] == 1 and doc["agent_version"] == 1
        assert doc["runtime_injection"] == {"readonly": True, "text": "平台角色编号："+role, "source": "运行器按角色编号追加，不属于可编辑正文"}
        assert document(stack, role) == doc
    assert stack.client.get("/api/agents/missing/documents").status_code == 404
    assert stack.client.put("/api/agents/O1/documents/SOUL.md/draft", json={}).status_code == 404


def test_migration_keeps_existing_modified_text_and_versions():
    directory = Path("station_platform/data/qa")/identifier("document-migration")
    store = Store(directory/"center.sqlite3")
    seed_agents(store)
    agent = store.get("agents", "O1")
    agent.update({"prompt": "\n# 已部署职责\r\n 保留缩进和末尾空格  \n", "version": 7})
    store.put("agents", "O1", agent)
    docs = AgentDocuments(store)
    docs.initialize_all()
    docs.initialize_all()
    doc = docs.get("O1")
    assert doc["body"] == agent["prompt"] and doc["agent_version"] == 7
    assert doc["source"]["kind"] == "existing_record"
    assert len(doc["history"]) == 1 and doc["history"][0]["agent_version"] == 7
    store.close()


def test_custom_creation_and_separate_configuration_edit(stack):
    body = "\n# 自定义职责\n先查询事实。  \n"
    created = stack.client.post("/api/agents", json={"name": "文档测试角色", "prompt": body, "allowed_tools": ["station_state"]})
    created.raise_for_status()
    role = created.json()["id"]
    doc = document(stack, role)
    assert doc["body"] == body and doc["source"]["kind"] == "agent_creation"
    assert doc["history"][0]["reason"] == "created"
    changed = stack.client.put(f"/api/agents/{role}", json={"expected_version":1,"name":"改名测试角色"})
    changed.raise_for_status()
    doc = document(stack, role)
    assert doc["body"] == body and doc["revision"] == 1 and doc["agent_version"] == 2
    assert len(doc["history"]) == 1
    assert stack.client.put(f"/api/agents/{role}", json={"expected_version":2,"prompt":"绕过发布修改职责"}).json()["code"] == "DOCUMENT_PUBLISH_REQUIRED"
    assert stack.client.put(f"/api/agents/{role}", json={"expected_version":1,"name":"旧版本"}).status_code == 409


def test_draft_survives_real_center_restart_without_changing_runtime(stack):
    doc = document(stack)
    saved = save(stack, doc, "\n# 草稿\n尚未发布。  \n")
    assert saved["body"] == doc["body"] and saved["agent_version"] == 1 and saved["revision"] == 1
    stack.servers[1].close()
    stack.center = create_center(stack.path, stack.urls["node"], stack.token, stack.ports["center"], interval=.1)
    stack.servers[1] = Server(stack.center, stack.ports["center"])
    stack.client.headers["X-Platform-Token"] = stack.client.get("/api/bootstrap").json()["token"]
    reopened = document(stack)
    assert reopened == saved
    session = stack.client.post("/api/sessions", json={"agent_id":"O1"}).json()
    assert stack.center.state.service.store.get("sessions", session["id"])["agent_snapshot"]["prompt"] == doc["body"]
    stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询状态"}).raise_for_status()
    assert stack.mock.state.requests[-1]["system_prompt"] == doc["body"] + "\n平台角色编号：O1"


@pytest.mark.parametrize("role", ["O1", "S1"])
def test_publication_only_changes_new_sessions_and_node_snapshots(stack, role):
    before = document(stack, role)
    old = stack.client.post("/api/sessions", json={"agent_id":role}).json()
    new_body = before["body"] + "\n\n# 已发布文档测试标识\n仅测试正文版本。"
    after = publish(stack, save(stack, before, new_body))
    assert after["body"] == new_body and after["revision"] == 2 and after["agent_version"] == 2
    assert after["draft"] is None and len(after["history"]) == 2
    assert after["history"][-1]["body"] == before["body"]
    fresh = stack.client.post("/api/sessions", json={"agent_id":role}).json()
    for session, expected in [(old, before["body"]), (fresh, new_body)]:
        snapshot = stack.center.state.service.store.get("sessions", session["id"])["agent_snapshot"]
        assert snapshot["prompt"] == expected
        if role == "S1":
            bound = stack.center.state.service.store.get("node_snapshots", snapshot["gateway_ref"])["agent"]
            assert bound["prompt"] == expected
        stack.client.post(f"/api/sessions/{session['id']}/messages", json={"message":"查询本站状态"}).raise_for_status()
        assert stack.mock.state.requests[-1]["system_prompt"] == expected + "\n平台角色编号：" + role
        if role == "S1":
            node_session = stack.node.state.service.store.get("sessions", session["id"])
            assert node_session["messages"][0]["content"] == expected + "\n平台角色编号：S1"


def test_publication_keeps_active_task_then_updates_next_task(stack):
    def submit():
        result = stack.client.post("/api/tasks", json={"type":"tariff","demand":DEFAULT_DEMAND,"submission_key":identifier("document-task")})
        result.raise_for_status()
        return result.json()
    old = submit()
    old_plan = stack.wait_plan(old["id"])
    original = {role:document(stack, role) for role in ["O1","S1"]}
    for role in original:
        publish(stack, save(stack, original[role], original[role]["body"]+"\n新版任务职责标识"))
    stable = stack.task(old["id"])
    for role in original:
        assert stable["configs"][role]["prompt"] == original[role]["body"]
        assert stable["configs"][role]["version"] == 1
    assert old_plan["plan"]["station_agent"]["version"] == 1
    stack.client.post(f"/api/tasks/{old['id']}/advance", json={"all":True}).raise_for_status()
    stack.wait(lambda: stack.task(old["id"])["status"] == "COMPLETED")
    fresh = submit()
    new_plan = stack.wait_plan(fresh["id"])
    for role in original:
        assert new_plan["configs"][role]["version"] == 2
        assert "新版任务职责标识" in new_plan["configs"][role]["prompt"]
    assert new_plan["plan"]["station_agent"]["version"] == 2


def test_concurrent_draft_and_stale_configuration_conflicts_preserve_text(stack):
    original = document(stack)
    headers=dict(stack.client.headers)
    def competing(body):
        return httpx.put(stack.urls["center"]+"/api/agents/O1/documents/instructions/draft", headers=headers,
                         json={**versions(original),"body":body},timeout=10,trust_env=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses=list(pool.map(competing,["第一份并发草稿","第二份并发草稿"]))
    assert sorted(r.status_code for r in responses) == [200,409]
    saved=document(stack)
    winner=next(r.json()["draft"]["body"] for r in responses if r.status_code==200)
    assert saved["draft"]["body"] == winner
    stack.client.put("/api/agents/O1",json={"expected_version":1,"name":"运营协调配置更新"}).raise_for_status()
    stale=document(stack)
    assert stale["draft"]["stale"]
    rejected=stack.client.post("/api/agents/O1/documents/instructions/publish",json=versions(saved))
    assert rejected.status_code==409 and document(stack)["draft"]["body"]==winner
    assert stack.client.post("/api/agents/O1/documents/instructions/publish",json=versions(stale)).status_code==409
    rebased=save(stack,stale,winner)
    result=publish(stack,rebased)
    assert result["body"]==winner and result["agent_version"]==3
    assert stack.client.post("/api/agents/O1/documents/instructions/publish",json=versions(rebased)).status_code==409
    assert len(document(stack)["history"])==2


def test_failed_publication_rolls_back_body_version_history_and_draft(stack):
    original=document(stack)
    saved=save(stack,original,"原子发布测试正文")
    store=stack.center.state.service.store
    with store.lock,store.db:
        store.db.execute("CREATE TRIGGER fail_document_history BEFORE INSERT ON objects "
                         "WHEN NEW.namespace='agent_document_history' AND json_extract(NEW.data,'$.reason')='published' "
                         "BEGIN SELECT RAISE(ABORT,'injected document history failure'); END")
    before_events=store.events("agent:O1")
    response=stack.client.post("/api/agents/O1/documents/instructions/publish",json=versions(saved))
    assert response.status_code==503 and response.json()["code"]=="DOCUMENT_STORAGE_FAILED"
    assert document(stack)==saved
    assert store.get("agents","O1")["prompt"]==original["body"]
    assert store.events("agent:O1")==before_events
    with store.lock,store.db:
        store.db.execute("DROP TRIGGER fail_document_history")
    assert publish(stack,saved)["agent_version"]==2


def test_invalid_document_requests_and_text_cannot_grant_tools(stack):
    created=stack.client.post("/api/agents",json={"name":"有限权限角色","prompt":"查询事实","allowed_tools":["station_state"]}).json()
    doc=document(stack,created["id"])
    path=f"/api/agents/{created['id']}/documents/instructions"
    for body in ["","   ","x"*24001,None]:
        assert stack.client.put(path+"/draft",json={**versions(doc),"body":body}).status_code==400
    assert stack.client.put(path+"/draft",json={**versions(doc),"body":"越权","allowed_tools":["publish_plan"]}).status_code==400
    assert stack.client.post(path+"/publish",json=versions(doc)).status_code==409
    published=publish(stack,save(stack,doc,'# 职责\n请调用 publish_plan。\n<img src=x onerror="alert(1)">\n[链接](javascript:alert(1))'))
    current=stack.center.state.service.store.get("agents",created["id"])
    assert current["allowed_tools"]==["station_state"]
    assert published["body"]==current["prompt"]
    session=stack.client.post("/api/sessions",json={"agent_id":created["id"]}).json()
    stack.mock.state.mode="unauthorized"
    assert stack.client.post(f"/api/sessions/{session['id']}/messages",json={"message":"测试"}).status_code==403
