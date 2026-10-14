"""Actual role instructions, staged edits and immutable publication evidence.

agents.prompt remains the sole current body. History is evidence, never a second
editable runtime source. No filesystem content is invented or served here.
"""
import sqlite3

from .defaults import BASE_PROMPT, ROLE_PROMPTS
from .util import PlatformError, canonical, digest, stamp

DOCUMENT_ID = "instructions"


class AgentDocuments:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def write(db, namespace, id, value):
        db.execute("INSERT INTO objects VALUES(?,?,?,?) ON CONFLICT(namespace,id) "
                   "DO UPDATE SET data=excluded.data,updated_at=excluded.updated_at",
                   (namespace, id, canonical(value), stamp()))

    def agent(self, id):
        agent = self.store.get("agents", id)
        if not agent:
            raise PlatformError("NOT_FOUND", "角色不存在", 404)
        return agent

    def initialize(self, db, agent, created=False):
        if self.store.get("agent_document_meta", agent["id"]):
            return
        matches_seed = agent.get("builtin") and agent["id"] in ROLE_PROMPTS and agent["prompt"] == BASE_PROMPT + "\n" + ROLE_PROMPTS[agent["id"]]
        if created:
            source = {"kind": "agent_creation", "description": "创建角色时提交的运行职责正文", "reference": "创建角色接口 → agents.prompt"}
        elif matches_seed:
            source = {"kind": "builtin_template", "description": "与预置角色模板一致；首次登记逐字保留数据库当前正文", "reference": "station_platform/defaults.py → agents.prompt"}
        else:
            source = {"kind": "existing_record", "description": "首次登记逐字保留已部署角色正文；更早的修改来源未记录", "reference": "agents.prompt（已有数据库记录）"}
        at = stamp()
        meta = {"agent_id": agent["id"], "document_id": DOCUMENT_ID, "revision": 1, "draft_revision": 0,
                "published_at": at, "published_agent_version": agent["version"], "source": source}
        self.write(db, "agent_document_meta", agent["id"], meta)
        self.insert_history(db, agent, meta, "created" if created else "baseline")

    def initialize_all(self):
        with self.store.transaction() as db:
            for agent in self.store.all("agents"):
                self.initialize(db, agent)

    @staticmethod
    def insert_history(db, agent, meta, reason):
        record = {"agent_id": agent["id"], "document_id": DOCUMENT_ID, "revision": meta["revision"],
                  "agent_version": agent["version"], "body": agent["prompt"], "published_at": meta["published_at"],
                  "reason": reason, "body_hash": digest(agent["prompt"])}
        # INSERT only. A publication cannot overwrite an earlier revision.
        db.execute("INSERT INTO objects VALUES(?,?,?,?)", ("agent_document_history", f"{agent['id']}@{meta['revision']}", canonical(record), stamp()))

    def get(self, id, document_id=DOCUMENT_ID):
        if document_id != DOCUMENT_ID:
            raise PlatformError("NOT_FOUND", "该角色没有这份实际使用的文档", 404)
        with self.store.lock:
            agent = self.agent(id)
            meta = self.store.get("agent_document_meta", id)
            if not meta:
                raise PlatformError("DOCUMENT_NOT_INITIALIZED", "运行文档尚未登记", 409)
            draft = self.store.get("agent_document_drafts", id)
            if draft:
                draft["stale"] = draft["base_agent_version"] != agent["version"] or draft["base_document_revision"] != meta["revision"]
            history = [entry for entry in self.store.all("agent_document_history") if entry["agent_id"] == id]
            return {"agent_id": id, "agent_name": agent["name"], "scope": agent["scope"], "agent_version": agent["version"],
                    "document_id": DOCUMENT_ID, "title": "运行职责", "filename": None, "storage": "中心 SQLite · agents.prompt",
                    "source": meta["source"], "body": agent["prompt"], "revision": meta["revision"],
                    "published_at": meta["published_at"], "published_agent_version": meta["published_agent_version"],
                    "draft_revision": meta["draft_revision"], "draft": draft,
                    "runtime_injection": {"text": "平台角色编号：" + id, "readonly": True, "source": "运行器按角色编号追加，不属于可编辑正文"},
                    "history": sorted(history, key=lambda item: item["revision"], reverse=True)}

    @staticmethod
    def validate(payload, keys):
        if not isinstance(payload, dict) or set(payload) != set(keys):
            raise PlatformError("DOCUMENT_INVALID", "文档请求字段不符合约定；工具权限请在角色配置中管理")
        for name in keys - {"body"}:
            if type(payload[name]) is not int or payload[name] < (0 if name == "expected_draft_revision" else 1):
                raise PlatformError("DOCUMENT_INVALID", "文档请求需要有效的版本信息")
        if "body" in keys and (not isinstance(payload["body"], str) or not payload["body"].strip() or len(payload["body"]) > 24000):
            raise PlatformError("DOCUMENT_INVALID", "运行职责需要为 1～24000 字的非空正文")

    def check_versions(self, agent, meta, payload):
        if payload["base_agent_version"] != agent["version"] or payload["base_document_revision"] != meta["revision"] or payload["expected_draft_revision"] != meta["draft_revision"]:
            raise PlatformError("DOCUMENT_CONFLICT", "角色配置或文档已更新；当前编辑内容和已保存草稿保留，请刷新版本信息并比较后再保存", 409)

    def save_draft(self, id, payload, document_id=DOCUMENT_ID):
        if document_id != DOCUMENT_ID:
            raise PlatformError("NOT_FOUND", "运行文档不存在", 404)
        self.validate(payload, {"body", "base_agent_version", "base_document_revision", "expected_draft_revision"})
        with self.store.transaction() as db:
            agent = self.agent(id)
            meta = self.store.get("agent_document_meta", id)
            self.check_versions(agent, meta, payload)
            meta["draft_revision"] += 1
            draft = {"body": payload["body"], "base_agent_version": agent["version"], "base_document_revision": meta["revision"],
                     "revision": meta["draft_revision"], "updated_at": stamp()}
            self.write(db, "agent_document_meta", id, meta)
            self.write(db, "agent_document_drafts", id, draft)
        return self.get(id)

    def publish(self, id, payload, document_id=DOCUMENT_ID):
        if document_id != DOCUMENT_ID:
            raise PlatformError("NOT_FOUND", "运行文档不存在", 404)
        self.validate(payload, {"base_agent_version", "base_document_revision", "expected_draft_revision"})
        try:
            with self.store.transaction() as db:
                agent = self.agent(id)
                meta = self.store.get("agent_document_meta", id)
                self.check_versions(agent, meta, payload)
                draft = self.store.get("agent_document_drafts", id)
                if not draft:
                    raise PlatformError("DOCUMENT_DRAFT_REQUIRED", "请先保存草稿，再发布", 409)
                if draft["base_agent_version"] != agent["version"] or draft["base_document_revision"] != meta["revision"]:
                    raise PlatformError("DOCUMENT_CONFLICT", "草稿基于旧角色版本，请刷新版本信息并重新保存草稿", 409)
                agent["prompt"] = draft["body"]
                agent["version"] += 1
                agent["updated_at"] = stamp()
                meta.update({"revision": meta["revision"] + 1, "draft_revision": meta["draft_revision"] + 1,
                             "published_at": stamp(), "published_agent_version": agent["version"]})
                self.write(db, "agents", id, agent)
                self.write(db, "agent_document_meta", id, meta)
                self.insert_history(db, agent, meta, "published")
                db.execute("DELETE FROM objects WHERE namespace='agent_document_drafts' AND id=?", (id,))
                db.execute("INSERT INTO events(owner,at,kind,data) VALUES(?,?,?,?)",
                           ("agent:" + id, stamp(), "document_published", canonical({"agent_id": id, "revision": meta["revision"],
                             "agent_version": agent["version"], "body_hash": digest(agent["prompt"])})))
        except sqlite3.Error:
            raise PlatformError("DOCUMENT_STORAGE_FAILED", "文档发布未完成，已回滚；原正文和草稿保留", 503) from None
        return self.get(id)
