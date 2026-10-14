import json
import sqlite3
import threading
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from .util import ACTIVE, PlatformError, canonical, public, redact, stamp

class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS objects (
                namespace TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY(namespace,id));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
                at TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_station ON objects
                (json_extract(data,'$.station_id')) WHERE namespace='tasks'
                AND json_extract(data,'$.status') IN
                ('QUEUED','RUNNING','DISPATCHING','WAITING_FEEDBACK','FINALIZING','CANCEL_REQUESTED');
        """)
        self.db.commit()

    def get(self, namespace, id, default=None):
        with self.lock:
            row = self.db.execute("SELECT data FROM objects WHERE namespace=? AND id=?", (namespace, id)).fetchone()
            return json.loads(row[0]) if row else deepcopy(default)

    @contextmanager
    def transaction(self):
        """One atomic multi-record write; callers use the yielded connection.

        Do not call put/event inside this scope: their connection contexts commit.
        """
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def all(self, namespace):
        with self.lock:
            return [json.loads(row[0]) for row in self.db.execute(
                "SELECT data FROM objects WHERE namespace=? ORDER BY updated_at DESC,id", (namespace,))]

    def put(self, namespace, id, value):
        with self.lock, self.db:
            try:
                self.db.execute("INSERT INTO objects VALUES(?,?,?,?) ON CONFLICT(namespace,id) DO UPDATE SET data=excluded.data,updated_at=excluded.updated_at",
                                (namespace, id, canonical(value), stamp()))
            except sqlite3.IntegrityError:
                raise PlatformError("STATION_BUSY", "本站已有活动任务，请完成或取消后再提交", 409) from None
        return deepcopy(value)

    def create_task(self, value):
        with self.lock, self.db:
            prior = self.get("submission_keys", value["submission_key"])
            if prior:
                if prior["fingerprint"] != value["submission_fingerprint"]:
                    raise PlatformError("IDEMPOTENCY_CONFLICT", "相同提交编号对应不同需求", 409)
                return self.get("tasks", prior["task_id"]), False
            try:
                self.db.execute("INSERT INTO objects VALUES(?,?,?,?)", ("tasks", value["id"], canonical(value), stamp()))
            except sqlite3.IntegrityError:
                raise PlatformError("STATION_BUSY", "本站已有活动任务，请完成或取消后再提交", 409) from None
            self.db.execute("INSERT INTO objects VALUES(?,?,?,?)", ("submission_keys", value["submission_key"],
                            canonical({"task_id": value["id"], "fingerprint": value["submission_fingerprint"]}), stamp()))
            return deepcopy(value), True

    def update(self, namespace, id, function):
        with self.lock, self.db:
            value = self.get(namespace, id)
            if value is None:
                raise PlatformError("NOT_FOUND", "记录不存在", 404)
            result = function(value)
            return self.put(namespace, id, value if result is None else result)

    def event(self, owner, kind, data=None, secrets=()):
        clean = redact(public(data or {}), secrets)
        with self.lock, self.db:
            cursor = self.db.execute("INSERT INTO events(owner,at,kind,data) VALUES(?,?,?,?)",
                                     (owner, stamp(), kind, canonical(clean)))
            return cursor.lastrowid

    def events(self, owner, after=0, limit=500):
        with self.lock:
            rows = self.db.execute("SELECT * FROM events WHERE owner=? AND id>? ORDER BY id LIMIT ?", (owner, after, limit))
            return [{"id": r["id"], "at": r["at"], "kind": r["kind"], "data": json.loads(r["data"])} for r in rows]

    def close(self):
        with self.lock:
            self.db.close()

    def global_events(self, after=0, limit=200):
        with self.lock:
            return [{"id": r["id"], "owner": r["owner"], "at": r["at"], "kind": r["kind"], "data": json.loads(r["data"])}
                    for r in self.db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (after, limit))]
