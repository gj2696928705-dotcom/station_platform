import socket
import threading
import time
from pathlib import Path
import httpx
import pytest
import uvicorn
from station_platform.center import create_center
from station_platform.node import create_node
from station_platform.util import identifier
from .mock_model import create_mock

def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

class Server:
    def __init__(self, app, port):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started: return
            if not self.thread.is_alive(): raise RuntimeError("Local test service failed to start")
            time.sleep(.01)
        raise RuntimeError("Local test service start timeout")
    def close(self):
        self.server.should_exit = True
        self.thread.join(5)

class Stack:
    def __init__(self, path, protocol="responses"):
        self.path = path
        self.ports = {name: port() for name in ["mock", "center", "node"]}
        self.urls = {k: f"http://127.0.0.1:{v}" for k,v in self.ports.items()}
        self.token = identifier("test-node-token")
        self.mock = create_mock()
        self.center = create_center(path, self.urls["node"], self.token, self.ports["center"], interval=.1)
        self.node = create_node(path, self.urls["center"], self.token, self.ports["node"], interval=.1)
        self.servers = []
        for app, key in [(self.mock, "mock"), (self.center, "center"), (self.node, "node")]:
            self.servers.append(Server(app, self.ports[key]))
        self.client = httpx.Client(base_url=self.urls["center"], timeout=15, trust_env=False)
        self.client.headers["X-Platform-Token"] = self.client.get("/api/bootstrap").json()["token"]
        self.wait(lambda: self.client.get("/api/station").json()["online"])
        self.secret = identifier("throwaway-test-api-key")
        self.connection = self.client.post("/api/models", json={"name": "自动测试替身（不是真实模型）", "model": "TEST_SUBSTITUTE",
            "protocol": protocol, "base_url": self.urls["mock"]+"/v1", "api_key": self.secret, "timeout_seconds": 5}).json()
        result = self.client.post(f"/api/models/{self.connection['id']}/test", json={})
        assert result.status_code == 200 and result.json()["verified"], result.text
        self.client.post("/api/settings", json={"default_connection": self.connection["id"]}).raise_for_status()
    def wait(self, predicate, timeout=10):
        end = time.monotonic()+timeout
        while time.monotonic()<end:
            result = predicate()
            if result: return result
            time.sleep(.05)
        raise AssertionError("Timed out waiting for test condition")
    def task(self, id): return self.client.get("/api/tasks/"+id).json()
    def wait_plan(self, id):
        def ready():
            task = self.task(id)
            if task["status"] in {"FAILED","BLOCKED","INTERRUPTED"}: raise AssertionError(task)
            return task if task.get("receipt", {}).get("status") == "SIMULATING" else None
        # This checks completion, not response speed. A full chain can exceed
        # ten seconds on Windows while every individual HTTP call succeeds.
        return self.wait(ready, timeout=30)
    def close(self):
        self.client.close()
        for s in reversed(self.servers): s.close()

@pytest.fixture
def stack(request):
    directory = Path("station_platform/data/qa")/identifier("case")
    directory.mkdir(parents=True)
    protocol = getattr(request, "param", "responses")
    item = Stack(directory, protocol)
    try: yield item
    finally: item.close()
