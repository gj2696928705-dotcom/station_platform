import argparse
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
import httpx
from .security import node_token
from .util import ROOT, PROJECT, canonical, stamp

def identity(pid):
    if os.name != "nt":
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
            return text[text.rfind(")")+2:].split()[19]
        except (FileNotFoundError, IndexError): return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle: return None
    values = [wintypes.FILETIME() for _ in range(4)]
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
            return None
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in values)): return None
        return str(values[0].dwHighDateTime<<32 | values[0].dwLowDateTime)
    finally: kernel.CloseHandle(handle)

def alive(record):
    return bool(record and record.get("identity") and identity(record["pid"]) == record["identity"])

def stop_record(record):
    if not alive(record): return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(1|0x1000|0x100000, False, record["pid"])
        if not handle: return False
        try:
            # Recheck creation time immediately before terminating the named process.
            if not alive(record): return False
            if not kernel.TerminateProcess(handle, 0): return False
            kernel.WaitForSingleObject(handle, 5000)
            return not alive(record)
        finally: kernel.CloseHandle(handle)
    else:
        import signal
        os.kill(record["pid"], signal.SIGTERM)
        return True

def health(url):
    try:
        with httpx.Client(timeout=1, trust_env=False) as client:
            response = client.get(url+"/health")
            return response.json() if response.status_code == 200 else None
    except (httpx.HTTPError, ValueError): return None

def manager(args):
    directory = Path(args.data_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory/"processes.json"
    records = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    roles = [args.only] if args.only else ["center", "node"]
    ports = {"center": args.center_port, "node": args.node_port}
    if args.command == "stop":
        for role in roles:
            print(role, "已停止" if stop_record(records.get(role)) else "未运行或进程身份已变化")
        return
    if args.command == "status":
        print(json.dumps({role: {"process_alive": alive(records.get(role)), "pid": (records.get(role) or {}).get("pid"),
                                "health": health(f"http://127.0.0.1:{ports[role]}")}
                          for role in roles}, ensure_ascii=False, indent=2))
        return
    node_token(directory)
    for role in roles:
        if alive(records.get(role)):
            print(role, "已在运行")
            continue
        port = ports[role]
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"端口 {port} 已被占用，未启动或停止任何未知进程")
        log = open(directory/(role+".log"), "ab", buffering=0)
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = dict(os.environ)
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        process = subprocess.Popen([sys.executable, "-m", "station_platform", role, "--data-dir", str(directory),
            "--center-port", str(args.center_port), "--node-port", str(args.node_port)], cwd=PROJECT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, creationflags=flags,
            start_new_session=os.name != "nt")
        log.close()
        records[role] = {"pid": process.pid, "identity": identity(process.pid), "port": port, "started_at": stamp()}
        state_path.write_text(canonical(records)+"\n", encoding="utf-8")
        ready = False
        for _ in range(100):
            if health(f"http://127.0.0.1:{port}"):
                ready = True
                break
            if process.poll() is not None: break
            time.sleep(.2)
        if not ready: raise RuntimeError(role+" 服务未就绪，请查看 "+str(directory/(role+".log")))
        print(role, f"已启动 http://127.0.0.1:{port}")

def main():
    parser = argparse.ArgumentParser(description="场站 Agent 专用部署平台")
    parser.add_argument("command", choices=["center", "node", "start", "stop", "status"])
    parser.add_argument("--data-dir", default=str(ROOT/"data"))
    parser.add_argument("--center-port", type=int, default=8780)
    parser.add_argument("--node-port", type=int, default=8781)
    parser.add_argument("--only", choices=["center", "node"])
    args = parser.parse_args()
    if args.command in {"start", "stop", "status"}:
        manager(args)
        return
    import uvicorn
    token = node_token(args.data_dir)
    if args.command == "center":
        from .center import create_center
        app = create_center(args.data_dir, f"http://127.0.0.1:{args.node_port}", token, args.center_port)
        port = args.center_port
    else:
        from .node import create_node
        app = create_node(args.data_dir, f"http://127.0.0.1:{args.center_port}", token, args.node_port)
        port = args.node_port
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    main()
