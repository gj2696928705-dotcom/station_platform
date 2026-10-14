"""Native regression for stopped Windows processes retaining creation metadata."""
import os
import subprocess
import sys
from station_platform.__main__ import alive, identity, stop_record

def test_native_process_identity_and_stop_wait():
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        record = {"pid": process.pid, "identity": identity(process.pid)}
        assert record["identity"] and alive(record)
        assert not stop_record({"pid":process.pid,"identity":"wrong-creation-time"})
        assert process.poll() is None
        assert stop_record(record)
        process.wait(timeout=3)
        assert not alive(record)
        assert identity(process.pid) is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
