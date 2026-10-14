"""Isolated browser QA fixture; does not configure the user-facing instance."""
import json
import time
from pathlib import Path
from .conftest import Stack
from ..util import identifier

if __name__ == "__main__":
    item = Stack(Path("station_platform/data/qa")/identifier("browser"))
    print(json.dumps({"url":item.urls["center"],"data_dir":str(item.path),"real_model":False,"purpose":"isolated_browser_qa"}), flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        item.close()
