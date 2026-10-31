"""Locate the platform only for integration examples, never in the numeric core."""
import os
from pathlib import Path
import sys


def enable_platform():
    root = Path(os.environ.get("A01_PLATFORM_ROOT", Path(__file__).resolve().parents[3]/"总平台")).resolve()
    if not (root/"station_platform/domain.py").is_file():
        raise RuntimeError("需要通过 A01_PLATFORM_ROOT 指定总平台源码目录")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
