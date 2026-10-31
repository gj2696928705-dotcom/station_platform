from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capacity_alignment.platform_support import enable_platform
enable_platform()

# Reuse the existing local HTTP/model substitute; keep all new evidence here.
from station_platform.tests.conftest import Stack
from station_platform.util import identifier


@pytest.fixture
def stack(request):
    directory = Path(__file__).resolve().parents[1]/"evidence"/"platform-test-data"/identifier("case")
    directory.mkdir(parents=True)
    item = Stack(directory, getattr(request, "param", "responses"))
    try:
        yield item
    finally:
        item.close()
