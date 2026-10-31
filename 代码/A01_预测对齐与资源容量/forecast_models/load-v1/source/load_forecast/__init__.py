"""Independent, candidate-only station load forecasting."""
from pathlib import Path
import sys

# Optional project-local wheel installation; never installs dependencies at runtime.
_deps = Path(__file__).resolve().parents[1] / ".deps"
if _deps.is_dir():
    sys.path.insert(0, str(_deps))

__version__ = "0.1.0"
