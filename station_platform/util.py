from __future__ import annotations
import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STATION_ID = "mock-station-001"
TERMINAL = {"COMPLETED", "FAILED", "BLOCKED", "CANCELLED", "INTERRUPTED"}
ACTIVE = {"QUEUED", "RUNNING", "DISPATCHING", "WAITING_FEEDBACK", "FINALIZING", "CANCEL_REQUESTED"}

class PlatformError(Exception):
    def __init__(self, code, message, status=400):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)

def now():
    return datetime.now(timezone.utc)

def stamp(value=None):
    return (value or now()).isoformat()

def parse_time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError()
        return result
    except (ValueError, TypeError, AttributeError):
        raise PlatformError("TIME_INVALID", "时间必须包含时区") from None

def identifier(prefix):
    return prefix + "-" + uuid.uuid4().hex

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def finite(value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlatformError("NUMBER_INVALID", "需要有效数值")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise PlatformError("NUMBER_INVALID", "数值超出范围")
    return value

def public(value):
    """Remove credential material recursively from all public representations."""
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k.lower() not in {
            "api_key", "credential", "credential_cipher", "node_token", "node_token_cipher", "authorization"}}
    if isinstance(value, list):
        return [public(v) for v in value]
    return value

def redact(value, secrets=()):
    if isinstance(value, dict):
        return {k: "[已隐藏]" if k.lower() in {"api_key", "credential", "credential_cipher", "node_token", "authorization"}
                else redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        result = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer [已隐藏]", value)
        result = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[已隐藏]", result)
        for secret in secrets:
            if secret:
                result = result.replace(secret, "[已隐藏]")
        return result
    return value
