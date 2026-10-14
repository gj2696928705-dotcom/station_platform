"""Windows per-user DPAPI; credentials are never stored as plaintext."""
import base64
import ctypes
import os
import secrets
from ctypes import wintypes
from pathlib import Path
from .util import PlatformError, canonical

class Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]

def crypt(value, decrypt=False):
    if os.name != "nt":
        raise PlatformError("SECRET_STORE_UNAVAILABLE", "本版凭据存储需要 Windows DPAPI；不能退回明文存储", 503)
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    output = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    function = library.CryptUnprotectData if decrypt else library.CryptProtectData
    function.restype = wintypes.BOOL
    ok = function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output))
    if not ok:
        raise PlatformError("SECRET_STORE_FAILED", "本机受保护凭据无法读取或保存", 503)
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        kernel.LocalFree(output.data)

class SecretStore:
    def encrypt(self, value):
        return base64.b64encode(crypt(value)).decode("ascii")

    def decrypt(self, value):
        return crypt(base64.b64decode(value), True).decode("utf-8")

def node_token(data_dir, vault=None):
    vault = vault or SecretStore()
    path = Path(data_dir) / "node-identity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import json
        return vault.decrypt(json.loads(path.read_text(encoding="utf-8"))["node_token_cipher"])
    token = secrets.token_urlsafe(40)
    # Identity is created by the start command before the two processes launch.
    path.write_text(canonical({"node_token_cipher": vault.encrypt(token)}) + "\n", encoding="utf-8")
    return token
