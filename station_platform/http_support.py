import secrets
from fastapi import Request
from fastapi.responses import JSONResponse
from .util import PlatformError

def install_errors(app):
    @app.exception_handler(PlatformError)
    async def platform_error(request, error):
        return JSONResponse({"code": error.code, "message": error.message}, status_code=error.status)

def require_node(request, token):
    actual = request.headers.get("Authorization", "")
    if not secrets.compare_digest(actual, "Bearer "+token):
        raise PlatformError("NODE_UNAUTHORIZED", "节点身份无效", 401)

def local_guard(app, port, browser_token=None):
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}", "testserver"}:
            return JSONResponse({"code": "HOST_INVALID", "message": "本版只允许本机访问"}, status_code=403)
        length = request.headers.get("content-length", "0")
        if not length.isdigit() or int(length) > 2_000_000:
            return JSONResponse({"message": "请求体过大或长度无效"}, status_code=413)
        if browser_token and request.method not in {"GET", "HEAD", "OPTIONS"} and not request.url.path.startswith("/internal/"):
            origin = request.headers.get("origin")
            if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}", "http://testserver"}:
                return JSONResponse({"message": "请求来源无效"}, status_code=403)
            if not secrets.compare_digest(request.headers.get("X-Platform-Token", ""), browser_token):
                return JSONResponse({"message": "请刷新本机页面后重试"}, status_code=403)
        return await call_next(request)
