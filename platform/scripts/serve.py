#!/usr/bin/env python3
"""Launch the HyperAgents Control Platform web dashboard + API.

Replicas-first: runs directly inside the workspace. Portable: the same
command works on any Docker host.

Offline mode (default, zero API cost):
    platform/.venv/bin/python platform/scripts/serve.py

With OpenRouter (requires explicit user approval + OPENROUTER_API_KEY):
    HACP_UPSTREAM=openrouter OPENROUTER_API_KEY=... platform/.venv/bin/python platform/scripts/serve.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from hacp.api import create_app  # noqa: E402
from hacp.config import ServiceConfig  # noqa: E402


def main():
    service = ServiceConfig.from_env()
    app = create_app(service)
    host = os.environ.get("HACP_BIND_HOST", "0.0.0.0" if service.host != "127.0.0.1" else "127.0.0.1")
    print(f"[hacp] upstream={service.upstream} home={service.home} shell_model={service.shell_model}")
    print(f"[hacp] listening on http://{host}:{service.port} (dashboard at /)")
    uvicorn.run(app, host=host, port=service.port, log_level="warning")


if __name__ == "__main__":
    main()
