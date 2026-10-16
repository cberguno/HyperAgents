"""Test configuration. All tests run offline: zero API calls, zero cost.

The upstream repository is used read-only: instrumentation tests copy files
into temporary directories; lifecycle tests use the stub loop + stub upstream.
"""
import sys
import threading
import time
from pathlib import Path

import pytest

PLATFORM_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PLATFORM_DIR.parent
sys.path.insert(0, str(PLATFORM_DIR))


@pytest.fixture(scope="session")
def upstream_repo() -> Path:
    """The pristine Meta HyperAgents checkout (read-only in all tests)."""
    assert (REPO_ROOT / "generate_loop.py").exists(), "upstream repo not found"
    return REPO_ROOT


@pytest.fixture
def store(tmp_path):
    """A store on the SAME database the app fixture uses (tmp_path/home), so
    tests can seed/inspect data through either instance."""
    from hacp.store import Store

    (tmp_path / "home" / "data").mkdir(parents=True, exist_ok=True)
    s = Store(tmp_path / "home" / "data" / "hacp.db")
    yield s
    s.close()


@pytest.fixture
def bus(store):
    from hacp.store import EventBus

    return EventBus(store)


@pytest.fixture
def service(tmp_path):
    from hacp.config import ServiceConfig

    return ServiceConfig(
        host="127.0.0.1",
        port=8499,  # not bound in TestClient tests
        home=str(tmp_path / "home"),
        upstream="stub",
        upstream_repo=str(REPO_ROOT),
        proxy_public_base="http://127.0.0.1:8499",  # overridden by live_server
        shell_model="stub/tiny",
    )


@pytest.fixture
def client(service):
    """ASGI test client (no real sockets; stub upstream)."""
    from fastapi.testclient import TestClient

    from hacp.api import create_app

    app = create_app(service)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def live_server_factory():
    """Real uvicorn server on an ephemeral port, for subprocess HTTP tests."""
    servers = []

    def make(service):
        import uvicorn

        from hacp.api import create_app

        app = create_app(service)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        service.proxy_public_base = base
        servers.append((server, thread))
        return base

    yield make

    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def live_server(service, live_server_factory, tmp_path):
    base = live_server_factory(service)
    yield base
