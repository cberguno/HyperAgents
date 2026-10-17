"""Option C — Docker socket proxy: env injection into container creates.

The proxy sits between the experiment's DOCKER_HOST and the real daemon and
rewrites POST /containers/create bodies to merge HACP_* env vars. All other
traffic (including hijacked exec streams) is forwarded transparently.

Tests: pure body-rewrite logic + a loopback integration against a fake docker
daemon on a unix socket (no real docker daemon required).
"""
import json
import os
import socket
import threading
import time
from pathlib import Path

import docker
import pytest

from hacp.docker_proxy import DockerEnvProxy, rewrite_create_body


# --------------------------------------------------------------------- pure logic
def test_rewrite_merges_and_replaces_env():
    body = json.dumps({
        "Image": "hyperagents",
        "Env": ["FOO=bar", "HACP_LLMBASE=http://stale", "PATH=/usr/bin"],
        "Cmd": ["tail", "-f", "/dev/null"],
    }).encode()
    out = rewrite_create_body(body, {"HACP_LLMBASE": "http://new", "HACP_META_MODEL": "z-ai/glm-5.3"})
    data = json.loads(out)
    env = dict(e.split("=", 1) for e in data["Env"])
    assert env["HACP_LLMBASE"] == "http://new"  # replaced, not duplicated
    assert env["HACP_META_MODEL"] == "z-ai/glm-5.3"  # added
    assert env["FOO"] == "bar" and env["PATH"] == "/usr/bin"  # untouched
    assert len(data["Env"]) == 4  # no duplicate HACP_LLMBASE


def test_rewrite_passthrough_for_non_json():
    assert rewrite_create_body(b"not json", {"A": "1"}) == b"not json"


def test_rewrite_adds_env_when_absent():
    out = rewrite_create_body(json.dumps({"Image": "x"}).encode(), {"A": "1"})
    assert json.loads(out)["Env"] == ["A=1"]


# ------------------------------------------------------------- fake docker daemon
class FakeDockerDaemon:
    """Minimal Docker API on a unix socket: _ping, version, containers/create,
    containers/json, and an exec hijack endpoint. Records created bodies."""

    def __init__(self, sock_path):
        self.sock_path = sock_path
        self.created_bodies = []
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(sock_path)
        self._srv.listen(16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._conn, args=(conn,), daemon=True).start()

    @staticmethod
    def _read_head(conn):
        buf = b""
        while b"\r\n\r\n" not in buf:
            data = conn.recv(65536)
            if not data:
                raise ConnectionError
            buf += data
        head, rest = buf.split(b"\r\n\r\n", 1)
        return head, rest

    def _conn(self, conn):
        try:
            while True:
                head, rest = self._read_head(conn)
                lines = head.decode().split("\r\n")
                method, path, _ = lines[0].split(" ")
                headers = {}
                for line in lines[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip().lower()] = v.strip()
                cl = int(headers.get("content-length", "0") or 0)
                body = rest
                while len(body) < cl:
                    body += conn.recv(65536)
                body = body[:cl]

                path_only = path.split("?")[0]
                if path_only == "/_ping" or path_only.endswith("/_ping"):
                    payload = b"OK"
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                                 + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
                elif path_only.endswith("/version"):
                    payload = json.dumps({"Version": "28.3.3", "ApiVersion": "1.51"}).encode()
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                 + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
                elif path_only.endswith("/containers/create"):
                    self.created_bodies.append(json.loads(body.decode()))
                    payload = json.dumps({"Id": "abc123", "Warnings": []}).encode()
                    conn.sendall(b"HTTP/1.1 201 Created\r\nContent-Type: application/json\r\n"
                                 + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
                elif path_only.endswith("/containers/json"):
                    payload = json.dumps([]).encode()
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                 + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
                else:
                    payload = json.dumps({"message": f"no route {path_only}"}).encode()
                    conn.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n"
                                 + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
                if headers.get("connection", "").lower() == "close":
                    return
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


@pytest.fixture
def fake_daemon(tmp_path):
    sock = str(tmp_path / "fake-docker.sock")
    daemon = FakeDockerDaemon(sock)
    yield daemon
    daemon.stop()


def _client_via(proxy: DockerEnvProxy):
    # NOTE: docker-py's zero-argument DockerClient() ignores DOCKER_HOST (only
    # from_env() reads it) — always pass base_url explicitly here. In the real
    # system the sitecustomize shim redirects upstream's no-arg construction.
    return docker.DockerClient(base_url=proxy.docker_host_url)


def test_sitecustomize_shim_redirects_no_arg_dockerclient(tmp_path, fake_daemon):
    """The delivery mechanism for upstream's docker.DockerClient(): with the
    shim on PYTHONPATH and HACP_DOCKER_PROXY_URL set, a *zero-argument*
    DockerClient() is redirected to the proxy and its container creates get
    the platform env — exactly how generate_loop.py constructs its client."""
    import subprocess
    import sys

    proxy = DockerEnvProxy(
        env={"HACP_LLMBASE": "http://127.0.0.1:8420/proxy/v1"},
        docker_socket=fake_daemon.sock_path,
        socket_dir=str(tmp_path / "prox3"),
    )
    proxy.start()
    shim_dir = str(Path(__file__).resolve().parent.parent / "hacp" / "docker_shim")
    try:
        code = (
            "import docker\n"
            "c = docker.DockerClient()  # zero-argument, like upstream generate_loop\n"
            "c.ping()\n"
            "c.api.create_container(image='shimtest', name='shim-n1', environment={'A': 'b'})\n"
            "print('SHIM-OK')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = shim_dir
        env["HACP_DOCKER_PROXY_URL"] = proxy.docker_host_url
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, timeout=120)
        assert "SHIM-OK" in r.stdout, r.stdout + r.stderr
        # the create landed on the FAKE daemon (through the proxy) with env
        assert len(fake_daemon.created_bodies) == 1
        env_list = dict(
            e.split("=", 1) for e in fake_daemon.created_bodies[0].get("Env", [])
        )
        assert env_list["HACP_LLMBASE"] == "http://127.0.0.1:8420/proxy/v1"
        assert env_list["A"] == "b"
    finally:
        proxy.stop()


def test_sitecustomize_shim_inert_without_env(tmp_path):
    """Without HACP_DOCKER_PROXY_URL the shim changes nothing (DockerClient()
    still uses the default socket)."""
    import subprocess
    import sys

    shim_dir = str(Path(__file__).resolve().parent.parent / "hacp" / "docker_shim")
    code = (
        "import docker\n"
        "c = docker.DockerClient()\n"
        "assert '/hacp-docker-' not in c.api.base_url\n"
        "print('INERT-OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = shim_dir
    env.pop("HACP_DOCKER_PROXY_URL", None)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=120)
    assert "INERT-OK" in r.stdout, r.stdout + r.stderr


def test_proxy_injects_env_into_container_creates(tmp_path, fake_daemon):
    proxy = DockerEnvProxy(
        env={"HACP_LLMBASE": "http://127.0.0.1:8420/proxy/v1",
             "HACP_META_MODEL": "openai/gpt-6-luna",
             "HACP_ROUTING_ACTIVE": "1"},
        docker_socket=fake_daemon.sock_path,
        socket_dir=str(tmp_path / "prox"),
    )
    sock = proxy.start()
    assert Path(sock).exists()
    try:
        client = _client_via(proxy)
        assert client.ping()  # transparent GET passthrough
        version = client.version()
        assert version["Version"] == "28.3.3"  # transparent JSON passthrough
        # create through the proxy -> env injected upstream at the daemon
        client.api.create_container(
            image="hyperagents", name="hyperagents-gl-container-run1",
            environment={"EXISTING": "kept"},
        )
        assert len(fake_daemon.created_bodies) == 1
        env_list = fake_daemon.created_bodies[0].get("Env", [])
        env = dict(e.split("=", 1) for e in env_list)
        assert env["HACP_LLMBASE"] == "http://127.0.0.1:8420/proxy/v1"
        assert env["HACP_META_MODEL"] == "openai/gpt-6-luna"
        assert env["HACP_ROUTING_ACTIVE"] == "1"
        assert env["EXISTING"] == "kept"  # caller env preserved
        assert client.containers.list() == []  # transparent list passthrough
    finally:
        proxy.stop()
    assert not Path(sock).exists()


def test_proxy_keeps_hijacked_streams_alive(fake_daemon, tmp_path):
    """Hijacked connections (exec attach, 101 Upgrade) must pass through raw.
    Simulated with a fake 101 endpoint driven by a second fake daemon socket."""
    class HijackDaemon(FakeDockerDaemon):
        def _conn(self, conn):
            head, rest = self._read_head(conn)
            lines = head.decode().split("\r\n")
            method, path, _ = lines[0].split(" ")
            if path.split("?")[0].endswith("/exec/xyz/start"):
                conn.sendall(b"HTTP/1.1 101 UPGRADED\r\nContent-Type: application/vnd.docker.raw-stream\r\n"
                             b"Connection: Upgrade\r\nUpgrade: tcp\r\n\r\n")
                # echo server: send back whatever arrives
                try:
                    while True:
                        data = conn.recv(65536)
                        if not data:
                            break
                        conn.sendall(b"echo:" + data)
                except OSError:
                    pass
                return
            super()._conn(conn)

    sock = str(tmp_path / "hijack.sock")
    daemon = HijackDaemon(sock)
    try:
        proxy = DockerEnvProxy(env={}, docker_socket=sock, socket_dir=str(tmp_path / "prox2"))
        proxy.start()
        try:
            # speak the hijack protocol manually through the proxy
            import urllib.parse

            req = (
                "POST /v1.51/exec/xyz/start HTTP/1.1\r\n"
                "Host: docker\r\n"
                "Connection: Upgrade\r\n"
                "Upgrade: tcp\r\n"
                "Content-Length: 0\r\n\r\n"
            )
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.connect(proxy.socket_path)
            c.sendall(req.encode())
            # read the 101 head
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += c.recv(65536)
            assert b"101" in buf
            # hijacked raw phase: send bytes, expect echo through the proxy
            c.sendall(b"hello-hijack")
            time.sleep(0.3)
            c.settimeout(5)
            got = c.recv(65536)
            assert got == b"echo:hello-hijack"
            c.close()
        finally:
            proxy.stop()
    finally:
        daemon.stop()
