"""Docker API socket proxy that injects platform env vars into container creates.

WHY THIS EXISTS: upstream ``utils/docker_utils.build_container()`` starts the
per-generation containers itself, with no way to pass environment variables,
and the only per-experiment config visible inside those containers is the
bind-mounted workspace — i.e. plain files, which a self-modifying meta-agent
can delete (observed in run ``e6e3168d3b3ca`` gen 2, which deleted
``platform_config.json`` and ``misc/hacp/`` to strip platform routing).

By pointing the experiment's ``DOCKER_HOST`` at this proxy, every container
upstream creates for the run is transparently given ``HACP_*`` environment
variables. A process environment cannot be removed by any file operation, so
LLM routing through the platform proxy survives exactly that attack. All
other Docker API traffic is forwarded byte-transparently (including hijacked
exec/attach streams).

Used together with the runner's pre-session check
(``hacp/runner.py:_guard_containers``): each generation container is probed
to confirm routing is active before the generation's work is trusted.
"""
from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULT_DOCKER_SOCKET = os.environ.get("HACP_DOCKER_SOCK", "/var/run/docker.sock")

_CREATE_PATH_RE = re.compile(r"(?:^|/)containers/create$")


def rewrite_create_body(body: bytes, env: dict) -> bytes:
    """Merge ``env`` into a /containers/create JSON body's Env list.

    Existing entries with the same key are replaced (platform wins), new keys
    are appended. Returns the original bytes untouched for non-JSON bodies.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(data, dict):
        return body
    existing = list(data.get("Env") or [])
    wanted = {str(k): str(v) for k, v in env.items()}
    out = []
    seen = set()
    for entry in existing:
        key = entry.split("=", 1)[0]
        if key in wanted:
            if key not in seen:
                out.append(f"{key}={wanted.pop(key)}")
                seen.add(key)
        else:
            out.append(entry)
    for k, v in wanted.items():
        out.append(f"{k}={v}")
    data["Env"] = out
    return json.dumps(data).encode("utf-8")


class _SockReader:
    """Buffered reader over a socket supporting header/body reads."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def _fill(self, n: int = 65536):
        chunk = self.sock.recv(n)
        if not chunk:
            raise ConnectionError("closed")
        self.buf += chunk

    def read_until(self, marker: bytes) -> bytes:
        while marker not in self.buf:
            self._fill()
        idx = self.buf.index(marker) + len(marker)
        out, self.buf = self.buf[:idx], self.buf[idx:]
        return out

    def read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def read_chunked_body(self) -> bytes:
        """Read a full chunked body (size-line framed), de-chunked."""
        out = b""
        while True:
            head = self.read_until(b"\r\n")
            size = int(head.strip().split(b";")[0], 16)
            if size == 0:
                # trailers until blank line
                while True:
                    line = self.read_until(b"\r\n")
                    if line in (b"\r\n", b""):
                        break
                return out
            out += self.read_exact(size)
            self.read_exact(2)  # CRLF after chunk


class DockerEnvProxy:
    """A unix-socket proxy in front of the real Docker daemon.

    ``start()`` returns the proxy socket path; point ``DOCKER_HOST`` at
    ``unix://<path>`` for any process whose container creates should carry the
    platform environment (the experiment's generate_loop subprocess).
    """

    def __init__(self, env: dict, docker_socket: str = DEFAULT_DOCKER_SOCKET,
                 socket_dir: Optional[str] = None):
        self.env = {str(k): str(v) for k, v in env.items()}
        self.docker_socket = docker_socket
        self._socket_dir = socket_dir
        self._sock_path: Optional[str] = None
        self._server: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> str:
        base = Path(self._socket_dir) if self._socket_dir else Path(
            os.environ.get("HACP_HOME", str(Path.home()))
        ) / "docker-proxies"
        base.mkdir(parents=True, exist_ok=True)
        # AF_UNIX paths are capped at ~108 chars: keep the name short
        name = f"hacp-{os.getpid()}-{int(time.time() * 1000)}.sock"
        path = str(base / name)
        if len(path) > 104:  # guard: fall back to /tmp
            path = f"/tmp/{name}"
            Path("/tmp").mkdir(exist_ok=True)
        self._sock_path = path
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self._sock_path)
        os.chmod(self._sock_path, 0o660)
        srv.listen(32)
        srv.settimeout(None)
        self._server = srv
        t = threading.Thread(target=self._accept_loop, daemon=True,
                             name="hacp-docker-proxy")
        t.start()
        return self._sock_path

    def stop(self):
        self._stopped.set()
        with self._lock:
            srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        if self._sock_path:
            try:
                os.unlink(self._sock_path)
            except OSError:
                pass

    @property
    def socket_path(self) -> str:
        if not self._sock_path:
            raise RuntimeError("proxy not started")
        return self._sock_path

    @property
    def docker_host_url(self) -> str:
        if not self._sock_path:
            raise RuntimeError("proxy not started")
        return f"unix://{self._sock_path}"

    # -- serving ---------------------------------------------------------------
    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, client: socket.socket):
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.settimeout(None)
            upstream.connect(self.docker_socket)
        except OSError:
            try:
                client.close()
            except OSError:
                pass
            return
        try:
            self._conversation(client, upstream)
        except (ConnectionError, OSError):
            pass
        finally:
            for s in (client, upstream):
                try:
                    s.close()
                except OSError:
                    pass

    def _conversation(self, client: socket.socket, upstream: socket.socket):
        """Sequential HTTP/1.1 conversation with keep-alive, chunked and
        hijack support. Requests are rewritten when they create containers."""
        cr = _SockReader(client)
        ur = _SockReader(upstream)
        while True:
            try:
                head = cr.read_until(b"\r\n\r\n")
            except ConnectionError:
                return
            # request line: METHOD /path HTTP/1.1
            lines = head.decode("latin-1", "replace").split("\r\n")
            req_line = lines[0]
            parts = req_line.split(" ")
            if len(parts) != 3:
                return
            method, path, _version = parts
            headers = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            # request body
            body = b""
            if headers.get("transfer-encoding", "").lower() == "chunked":
                body = cr.read_chunked_body()
                # we forward content-length framed
                fwd_headers = {k: v for k, v in headers.items()
                               if k != "transfer-encoding"}
                fwd_headers["content-length"] = str(len(body))
                headers = fwd_headers
            else:
                cl = int(headers.get("content-length", "0") or 0)
                if cl > 0:
                    body = cr.read_exact(cl)

            # rewrite container creates
            path_only = path.split("?", 1)[0]
            if method.upper() == "POST" and _CREATE_PATH_RE.search(path_only):
                body = rewrite_create_body(body, self.env)

            # forward request
            out = [req_line]
            for k, v in headers.items():
                if k == "content-length":
                    v = str(len(body))
                out.append(f"{k}: {v}")
            out.append("")
            req = ("\r\n".join(out) + "\r\n").encode("latin-1") + body
            upstream.sendall(req)

            # response head
            try:
                rhead = ur.read_until(b"\r\n\r\n")
            except ConnectionError:
                return
            rlines = rhead.decode("latin-1", "replace").split("\r\n")
            status_line = rlines[0]
            rheaders = {}
            for line in rlines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    rheaders[k.strip().lower()] = v.strip()

            upgrade = "upgrade" in rheaders.get("connection", "").lower()
            try:
                _status_code = status_line.split()[1]
            except IndexError:
                _status_code = ""
            if upgrade or _status_code == "101":
                # hijacked stream: raw bidirectional pump until close
                client.sendall(rhead)
                self._pump(client, upstream)
                return

            # Bodiless responses (204 No Content, 304 Not Modified, 1xx, and
            # HEAD replies) carry no framing headers. The daemon keeps the
            # connection alive, so we must NOT fall through to the
            # read-until-upstream-closes branch: doing so parks this thread
            # forever while the client (docker-py connection pooling) reuses
            # the connection for its NEXT request — which then never gets
            # forwarded and the client blocks awaiting a response. This
            # exactly stalled POST /containers/<id>/start (upstream calls
            # container.start() after containers.run() already started it) in
            # run e0d7745dc1575 attempts 2 and 3.
            if (
                _status_code in ("204", "304")
                or _status_code.startswith("1")
                or method.upper() == "HEAD"
            ):
                client.sendall(rhead)
                continue  # keep-alive: next request

            # response body framing
            if rheaders.get("transfer-encoding", "").lower() == "chunked":
                client.sendall(rhead)
                # forward chunks incrementally (streams!) then trailers
                self._forward_chunked(ur, client)
                continue  # keep-alive: next request
            rcl = rheaders.get("content-length")
            if rcl is not None:
                client.sendall(rhead)
                remaining = int(rcl)
                while remaining > 0:
                    chunk = ur.read_exact(min(remaining, 262144))
                    client.sendall(chunk)
                    remaining -= len(chunk)
                continue  # keep-alive
            # no framing: read until upstream closes this response
            client.sendall(rhead)
            while True:
                try:
                    data = ur.sock.recv(65536)
                except OSError:
                    break
                if not data:
                    break
                client.sendall(data)
            return  # cannot keep-alive safely; close

    @staticmethod
    def _pump(a: socket.socket, b: socket.socket):
        """Raw bidirectional byte pump until either side closes."""

        def one(src, dst, errors: list):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                errors.append(1)
                try:
                    dst.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        errs: list = []
        t1 = threading.Thread(target=one, args=(a, b, errs), daemon=True)
        t2 = threading.Thread(target=one, args=(b, a, errs), daemon=True)
        t1.start()
        t2.start()
        while not errs:
            time.sleep(0.05)
        # give the other thread a moment to unblock, then return
        t1.join(timeout=2)
        t2.join(timeout=2)

    @staticmethod
    def _forward_chunked(reader: _SockReader, client: socket.socket):
        """Stream a chunked response through, chunk by chunk (keeps streaming
        build/exec logs flowing), then the terminal chunk + trailers."""
        while True:
            head = reader.read_until(b"\r\n")
            client.sendall(head)
            size = int(head.strip().split(b";")[0], 16)
            if size == 0:
                # trailers until blank line; forward as they come
                while True:
                    line = reader.read_until(b"\r\n")
                    client.sendall(line)
                    if line in (b"\r\n", b""):
                        break
                return
            data = reader.read_exact(size)
            client.sendall(data)
            crlf = reader.read_exact(2)
            client.sendall(crlf)
