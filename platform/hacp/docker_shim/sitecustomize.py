"""Platform docker shim: routes upstream's DockerClient through our proxy.

Loaded automatically by Python's ``site`` machinery when the platform puts
this directory on ``PYTHONPATH`` for the experiment subprocess
(``HACP_DOCKER_PROXY_URL`` signals activation). Why this is needed: upstream
``generate_loop.py`` constructs ``docker.DockerClient()`` with no arguments,
and docker-py's zero-argument ``DockerClient()`` ignores the ``DOCKER_HOST``
environment variable (only ``from_env()`` reads it). This shim patches the
no-argument construction to point at the platform's Docker socket proxy,
which transparently injects ``HACP_*`` environment variables into every
container the run creates — a channel file operations inside containers
cannot remove.

No upstream file is modified; the patch lives entirely in the process
environment the platform controls. It is inert unless
``HACP_DOCKER_PROXY_URL`` is set.
"""
import os

_PROXY_URL = os.environ.get("HACP_DOCKER_PROXY_URL", "")

if _PROXY_URL:
    try:
        import docker as _docker

        _orig_init = _docker.DockerClient.__init__

        def _hacp_init(self, *args, **kwargs):
            # only redirect the no-argument construction upstream uses;
            # explicit base_url callers (tests, guards) are left untouched
            if not args and not kwargs.get("base_url"):
                kwargs["base_url"] = _PROXY_URL
            _orig_init(self, *args, **kwargs)

        _docker.DockerClient.__init__ = _hacp_init
    except Exception:  # never break the host process over the shim
        pass
