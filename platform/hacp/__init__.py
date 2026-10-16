"""HyperAgents Control Platform (HACP) — M1 control plane.

Wraps Meta's HyperAgents research core without modifying it:
experiment registry, LLM routing proxy (OpenRouter / offline stub),
usage + cost accounting, budget kill-switches, live monitoring,
and the foundation for the web dashboard and conversational shell.
"""

__version__ = "0.1.0"
