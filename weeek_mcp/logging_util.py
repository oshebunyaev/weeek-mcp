"""Redacted audit logging to stderr, optionally mirrored to a file.

Opt-in via WEEEK_DEBUG_LOG=1 (see config.py): MCP hosts vary in whether they
capture a locally-run extension's stderr; Claude Desktop was observed wiring
the subprocess's fd 2 to /dev/null, which silently swallows debug output.
Writing to a file we control means it survives regardless. Stderr remains on so
production container logs always contain tool outcome and duration records.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

_BEARER = re.compile(r"(?i)(authorization\s*[:=]?\s*bearer|bearer)\s+[^\s,;]+")
_COOKIE = re.compile(r"(?i)(cookie|set-cookie)\s*[:=]\s*[^\n]+")


def redact(message: str) -> str:
    text = _BEARER.sub(r"\1 [REDACTED]", str(message))
    text = _COOKIE.sub(r"\1=[REDACTED]", text)
    for name in ("WEEEK_API_TOKEN", "MCP_AUTH_TOKEN", "WEEEK_PASSWORD"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def make_logger(log_path: Path | None, tag: str):
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{tag}] {redact(msg)}"
        if log_path is not None:
            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except (OSError, UnicodeError):
                pass
        print(line, file=sys.stderr, flush=True)

    return _log
