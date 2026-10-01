"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present; real environment variables always win.
load_dotenv()

DEFAULT_API_BASE = "https://api.weeek.net/public/v1"
DEFAULT_APP_BASE = "https://app.weeek.net"
# Internal (non-public) API the web app uses for the knowledge base.
DEFAULT_INTERNAL_API_BASE = "https://api.weeek.net"
# Hocuspocus server the editors sync document bodies through.
DEFAULT_COLLAB_BASE = "wss://collabws.weeek.net"


def _state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / "weeek-mcp"


def _default_state_path() -> Path:
    """Where the Playwright login session (storageState) is cached."""
    return _state_dir() / "storage_state.json"


def _default_log_path() -> Path | None:
    """Where diagnostic timing/step logs are written, if enabled.

    Opt-in via WEEEK_DEBUG_LOG=1: MCP hosts commonly discard a locally-run
    extension's stderr (observed with Claude Desktop: the subprocess's fd 2 is
    wired to /dev/null), so debug output has to go to a file we control to
    survive at all — but that shouldn't happen by default for every user.
    """
    if os.environ.get("WEEEK_DEBUG_LOG", "").lower() not in ("1", "true"):
        return None
    return _state_dir() / "debug.log"


def _bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _ids(name: str) -> frozenset[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    try:
        return frozenset(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integer ids") from exc


@dataclass(frozen=True)
class Config:
    # --- MCP transport and security ---
    transport: str
    http_host: str
    http_port: int
    auth_token: str | None
    allow_write: bool
    allowed_workspace_id: str | None
    read_project_ids: frozenset[int]
    write_project_ids: frozenset[int]
    proposals_db_path: Path
    proposal_ttl_seconds: int

    # --- Task management (public REST API) ---
    api_token: str | None
    api_base: str

    # --- Knowledge base (internal API + Playwright for login) ---
    app_base: str
    internal_api_base: str
    collab_base: str
    workspace_id: str | None  # auto-detected via /ws when not set
    email: str | None
    password: str | None
    storage_state_path: Path
    log_path: Path | None
    headless: bool
    kb_cache_ttl: int  # seconds to cache the KB document tree for resources/list
    kb_auto_login: bool

    @property
    def has_api(self) -> bool:
        return bool(self.api_token)

    @property
    def has_kb_credentials(self) -> bool:
        return bool(self.email and self.password)

    @classmethod
    def from_env(cls) -> "Config":
        state = os.environ.get("WEEEK_STORAGE_STATE")
        transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
        if transport not in ("stdio", "http"):
            raise ValueError("MCP_TRANSPORT must be 'stdio' or 'http'")
        allowed_workspace_id = os.environ.get("WEEEK_ALLOWED_WORKSPACE_ID")
        auth_token = os.environ.get("MCP_AUTH_TOKEN")
        if transport == "http" and not auth_token:
            raise ValueError("MCP_AUTH_TOKEN is required when MCP_TRANSPORT=http")
        if transport == "http" and not allowed_workspace_id:
            raise ValueError("WEEEK_ALLOWED_WORKSPACE_ID is required when MCP_TRANSPORT=http")
        return cls(
            transport=transport,
            http_host=os.environ.get("MCP_HTTP_HOST", "127.0.0.1"),
            http_port=int(os.environ.get("MCP_HTTP_PORT", "8000")),
            auth_token=auth_token,
            allow_write=_bool("WEEEK_ALLOW_WRITE"),
            allowed_workspace_id=allowed_workspace_id,
            read_project_ids=_ids("WEEEK_READ_PROJECT_IDS"),
            write_project_ids=_ids("WEEEK_WRITE_PROJECT_IDS"),
            proposals_db_path=Path(os.environ.get("WEEEK_PROPOSALS_DB", str(_state_dir() / "proposals.sqlite3"))),
            proposal_ttl_seconds=int(os.environ.get("WEEEK_PROPOSAL_TTL", "600")),
            api_token=os.environ.get("WEEEK_API_TOKEN"),
            api_base=os.environ.get("WEEEK_API_BASE", DEFAULT_API_BASE).rstrip("/"),
            app_base=os.environ.get("WEEEK_APP_BASE", DEFAULT_APP_BASE).rstrip("/"),
            internal_api_base=os.environ.get("WEEEK_INTERNAL_API_BASE", DEFAULT_INTERNAL_API_BASE).rstrip("/"),
            collab_base=os.environ.get("WEEEK_COLLAB_BASE", DEFAULT_COLLAB_BASE).rstrip("/"),
            workspace_id=allowed_workspace_id or os.environ.get("WEEEK_WORKSPACE_ID"),
            email=os.environ.get("WEEEK_EMAIL"),
            password=os.environ.get("WEEEK_PASSWORD"),
            storage_state_path=Path(state) if state else _default_state_path(),
            log_path=_default_log_path(),
            headless=os.environ.get("WEEEK_HEADLESS", "true").lower() != "false",
            kb_cache_ttl=int(os.environ.get("WEEEK_KB_CACHE_TTL", "300")),
            kb_auto_login=_bool("WEEEK_KB_AUTO_LOGIN"),
        )
