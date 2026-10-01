from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from weeek_mcp.access import AccessDenied, AccessPolicy
from weeek_mcp.config import Config
from weeek_mcp.logging_util import make_logger
from weeek_mcp.proposals import ProposalError, ProposalStore
from weeek_mcp.server import HTTPApplication, WeeekServer
from weeek_mcp.tools import DELETE_TOOL_NAMES, proposal_tools, read_tools


def config(tmp_path: Path, **changes) -> Config:
    base = Config(
        transport="http",
        http_host="127.0.0.1",
        http_port=8000,
        auth_token="mcp-secret",
        allow_write=False,
        allowed_workspace_id="ws-1",
        read_project_ids=frozenset({1, 2}),
        write_project_ids=frozenset({1}),
        proposals_db_path=tmp_path / "pending" / "proposals.sqlite3",
        proposal_ttl_seconds=600,
        proposal_max_payload_bytes=65536,
        proposal_max_pending=100,
        proposal_max_total=1000,
        proposal_retention_seconds=86400,
        http_max_body_bytes=1048576,
        mcp_domain="test",
        allowed_origins=(),
        allow_workspace_reads=False,
        api_token="api-secret",
        api_base="https://api.invalid",
        app_base="https://app.invalid",
        internal_api_base="https://api.invalid",
        collab_base="wss://collab.invalid",
        workspace_id="ws-1",
        email=None,
        password=None,
        storage_state_path=tmp_path / "state.json",
        log_path=tmp_path / "server.log",
        headless=True,
        kb_cache_ttl=300,
        kb_auto_login=False,
    )
    return replace(base, **changes)


@pytest.mark.asyncio
async def test_http_auth_and_public_health(tmp_path):
    app = HTTPApplication(WeeekServer(config(tmp_path)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.post("/mcp", json={})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_read_only_surface_and_delete_tools_absent():
    names = {tool.name for tool in read_tools(task=True, kb=True)}
    assert "confirm_write" not in names
    assert not names & DELETE_TOOL_NAMES
    writable = {tool.name for tool in proposal_tools(task=True, kb=True)}
    assert not writable & DELETE_TOOL_NAMES
    assert not {f"propose_{name}" for name in DELETE_TOOL_NAMES} & writable


def test_write_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WEEEK_ALLOW_WRITE", raising=False)
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    assert Config.from_env().allow_write is False


def test_write_whitelist():
    policy = AccessPolicy(config(Path("/tmp")))
    policy.check_projects({1}, write=True)
    with pytest.raises(AccessDenied):
        policy.check_projects({2}, write=True)


class FakeAPI:
    def __init__(self):
        self.created = []

    async def create_task(self, body):
        self.created.append(body)
        return {"success": True, "task": {"id": 10}}

    async def list_projects(self):
        return {"projects": [{"id": 1}, {"id": 2}]}

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_proposal_does_not_write_and_confirm_uses_saved_payload(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True))
    api = FakeAPI()
    server._api = api
    args = {"title": "original", "project_id": 1}
    proposed = await server._propose("weeek_create_task", args)
    assert api.created == []
    args["title"] = "tampered"
    await server._confirm({"confirmation_token": proposed["confirmation_token"]})
    assert api.created[0]["title"] == "original"
    assert await server._confirm({"confirmation_token": proposed["confirmation_token"]}) == {
        "success": True,
        "task": {"id": 10},
    }
    assert len(api.created) == 1


def test_expired_confirmation_token(tmp_path):
    store = ProposalStore(tmp_path / "proposals.sqlite3", 600)
    token, _ = store.create("weeek_create_task", {"project_id": 1}, "ws-1", {1}, "preview")
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE proposals SET expires_at = 0")
    with pytest.raises(ProposalError, match="expired"):
        store.consume(token)


@pytest.mark.asyncio
async def test_workspace_isolation(tmp_path):
    server = WeeekServer(config(tmp_path))

    class WrongKB:
        async def workspace(self):
            return "ws-other"

    server._kb = WrongKB()
    with pytest.raises(AccessDenied, match="different workspace"):
        await server._check_kb_workspace()


def test_secrets_are_redacted_from_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("WEEEK_API_TOKEN", "api-secret")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "mcp-secret")
    monkeypatch.setenv("WEEEK_PASSWORD", "password-secret")
    path = tmp_path / "server.log"
    make_logger(path, "test")("Authorization: Bearer mcp-secret api-secret password-secret Cookie: sid=cookie-secret")
    contents = path.read_text(encoding="utf-8")
    for secret in ("mcp-secret", "api-secret", "password-secret", "cookie-secret"):
        assert secret not in contents
