from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import sqlite3
from dataclasses import replace

import httpx
import pytest
from mcp import types

from tests.test_security import FakeAPI, config
from weeek_mcp.access import AccessDenied, AccessPolicy
from weeek_mcp.kb.client import KBError, WeeekKB
from weeek_mcp.proposals import ProposalError, ProposalStore
from weeek_mcp.server import HTTPApplication, WeeekServer
from weeek_mcp.tools import kb_doc_id_from_uri
from weeek_mcp.validation import validate_tool_ids
from weeek_mcp.weeek_api import WeeekAPI, WeeekAPIError

BAD_IDS = ["..", "%2e%2e", "a/b", r"a\b", "a?x", "a#x", "a%x", "a\nheader"]


@pytest.mark.parametrize("bad", BAD_IDS)
async def test_attachment_id_rejected_before_request(bad):
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"success": True})

    api = WeeekAPI("secret", "https://api.invalid/public/v1")
    await api._client.aclose()
    api._client = httpx.AsyncClient(base_url="https://api.invalid/public/v1", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        await api.get_attachment(bad)
    assert calls == []
    await api.aclose()


async def test_canonical_attachment_has_exact_endpoint_and_method():
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"success": True})

    api = WeeekAPI("secret", "https://api.invalid/public/v1")
    await api._client.aclose()
    api._client = httpx.AsyncClient(base_url="https://api.invalid/public/v1", transport=httpx.MockTransport(handler))
    file_id = "00000000-0000-4000-8000-000000000000"
    await api.get_attachment(file_id)
    assert calls == [("GET", f"https://api.invalid/public/v1/ws/attachments/{file_id}")]
    await api.aclose()


@pytest.mark.parametrize("bad", BAD_IDS)
async def test_kb_doc_id_rejected_before_request(tmp_path, bad):
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"article": {}})

    kb = WeeekKB(config(tmp_path))
    kb._ws = "ws-1"
    kb._client = httpx.AsyncClient(base_url="https://api.invalid", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError):
        await kb.read_document(bad)
    assert calls == []
    await kb.aclose()


async def test_kb_canonical_ids_keep_exact_workspace_endpoint(tmp_path):
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, json={"article": {"id": "123", "name": "N"}})

    kb = WeeekKB(config(tmp_path))
    kb._ws = "ws-1"
    kb._client = httpx.AsyncClient(base_url="https://api.invalid", transport=httpx.MockTransport(handler))
    await kb.rename_document("123", "N")
    assert calls == [("PUT", "https://api.invalid/ws/ws-1/kb/articles/123")]
    await kb.aclose()


@pytest.mark.parametrize(
    "uri",
    [
        "http://1",
        "weeek-kb://1/2",
        "weeek-kb://1?x=/../",
        "weeek-kb://1#x",
        "weeek-kb://../1",
    ],
)
def test_kb_resource_uri_is_exact(uri):
    with pytest.raises(ValueError):
        kb_doc_id_from_uri(uri)


def test_all_tool_path_ids_reject_meta_characters():
    for key in ("task_id", "comment_id", "project_id", "board_id", "board_column_id", "parent_id"):
        with pytest.raises(ValueError):
            validate_tool_ids({key: "../1"})


async def test_attachment_is_fail_closed_with_project_allowlist(tmp_path):
    server = WeeekServer(config(tmp_path))
    with pytest.raises(AccessDenied, match="ownership cannot be proven"):
        await server._check_task_read("weeek_get_attachment", {"file_id": "00000000-0000-4000-8000-000000000000"})


async def test_kb_workspace_is_derived_from_live_session(tmp_path):
    kb = WeeekKB(config(tmp_path, workspace_id="ws-1"))
    kb._client = httpx.AsyncClient(
        base_url="https://api.invalid",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"workspaces": [{"id": "other"}]})),
    )
    with pytest.raises(Exception, match="does not have access"):
        await kb.workspace()
    await kb.aclose()


async def test_kb_parent_fingerprint_never_treats_doc_id_as_task_id(tmp_path):
    server = WeeekServer(config(tmp_path, api_token=None))

    class FakeKB:
        async def workspace(self):
            return "ws-1"

        async def read_document(self, doc_id):
            return f"document {doc_id}"

    server._kb = FakeKB()
    fingerprint = await server._state_fingerprint("weeek_kb_move", {"doc_id": "1", "parent_id": 2})
    assert len(fingerprint) == 64


class ScopeAPI:
    def __init__(self, scopes):
        self.scopes = scopes

    async def get_task(self, task_id):
        return {"task": {"id": task_id, "locations": [{"projectId": p} for p in self.scopes[task_id]]}}

    async def list_projects(self):
        return {"projects": [{"id": 1}, {"id": 2}]}


async def test_related_parent_after_before_must_all_be_allowed(tmp_path):
    policy = AccessPolicy(config(tmp_path))
    api = ScopeAPI({1: {1}, 2: {2}, 3: {2}, 4: {2}})
    for key, related in (("parent_id", 2), ("after", 3), ("before", 4)):
        args = {"task_id": 1, "parent_id": None, "after": None, "before": None, key: related}
        projects = await policy.project_ids_for("weeek_set_task_parent", args, api, write=True)
        with pytest.raises(AccessDenied):
            policy.check_projects(projects, write=True)


async def test_create_with_denied_parent_is_rejected(tmp_path):
    policy = AccessPolicy(config(tmp_path))
    api = ScopeAPI({2: {2}})
    projects = await policy.project_ids_for("weeek_create_task", {"project_id": 1, "parent_id": 2}, api, write=True)
    with pytest.raises(AccessDenied):
        policy.check_projects(projects, write=True)


async def test_unknown_and_multi_project_reads_fail_closed(tmp_path):
    server = WeeekServer(config(tmp_path))
    server._api = ScopeAPI({1: set(), 2: {1, 3}})
    with pytest.raises(AccessDenied):
        await server._check_task_read("weeek_get_task", {"task_id": 1})
    with pytest.raises(AccessDenied):
        await server._check_task_read("weeek_get_task", {"task_id": 2})
    filtered = server._filter_task_read(
        "weeek_list_tasks",
        {
            "tasks": [
                {"id": 1, "locations": []},
                {"id": 2, "locations": [{"projectId": 1}, {"projectId": 3}]},
                {"id": 3, "locations": [{"projectId": 1}]},
            ]
        },
    )
    assert [task["id"] for task in filtered["tasks"]] == [3]


def test_proposal_quota_payload_and_expired_unused_cleanup(tmp_path):
    store = ProposalStore(tmp_path / "p.sqlite", max_pending=1, max_payload_bytes=20)
    store.create("x", {"a": 1}, "ws", set(), "p")
    with pytest.raises(ProposalError, match="quota"):
        store.create("x", {"b": 2}, "ws", set(), "p")
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE proposals SET expires_at=0")
    store.create("x", {"b": 2}, "ws", set(), "p")
    with pytest.raises(ProposalError, match="too large"):
        store.create("x", {"long": "x" * 100}, "ws", set(), "p")


def test_terminal_proposal_retention_is_row_bounded(tmp_path):
    store = ProposalStore(tmp_path / "p.sqlite", max_pending=1, max_total=3)
    for index in range(6):
        token, _ = store.create("x", {"index": index}, "ws", set(), "p")
        proposal = store.claim(token)
        store.finish(proposal, "succeeded", {"index": index})
    # One more create runs cleanup before insert: at most two terminal + one pending.
    store.create("x", {"last": True}, "ws", set(), "p")
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] <= 3


def test_cleanup_never_removes_executing_proposals(tmp_path):
    store = ProposalStore(tmp_path / "p.sqlite", max_pending=4, max_total=4)
    executing = []
    for index in range(3):
        token, _ = store.create("x", {"index": index}, "ws", set(), "p")
        executing.append(store.claim(token))

    store.create("x", {"pending": 1}, "ws", set(), "p")
    # The store is at max_total, and this create invokes cleanup again. Active
    # records may temporarily make the database exceed its terminal-row budget.
    store.create("x", {"pending": 2}, "ws", set(), "p")
    with sqlite3.connect(store.path) as db:
        statuses = dict(db.execute("SELECT token_hash, status FROM proposals"))
    assert all(statuses[proposal.token_hash] == "executing" for proposal in executing)

    for index, proposal in enumerate(executing):
        store.finish(proposal, "succeeded", {"index": index})


def test_legacy_consumed_tokens_migrate_to_indeterminate(tmp_path):
    path = tmp_path / "legacy.sqlite"
    token = "legacy-token"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE proposals (
            token_hash TEXT PRIMARY KEY, tool_name TEXT NOT NULL, arguments_json TEXT NOT NULL,
            workspace_id TEXT NOT NULL, project_ids_json TEXT NOT NULL, preview TEXT NOT NULL,
            expires_at INTEGER NOT NULL, used_at INTEGER)""")
        db.execute(
            "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?)",
            (hashlib.sha256(token.encode()).hexdigest(), "x", "{}", "ws", "[]", "p", 9999999999, 1),
        )
    store = ProposalStore(path)
    with pytest.raises(ProposalError, match="indeterminate"):
        store.claim(token)


def test_parallel_claim_executes_once(tmp_path):
    store = ProposalStore(tmp_path / "p.sqlite")
    token, _ = store.create("x", {}, "ws", set(), "p")

    def claim():
        try:
            store.claim(token)
            return True
        except ProposalError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        assert sum(pool.map(lambda _: claim(), range(20))) == 1


async def test_token_rotation_invalidates_existing_proposal(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True, read_project_ids=frozenset()))
    server._api = FakeAPI()
    proposed = await server._propose("weeek_create_task", {"title": "x", "project_id": 1})
    server.cfg = replace(server.cfg, api_token="rotated")
    with pytest.raises(AccessDenied, match="credential changed"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})


async def test_confirmation_revalidates_stored_ids_before_any_upstream_request(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True))
    token, _ = server._proposals.create(
        "weeek_kb_update",
        {"doc_id": "../../ws/other/kb/articles/9?", "title": "owned"},
        "ws-1",
        set(),
        "legacy malicious proposal",
        state_fingerprint="irrelevant",
        credential_fingerprint=server._credential_fingerprint(),
    )
    with pytest.raises(ValueError, match="doc_id"):
        await server._confirm({"confirmation_token": token})


class MutableTaskAPI:
    def __init__(self):
        self.title = "before"
        self.writes = 0
        self.fail_reads = False

    async def list_projects(self):
        return {"projects": [{"id": 1}, {"id": 2}]}

    async def get_task(self, task_id):
        if self.fail_reads:
            raise httpx.ReadTimeout("preflight timeout")
        return {"task": {"id": task_id, "title": self.title, "locations": [{"projectId": 1}]}}

    async def update_task(self, task_id, body):
        self.writes += 1
        return {"task": {"id": task_id, **body}}


async def test_proposal_rejects_state_change_while_building_preview(tmp_path, monkeypatch):
    server = WeeekServer(config(tmp_path, allow_write=True))
    api = MutableTaskAPI()
    server._api = api
    original_preview = server._preview

    async def preview_then_external_change(name, args):
        preview = await original_preview(name, args)
        api.title = "changed while preview was being prepared"
        return preview

    monkeypatch.setattr(server, "_preview", preview_then_external_change)
    with pytest.raises(AccessDenied, match="changed while preparing proposal"):
        await server._propose("weeek_update_task", {"task_id": 7, "title": "after"})

    with sqlite3.connect(server._proposals.path) as db:
        assert db.execute("SELECT COUNT(*) FROM proposals").fetchone()[0] == 0
    assert api.writes == 0


async def test_stale_upstream_state_rejects_before_write(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True))
    api = MutableTaskAPI()
    server._api = api
    proposed = await server._propose("weeek_update_task", {"task_id": 7, "title": "after"})
    api.title = "changed elsewhere"
    with pytest.raises(AccessDenied, match="state changed"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})
    assert api.writes == 0


async def test_related_scope_change_before_confirmation_is_rejected(tmp_path):
    class RelatedAPI:
        def __init__(self):
            self.scopes = {1: {1}, 2: {1}}
            self.writes = 0

        async def list_projects(self):
            return {"projects": [{"id": 1}, {"id": 2}]}

        async def get_task(self, task_id):
            return {"task": {"id": task_id, "locations": [{"projectId": p} for p in self.scopes[task_id]]}}

        async def set_task_parent(self, task_id, body):
            self.writes += 1
            return {"success": True}

    server = WeeekServer(config(tmp_path, allow_write=True))
    api = RelatedAPI()
    server._api = api
    proposed = await server._propose("weeek_set_task_parent", {"task_id": 1, "parent_id": 2})
    api.scopes[2] = {2}
    with pytest.raises(AccessDenied, match="state changed"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})
    assert api.writes == 0


async def test_timeout_during_preflight_is_definite_and_never_writes(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True))
    api = MutableTaskAPI()
    server._api = api
    proposed = await server._propose("weeek_update_task", {"task_id": 7, "title": "after"})
    api.fail_reads = True
    with pytest.raises(httpx.ReadTimeout):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})
    assert api.writes == 0
    with pytest.raises(ProposalError, match="failed_definite"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})


class TimeoutWriteAPI(FakeAPI):
    async def create_task(self, body):
        self.created.append(body)
        raise httpx.ReadTimeout("response lost after possible send")


class ServerErrorWriteAPI(FakeAPI):
    async def create_task(self, body):
        self.created.append(body)
        raise WeeekAPIError(500, {"debug": "unknown after send"})


async def test_post_send_timeout_becomes_indeterminate_and_is_not_retried(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True))
    api = TimeoutWriteAPI()
    server._api = api
    proposed = await server._propose("weeek_create_task", {"title": "x", "project_id": 1})
    with pytest.raises(ValueError, match="indeterminate"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})
    with pytest.raises(ProposalError, match="indeterminate"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})
    assert len(api.created) == 1


async def test_upstream_5xx_becomes_explicitly_indeterminate(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True))
    server._api = ServerErrorWriteAPI()
    proposed = await server._propose("weeek_create_task", {"title": "x", "project_id": 1})
    with pytest.raises(ValueError, match="indeterminate"):
        await server._confirm({"confirmation_token": proposed["confirmation_token"]})


async def test_parallel_confirmations_execute_exactly_one_write(tmp_path):
    class SlowAPI(FakeAPI):
        async def create_task(self, body):
            self.created.append(body)
            await asyncio.sleep(0.02)
            return {"success": True, "task": {"id": 10}}

    server = WeeekServer(config(tmp_path, allow_write=True))
    api = SlowAPI()
    server._api = api
    proposed = await server._propose("weeek_create_task", {"title": "x", "project_id": 1})

    async def confirm():
        try:
            return await server._confirm({"confirmation_token": proposed["confirmation_token"]})
        except ProposalError:
            return None

    results = await asyncio.gather(*(confirm() for _ in range(20)))
    assert sum(result is not None for result in results) == 1
    assert len(api.created) == 1


async def test_secret_bearing_upstream_error_is_not_returned_to_mcp_client(tmp_path):
    server = WeeekServer(config(tmp_path, read_project_ids=frozenset()))

    class ErrorAPI:
        async def list_projects(self):
            return {"projects": [{"id": 1}]}

        async def get_task(self, task_id):
            raise WeeekAPIError(500, {"debug": "WEEEK_API_TOKEN=super-secret"})

    server._api = ErrorAPI()
    handler = server.server.request_handlers[types.CallToolRequest]
    response = await handler(
        types.CallToolRequest(
            params=types.CallToolRequestParams(
                name="weeek_get_task",
                arguments={"task_id": 1},
            )
        )
    )
    text = response.root.content[0].text
    assert response.root.isError is True
    assert "super-secret" not in text and "debug" not in text
    assert "upstream_api_error" in text


async def test_secret_bearing_kb_error_is_not_returned_to_mcp_client(tmp_path):
    server = WeeekServer(config(tmp_path))

    class ErrorKB:
        async def workspace(self):
            return "ws-1"

        async def read_document(self, doc_id):
            raise KBError("cookie=sid-secret raw-document-content")

    server._kb = ErrorKB()
    handler = server.server.request_handlers[types.CallToolRequest]
    response = await handler(
        types.CallToolRequest(
            params=types.CallToolRequestParams(
                name="weeek_kb_read",
                arguments={"doc_id": "1"},
            )
        )
    )
    text = response.root.content[0].text
    assert response.root.isError is True
    assert "sid-secret" not in text and "raw-document-content" not in text
    assert "knowledge_base_error" in text


async def test_oversized_proposal_is_rejected_before_upstream_calls(tmp_path):
    server = WeeekServer(config(tmp_path, allow_write=True, proposal_max_payload_bytes=32))

    class CountingAPI(FakeAPI):
        def __init__(self):
            super().__init__()
            self.reads = 0

        async def list_projects(self):
            self.reads += 1
            return await super().list_projects()

    api = CountingAPI()
    server._api = api
    with pytest.raises(ValueError, match="too large"):
        await server._propose("weeek_create_task", {"title": "x" * 100, "project_id": 1})
    assert api.reads == 0 and api.created == []


async def test_project_binding_fails_when_token_cannot_see_configured_projects(tmp_path):
    server = WeeekServer(config(tmp_path))

    class WrongWorkspaceAPI:
        async def list_projects(self):
            return {"projects": [{"id": 999}]}

    server._api = WrongWorkspaceAPI()
    with pytest.raises(AccessDenied, match="cannot prove access"):
        await server._check_public_project_binding()


async def test_http_rejects_duplicate_auth_query_invalid_host_origin_and_large_body(tmp_path):
    app = HTTPApplication(WeeekServer(config(tmp_path, http_max_body_bytes=8)))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        duplicate = await client.post(
            "/mcp",
            headers=[("Authorization", "Bearer mcp-secret"), ("Authorization", "Bearer mcp-secret")],
            content=b"{}",
        )
        assert duplicate.status_code == 401
        assert (
            await client.post("/mcp?token=x", headers={"Authorization": "Bearer mcp-secret"}, content=b"{}")
        ).status_code == 401
        assert (
            await client.post("http://evil/mcp", headers={"Authorization": "Bearer mcp-secret"}, content=b"{}")
        ).status_code == 421
        assert (
            await client.post(
                "/mcp", headers={"Authorization": "Bearer mcp-secret", "Origin": "https://evil"}, content=b"{}"
            )
        ).status_code == 403
        assert (
            await client.post("/mcp", headers={"Authorization": "Bearer mcp-secret"}, content=b"123456789")
        ).status_code == 413


async def test_correct_bearer_completes_streamable_http_handshake(tmp_path):
    app = HTTPApplication(WeeekServer(config(tmp_path)))
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "security-test", "version": "1"},
        },
    }
    async with app.manager.run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer mcp-secret",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "weeek-mcp"


async def test_direct_legacy_writes_and_deletes_are_rejected_server_side(tmp_path):
    server = WeeekServer(config(tmp_path))
    handler = server.server.request_handlers[types.CallToolRequest]
    for name, arguments in (
        ("weeek_delete_task", {"task_id": 1}),
        ("weeek_delete_task_comment", {"task_id": 1, "comment_id": 1}),
        ("weeek_kb_delete", {"doc_id": "1"}),
        ("weeek_manage_tags", {"action": "delete", "tag_id": 1}),
    ):
        response = await handler(
            types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=arguments))
        )
        assert response.root.isError is True


async def test_scoped_tool_advertisement_hides_unprovable_and_workspace_reads(tmp_path):
    server = WeeekServer(config(tmp_path))
    handler = server.server.request_handlers[types.ListToolsRequest]
    response = await handler(types.ListToolsRequest())
    names = {tool.name for tool in response.root.tools}
    assert "weeek_get_attachment" not in names
    assert "weeek_list_custom_fields" not in names
    assert not names & {"weeek_whoami", "weeek_list_members"}


def test_log_fields_cannot_inject_new_lines(tmp_path):
    server = WeeekServer(config(tmp_path))
    rendered = server._log_ids({"task_id": "1\nFAKE success"})
    assert "\n" not in rendered and "FAKE success" not in rendered
