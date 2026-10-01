"""Weeek MCP server (stdio).

Exposes:
  * task-management tools backed by the Weeek public REST API
  * knowledge base tools + MCP Resources backed by Playwright

Capabilities are advertised based on configuration: task tools require an API
token; knowledge base tools/resources require login credentials or a cached
session. This keeps the tool list clean for whatever the user has set up.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import time
from typing import Any

import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyUrl
from starlette.responses import JSONResponse, PlainTextResponse

from . import __version__
from .access import AccessDenied, AccessPolicy
from .config import Config
from .kb.client import KBError, WeeekKB
from .logging_util import make_logger
from .proposals import ProposalStore
from .tools import (
    ALL_TOOLS,
    KB_TOOL_NAMES,
    PROPOSABLE_KB_TOOL_NAMES,
    PROPOSABLE_TASK_TOOL_NAMES,
    TASK_TOOL_NAMES,
    handle_kb_tool,
    handle_task_tool,
    kb_doc_id_from_uri,
    kb_uri,
    proposal_tools,
    read_tools,
)
from .validation import positive_int, validate_tool_ids
from .weeek_api import WeeekAPI, WeeekAPIError


class WeeekServer:
    def __init__(self, config: Config):
        self.cfg = config
        self.server: Server = Server("weeek-mcp", version=__version__)
        self._api: WeeekAPI | None = None
        self._kb: WeeekKB | None = None
        self.kb_available = config.storage_state_path.exists() or (config.kb_auto_login and config.has_kb_credentials)
        self._log = make_logger(config.log_path, "weeek-mcp")
        self._policy = AccessPolicy(config)
        self._proposals = ProposalStore(
            config.proposals_db_path,
            config.proposal_ttl_seconds,
            max_payload_bytes=config.proposal_max_payload_bytes,
            max_pending=config.proposal_max_pending,
            max_total=config.proposal_max_total,
            retention_seconds=config.proposal_retention_seconds,
        )
        self._register()

    # ------------------------------------------------------------- lazy deps
    def _get_api(self) -> WeeekAPI:
        if not self.cfg.has_api:
            raise ValueError("Task tools require WEEEK_API_TOKEN. Set it in the environment.")
        if self._api is None:
            self._api = WeeekAPI(self.cfg.api_token or "", self.cfg.api_base)
        return self._api

    def _get_kb(self) -> WeeekKB:
        if self._kb is None:
            self._kb = WeeekKB(self.cfg)
        return self._kb

    # ------------------------------------------------------------- handlers
    def _register(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            tools = read_tools(task=self.cfg.has_api, kb=self.kb_available)
            if self.cfg.read_project_ids:
                tools = [
                    tool for tool in tools if tool.name not in {"weeek_get_attachment", "weeek_list_custom_fields"}
                ]
            if not self.cfg.allow_workspace_reads:
                tools = [tool for tool in tools if tool.name not in {"weeek_whoami", "weeek_list_members"}]
            if self.cfg.allow_write:
                tools += proposal_tools(task=self.cfg.has_api, kb=self.kb_available)
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.ContentBlock]:
            args = arguments or {}
            t0 = time.monotonic()
            self._log(f"{name}: start {self._log_ids(args)}")
            try:
                if name.startswith("propose_"):
                    result = await self._propose(name.removeprefix("propose_"), args)
                elif name == "confirm_write":
                    result = await self._confirm(args)
                elif name in TASK_TOOL_NAMES:
                    if name not in {t.name for t in read_tools(task=True, kb=False)}:
                        raise AccessDenied("Direct write tools are disabled; use propose_* then confirm_write")
                    args = validate_tool_ids(args)
                    await self._check_task_read(name, args)
                    # Task descriptions are only writable through the browser session,
                    # so the KB client rides along when one is available.
                    kb = self._get_kb() if self.kb_available else None
                    result = await handle_task_tool(name, args, self._get_api(), kb)
                    result = self._filter_task_read(name, result)
                elif name in KB_TOOL_NAMES:
                    if name not in {t.name for t in read_tools(task=False, kb=True)}:
                        raise AccessDenied("Direct write tools are disabled; use propose_* then confirm_write")
                    await self._check_kb_workspace()
                    result = await handle_kb_tool(name, args, self._get_kb())
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except WeeekAPIError as exc:
                request_id = secrets.token_hex(6)
                self._log(f"{name}: failed request_id={request_id} status={exc.status_code}")
                raise ValueError(f"upstream_api_error status={exc.status_code} request_id={request_id}") from exc
            except KBError as exc:
                request_id = secrets.token_hex(6)
                self._log(f"{name}: failure request_id={request_id} category=knowledge_base_error")
                raise ValueError(f"knowledge_base_error request_id={request_id}") from exc
            except Exception as exc:
                self._log(f"{name}: failure after {time.monotonic() - t0:.1f}s: {type(exc).__name__}")
                raise

            self._log(f"{name}: success in {time.monotonic() - t0:.1f}s {self._log_ids(args)}")
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
            return [types.TextContent(type="text", text=text)]

        @self.server.list_resources()
        async def list_resources() -> list[types.Resource]:
            if not self.kb_available:
                return []
            try:
                await self._check_kb_workspace()
                docs = await self._get_kb().list_documents()
            except Exception as exc:  # noqa: BLE001 — never let KB break the session
                self._log(f"list_resources failure: {type(exc).__name__}")
                return []
            return [
                types.Resource(
                    uri=AnyUrl(kb_uri(d.id)),
                    name=d.title,
                    description=d.path or f"Weeek knowledge base document ({d.id})",
                    mimeType="text/markdown",
                )
                for d in docs
            ]

        @self.server.read_resource()
        async def read_resource(uri: Any) -> str:
            doc_id = kb_doc_id_from_uri(str(uri))
            try:
                await self._check_kb_workspace()
                return await self._get_kb().read_document(doc_id)
            except KBError as exc:
                request_id = secrets.token_hex(6)
                self._log(f"read_resource failure request_id={request_id} category=knowledge_base_error")
                raise ValueError(f"knowledge_base_error request_id={request_id}") from exc

    async def _check_kb_workspace(self) -> None:
        actual = await self._get_kb().workspace()
        if actual != self._policy.workspace_id:
            raise AccessDenied("Knowledge Base session belongs to a different workspace")

    def _log_ids(self, args: dict[str, Any]) -> str:
        fields = ["project_id", "task_id", "doc_id", "comment_id", "board_id", "board_column_id"]
        ids = [f"workspace_id={self.cfg.allowed_workspace_id}"] if self.cfg.allowed_workspace_id else []

        def safe(value: Any) -> str:
            return re.sub(r"[^A-Za-z0-9_.:-]", "?", str(value))[:128]

        ids.extend(f"{key}={safe(args[key])}" for key in fields if args.get(key) is not None)
        return " ".join(ids)

    @staticmethod
    def _validate_required(tool: types.Tool, args: dict[str, Any]) -> None:
        missing = [key for key in tool.inputSchema.get("required", []) if key not in args]
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    @staticmethod
    def _validated_arguments(name: str, args: dict[str, Any]) -> dict[str, Any]:
        validated = validate_tool_ids(args)
        if name == "weeek_set_task_parent":
            for key in ("after", "before"):
                if validated.get(key) is not None:
                    validated[key] = positive_int(validated[key], key)
        return validated

    async def _task_projects(self, name: str, args: dict[str, Any], *, write: bool) -> set[int]:
        return await self._policy.project_ids_for(name, args, self._get_api(), write=write)

    async def _check_task_read(self, name: str, args: dict[str, Any]) -> None:
        if name in {"weeek_whoami", "weeek_list_members"} and not self.cfg.allow_workspace_reads:
            raise AccessDenied("Workspace-wide reads require WEEEK_ALLOW_WORKSPACE_READS=true")
        if name == "weeek_get_attachment" and self.cfg.read_project_ids:
            raise AccessDenied("Attachment ownership cannot be proven while a project read allowlist is enabled")
        if name == "weeek_list_custom_fields" and self.cfg.read_project_ids:
            raise AccessDenied("Custom-field ownership cannot be proven while a project read allowlist is enabled")
        await self._check_public_project_binding()
        projects = await self._task_projects(name, args, write=False)
        if self.cfg.read_project_ids and (projects or name not in {"weeek_list_projects", "weeek_list_tasks"}):
            self._policy.check_projects(projects, write=False)

    async def _check_public_project_binding(self) -> None:
        """Prove that the credential can see every configured project ID.

        The documented Public API exposes no workspace identifier, so this is the
        strongest available check; it does not turn the operator-provided workspace
        label into a token-derived identity.
        """
        required = self.cfg.read_project_ids | self.cfg.write_project_ids
        if not required:
            return
        data = await self._get_api().list_projects()
        visible = {int(project["id"]) for project in data.get("projects") or [] if project.get("id") is not None}
        missing = required - visible
        if missing:
            raise AccessDenied(f"WEEEK API credential cannot prove access to configured projects: {sorted(missing)}")

    def _filter_task_read(self, name: str, result: Any) -> Any:
        allowed = self.cfg.read_project_ids
        if not allowed or not isinstance(result, dict):
            return result
        if name == "weeek_list_projects":
            result = dict(result)
            result["projects"] = [p for p in result.get("projects") or [] if int(p.get("id", -1)) in allowed]
        elif name == "weeek_list_tasks":

            def allowed_task(task: dict[str, Any]) -> bool:
                scope = {
                    int(location["projectId"])
                    for location in task.get("locations") or []
                    if location.get("projectId") is not None
                }
                return bool(scope) and scope.issubset(allowed)

            result = dict(result)
            result["tasks"] = [task for task in result.get("tasks") or [] if allowed_task(task)]
            result.pop("hasMore", None)
        return result

    def _credential_fingerprint(self) -> str:
        return hashlib.sha256((self.cfg.api_token or "no-public-api").encode()).hexdigest()

    async def _state_fingerprint(self, name: str, args: dict[str, Any]) -> str:
        state: dict[str, Any] = {}
        task_ids = (
            {args[key] for key in ("task_id", "parent_id", "after", "before") if args.get(key) is not None}
            if name in PROPOSABLE_TASK_TOOL_NAMES
            else set()
        )
        for task_id in sorted(task_ids):
            state[f"task:{task_id}"] = await self._get_api().get_task(int(task_id))
        if name == "weeek_update_task_comment" and args.get("comment_id") is not None:
            comments = await self._get_kb().list_task_comments(int(args["task_id"]))
            state["comment"] = next((c for c in comments if str(c.get("id")) == str(args["comment_id"])), None)
        if name in PROPOSABLE_KB_TOOL_NAMES:
            await self._check_kb_workspace()
            for key in ("doc_id", "parent_id"):
                if args.get(key) is not None:
                    state[f"kb:{key}"] = await self._get_kb().read_document(str(args[key]))
        encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    async def _preview(self, name: str, args: dict[str, Any]) -> str:
        def compact(value: Any) -> Any:
            if isinstance(value, str) and len(value) > 120:
                digest = hashlib.sha256(value.encode()).hexdigest()[:12]
                return f"<{len(value)} chars, sha256:{digest}>"
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [compact(v) for v in value]
            return value

        before = "new object"
        if args.get("task_id") is not None:
            task = (await self._get_api().get_task(int(args["task_id"]))).get("task") or {}
            before = (
                f"task {args['task_id']} title={task.get('title')!r} projects={sorted(self._policy_project_ids(task))}"
            )
            if name == "weeek_update_task_comment":
                comments = await self._get_kb().list_task_comments(int(args["task_id"]))
                current = next((c for c in comments if str(c.get("id")) == str(args.get("comment_id"))), None)
                if current is None:
                    raise ValueError(f"Comment {args.get('comment_id')} not found on task {args['task_id']}")
                body = json.dumps(current.get("content") or {}, ensure_ascii=False, sort_keys=True)
                before += (
                    f" comment={args.get('comment_id')} current_sha256={hashlib.sha256(body.encode()).hexdigest()[:12]}"
                )
        elif name in PROPOSABLE_KB_TOOL_NAMES:
            await self._check_kb_workspace()
            if args.get("doc_id") is not None:
                document = await self._get_kb().read_document(str(args["doc_id"]))
                before = (
                    f"KB document {args['doc_id']} current_sha256={hashlib.sha256(document.encode()).hexdigest()[:12]}"
                )
        return f"Proposed {name}\nCurrent: {before}\nRequested: {json.dumps(compact(args), ensure_ascii=False, sort_keys=True)}"

    @staticmethod
    def _policy_project_ids(task: dict[str, Any]) -> set[int]:
        return {int(x["projectId"]) for x in task.get("locations") or [] if x.get("projectId") is not None}

    async def _propose(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.cfg.allow_write:
            raise AccessDenied("Writes are disabled (WEEEK_ALLOW_WRITE=false)")
        if name not in PROPOSABLE_TASK_TOOL_NAMES | PROPOSABLE_KB_TOOL_NAMES:
            raise AccessDenied(f"{name} is not available for safe writes")
        args = self._validated_arguments(name, args)
        if (
            len(json.dumps(args, ensure_ascii=False, separators=(",", ":")).encode())
            > self.cfg.proposal_max_payload_bytes
        ):
            raise ValueError("Proposal payload is too large")
        if name in {"weeek_add_task_comment", "weeek_update_task_comment"} and not self.kb_available:
            raise ValueError("Comment writes require a valid Knowledge Base browser session")
        if name == "weeek_update_task" and args.get("description") is not None and not self.kb_available:
            raise ValueError("Updating a task description requires a valid Knowledge Base browser session")
        source = next((t for t in ALL_TOOLS if t.name == name), None)
        if source is None:
            raise ValueError(f"Unknown tool: {name}")
        self._validate_required(source, args)
        projects: set[int] = set()
        if name in PROPOSABLE_TASK_TOOL_NAMES:
            await self._check_public_project_binding()
            if name in {"weeek_add_task_comment", "weeek_update_task_comment"} or (
                name == "weeek_update_task" and args.get("description") is not None
            ):
                await self._check_kb_workspace()
            projects = await self._task_projects(name, args, write=True)
            self._policy.check_projects(projects, write=True)
        else:
            await self._check_kb_workspace()
        preview = await self._preview(name, args)
        state_fingerprint = await self._state_fingerprint(name, args)
        token, expires_at = self._proposals.create(
            name,
            args,
            self._policy.workspace_id,
            projects,
            preview,
            state_fingerprint=state_fingerprint,
            credential_fingerprint=self._credential_fingerprint(),
        )
        return {"preview": preview, "confirmation_token": token, "expires_at": expires_at}

    async def _confirm(self, args: dict[str, Any]) -> Any:
        if not self.cfg.allow_write:
            raise AccessDenied("Writes are disabled (WEEEK_ALLOW_WRITE=false)")
        proposal = self._proposals.claim(str(args.get("confirmation_token") or ""))
        if proposal.status == "succeeded":
            return proposal.result
        try:
            await self._validate_claimed(proposal)
        except Exception:
            self._proposals.finish(proposal, "failed_definite")
            raise
        try:
            result = await self._execute_claimed(proposal)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            self._proposals.finish(proposal, "indeterminate")
            raise ValueError("write_outcome_indeterminate; do not retry this proposal") from exc
        except KBError as exc:
            self._proposals.finish(proposal, "indeterminate")
            raise ValueError("write_outcome_indeterminate; do not retry this proposal") from exc
        except WeeekAPIError as exc:
            status = "indeterminate" if exc.status_code >= 500 else "failed_definite"
            self._proposals.finish(proposal, status)
            if status == "indeterminate":
                raise ValueError("write_outcome_indeterminate; do not retry this proposal") from exc
            raise
        except Exception as exc:
            self._proposals.finish(proposal, "indeterminate")
            raise ValueError("write_outcome_indeterminate; do not retry this proposal") from exc
        self._proposals.finish(proposal, "succeeded", result)
        return result

    async def _validate_claimed(self, proposal: Any) -> None:
        validated = self._validated_arguments(proposal.tool_name, proposal.arguments)
        if validated != proposal.arguments:
            raise AccessDenied("Proposal contains non-canonical identifiers; create a new proposal")
        if proposal.workspace_id != self._policy.workspace_id:
            raise AccessDenied("Proposal workspace no longer matches server scope")
        if proposal.credential_fingerprint != self._credential_fingerprint():
            raise AccessDenied("WEEEK API credential changed after proposal; create a new proposal")
        current_fingerprint = await self._state_fingerprint(proposal.tool_name, proposal.arguments)
        if proposal.state_fingerprint != current_fingerprint:
            raise AccessDenied("Relevant upstream state changed after proposal; create a new proposal")
        if proposal.tool_name in PROPOSABLE_TASK_TOOL_NAMES:
            await self._check_public_project_binding()
            if proposal.tool_name in {"weeek_add_task_comment", "weeek_update_task_comment"} or (
                proposal.tool_name == "weeek_update_task" and proposal.arguments.get("description") is not None
            ):
                await self._check_kb_workspace()
            projects = await self._task_projects(proposal.tool_name, proposal.arguments, write=True)
            self._policy.check_projects(projects, write=True)
            if projects != set(proposal.project_ids):
                raise AccessDenied("Object project scope changed after proposal; create a new proposal")
        else:
            await self._check_kb_workspace()

    async def _execute_claimed(self, proposal: Any) -> Any:
        if proposal.tool_name in PROPOSABLE_TASK_TOOL_NAMES:
            kb = self._get_kb() if self.kb_available else None
            return await handle_task_tool(proposal.tool_name, proposal.arguments, self._get_api(), kb)
        return await handle_kb_tool(proposal.tool_name, proposal.arguments, self._get_kb())

    async def aclose(self) -> None:
        if self._api is not None:
            await self._api.aclose()
        if self._kb is not None:
            await self._kb.aclose()

    # ------------------------------------------------------------- run
    async def run_stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            try:
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
            finally:
                await self.aclose()


class HTTPApplication:
    """Tiny ASGI shell around the SDK's official Streamable HTTP manager."""

    def __init__(self, weeek: WeeekServer):
        self.weeek = weeek
        domain = weeek.cfg.mcp_domain or weeek.cfg.http_host
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[domain, f"{domain}:*"],
            allowed_origins=list(weeek.cfg.allowed_origins),
        )
        self.manager = StreamableHTTPSessionManager(
            weeek.server,
            json_response=True,
            stateless=True,
            security_settings=security,
        )

    async def _bounded_body(self, receive: Any) -> tuple[Any | None, bool]:
        """Buffer at most the configured limit and return a replay receive callable."""
        messages: list[dict[str, Any]] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.request":
                total += len(message.get("body", b""))
                if total > self.weeek.cfg.http_max_body_bytes:
                    return None, True
                if not message.get("more_body", False):
                    break
            elif message.get("type") == "http.disconnect":
                break
        index = 0

        async def replay() -> dict[str, Any]:
            nonlocal index
            if index < len(messages):
                result = messages[index]
                index += 1
                return result
            return {"type": "http.disconnect"}

        return replay, False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            async with self.manager.run():
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await self.weeek.aclose()
                        await send({"type": "lifespan.shutdown.complete"})
                        return
        if scope["type"] != "http":
            return
        path = scope.get("path")
        if path == "/health":
            await JSONResponse({"status": "ok"})(scope, receive, send)
            return
        if path != "/mcp":
            await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
            return
        auth_headers = [v for k, v in scope.get("headers", []) if k.lower() == b"authorization"]
        if len(auth_headers) != 1 or scope.get("query_string"):
            await JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})(
                scope, receive, send
            )
            return
        supplied = auth_headers[0].decode("latin-1")
        expected = f"Bearer {self.weeek.cfg.auth_token or ''}"
        if not hmac.compare_digest(supplied, expected):
            await JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
            return
        header_pairs = scope.get("headers", [])
        hosts = [v.decode("latin-1") for k, v in header_pairs if k.lower() == b"host"]
        origins = [v.decode("latin-1") for k, v in header_pairs if k.lower() == b"origin"]
        domain = self.weeek.cfg.mcp_domain or self.weeek.cfg.http_host
        host_ok = len(hosts) == 1 and re.fullmatch(re.escape(domain) + r"(?::[0-9]{1,5})?", hosts[0])
        if not host_ok:
            await PlainTextResponse("Invalid Host header", status_code=421)(scope, receive, send)
            return
        if len(origins) > 1 or (origins and origins[0] not in self.weeek.cfg.allowed_origins):
            await PlainTextResponse("Invalid Origin header", status_code=403)(scope, receive, send)
            return
        content_lengths = [v for k, v in scope.get("headers", []) if k.lower() == b"content-length"]
        try:
            declared = int(content_lengths[0]) if len(content_lengths) == 1 else 0
        except ValueError:
            declared = self.weeek.cfg.http_max_body_bytes + 1
        if len(content_lengths) > 1 or declared > self.weeek.cfg.http_max_body_bytes:
            await JSONResponse({"error": "request_too_large"}, status_code=413)(scope, receive, send)
            return
        bounded_receive, too_large = await self._bounded_body(receive)
        if too_large or bounded_receive is None:
            await JSONResponse({"error": "request_too_large"}, status_code=413)(scope, receive, send)
            return
        await self.manager.handle_request(scope, bounded_receive, send)


async def _amain() -> None:
    cfg = Config.from_env()
    log = make_logger(cfg.log_path, "weeek-mcp")
    log(f"Initializing server (log file: {cfg.log_path})")
    if not cfg.has_api and not (cfg.has_kb_credentials or cfg.storage_state_path.exists()):
        log("Warning: neither WEEEK_API_TOKEN nor KB credentials/session found. Tools will error until configured.")
    weeek = WeeekServer(cfg)
    if cfg.transport == "stdio":
        await weeek.run_stdio()
    else:
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(HTTPApplication(weeek), host=cfg.http_host, port=cfg.http_port, log_level="info")
        )
        await server.serve()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
