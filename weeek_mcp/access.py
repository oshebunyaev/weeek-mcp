"""Server-side workspace and project access policy."""

from __future__ import annotations

from typing import Any

from .config import Config
from .weeek_api import WeeekAPI


class AccessDenied(ValueError):
    pass


def task_project_ids(task_response: Any) -> set[int]:
    task = (task_response or {}).get("task") or task_response or {}
    return {
        int(location["projectId"])
        for location in task.get("locations") or []
        if isinstance(location, dict) and location.get("projectId") is not None
    }


class AccessPolicy:
    def __init__(self, config: Config):
        self.cfg = config

    @property
    def workspace_id(self) -> str:
        if not self.cfg.allowed_workspace_id:
            raise AccessDenied("WEEEK_ALLOWED_WORKSPACE_ID is required for scoped access")
        return self.cfg.allowed_workspace_id

    def check_projects(self, project_ids: set[int], *, write: bool) -> None:
        allowed = self.cfg.write_project_ids if write else self.cfg.read_project_ids
        if write and not allowed:
            raise AccessDenied("Writes require a non-empty WEEEK_WRITE_PROJECT_IDS whitelist")
        if allowed and (not project_ids or not project_ids.issubset(allowed)):
            denied = sorted(project_ids - allowed) if project_ids else ["unknown"]
            raise AccessDenied(f"Project scope is not allowed: {denied}")

    async def project_ids_for(
        self, tool_name: str, args: dict[str, Any], api: WeeekAPI | None, *, write: bool
    ) -> set[int]:
        ids: set[int] = set()
        if args.get("project_id") is not None:
            ids.add(int(args["project_id"]))
        if args.get("scope") == "project" and args.get("scope_id") is not None:
            ids.add(int(args["scope_id"]))
        task_id = args.get("task_id")
        if task_id is not None:
            if api is None:
                raise AccessDenied("Task scope cannot be verified without WEEEK_API_TOKEN")
            ids.update(task_project_ids(await api.get_task(int(task_id))))
        if args.get("board_id") is not None:
            candidates = self.cfg.write_project_ids if write else self.cfg.read_project_ids
            if not candidates:
                return ids
            if api is None:
                raise AccessDenied("Board scope cannot be verified without WEEEK_API_TOKEN")
            ids.add(await self._project_for_board(int(args["board_id"]), api, candidates))
        if args.get("board_column_id") is not None:
            candidates = self.cfg.write_project_ids if write else self.cfg.read_project_ids
            if not candidates:
                return ids
            if api is None:
                raise AccessDenied("Board-column scope cannot be verified without WEEEK_API_TOKEN")
            ids.add(await self._project_for_column(int(args["board_column_id"]), api, candidates))
        return ids

    async def _project_for_board(self, board_id: int, api: WeeekAPI, candidates: frozenset[int]) -> int:
        for project_id in candidates:
            data = await api.list_boards(project_id)
            if any(int(board.get("id", -1)) == board_id for board in data.get("boards") or []):
                return project_id
        raise AccessDenied(f"Board {board_id} is outside the configured project whitelists")

    async def _project_for_column(self, column_id: int, api: WeeekAPI, candidates: frozenset[int]) -> int:
        for project_id in candidates:
            boards = (await api.list_boards(project_id)).get("boards") or []
            for board in boards:
                data = await api.list_board_columns(int(board["id"]))
                columns = data.get("boardColumns") or data.get("columns") or []
                if any(int(column.get("id", -1)) == column_id for column in columns):
                    return project_id
        raise AccessDenied(f"Board column {column_id} is outside the configured project whitelists")
