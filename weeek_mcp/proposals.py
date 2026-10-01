"""Durable, one-time write proposals backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProposalError(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    tool_name: str
    arguments: dict[str, Any]
    workspace_id: str
    project_ids: frozenset[int]
    preview: str
    expires_at: int


class ProposalStore:
    def __init__(self, path: Path, ttl_seconds: int = 600):
        if ttl_seconds <= 0:
            raise ValueError("Proposal TTL must be positive")
        self.path = path
        self.ttl_seconds = ttl_seconds
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS proposals (
                    token_hash TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    project_ids_json TEXT NOT NULL,
                    preview TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    used_at INTEGER
                )"""
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_id: str,
        project_ids: set[int] | frozenset[int],
        preview: str,
    ) -> tuple[str, int]:
        token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + self.ttl_seconds
        with self._connect() as db:
            db.execute(
                "INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    self._hash(token),
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    workspace_id,
                    json.dumps(sorted(project_ids)),
                    preview,
                    expires_at,
                ),
            )
            db.execute(
                "DELETE FROM proposals WHERE expires_at < ? AND used_at IS NOT NULL", (int(time.time()) - 86400,)
            )
        return token, expires_at

    def consume(self, token: str) -> Proposal:
        if not token:
            raise ProposalError("confirmation_token is required")
        now = int(time.time())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM proposals WHERE token_hash = ?", (self._hash(token),)).fetchone()
            if row is None:
                db.rollback()
                raise ProposalError("Unknown confirmation token")
            if row["used_at"] is not None:
                db.rollback()
                raise ProposalError("Confirmation token has already been used")
            if row["expires_at"] < now:
                db.rollback()
                raise ProposalError("Confirmation token has expired")
            updated = db.execute(
                "UPDATE proposals SET used_at = ? WHERE token_hash = ? AND used_at IS NULL",
                (now, self._hash(token)),
            )
            if updated.rowcount != 1:
                db.rollback()
                raise ProposalError("Confirmation token has already been used")
            db.commit()
        return Proposal(
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments_json"]),
            workspace_id=row["workspace_id"],
            project_ids=frozenset(json.loads(row["project_ids_json"])),
            preview=row["preview"],
            expires_at=row["expires_at"],
        )
