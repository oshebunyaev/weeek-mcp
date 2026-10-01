"""Bounded durable write proposals with an explicit execution state machine."""

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
    token_hash: str
    tool_name: str
    arguments: dict[str, Any]
    workspace_id: str
    project_ids: frozenset[int]
    preview: str
    expires_at: int
    state_fingerprint: str
    credential_fingerprint: str
    status: str = "executing"
    result: Any = None


class ProposalStore:
    TERMINAL = frozenset({"succeeded", "failed_definite", "indeterminate"})

    def __init__(
        self,
        path: Path,
        ttl_seconds: int = 600,
        *,
        max_payload_bytes: int = 65_536,
        max_pending: int = 100,
        max_total: int = 1000,
        retention_seconds: int = 86_400,
    ):
        if min(ttl_seconds, max_payload_bytes, max_pending, max_total, retention_seconds) <= 0:
            raise ValueError("Proposal limits must be positive")
        if max_total < max_pending:
            raise ValueError("Proposal total limit cannot be smaller than pending limit")
        self.path, self.ttl_seconds = path, ttl_seconds
        self.max_payload_bytes, self.max_pending = max_payload_bytes, max_pending
        self.max_total = max_total
        self.retention_seconds = retention_seconds
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
            db.execute("""CREATE TABLE IF NOT EXISTS proposals (
                token_hash TEXT PRIMARY KEY, tool_name TEXT NOT NULL, arguments_json TEXT NOT NULL,
                workspace_id TEXT NOT NULL, project_ids_json TEXT NOT NULL, preview TEXT NOT NULL,
                expires_at INTEGER NOT NULL, used_at INTEGER, status TEXT NOT NULL DEFAULT 'pending',
                state_fingerprint TEXT NOT NULL DEFAULT '', credential_fingerprint TEXT NOT NULL DEFAULT '',
                result_json TEXT, finished_at INTEGER)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(proposals)")}
            for name, declaration in {
                "status": "TEXT NOT NULL DEFAULT 'pending'",
                "state_fingerprint": "TEXT NOT NULL DEFAULT ''",
                "credential_fingerprint": "TEXT NOT NULL DEFAULT ''",
                "result_json": "TEXT",
                "finished_at": "INTEGER",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE proposals ADD COLUMN {name} {declaration}")
            # Rows consumed by the legacy consume-before-write implementation have
            # an unknowable outcome. Never make them executable again after upgrade.
            db.execute(
                """UPDATE proposals SET status='indeterminate', finished_at=used_at
                   WHERE used_at IS NOT NULL AND status='pending'"""
            )
            db.execute(
                "UPDATE proposals SET status='indeterminate', finished_at=? WHERE status='executing'",
                (int(time.time()),),
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def _cleanup(self, db: sqlite3.Connection, now: int) -> None:
        db.execute("DELETE FROM proposals WHERE status='pending' AND expires_at < ?", (now,))
        db.execute(
            "DELETE FROM proposals WHERE status != 'pending' AND COALESCE(finished_at, expires_at) < ?",
            (now - self.retention_seconds,),
        )
        db.execute(
            """DELETE FROM proposals WHERE token_hash IN (
                SELECT token_hash FROM proposals WHERE status != 'pending'
                ORDER BY COALESCE(finished_at, expires_at) DESC LIMIT -1 OFFSET ?
            )""",
            (self.max_total - self.max_pending,),
        )

    def create(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_id: str,
        project_ids: set[int] | frozenset[int],
        preview: str,
        *,
        state_fingerprint: str = "",
        credential_fingerprint: str = "",
    ) -> tuple[str, int]:
        arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if len(arguments_json.encode()) > self.max_payload_bytes:
            raise ProposalError("Proposal payload is too large")
        token, now = secrets.token_urlsafe(32), int(time.time())
        expires_at = now + self.ttl_seconds
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._cleanup(db, now)
            pending = db.execute("SELECT COUNT(*) FROM proposals WHERE status='pending'").fetchone()[0]
            if pending >= self.max_pending:
                db.rollback()
                raise ProposalError("Pending proposal quota exceeded")
            db.execute(
                """INSERT INTO proposals
                (token_hash,tool_name,arguments_json,workspace_id,project_ids_json,preview,expires_at,used_at,
                 status,state_fingerprint,credential_fingerprint)
                VALUES (?,?,?,?,?,?,?,NULL,'pending',?,?)""",
                (
                    self._hash(token),
                    tool_name,
                    arguments_json,
                    workspace_id,
                    json.dumps(sorted(project_ids)),
                    preview,
                    expires_at,
                    state_fingerprint,
                    credential_fingerprint,
                ),
            )
            db.commit()
        return token, expires_at

    @staticmethod
    def _proposal(row: sqlite3.Row) -> Proposal:
        return Proposal(
            row["token_hash"],
            row["tool_name"],
            json.loads(row["arguments_json"]),
            row["workspace_id"],
            frozenset(json.loads(row["project_ids_json"])),
            row["preview"],
            row["expires_at"],
            row["state_fingerprint"],
            row["credential_fingerprint"],
            row["status"],
            json.loads(row["result_json"]) if row["result_json"] else None,
        )

    def claim(self, token: str) -> Proposal:
        if not token:
            raise ProposalError("confirmation_token is required")
        now, token_hash = int(time.time()), self._hash(token)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM proposals WHERE token_hash=?", (token_hash,)).fetchone()
            if row is None:
                db.rollback()
                raise ProposalError("Unknown confirmation token")
            if row["status"] == "succeeded":
                db.rollback()
                return self._proposal(row)
            if row["status"] != "pending":
                db.rollback()
                raise ProposalError(f"Proposal outcome is {row['status']}; it will not be executed again")
            if row["expires_at"] < now:
                db.execute("DELETE FROM proposals WHERE token_hash=?", (token_hash,))
                db.commit()
                raise ProposalError("Confirmation token has expired")
            updated = db.execute(
                "UPDATE proposals SET status='executing',used_at=? WHERE token_hash=? AND status='pending'",
                (now, token_hash),
            )
            if updated.rowcount != 1:
                db.rollback()
                raise ProposalError("Confirmation token has already been claimed")
            db.commit()
            row = db.execute("SELECT * FROM proposals WHERE token_hash=?", (token_hash,)).fetchone()
        return self._proposal(row)

    def consume(self, token: str) -> Proposal:
        proposal = self.claim(token)
        if proposal.status == "succeeded":
            raise ProposalError("Confirmation token has already been used")
        return proposal

    def finish(self, proposal: Proposal, status: str, result: Any = None) -> None:
        if status not in self.TERMINAL:
            raise ValueError("Invalid terminal proposal status")
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")) if result is not None else None
        with self._connect() as db:
            updated = db.execute(
                "UPDATE proposals SET status=?,result_json=?,finished_at=? WHERE token_hash=? AND status='executing'",
                (status, encoded, int(time.time()), proposal.token_hash),
            )
            if updated.rowcount != 1:
                raise ProposalError("Proposal is not executing")
