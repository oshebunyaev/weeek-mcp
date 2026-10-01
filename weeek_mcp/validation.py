"""Strict validation for every caller-controlled identifier used in an endpoint."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlsplit

_UUID = re.compile(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.IGNORECASE)
_OPAQUE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_INTEGER_KEYS = {
    "task_id",
    "comment_id",
    "project_id",
    "board_id",
    "board_column_id",
    "parent_id",
    "portfolio_id",
    "scope_id",
    "target_id",
    "upper_board_id",
    "upper_board_column_id",
}
_UUID_KEYS = {"file_id"}
_OPAQUE_KEYS = {"field_id", "option_id", "entry_id"}


def positive_int(value: Any, name: str) -> int:
    """Return a canonical positive integer, rejecting bools and string tricks."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")  # noqa: TRY004 - public input validation
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        result = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def uuid_id(value: Any, name: str) -> str:
    result = str(value)
    if not _UUID.fullmatch(result):
        raise ValueError(f"{name} must be a canonical UUID")
    return result.lower()


def kb_doc_id(value: Any, name: str = "doc_id") -> str:
    return str(positive_int(value, name))


def opaque_id(value: Any, name: str) -> str:
    result = str(value)
    if not _OPAQUE.fullmatch(result):
        raise ValueError(f"{name} must contain only ASCII letters, digits, underscore, or hyphen")
    return result


def path_segment(value: str) -> str:
    """Encode a value already accepted by a strict canonical validator."""
    return quote(value, safe="")


def validate_tool_ids(args: dict[str, Any]) -> dict[str, Any]:
    """Validate endpoint identifiers before any network request is possible."""
    clean = dict(args)
    for key in _INTEGER_KEYS:
        if clean.get(key) is not None:
            clean[key] = positive_int(clean[key], key)
    for key in _UUID_KEYS:
        if clean.get(key) is not None:
            clean[key] = uuid_id(clean[key], key)
    for key in _OPAQUE_KEYS:
        if clean.get(key) is not None:
            clean[key] = opaque_id(clean[key], key)
    if clean.get("doc_id") is not None:
        clean["doc_id"] = kb_doc_id(clean["doc_id"])
    if clean.get("tag_id") is not None:
        clean["tag_id"] = positive_int(clean["tag_id"], "tag_id")
    return clean


def parse_kb_uri(uri: str) -> str:
    """Accept only weeek-kb://<positive integer>, without path/query/fragment."""
    parsed = urlsplit(uri)
    if parsed.scheme != "weeek-kb" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("KB resource URI must be exactly weeek-kb://<document-id>")
    return kb_doc_id(parsed.netloc)
