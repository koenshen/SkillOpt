"""BIRD text-to-SQL evaluation: SQL extraction and execution accuracy (EX).

Both parts follow the reference pipeline in bird-text-to-sql-260820:

* :func:`extract_sql` is ``extract_sql_content`` from ``src/run_prompt.py``.
* :func:`execution_match` follows ``src/evaluation/evaluation_v2.py``: run the
  predicted SQL, then the gold SQL, on the same SQLite database; the item is
  correct when ``set(pred_rows) == set(gold_rows)``.  One deadline
  (``timeout`` seconds, 30 by default) covers both queries; a timeout or any
  error scores 0.

Unlike the reference, databases are opened read-only (``mode=ro`` and
``immutable=1``) so a generated ``UPDATE``/``DELETE`` can never change the
shared database files; such statements fail and score 0.  Every gold query
returns rows, so the reference would score them 0 as well.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from urllib.parse import quote

DEFAULT_TIMEOUT = 30.0


def extract_sql(content: str) -> str:
    """Extract one SQL statement from an LLM response (reference pipeline)."""
    if not content:
        return ""

    # Thinking models commonly put their reasoning before the final answer.
    if " response" in content:
        content = content.rsplit(" response", 1)[1]

    # A fenced SQL block provides the clearest start and end boundaries.
    fenced_sql = re.search(
        r"```(?:sql|sqlite|mysql|postgresql)?\s*([\s\S]*?)```",
        content,
        flags=re.IGNORECASE,
    )
    if fenced_sql:
        content = fenced_sql.group(1)

    # Otherwise start at the first SQL statement keyword at the start of a line.
    sql_start = re.search(
        r"(?im)^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b",
        content,
    )
    if sql_start:
        content = content[sql_start.start():]

    content = content.strip()

    # Stop at the first semicolon outside quoted SQL values/identifiers.
    quote_char = None
    index = 0
    while index < len(content):
        char = content[index]
        if quote_char:
            if char == quote_char:
                # SQL escapes quote characters by doubling them.
                if index + 1 < len(content) and content[index + 1] == quote_char:
                    index += 2
                    continue
                quote_char = None
        elif char in ("'", '"', "`"):
            quote_char = char
        elif char == ";":
            return content[: index + 1].strip()
        index += 1

    return content


def db_path(db_root: str, db_id: str) -> str:
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


def _connect_read_only(path: str) -> sqlite3.Connection:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BIRD database not found: {path}")
    uri = f"file:{quote(os.path.abspath(path))}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds
        self.hit = False

    def __call__(self) -> int:
        if time.monotonic() > self.end:
            self.hit = True
            return 1  # non-zero aborts the running statement
        return 0


def execution_match(
    pred_sql: str,
    gold_sql: str,
    path: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Execute predicted and gold SQL; return EX plus result details.

    Keys: ``res`` (0/1), ``status`` (``ok`` / ``timeout`` / ``error``),
    ``error`` (message or ""), ``pred_rows`` and ``gold_rows`` (lists or None).
    """
    out = {"res": 0, "status": "ok", "error": "", "pred_rows": None, "gold_rows": None}
    if not pred_sql or not pred_sql.strip():
        out.update(status="error", error="empty SQL")
        return out
    deadline = _Deadline(timeout)
    conn = None
    stage = "connect"
    try:
        conn = _connect_read_only(path)
        conn.set_progress_handler(deadline, 1000)
        stage = "predicted SQL"
        pred_rows = conn.execute(pred_sql).fetchall()
        out["pred_rows"] = pred_rows
        stage = "gold SQL"
        gold_rows = conn.execute(gold_sql).fetchall()
        out["gold_rows"] = gold_rows
        out["res"] = int(set(pred_rows) == set(gold_rows))
    except Exception as exc:  # noqa: BLE001 — any failure scores 0, like the reference
        if deadline.hit:
            out.update(status="timeout", error=f"timeout after {timeout:g}s ({stage})")
        else:
            out.update(status="error", error=f"{stage}: {type(exc).__name__}: {exc}")
    finally:
        if conn is not None:
            conn.close()
    return out


def preview_rows(rows, limit: int = 10, max_chars: int = 600) -> str:
    """Short text preview of a result set for logs and the analyst."""
    if rows is None:
        return "None"
    text = repr(list(rows[:limit]))
    if len(rows) > limit:
        text += f" ... ({len(rows)} rows total)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"... ({len(rows)} rows total)"
    return text


def evaluate(response: str, gold_sql: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Extract SQL from ``response`` and score it against ``gold_sql``."""
    pred_sql = extract_sql(response)
    detail = execution_match(pred_sql, gold_sql, path, timeout=timeout)
    detail["pred_sql"] = pred_sql
    return detail
