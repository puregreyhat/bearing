"""SQLite persistence for diagnosis test logs."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "bearing_health.db"


def _connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS test_logs (
                test_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_filename TEXT NOT NULL,
                model_family TEXT NOT NULL,
                predicted_result TEXT NOT NULL
            )
            """
        )


def create_test_log(input_filename: str, model_family: str, predicted_result: str) -> dict[str, str]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    test_id = f"TEST-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8].upper()}"
    record = {
        "test_id": test_id,
        "created_at": created_at,
        "input_filename": input_filename,
        "model_family": model_family,
        "predicted_result": predicted_result,
    }
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO test_logs
                (test_id, created_at, input_filename, model_family, predicted_result)
            VALUES (?, ?, ?, ?, ?)
            """,
            tuple(record.values()),
        )
    return record


def list_test_logs(limit: int = 50, offset: int = 0, search: str = "") -> tuple[list[dict[str, str]], int]:
    pattern = f"%{search.strip()}%"
    with _connection() as connection:
        total = connection.execute(
            """
            SELECT COUNT(*) FROM test_logs
            WHERE test_id LIKE ? OR input_filename LIKE ? OR predicted_result LIKE ?
            """,
            (pattern, pattern, pattern),
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT test_id, created_at, input_filename, model_family, predicted_result
            FROM test_logs
            WHERE test_id LIKE ? OR input_filename LIKE ? OR predicted_result LIKE ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (pattern, pattern, pattern, limit, offset),
        ).fetchall()
    return [dict(row) for row in rows], total


def get_test_log(test_id: str) -> dict[str, str] | None:
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT test_id, created_at, input_filename, model_family, predicted_result
            FROM test_logs WHERE test_id = ?
            """,
            (test_id,),
        ).fetchone()
    return dict(row) if row else None
