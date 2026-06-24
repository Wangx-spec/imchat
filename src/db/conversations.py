from __future__ import annotations
import logging
from datetime import datetime
from db.connection import get_postgres_pool

logger = logging.getLogger(__name__)

DEFAULT_USER = "default-user"

def _row_to_dict(row) -> dict:
    return {
        "session_id": row[0],
        "user_id": row[1],
        "title": row[2],
        "created_at": row[3].isoformat() if isinstance(row[3], datetime) else row[3],
        "updated_at": row[4].isoformat() if isinstance(row[4], datetime) else row[4],
    }

def create_conversation(session_id: str, title: str | None = None) -> dict:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations (session_id, user_id, title)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (session_id) DO NOTHING
                   RETURNING session_id, user_id, title, created_at, updated_at""",
                (session_id, DEFAULT_USER, title),
            )
            row = cur.fetchone()
            conn.commit()
    if row is None:
        return {"session_id": session_id, "exists": True}
    return _row_to_dict(row)

def list_conversations(limit: int = 50) -> list[dict]:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT session_id, user_id, title, created_at, updated_at
                   FROM conversations
                   WHERE user_id = %s
                   ORDER BY updated_at DESC
                   LIMIT %s""",
                (DEFAULT_USER, limit),
            )
            return [_row_to_dict(r) for r in cur.fetchall()]

def update_conversation(session_id: str) -> None:
    """每条新消息时更新 updated_at。"""
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET updated_at = now() WHERE session_id = %s",
                (session_id,),
            )
            conn.commit()

def update_title(session_id: str, title: str) -> None:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s WHERE session_id = %s",
                (title, session_id),
            )
            conn.commit()


def delete_conversation(session_id: str) -> bool:
    """Delete a conversation and cascade-delete its messages."""
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE session_id = %s AND user_id = %s",
                (session_id, DEFAULT_USER),
            )
            deleted = cur.rowcount > 0
            conn.commit()
    return deleted