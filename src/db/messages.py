from __future__ import annotations
import logging
from datetime import datetime
from db.connection import get_postgres_pool

logger = logging.getLogger(__name__)

def save_message(session_id: str, role: str, content: str) -> None:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
            conn.commit()

def list_messages(session_id: str, limit: int = 200) -> list[dict]:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, session_id, role, content, created_at
                   FROM messages
                   WHERE session_id = %s
                   ORDER BY created_at ASC
                   LIMIT %s""",
                (session_id, limit),
            )
            return [_row_to_dict(r) for r in cur.fetchall()]


def list_recent_user_messages(user_id: str, limit: int = 20) -> list[dict]:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT m.id, m.session_id, m.role, m.content, m.created_at
                   FROM messages m
                   JOIN conversations c ON c.session_id = m.session_id
                   WHERE c.user_id = %s
                   ORDER BY m.created_at DESC
                   LIMIT %s""",
                (user_id, limit),
            )
            rows = [_row_to_dict(r) for r in cur.fetchall()]
    return list(reversed(rows))

def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "session_id": row[1],
        "role": row[2],
        "content": row[3],
        "created_at": row[4].isoformat() if isinstance(row[4], datetime) else row[4],
    }