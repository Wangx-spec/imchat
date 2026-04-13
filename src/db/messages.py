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

def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "session_id": row[1],
        "role": row[2],
        "content": row[3],
        "created_at": row[4].isoformat() if isinstance(row[4], datetime) else row[4],
    }