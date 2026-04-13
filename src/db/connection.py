from __future__ import annotations
import logging
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

__pool: ConnectionPool | None = None

def init_postgres_pool(postgres_uri: str, min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    global __pool
    if __pool is not None:
        return __pool
    __pool = ConnectionPool(
        conninfo=postgres_uri,
        min_size=min_size,
        max_size=max_size,
    )
    logger.info("PostgreSQL connection pool created (min=%d, max=%d)", min_size, max_size)
    return __pool

def get_postgres_pool() -> ConnectionPool:
    if __pool is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call init_postgres_pool() first.")
    return __pool