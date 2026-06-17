import logging

from config.settings import Settings


logger = logging.getLogger(__name__)

_CHECKPOINTER = None
_PG_CONN = None


def build_checkpointer(settings: Settings):
    global _CHECKPOINTER, _PG_CONN
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    if settings.postgres_uri:
        from psycopg import Connection
        from langgraph.checkpoint.postgres import PostgresSaver

        _PG_CONN = Connection.connect(
            settings.postgres_uri,
            autocommit=True,
            prepare_threshold=0,
        )
        _CHECKPOINTER = PostgresSaver(_PG_CONN)
        _CHECKPOINTER.setup()
        logger.info("Checkpointer: PostgresSaver")
    else:
        from langgraph.checkpoint.memory import MemorySaver

        _CHECKPOINTER = MemorySaver()
        logger.info("Checkpointer: MemorySaver (in-memory fallback)")
    return _CHECKPOINTER
