"""Connection pool and schema initialization for PostgreSQL + pgvector."""

import os
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://pgvector:pgvector@postgres:5432/vla"
)

_pool: ConnectionPool | None = None

_INIT_SQL_PATH = os.path.join(os.path.dirname(__file__), "init.sql")


def startup():
    """Create the connection pool and ensure the schema exists."""
    global _pool
    _pool = ConnectionPool(DATABASE_URL, min_size=2, max_size=10, open=True)

    with _pool.connection() as conn:
        with open(_INIT_SQL_PATH) as f:
            conn.execute(f.read())
        conn.commit()


def shutdown():
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def check_health() -> dict:
    """Return DB connection status."""
    if _pool is None:
        return {"status": "error", "detail": "pool not initialized"}
    try:
        with _pool.connection() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@contextmanager
def get_conn():
    """Context manager that yields a psycopg connection from the pool."""
    if _pool is None:
        raise RuntimeError("Connection pool not initialized — call startup() first")
    with _pool.connection() as conn:
        yield conn
