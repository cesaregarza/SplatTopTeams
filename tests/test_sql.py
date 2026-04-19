from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy.exc import SQLAlchemyError

from team_api.sql import _execute_ddl_in_savepoint


class _FakeConnection:
    def __init__(self, failing_statements: set[str] | None = None) -> None:
        self.failing_statements = failing_statements or set()
        self.executed: list[str] = []
        self.savepoint_entries = 0

    @contextmanager
    def begin_nested(self):
        self.savepoint_entries += 1
        yield

    def execute(self, statement):
        text_value = str(statement)
        self.executed.append(text_value)
        if any(fragment in text_value for fragment in self.failing_statements):
            raise SQLAlchemyError("boom")


def test_execute_ddl_in_savepoint_returns_false_on_error():
    conn = _FakeConnection({"CREATE EXTENSION"})

    ok = _execute_ddl_in_savepoint(conn, "CREATE EXTENSION IF NOT EXISTS vector")

    assert ok is False
    assert conn.savepoint_entries == 1


def test_execute_ddl_in_savepoint_allows_following_calls():
    conn = _FakeConnection({"CREATE EXTENSION"})

    first = _execute_ddl_in_savepoint(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    second = _execute_ddl_in_savepoint(
        conn, "CREATE INDEX IF NOT EXISTS ix_demo ON demo(id)"
    )

    assert first is False
    assert second is True
    assert conn.savepoint_entries == 2
