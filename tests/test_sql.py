from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy.exc import SQLAlchemyError

from team_api.sql import _execute_ddl_in_savepoint, _require_existing_schema


class _FakeResult:
    def __init__(self, scalar_value=None) -> None:
        self._scalar_value = scalar_value

    def scalar_one(self):
        return self._scalar_value

    def scalar_one_or_none(self):
        return self._scalar_value


class _FakeConnection:
    def __init__(
        self,
        failing_statements: set[str] | None = None,
        schema_exists: bool = True,
    ) -> None:
        self.failing_statements = failing_statements or set()
        self.schema_exists = schema_exists
        self.executed: list[str] = []
        self.savepoint_entries = 0

    @contextmanager
    def begin_nested(self):
        self.savepoint_entries += 1
        yield

    def execute(self, statement, _params=None):
        text_value = str(statement)
        self.executed.append(text_value)
        if any(fragment in text_value for fragment in self.failing_statements):
            raise SQLAlchemyError("boom")
        if "FROM information_schema.schemata" in text_value:
            return _FakeResult(self.schema_exists)
        return _FakeResult(None)


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


def test_require_existing_schema_accepts_existing_schema():
    conn = _FakeConnection(schema_exists=True)

    _require_existing_schema(conn, "comp_rankings")

    assert any("FROM information_schema.schemata" in stmt for stmt in conn.executed)


def test_require_existing_schema_raises_for_missing_schema():
    conn = _FakeConnection(schema_exists=False)

    try:
        _require_existing_schema(conn, "comp_rankings")
    except RuntimeError as exc:
        assert "does not exist" in str(exc)
        assert "comp_rankings" in str(exc)
    else:
        raise AssertionError("Expected missing schema to raise RuntimeError")
