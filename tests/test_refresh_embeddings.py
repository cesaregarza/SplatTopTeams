from __future__ import annotations

from sqlalchemy.exc import SQLAlchemyError

from team_jobs import refresh_embeddings as re


def test_run_refresh_skips_schema_bootstrap_on_permission_denied(monkeypatch):
    bootstrap_calls = []
    completed = {}

    monkeypatch.setattr(re, "create_engine", lambda *_args, **_kwargs: object())

    def fake_ensure_search_tables(*_args, **_kwargs):
        bootstrap_calls.append(True)
        raise SQLAlchemyError("permission denied for database rankings_db")

    monkeypatch.setattr(re, "ensure_search_tables", fake_ensure_search_tables)
    monkeypatch.setattr(re, "_start_run", lambda *_args, **_kwargs: 42)
    monkeypatch.setattr(re, "_fetch_lineups", lambda *_args, **_kwargs: [])

    def fake_mark_run(_db_url, _schema, run_id, **kwargs):
        completed["run_id"] = run_id
        completed.update(kwargs)

    monkeypatch.setattr(re, "_mark_run", fake_mark_run)

    run_id = re.run_refresh(
        source_db_url="postgresql://source",
        target_db_url="postgresql://target",
        schema="comp_rankings",
        days=540,
        until_days=None,
        min_players=4,
        max_players=4,
        semantic_dim=64,
        identity_dim=256,
        identity_beta=3.0,
        identity_idf_cap=3.0,
        max_cluster_teams=2500,
        strict_threshold=0.94,
        explore_threshold=0.90,
        keep_runs=5,
    )

    assert bootstrap_calls == [True]
    assert run_id == 42
    assert completed["run_id"] == 42
    assert completed["status"] == "completed"
    assert completed["message"] == "No lineups found in requested window."


def test_run_refresh_raises_non_permission_bootstrap_error(monkeypatch):
    monkeypatch.setattr(re, "create_engine", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        re,
        "ensure_search_tables",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SQLAlchemyError("some other ddl failure")
        ),
    )

    try:
        re.run_refresh(
            source_db_url="postgresql://source",
            target_db_url="postgresql://target",
            schema="comp_rankings",
            days=540,
            until_days=None,
            min_players=4,
            max_players=4,
            semantic_dim=64,
            identity_dim=256,
            identity_beta=3.0,
            identity_idf_cap=3.0,
            max_cluster_teams=2500,
            strict_threshold=0.94,
            explore_threshold=0.90,
            keep_runs=5,
        )
    except SQLAlchemyError as exc:
        assert "some other ddl failure" in str(exc)
    else:
        raise AssertionError("Expected non-permission bootstrap error to be raised")
