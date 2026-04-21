from __future__ import annotations

from dataclasses import replace

from sqlalchemy.exc import SQLAlchemyError

from team_jobs import refresh_embeddings as re
from team_api.healbook_reference import build_healbook_reference_payloads


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


def test_resolve_refresh_db_urls_supports_target_override(monkeypatch):
    args = re.build_parser().parse_args([])

    monkeypatch.setenv("DATABASE_URL", "postgresql://shared")
    monkeypatch.setenv("TARGET_DATABASE_URL", "postgresql://target")

    source_db_url, target_db_url = re._resolve_refresh_db_urls(args)

    assert source_db_url == "postgresql://shared"
    assert target_db_url == "postgresql://target"


def test_run_refresh_builds_family_clusters(monkeypatch):
    persisted = {}

    monkeypatch.setattr(re, "create_engine", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(re, "ensure_search_tables", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(re, "_start_run", lambda *_args, **_kwargs: 42)
    monkeypatch.setattr(re, "_fetch_lineups", lambda *_args, **_kwargs: [{"match_id": 1}])
    monkeypatch.setattr(
        re,
        "_index_lineups",
        lambda *_args, **_kwargs: ({}, {}, {}, {}),
    )
    monkeypatch.setattr(re, "_build_semantic_vectors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(re, "_build_identity_vectors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(re, "_fetch_team_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(re, "_fetch_player_names", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        re,
        "_build_payloads",
        lambda *_args, **_kwargs: build_healbook_reference_payloads(),
    )
    monkeypatch.setattr(re, "_cleanup_old_runs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(re, "_mark_run", lambda *_args, **_kwargs: None)

    def fake_persist_snapshot(
        _db_url,
        _schema,
        _run_id,
        _identity_beta,
        _payloads,
        _strict_clusters,
        _explore_clusters,
        family_clusters,
        *,
        final_vector_dim,
    ):
        persisted["family_clusters"] = family_clusters
        persisted["final_vector_dim"] = final_vector_dim

    monkeypatch.setattr(re, "_persist_snapshot", fake_persist_snapshot)

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

    family_clusters = persisted["family_clusters"]
    assert run_id == 42
    assert persisted["final_vector_dim"] == 320
    assert family_clusters[46624]["cluster_id"] == family_clusters[49106]["cluster_id"]
    assert family_clusters[46624]["cluster_id"] == family_clusters[54749]["cluster_id"]
    assert 47527 not in family_clusters


def test_normalize_family_name_bucket_strips_bracketed_variants():
    assert re._normalize_family_name_bucket("Healbook (SBR Edition)") == "healbook"
    assert re._normalize_family_name_bucket(" HEALBOOK ") == "healbook"
    assert re._normalize_family_name_bucket("Heal-book") == "heal book"


def test_representative_cluster_name_prefers_clean_family_label():
    payloads = build_healbook_reference_payloads()
    mutated = []
    for payload in payloads:
        if payload.team_id == 46624:
            mutated.append(replace(payload, team_name="Healbook (Main Roster)"))
        elif payload.team_id == 49106:
            mutated.append(replace(payload, team_name="Healbook"))
        elif payload.team_id == 54749:
            mutated.append(replace(payload, team_name="HEALBOOK"))
        else:
            mutated.append(payload)

    representative = re._representative_cluster_name(
        "family",
        [payload for payload in mutated if payload.team_id in {46624, 49106, 54749}],
    )

    assert representative == "Healbook"
