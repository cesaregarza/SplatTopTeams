from __future__ import annotations

from fastapi.testclient import TestClient

from team_api.app import app
from team_api.search_logic import rank_similar_teams
from team_api.dependencies import get_store


class FakeStore:
    def ping(self):
        return True

    def latest_snapshot(self):
        return {"run_id": 7, "teams_indexed": 3}

    def load_embeddings(self, snapshot_id):
        from team_api.store import EmbeddingRow
        import numpy as np

        rows = []
        for tid, name, vec in [
            (1, "Alpha", [1.0, 0.0]),
            (2, "Bravo", [0.9, 0.1]),
            (3, "Gamma", [0.1, 0.9]),
        ]:
            arr = np.asarray(vec, dtype=float)
            rows.append(
                EmbeddingRow(
                    team_id=tid,
                    tournament_id=11,
                    team_name=name,
                    event_time_ms=1000,
                    lineup_count=10,
                    semantic_vector=arr,
                    identity_vector=arr,
                    final_vector=arr,
                    top_lineup_summary="",
                )
            )
        return rows

    def match_targets(self, snapshot_id, query, limit=25):
        return [1] if "alp" in query.lower() else []

    def load_cluster_map(self, snapshot_id, profile):
        return {1: {"cluster_id": 3, "cluster_size": 5, "representative_team_name": "Alpha"}}

    def search_similar_teams(
        self,
        snapshot_id,
        query,
        top_n,
        min_relevance,
        cluster_mode,
        include_clusters,
        consolidate=True,
        consolidate_min_overlap=0.8,
    ):
        rows = self.load_embeddings(snapshot_id)
        target_ids = self.match_targets(snapshot_id, query)
        cluster_map = self.load_cluster_map(snapshot_id, cluster_mode) if include_clusters else {}
        return rank_similar_teams(
            embeddings=rows,
            target_team_ids=target_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=min_relevance,
        )

    def list_clusters(self, snapshot_id, profile, query, limit):
        return [
            {
                "cluster_id": 3,
                "cluster_size": 5,
                "representative_team_name": "Alpha",
                "top_teams": [],
                "stability_hint": "stable",
            }
        ]

    def cluster_detail(self, snapshot_id, profile, cluster_id):
        if cluster_id != 3:
            return None
        return {
            "cluster_id": 3,
            "cluster_size": 5,
            "representative_team_name": "Alpha",
            "members": [{"team_id": 1, "team_name": "Alpha"}],
        }

    def analytics_roster_diversity(
        self,
        snapshot_id,
        profile,
        min_similarity,
        max_player_overlap,
        min_cluster_size,
        limit,
    ):
        from team_api.analytics_logic import compute_roster_diversity_candidates

        rows = self.load_embeddings(snapshot_id)
        cluster_map = self.load_cluster_map(snapshot_id, profile)
        return compute_roster_diversity_candidates(
            rows,
            cluster_map,
            min_similarity=float(min_similarity),
            max_player_overlap=float(max_player_overlap),
            min_cluster_size=int(min_cluster_size),
            limit=int(limit),
        )


def test_health_and_team_search_contracts():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    search = client.get("/api/team-search", params={"q": "alp"})
    assert search.status_code == 200
    payload = search.json()
    assert payload["snapshot_id"] == 7
    assert payload["results"][0]["team_id"] == 1

    app.dependency_overrides.clear()


def test_cluster_detail_404():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/clusters/999")
    assert resp.status_code == 404
    app.dependency_overrides.clear()


def test_roster_diversity_contract():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get(
        "/api/analytics/roster-overlap",
        params={
            "cluster_mode": "explore",
            "min_similarity": "0.75",
            "max_player_overlap": "0.5",
            "min_cluster_size": "2",
            "limit": "10",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "cohorts" in payload
    assert "pairs" in payload
    app.dependency_overrides.clear()
