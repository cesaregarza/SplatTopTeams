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

    def match_targets(self, snapshot_id, query, limit=25, tournament_id=None):
        return [1] if "alp" in query.lower() else []

    def load_cluster_map(self, snapshot_id, profile):
        return {1: {"cluster_id": 3, "cluster_size": 5, "representative_team_name": "Alpha"}}

    def suggest_team_names(self, snapshot_id, query, limit=8):
        names = ["Alpha", "Bravo", "Gamma"]
        q = query.lower()
        return [
            {"team_id": i + 1, "team_name": name, "lineup_count": 10}
            for i, name in enumerate(names)
            if q in name.lower()
        ][:limit]

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
        tournament_id=None,
        seed_player_ids=None,
        recency_weight=0.0,
    ):
        rows = self.load_embeddings(snapshot_id)
        target_ids = self.match_targets(
            snapshot_id,
            query,
            tournament_id=tournament_id,
        )
        cluster_map = self.load_cluster_map(snapshot_id, cluster_mode) if include_clusters else {}
        return rank_similar_teams(
            embeddings=rows,
            target_team_ids=target_ids,
            cluster_map=cluster_map,
            top_n=top_n,
            min_relevance=min_relevance,
            recency_weight=recency_weight,
        )

    def list_tournament_teams(self, snapshot_id, tournament_id, query, limit):
        return {
            "tournament_id": int(tournament_id),
            "source": "dataset",
            "teams": [
                {
                    "team_id": 1,
                    "team_name": "Alpha",
                    "lineup_count": 10,
                    "event_time_ms": 1000,
                    "source": "dataset",
                }
            ],
        }

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

    def suggest_players(self, snapshot_id, query, limit=10):
        players = [
            {"player_id": 101, "display_name": "PlayerAlpha", "team_count": 3},
            {"player_id": 102, "display_name": "PlayerBravo", "team_count": 2},
        ]
        q = query.lower()
        return [p for p in players if q in p["display_name"].lower()][:limit]

    def get_player_teams(self, snapshot_id, player_id, limit=50):
        if player_id == 101:
            return {
                "player_id": 101,
                "display_name": "PlayerAlpha",
                "team_count": 1,
                "teams": [
                    {
                        "team_id": 1,
                        "team_name": "Alpha",
                        "lineup_count": 10,
                        "event_time_ms": 1000,
                        "player_match_count": 5,
                        "roster_player_names": ["PlayerAlpha", "PlayerBravo"],
                    }
                ],
            }
        return {
            "player_id": player_id,
            "display_name": str(player_id),
            "team_count": 0,
            "teams": [],
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

    def analytics_team_lab(self, snapshot_id, profile, team_id, neighbors):
        if int(team_id) != 1:
            return None
        return {
            "snapshot_id": snapshot_id,
            "cluster_mode": profile,
            "team": {
                "team_id": 1,
                "team_name": "Alpha",
                "lineup_count": 10,
                "distinct_lineup_count": 2,
                "lineup_entropy": 0.4,
                "top_lineup_share_pct": 60.0,
                "effective_lineups": 1.5,
                "volatility_score": 0.2,
                "uniqueness_score": 0.1,
                "cluster_id": 3,
                "cluster_size": 5,
            },
            "neighbors": [
                {
                    "team_id": 2,
                    "team_name": "Bravo",
                    "sim_to_query": 0.9,
                    "sim_semantic": 0.92,
                    "sim_identity": 0.88,
                    "identity_delta": -0.04,
                    "cluster_id": 3,
                }
            ],
            "match_summary": {
                "matches": 3,
                "wins": 2,
                "win_rate": 0.6667,
                "same_cluster_matches": 1,
                "same_cluster_win_rate": 1.0,
                "cross_cluster_matches": 2,
                "cross_cluster_win_rate": 0.5,
            },
            "opponent_cluster_breakdown": [
                {"cluster_id": 3, "matches": 1, "wins": 1, "win_rate": 1.0}
            ],
        }

    def analytics_team_matches(self, snapshot_id, team_ids, limit):
        return {
            "snapshot_id": snapshot_id,
            "summary": {
                "primary_team_id": int(team_ids[0]),
                "primary_team_name": "Alpha",
                "team_ids": [int(team_id) for team_id in team_ids],
                "team_names": ["Alpha", "Alpha Prime"][: len(team_ids)],
                "selected_team_count": len(team_ids),
                "total_matches": 1,
                "wins": 1,
                "losses": 0,
                "unresolved_matches": 0,
                "decided_matches": 1,
                "win_rate": 1.0,
                "tournaments": 1,
                "tournament_tier_distribution": {
                    "X": 0,
                    "S+": 1,
                    "S": 0,
                    "A+": 0,
                    "A": 0,
                    "A-": 0,
                    "Unscored": 0,
                },
                "tournament_tier_match_distribution": {
                    "X": 0,
                    "S+": 1,
                    "S": 0,
                    "A+": 0,
                    "A": 0,
                    "A-": 0,
                    "Unscored": 0,
                },
            },
            "matches": [
                {
                    "match_id": 77,
                    "team_id": int(team_ids[0]),
                    "team_name": "Alpha",
                    "opponent_team_id": 2,
                    "opponent_team_name": "Bravo",
                    "tournament_id": 11,
                    "tournament_name": "Summer Cup",
                    "tournament_score_tier": "S+",
                    "winner_side": "team",
                    "team_score": 3.0,
                    "opponent_score": 1.0,
                    "team_roster": [{"player_id": 101, "player_name": "PlayerAlpha"}],
                    "opponent_roster": [{"player_id": 201, "player_name": "PlayerBravo"}],
                    "event_time_ms": 1000,
                    "match_rounds": [],
                    "team_is_winner": True,
                    "opponent_is_winner": False,
                }
            ],
        }


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


def test_tournament_team_lookup_contract():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/tournaments/3192/teams", params={"q": "alp", "limit": 20})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["tournament_id"] == 3192
    assert payload["source"] == "dataset"
    assert payload["count"] == 1
    assert payload["teams"][0]["team_name"] == "Alpha"
    app.dependency_overrides.clear()


def test_suggest_endpoint_contract():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/team-search/suggest", params={"q": "alp"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["snapshot_id"] == 7
    assert payload["query"] == "alp"
    assert len(payload["suggestions"]) == 1
    assert payload["suggestions"][0]["team_name"] == "Alpha"
    app.dependency_overrides.clear()


def test_suggest_endpoint_no_match():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/team-search/suggest", params={"q": "zzz"})
    assert resp.status_code == 200
    assert resp.json()["suggestions"] == []
    app.dependency_overrides.clear()


def test_search_accepts_recency_weight():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get(
        "/api/team-search",
        params={"q": "alp", "recency_weight": "0.3"},
    )
    assert resp.status_code == 200
    assert resp.json()["result_count"] >= 1
    app.dependency_overrides.clear()


def test_player_suggest_endpoint():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/players/suggest", params={"q": "alpha"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["snapshot_id"] == 7
    assert payload["query"] == "alpha"
    assert len(payload["suggestions"]) == 1
    assert payload["suggestions"][0]["display_name"] == "PlayerAlpha"
    assert payload["suggestions"][0]["team_count"] == 3
    app.dependency_overrides.clear()


def test_player_suggest_no_match():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/players/suggest", params={"q": "zzz"})
    assert resp.status_code == 200
    assert resp.json()["suggestions"] == []
    app.dependency_overrides.clear()


def test_player_teams_endpoint():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/players/101/teams")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["snapshot_id"] == 7
    assert payload["player_id"] == 101
    assert payload["display_name"] == "PlayerAlpha"
    assert payload["team_count"] == 1
    assert len(payload["teams"]) == 1
    assert payload["teams"][0]["team_name"] == "Alpha"
    assert payload["teams"][0]["player_match_count"] == 5
    app.dependency_overrides.clear()


def test_player_teams_empty():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get("/api/players/999/teams")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["team_count"] == 0
    assert payload["teams"] == []
    app.dependency_overrides.clear()


def test_team_matches_endpoint_contract():
    app.dependency_overrides[get_store] = lambda: FakeStore()
    client = TestClient(app)
    resp = client.get(
        "/api/analytics/team/1/matches",
        params={"team_ids": "1,3", "limit": 10},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["snapshot_id"] == 7
    assert payload["summary"]["primary_team_id"] == 1
    assert payload["summary"]["team_ids"] == [1, 3]
    assert payload["summary"]["selected_team_count"] == 2
    assert payload["matches"][0]["opponent_team_name"] == "Bravo"
    app.dependency_overrides.clear()
