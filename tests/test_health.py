from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_expected_shape() -> None:
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"status", "db_ok", "index_count"}
    assert isinstance(body["db_ok"], bool)
    assert body["status"] in {"ok", "degraded"}
    if body["db_ok"]:
        assert isinstance(body["index_count"], int)
    else:
        assert body["index_count"] is None


def test_unknown_route_is_404() -> None:
    res = client.get("/api/v1/does-not-exist")
    assert res.status_code == 404


def test_routers_are_mounted() -> None:
    paths = set(app.openapi()["paths"])
    assert "/api/v1/health" in paths
