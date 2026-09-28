import pytest

from sih import config as C

pytestmark = pytest.mark.skipif(not (C.OUT / "products.json").exists(), reason="run run_demo.py first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from sih.api.app import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_core_and_point_agree(client):
    core = client.get("/alerts/core").json()
    lon, lat = core["geometry"]["coordinates"]
    p = client.get(f"/alerts/point?lat={lat}&lon={lon}").json()
    assert p["prob"] == core["properties"]["prob"]


def test_bad_lead_is_404(client):
    assert client.get("/alerts/core?lead_hours=7").status_code == 404
