"""Call every API endpoint in-process and print the responses (run after run_demo.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python scripts/smoke_test_api.py` finds sih

from fastapi.testclient import TestClient  # noqa: E402

from sih.api.app import app  # noqa: E402

c = TestClient(app)


def show(path):
    r = c.get(path)
    body = r.json()
    text = json.dumps(body)
    print(f"GET {path} -> {r.status_code}\n  {text[:400]}{' ...' if len(text) > 400 else ''}")
    assert r.status_code == 200, text
    return body


show("/health")
tracks = show("/tracks")
show(f"/tracks/{tracks[0]['id']}")
core = show("/alerts/core")
lat, lon = core["geometry"]["coordinates"][1], core["geometry"]["coordinates"][0]
show(f"/alerts/point?lat={lat}&lon={lon}")
show(f"/alerts/point?lat={lat + 1.5}&lon={lon + 1.5}&radius_km=10")
for L in show("/health")["downscaled_leads_h"]:
    show(f"/alerts/core?lead_hours={L}")
r = c.get("/alerts/first-warning")  # 404 when no lead reaches an alert level
print(f"GET /alerts/first-warning -> {r.status_code}\n  {json.dumps(r.json())[:300]}")
assert r.status_code in (200, 404)
z = show("/alerts/zones")
print("zones by level:", {k: sum(f["properties"]["level"] == k for f in z["features"]) for k in ("low", "moderate", "severe")})
assert c.get("/dashboard").status_code == 200
print("all endpoints OK")
