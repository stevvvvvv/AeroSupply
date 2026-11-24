def test_overview(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    k = r.json()["kpis"]
    assert k["tier1"] > 0 and 0 <= k["otd_90d"] <= 1


def test_suppliers_and_detail(client):
    rows = client.get("/api/suppliers").json()
    assert {r["tier"] for r in rows} == {1, 2, 3}
    t1 = next(r for r in rows if r["tier"] == 1)
    assert 0 <= t1["risk_prob"] <= 1
    d = client.get(f"/api/suppliers/{t1['supplier_id']}").json()
    assert d["drivers"] and "parts_supplied" in d
    assert client.get(f"/api/suppliers/{t1['supplier_id']}/weekly").status_code == 200
    assert client.get("/api/suppliers/NOPE").status_code == 404


def test_parts(client):
    rows = client.get("/api/parts").json()
    p = rows[0]
    assert 0 <= p["p_stockout_60"] <= 1
    d = client.get(f"/api/parts/{p['part_id']}").json()
    assert len(d["forecast"]["p50"]) == len(d["forecast"]["dates"])


def test_network_and_simulation(client):
    net = client.get("/api/network").json()
    assert len(net["nodes"]) > 300 and len(net["edges"]) > 500
    sid = next(n["key"] for n in net["nodes"] if n["kind"] == "supplier" and n.get("tier") == 2)
    r = client.post("/api/simulate", json={"supplier_id": sid, "duration_days": 60, "capacity_loss": 1.0, "save": True})
    assert r.status_code == 200
    res = r.json()
    assert res["impact_lag_days"] == 30
    r2 = client.post("/api/simulate", json={"supplier_id": sid, "duration_days": 120, "capacity_loss": 1.0}).json()
    assert r2["aircraft_at_risk"] >= res["aircraft_at_risk"]          # longer outage never hurts less
    assert any(s["supplier_id"] == sid for s in client.get("/api/simulations").json())


def test_mitigation_and_model(client):
    m = client.get("/api/mitigation?budget_meur=2").json()
    assert m["spent_eur"] <= 2e6
    card = client.get("/api/model").json()
    assert "lead_time" in card["metrics"] and card["last_run"] is not None
