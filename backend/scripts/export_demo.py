"""Export every API response the dashboard needs as static JSON (GitHub Pages demo).

    python -m scripts.export_demo            # -> ../frontend/public/demo
    cd ../frontend && npm run build:demo     # static site in dist/

The demo is the real pipeline's output frozen at AS_OF; what-if scenarios are pre-computed on a
grid (duration x capacity loss) and the expediting optimiser on a set of budgets.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from aerosupply.api.main import app

OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "demo"
DURATIONS = (30, 60, 90, 120, 180)
LOSSES = (50, 100)
BUDGETS = (0.5, 1, 2, 5, 10, 20, 50)


def _round(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4) if abs(v) < 100 else round(v, 1)
    if isinstance(v, list):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


def write(rel: str, data) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_round(data), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    c = TestClient(app)
    get = lambda u: c.get(u).raise_for_status().json()  # noqa: E731
    for name, url in [("meta", "/api/meta"), ("overview", "/api/overview"), ("suppliers", "/api/suppliers"),
                      ("parts", "/api/parts"), ("network", "/api/network"), ("spofs", "/api/network/spofs"),
                      ("commodities", "/api/commodities"), ("model", "/api/model")]:
        write(f"{name}.json", get(url))
    sups = get("/api/suppliers")
    for s in sups:
        sid = s["supplier_id"]
        write(f"suppliers/{sid}.json", get(f"/api/suppliers/{sid}"))
        write(f"weekly/{sid}.json", get(f"/api/suppliers/{sid}/weekly"))
        for d in DURATIONS:
            for l in LOSSES:
                r = c.post("/api/simulate", json={"supplier_id": sid, "duration_days": d, "capacity_loss": l / 100}).json()
                r["parts"] = r["parts"][:40]
                write(f"sim/{sid}_{d}_{l}.json", r)
    for p in get("/api/parts"):
        write(f"parts/{p['part_id']}.json", get(f"/api/parts/{p['part_id']}"))
    for b in BUDGETS:
        write(f"mitigation/{b}.json", get(f"/api/mitigation?budget_meur={b}"))
    size = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"demo exported to {OUT} — {sum(1 for _ in OUT.rglob('*.json'))} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
