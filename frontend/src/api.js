// Live mode talks to the FastAPI backend. Demo mode (`vite build --mode demo`, GitHub Pages) reads
// the same responses exported as static JSON by backend/scripts/export_demo.py.
export const DEMO = import.meta.env.MODE === "demo";
const BASE = import.meta.env.VITE_API_URL || "";
const DEMO_BASE = `${import.meta.env.BASE_URL}demo/`;

// grids pre-computed for the static demo
export const DEMO_DURATIONS = [30, 60, 90, 120, 180];
export const DEMO_LOSSES = [50, 100];
export const DEMO_BUDGETS = [0.5, 1, 2, 5, 10, 20, 50];
const SAVED_KEY = "aerosupply.demo.simulations";
const nearest = (v, grid) => grid.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));

async function request(path, options = {}) {
  const res = await fetch(`${BASE}/api${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function file(rel) {
  const res = await fetch(`${DEMO_BASE}${rel}`);
  if (!res.ok) throw new Error(`demo data not found: ${rel}`);
  return res.json();
}

const qs = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)).toString();

function loadSaved() {
  try { return JSON.parse(localStorage.getItem(SAVED_KEY) || "[]"); } catch { return []; }
}

const liveApi = {
  meta: () => request("/meta"),
  overview: () => request("/overview"),
  suppliers: (params = {}) => request(`/suppliers?${qs(params)}`),
  supplier: (id) => request(`/suppliers/${id}`),
  supplierWeekly: (id) => request(`/suppliers/${id}/weekly`),
  parts: (params = {}) => request(`/parts?${qs(params)}`),
  part: (id) => request(`/parts/${id}`),
  network: () => request("/network"),
  spofs: () => request("/network/spofs"),
  simulate: (body) => request("/simulate", { method: "POST", body: JSON.stringify(body) }),
  simulations: () => request("/simulations"),
  mitigation: (budget) => request(`/mitigation?budget_meur=${budget}`),
  commodities: () => request("/commodities"),
  model: () => request("/model"),
};

const demoApi = {
  meta: () => file("meta.json"),
  overview: () => file("overview.json"),
  suppliers: () => file("suppliers.json"),
  supplier: (id) => file(`suppliers/${id}.json`),
  supplierWeekly: (id) => file(`weekly/${id}.json`).catch(() => []),
  parts: () => file("parts.json"),
  part: (id) => file(`parts/${id}.json`),
  network: () => file("network.json"),
  spofs: () => file("spofs.json"),
  simulate: async (body) => {
    const d = nearest(body.duration_days, DEMO_DURATIONS);
    const l = nearest(Math.round(body.capacity_loss * 100), DEMO_LOSSES);
    const res = await file(`sim/${body.supplier_id}_${d}_${l}.json`);
    if (body.save) {
      const saved = loadSaved();
      saved.unshift({ label: `${res.supplier_name} — ${d} d, ${l} %`, supplier_id: res.supplier_id,
                      created_at: new Date().toISOString().slice(0, 19), aircraft_at_risk: res.aircraft_at_risk,
                      parts_short: res.parts_short });
      try { localStorage.setItem(SAVED_KEY, JSON.stringify(saved.slice(0, 20))); } catch { /* private mode */ }
    }
    return res;
  },
  simulations: async () => loadSaved(),
  mitigation: (budget) => file(`mitigation/${nearest(Number(budget), DEMO_BUDGETS)}.json`),
  commodities: () => file("commodities.json"),
  model: () => file("model.json"),
};

export const api = DEMO ? demoApi : liveApi;
