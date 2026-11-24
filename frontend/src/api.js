const BASE = import.meta.env.VITE_API_URL || "";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}/api${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

const qs = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== "" && v != null)).toString();

export const api = {
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
  signals: (params = {}) => request(`/signals?${qs(params)}`),
  model: () => request("/model"),
};
