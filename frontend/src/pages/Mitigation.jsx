import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Meter, Stat, fmt, label } from "../components/ui.jsx";
import { C, baseLayout } from "../theme.js";

export default function Mitigation() {
  const [budget, setBudget] = useState(5);
  const [applied, setApplied] = useState(5);
  const { data, error, loading } = useFetch(() => api.mitigation(applied), [applied]);
  const { data: spofs } = useFetch(api.spofs);
  return (
    <>
      <div className="page-head"><div><div className="crumbs">Decisions</div><h1>Mitigation plan</h1></div></div>
      <Card title="Expediting budget allocation"
            subtitle="Which parts to buy as expedited / spot orders (35 % premium) to protect the most aircraft? Optimised on the Monte-Carlo shortage scenarios: greedy marginal allocation, optimal for this separable concave problem.">
        <div className="filters">
          <label className="check">Budget (€m)
            <input type="number" min="0.5" max="200" step="0.5" value={budget} onChange={(e) => setBudget(e.target.value)} style={{ width: 90 }} />
          </label>
          <button className="btn primary" onClick={() => setApplied(Number(budget) || 5)}>Optimise</button>
        </div>
      </Card>
      {loading || error ? <Loading error={error} /> : (
        <>
          <div className="stats">
            <Stat label="Budget used" value={`€${fmt(data.spent_eur / 1e6, 2)}m`} sub={`of €${fmt(data.budget_eur / 1e6, 1)}m`} />
            <Stat label="Expected shortage before" value={fmt(data.expected_shortage_aircraft_sets_before, 0)} sub="aircraft-sets of missing parts, next 120 d" />
            <Stat label="Protected by the plan" value={fmt(data.expected_aircraft_sets_protected, 0)} sub="expected aircraft-sets" />
            <Stat label="Parts in the plan" value={data.parts.length} />
          </div>
          <div className="grid-2">
            <Card title="Diminishing returns" subtitle="Expected aircraft-sets protected as the budget grows.">
              <Plot height={260}
                data={[{ x: data.curve.map((c) => c.budget_meur), y: data.curve.map((c) => c.protected), type: "scatter", mode: "lines+markers",
                         line: { color: C.series[0], width: 2 }, marker: { size: 8, line: { color: C.surface, width: 2 } },
                         hovertemplate: "€%{x}m → %{y:.0f}<extra></extra>" }]}
                layout={baseLayout({ showlegend: false, xaxis: { title: { text: "Budget (€m)" }, rangemode: "tozero" },
                                     yaxis: { title: { text: "Aircraft-sets protected" }, rangemode: "tozero" } })} />
            </Card>
            <Card title="Structural actions" subtitle="Expediting buys time; it does not remove the cause.">
              <ul className="policy">
                <li>Qualify a second sub-tier source for the <strong>{data.structural.hidden_spof_parts}</strong> parts behind a hidden single point of failure (list below).</li>
                <li>Raise safety stock on parts whose sources score <em>Critical</em> and have less than 30 days of cover.</li>
                <li>Supplier development / on-site support for tier-1 suppliers in an ongoing anomaly episode.</li>
              </ul>
            </Card>
          </div>
          <Card title="Allocation">
            <table className="table"><thead><tr><th>Part</th><th>Commodity</th><th>P(stock-out 60 d)</th><th className="num">Units</th><th className="num">Cost (€k)</th>
              <th className="num">Expected shortage before</th><th className="num">After</th><th className="num">Protected</th></tr></thead>
              <tbody>{data.parts.map((p) => (
                <tr key={p.part_id}><td><Link to={`/parts/${p.part_id}`}>{p.name}</Link></td><td>{label(p.commodity)}</td><td><Meter value={p.p_stockout_60} /></td>
                  <td className="num">{fmt(p.units)}</td><td className="num">{fmt(p.cost_eur / 1000)}</td>
                  <td className="num">{fmt(p.expected_shortage_before, 1)}</td><td className="num">{fmt(p.expected_shortage_after, 1)}</td>
                  <td className="num strong">{fmt(p.expected_aircraft_protected, 1)}</td></tr>))}</tbody></table>
            <p className="note" style={{ marginTop: 10 }}>Quantities in aircraft-sets: units of the part needed per aircraft (BOM quantity weighted by current production rates). Summed over parts they overstate missing aircraft, since one aircraft can lack several parts.</p>
          </Card>
          {spofs && spofs.length > 0 && (
            <Card title="Second-source candidates (hidden SPOF)">
              <table className="table"><thead><tr><th>Part</th><th>Current tier-1 sources</th><th>Shared sub-tier supplier(s)</th></tr></thead>
                <tbody>{spofs.map((r) => (<tr key={r.part_id}><td><Link to={`/parts/${r.part_id}`}>{r.name}</Link> <span className="muted small">{r.part_id}</span></td>
                  <td className="small">{r.source_names.join(" · ")}</td>
                  <td className="small">{r.common_upstream.map((u, i) => <span key={u.supplier_id}>{i > 0 && " · "}<Link to={`/suppliers/${u.supplier_id}`}>{u.name}</Link> (T{u.tier})</span>)}</td></tr>))}</tbody></table>
            </Card>
          )}
        </>
      )}
    </>
  );
}
