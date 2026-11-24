import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Meter, StatusBadge, TierTag, fmt, label, pct } from "../components/ui.jsx";
import { C, STATUS, baseLayout } from "../theme.js";

const LEVELS = ["CRITICAL", "WARNING", "WATCH", "HEALTHY"];

export default function Suppliers() {
  const { data, error, loading } = useFetch(() => api.suppliers());
  const [params] = useSearchParams();
  const [tier, setTier] = useState(params.get("tier") || "");
  const [level, setLevel] = useState("");
  const [q, setQ] = useState("");
  const nav = useNavigate();
  const rows = useMemo(() => (data || []).filter((s) =>
    (!tier || String(s.tier) === tier) && (!level || s.risk_level === level) &&
    (!q || `${s.name} ${s.supplier_id} ${s.commodity}`.toLowerCase().includes(q.toLowerCase()))), [data, tier, level, q]);
  if (loading || error) return <Loading error={error} />;
  const t1 = data.filter((s) => s.tier === 1);
  return (
    <>
      <div className="page-head">
        <div><div className="crumbs">Suppliers</div><h1>Supplier risk</h1></div>
      </div>
      <Card title="Risk matrix — tier-1 suppliers"
            subtitle="Probability of a severe delivery failure in the next 90 days (model) against the impact of losing the supplier for 90 days (network simulation). Marker size = 12-month spend. Click a point to open the supplier.">
        <Plot height={380}
          onClick={(e) => e.points?.[0] && nav(`/suppliers/${e.points[0].customdata[0]}`)}
          data={LEVELS.map((lv) => {
            const pts = t1.filter((s) => s.risk_level === lv);
            return {
              x: pts.map((s) => s.risk_prob), y: pts.map((s) => s.impact_aircraft_90d), type: "scatter", mode: "markers",
              name: STATUS[lv].label, customdata: pts.map((s) => [s.supplier_id, s.name, label(s.commodity)]),
              marker: { color: lv === "HEALTHY" ? "#aab3c2" : STATUS[lv].color, size: pts.map((s) => 8 + Math.sqrt(s.spend_12m_meur || 0) * 1.6),
                        line: { color: C.surface, width: 2 }, opacity: 0.9 },
              hovertemplate: "<b>%{customdata[1]}</b><br>%{customdata[2]}<br>P(failure) %{x:.0%}<br>Aircraft at risk if lost: %{y}<extra></extra>",
            };
          })}
          layout={baseLayout({
            xaxis: { title: { text: "P(severe delivery failure, next 90 days)" }, tickformat: ".0%", range: [-0.03, 1.03] },
            yaxis: { title: { text: "Aircraft at risk if lost 90 days" }, rangemode: "tozero" },
            margin: { l: 60, r: 16, t: 30, b: 50 },
          })} />
      </Card>
      <Card title={`${rows.length} suppliers`}
            actions={
              <div className="filters">
                <input placeholder="Search name, id, commodity" value={q} onChange={(e) => setQ(e.target.value)} />
                <select value={tier} onChange={(e) => setTier(e.target.value)}>
                  <option value="">All tiers</option><option value="1">Tier 1</option><option value="2">Tier 2</option><option value="3">Tier 3</option>
                </select>
                <select value={level} onChange={(e) => setLevel(e.target.value)}>
                  <option value="">All levels</option>{LEVELS.map((l) => <option key={l} value={l}>{STATUS[l].label}</option>)}
                </select>
              </div>
            }>
        <table className="table">
          <thead><tr><th>Supplier</th><th>Tier</th><th>Commodity</th><th>Status</th><th>Risk</th>
            <th className="num">OTD 90 d</th><th className="num">Parts dependent</th><th className="num">Hidden SPOF</th>
            <th className="num">Aircraft at risk if lost</th><th className="num">Spend 12 m (€m)</th></tr></thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.supplier_id} onClick={() => nav(`/suppliers/${s.supplier_id}`)}>
                <td><span className="strong">{s.name}</span> <span className="muted small">{s.supplier_id} · {s.country}</span></td>
                <td><TierTag tier={s.tier} /></td>
                <td>{label(s.commodity)}</td>
                <td><StatusBadge status={s.risk_level} /></td>
                <td>{s.tier === 1 ? <Meter value={s.risk_prob} title="P(severe failure in 90 days)" />
                  : <span className="small" title="Rule-based screening score (no delivery data on sub-tier)">score {fmt(s.screening_score)}</span>}</td>
                <td className="num">{pct(s.otd_90d)}</td>
                <td className="num">{s.parts_dependent}</td>
                <td className="num">{s.hidden_spof_parts || ""}</td>
                <td className="num">{fmt(s.impact_aircraft_90d)}</td>
                <td className="num">{s.tier === 1 ? fmt(s.spend_12m_meur, 1) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}
