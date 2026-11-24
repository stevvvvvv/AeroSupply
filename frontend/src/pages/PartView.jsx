import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Meter, Stat, StatusBadge, fmt, label, pct } from "../components/ui.jsx";
import { C, baseLayout } from "../theme.js";

export default function PartView() {
  const { id } = useParams();
  const { data: p, error, loading } = useFetch(() => api.part(id), [id]);
  if (loading || error) return <Loading error={error} />;
  const f = p.forecast;
  const pos = p.open_purchase_orders;
  return (
    <>
      <div className="page-head">
        <div>
          <div className="crumbs"><Link to="/parts">Parts</Link> / {p.part_id}</div>
          <h1>{p.name}</h1>
          <p className="muted">{p.part_number} · {label(p.commodity)} · used on {p.programs} · unit cost €{fmt(p.unit_cost)}</p>
        </div>
      </div>
      <div className="stats">
        <Stat label="On hand" value={fmt(p.on_hand)} sub={p.in_shortage_now ? "short — parts missing on the line" : `${fmt(p.days_of_cover)} days of cover`} />
        <Stat label="P(stock-out ≤ 30 d)" value={pct(p.p_stockout_30)} />
        <Stat label="P(stock-out ≤ 60 d)" value={pct(p.p_stockout_60)} sub={p.mrp_stockout_60 ? "ERP also flags it" : "ERP projection: no shortage"} />
        <Stat label="P(stock-out ≤ 90 d)" value={pct(p.p_stockout_90)} />
        <Stat label="Open POs" value={p.open_pos} sub={`${p.overdue_pos} overdue · ${fmt(p.open_qty)} units`} />
      </div>
      <Card title="Inventory: last 12 months and 120-day forecast"
            subtitle="Band: 10th–90th percentile of the Monte-Carlo scenarios; solid: median; dashed: ERP projection with every PO on its promised date. Below zero = parts missing on the line.">
        <Plot height={340}
          data={[
            { x: p.history.dates, y: p.history.on_hand, name: "Actual on hand", type: "scatter", mode: "lines", line: { color: C.text2, width: 1.5 } },
            { x: f.dates, y: f.p90, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
            { x: f.dates, y: f.p10, name: "P10–P90", type: "scatter", mode: "lines", fill: "tonexty", fillcolor: C.band, line: { width: 0 },
              hoverinfo: "skip" },
            { x: f.dates, y: f.p50, name: "Forecast (median)", type: "scatter", mode: "lines", line: { color: C.series[0], width: 2 },
              hovertemplate: "%{x}<br>median %{y:.0f}<extra></extra>" },
            { x: f.dates, y: f.mrp, name: "ERP projection", type: "scatter", mode: "lines", line: { color: C.series[1], width: 2, dash: "dash" },
              hovertemplate: "%{x}<br>ERP %{y:.0f}<extra></extra>" },
          ]}
          layout={baseLayout({
            yaxis: { title: { text: "Units" } },
            shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: "#d03b3b", width: 1 } },
                     { type: "line", xref: "x", yref: "paper", x0: f.dates[0], x1: f.dates[0], y0: 0, y1: 1, line: { color: C.axis, width: 1, dash: "dot" } }],
            annotations: [{ x: f.dates[0], y: 1, yref: "paper", text: "today", showarrow: false, xanchor: "left", font: { color: C.muted, size: 11 } }],
          })} />
      </Card>
      <Card title="Open purchase orders" subtitle="Promised date vs. arrival predicted by the lead-time model (P10 – P50 – P90).">
        {pos.length === 0 ? <p className="muted">No open order.</p> : (
          <>
            <Plot height={Math.max(160, 34 * pos.length + 60)}
              data={[
                { y: pos.map((o) => o.po_id), x: pos.map((o) => o.promised_date), type: "scatter", mode: "markers", name: "Promised",
                  marker: { symbol: "line-ns-open", size: 16, color: C.text2, line: { width: 2, color: C.text2 } } },
                ...pos.map((o, i) => ({ y: [o.po_id, o.po_id], x: [o.eta_p10, o.eta_p90], type: "scatter", mode: "lines", showlegend: i === 0,
                  name: "P10–P90", line: { color: C.series[0], width: 6 }, opacity: 0.45, hoverinfo: "skip" })),
                { y: pos.map((o) => o.po_id), x: pos.map((o) => o.eta_p50), type: "scatter", mode: "markers", name: "Predicted (median)",
                  marker: { color: C.series[0], size: 10, line: { color: C.surface, width: 2 } },
                  customdata: pos.map((o) => [o.supplier_name, o.qty]), hovertemplate: "%{y} · %{customdata[0]}<br>ETA %{x}<br>%{customdata[1]} units<extra></extra>" },
              ]}
              layout={baseLayout({ margin: { l: 90, r: 16, t: 30, b: 30 }, yaxis: { autorange: "reversed" }, xaxis: { type: "date" } })} />
            <table className="table">
              <thead><tr><th>PO</th><th>Supplier</th><th>Ordered</th><th className="num">Qty</th><th>Promised</th><th>Predicted (P10 / P50 / P90)</th></tr></thead>
              <tbody>{pos.map((o) => (
                <tr key={o.po_id}><td className="mono">{o.po_id}</td><td><Link to={`/suppliers/${o.supplier_id}`}>{o.supplier_name}</Link></td>
                  <td>{o.order_date}</td><td className="num">{fmt(o.qty)}</td>
                  <td>{o.promised_date}{o.overdue_days > 0 && <span className="muted small"> · {o.overdue_days} d overdue</span>}</td>
                  <td className="mono small">{o.eta_p10} / <span className="strong">{o.eta_p50}</span> / {o.eta_p90}</td></tr>))}</tbody>
            </table>
          </>
        )}
      </Card>
      <div className="grid-2">
        <Card title="Tier-1 sources">
          <table className="table"><thead><tr><th>Supplier</th><th className="num">Share</th><th>Status</th><th>P(failure 90 d)</th></tr></thead>
            <tbody>{p.sources.map((s) => (
              <tr key={s.supplier_id}><td><Link to={`/suppliers/${s.supplier_id}`}>{s.name}</Link></td><td className="num">{pct(s.share)}</td>
                <td><StatusBadge status={s.level} /></td><td><Meter value={s.risk_prob} /></td></tr>))}</tbody></table>
        </Card>
        <Card title="Sub-tier dependence" subtitle="Share of this part's supply lost if the sub-tier supplier stopped (inputs are complements, sources substitutes).">
          <table className="table"><thead><tr><th>Sub-tier supplier</th><th>Dependence</th><th>Status</th></tr></thead>
            <tbody>{p.subtier_dependence.map((s) => (
              <tr key={s.supplier_id}><td><Link to={`/suppliers/${s.supplier_id}`}>{s.name}</Link></td><td><Meter value={s.dependence} /></td>
                <td><StatusBadge status={s.level} /></td></tr>))}</tbody></table>
        </Card>
      </div>
    </>
  );
}
