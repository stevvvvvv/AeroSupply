import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Meter, label } from "../components/ui.jsx";
import { C, PROGRAM_COLOR, STATUS, baseLayout } from "../theme.js";

const COLS = ["Tier 3 · raw materials", "Tier 2 · sub-tier", "Tier 1", "Parts", "Programs"];

function downstream(edges, start) {
  const out = {};
  edges.forEach((e) => { (out[e.source] ||= []).push(e.target); });
  const seen = new Set([start]);
  const stack = [start];
  while (stack.length) for (const n of out[stack.pop()] || []) if (!seen.has(n)) { seen.add(n); stack.push(n); }
  return seen;
}

export default function Network() {
  const { data, error, loading } = useFetch(api.network);
  const { data: spofs } = useFetch(api.spofs);
  const [focus, setFocus] = useState("");
  const nav = useNavigate();
  const reach = useMemo(() => (data && focus ? downstream(data.edges, focus) : null), [data, focus]);
  if (loading || error) return <Loading error={error} />;
  const pos = Object.fromEntries(data.nodes.map((n) => [n.id, n]));
  const on = (id) => !reach || reach.has(id);
  const edgeTrace = (active) => {
    const xs = [], ys = [];
    data.edges.forEach((e) => {
      const isOn = !reach || (reach.has(e.source) && reach.has(e.target));
      if (isOn === active) {
        xs.push(pos[e.source].x, pos[e.target].x, null);
        ys.push(pos[e.source].y, pos[e.target].y, null);
      }
    });
    return { x: xs, y: ys, type: "scatter", mode: "lines", hoverinfo: "skip", showlegend: false,
             line: { color: active ? (reach ? C.edgeHi : C.edge) : C.edgeDim, width: 1 } };
  };
  const sup = data.nodes.filter((n) => n.kind === "supplier");
  const parts = data.nodes.filter((n) => n.kind === "part");
  const progs = data.nodes.filter((n) => n.kind === "program");
  const op = (n) => (on(n.id) ? 1 : 0.15);
  const partColor = (p) => (p.p60 >= 0.5 ? STATUS.CRITICAL.color : p.p60 >= 0.2 ? STATUS.WARNING.color : "#cbd2dd");
  // healthy nodes stay neutral so the network reads by exception
  const nodeColor = (lv) => (lv === "HEALTHY" ? "#aab3c2" : STATUS[lv].color);
  const traces = [
    edgeTrace(false), edgeTrace(true),
    ...Object.keys(STATUS).map((lv) => {
      const n = sup.filter((s) => (s.level || "HEALTHY") === lv);
      return { x: n.map((s) => s.x), y: n.map((s) => s.y), type: "scatter", mode: "markers", name: `Supplier · ${STATUS[lv].label}`,
               customdata: n.map((s) => [s.id, s.label, label(s.commodity), s.impact ?? 0]),
               marker: { size: n.map((s) => 7 + Math.sqrt(s.impact || 0) * 0.8), color: nodeColor(lv), opacity: n.map(op),
                         symbol: n.map((s) => (s.hidden_spof_parts > 0 ? "diamond" : "circle")), line: { color: C.surface, width: 1.5 } },
               hovertemplate: "<b>%{customdata[1]}</b><br>%{customdata[2]}<br>aircraft at risk if lost 90 d: %{customdata[3]}<extra></extra>" };
    }),
    { x: parts.map((p) => p.x), y: parts.map((p) => p.y), type: "scatter", mode: "markers", name: "Part (colour = stock-out risk)",
      customdata: parts.map((p) => [p.id, p.label, Math.round((p.p60 || 0) * 100)]),
      marker: { size: 5, color: parts.map(partColor), opacity: parts.map(op) },
      hovertemplate: "<b>%{customdata[1]}</b><br>P(stock-out 60 d) %{customdata[2]}%<extra></extra>" },
    { x: progs.map((p) => p.x), y: progs.map((p) => p.y), type: "scatter", mode: "markers+text", name: "Program", showlegend: false,
      text: progs.map((p) => p.key), textposition: "middle right", textfont: { color: C.text2 },
      customdata: progs.map((p) => [p.id, p.label]),
      marker: { size: 16, color: progs.map((p) => PROGRAM_COLOR[p.key]), line: { color: C.surface, width: 2 } },
      hovertemplate: "<b>%{customdata[1]}</b><extra></extra>" },
  ];
  return (
    <>
      <div className="page-head">
        <div><div className="crumbs">Network</div><h1>Multi-tier dependency network</h1></div>
        <div className="filters">
          <select value={focus} onChange={(e) => setFocus(e.target.value)}>
            <option value="">Highlight what depends on…</option>
            {[3, 2, 1].map((t) => (
              <optgroup key={t} label={`Tier ${t}`}>
                {sup.filter((s) => s.tier === t).map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
              </optgroup>
            ))}
          </select>
          {focus && <button className="btn" onClick={() => setFocus("")}>Clear</button>}
        </div>
      </div>
      <Card title="Raw materials → sub-tier → tier-1 → parts → aircraft programs"
            subtitle={`${sup.length} suppliers, ${parts.length} parts, ${data.edges.length} dependencies (NetworkX). Diamond = sub-tier supplier behind a hidden single point of failure. Size = aircraft at risk if lost for 90 days. Click a node to open it.`}>
        <Plot height={640}
          onClick={(e) => {
            const id = e.points?.[0]?.customdata?.[0];
            if (!id) return;
            if (id.startsWith("S:")) nav(`/suppliers/${id.slice(2)}`);
            if (id.startsWith("P:")) nav(`/parts/${id.slice(2)}`);
          }}
          data={traces}
          layout={baseLayout({
            margin: { l: 10, r: 10, t: 40, b: 10 }, hovermode: "closest",
            xaxis: { visible: false, range: [-0.3, 4.6] }, yaxis: { visible: false, range: [-0.02, 1.07] },
            legend: { orientation: "h", y: 1.07, x: 0, font: { color: C.text2 } },
            annotations: COLS.map((t, i) => ({ x: i, y: 1.035, yref: "y", text: t, showarrow: false, font: { color: C.muted, size: 12 } })),
          })} />
      </Card>
      <Card title="Hidden single points of failure"
            subtitle="Parts that look dual-sourced at tier 1, but whose sources all depend on the same sub-tier supplier — dual-sourcing does not protect them.">
        {!spofs ? <Loading /> : (
          <table className="table">
            <thead><tr><th>Part</th><th>Programs</th><th>Tier-1 sources</th><th>Sub-tier supplier(s) every source depends on</th><th>P(stock-out 60 d)</th></tr></thead>
            <tbody>{spofs.map((r) => (
              <tr key={r.part_id} onClick={() => nav(`/parts/${r.part_id}`)}>
                <td className="strong">{r.name} <span className="muted small">{r.part_id}</span></td>
                <td>{r.programs}</td>
                <td className="small">{r.source_names.join(" · ")}</td>
                <td><div className="chain-tags">{r.common_upstream.map((u) => (
                  <Link key={u.supplier_id} to={`/suppliers/${u.supplier_id}`} className="tag" onClick={(e) => e.stopPropagation()}>
                    {u.name} · T{u.tier} · {u.country} <span style={{ color: STATUS[u.level]?.color }}>{STATUS[u.level]?.icon}</span></Link>))}</div></td>
                <td><Meter value={r.p_stockout_60} /></td>
              </tr>))}</tbody>
          </table>
        )}
      </Card>
    </>
  );
}
