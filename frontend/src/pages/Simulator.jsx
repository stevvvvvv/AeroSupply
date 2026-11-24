import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Stat, StatusBadge, fmt } from "../components/ui.jsx";
import { C, PROGRAM_COLOR, baseLayout } from "../theme.js";

export default function Simulator() {
  const [params] = useSearchParams();
  const { data: sups, error, loading } = useFetch(() => api.suppliers());
  const saved = useFetch(api.simulations);
  const [sid, setSid] = useState(params.get("supplier") || "");
  const [duration, setDuration] = useState(90);
  const [loss, setLoss] = useState(100);
  const [start, setStart] = useState(0);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const nav = useNavigate();

  useEffect(() => { if (!sid && sups?.length) setSid(sups.find((s) => s.tier === 2)?.supplier_id || sups[0].supplier_id); }, [sups]);
  const run = (save = false) => {
    setBusy(true); setErr(null);
    api.simulate({ supplier_id: sid, duration_days: duration, capacity_loss: loss / 100, start_in_days: start, save })
      .then((r) => { setRes(r); if (save) saved.reload(); }).catch(setErr).finally(() => setBusy(false));
  };
  useEffect(() => { if (sid) run(); }, [sid]);
  if (loading || error) return <Loading error={error} />;
  const sel = sups.find((s) => s.supplier_id === sid);
  return (
    <>
      <div className="page-head"><div><div className="crumbs">What-if</div><h1>Disruption simulator</h1></div></div>
      <Card title="Scenario" subtitle="Lose part of a supplier's output for a period. The shortage propagates through the network (a sub-tier failure hits tier-1 after 30–60 days of downstream buffer), alternative qualified sources ramp up after 60 days, and the result is compared with the baseline stock forecast.">
        <div className="controls">
          <label>Supplier
            <select value={sid} onChange={(e) => setSid(e.target.value)}>
              {[1, 2, 3].map((t) => (
                <optgroup key={t} label={`Tier ${t}`}>
                  {sups.filter((s) => s.tier === t).map((s) => <option key={s.supplier_id} value={s.supplier_id}>{s.name} — {s.commodity.replaceAll("_", " ")}</option>)}
                </optgroup>))}
            </select>
          </label>
          <label>Duration: {duration} days<input type="range" min="7" max="180" step="1" value={duration} onChange={(e) => setDuration(+e.target.value)} /></label>
          <label>Capacity lost: {loss} %<input type="range" min="10" max="100" step="5" value={loss} onChange={(e) => setLoss(+e.target.value)} /></label>
          <label>Starts in: {start} days<input type="range" min="0" max="60" step="1" value={start} onChange={(e) => setStart(+e.target.value)} /></label>
          <div className="filters"><button className="btn primary" disabled={busy} onClick={() => run(false)}>{busy ? "Running…" : "Run"}</button>
            <button className="btn" disabled={busy || !res} onClick={() => run(true)}>Save scenario</button></div>
        </div>
        {sel && <p className="small muted" style={{ marginTop: 12 }}>
          <Link to={`/suppliers/${sel.supplier_id}`}>{sel.name}</Link> · tier {sel.tier} · <StatusBadge status={sel.risk_level} /> · {sel.parts_dependent} parts depend on it
          {sel.hidden_spof_parts ? ` · hidden SPOF for ${sel.hidden_spof_parts} parts` : ""}</p>}
        {err && <div className="state error">{String(err.message)}</div>}
      </Card>
      {res && (
        <>
          <div className="stats">
            <Stat label="Aircraft deliveries at risk" value={fmt(res.aircraft_at_risk)} sub="peak aircraft that cannot be completed" />
            <Stat label="Value at list price" value={`€${fmt(res.value_at_risk_meur / 1000, 2)} bn`} sub="illustrative list prices" />
            <Stat label="Parts going short" value={`${res.parts_short} / ${res.parts_affected}`} sub="incremental vs baseline forecast" />
            <Stat label="First impact on the line" value={(() => { const d = res.programs.map((p) => p.first_impact_day).filter((x) => x != null); return d.length ? `day ${Math.min(...d)}` : "none"; })()}
                  sub={`impact lag ${res.impact_lag_days} d (downstream buffer)`} />
          </div>
          <div className="grid-2">
            <Card title="Aircraft that cannot be completed, by program" subtitle="Driven by the scarcest part of each program (an aircraft needs every part).">
              <Plot height={300}
                data={Object.entries(res.timeline).map(([prog, ys]) => ({
                  x: res.dates, y: ys, name: prog, type: "scatter", mode: "lines", line: { color: PROGRAM_COLOR[prog], width: 2, shape: "hv" },
                  hovertemplate: `${prog}<br>%{x}<br>%{y} aircraft<extra></extra>` }))}
                layout={baseLayout({ yaxis: { title: { text: "Aircraft" }, rangemode: "tozero" } })} />
            </Card>
            <Card title="Programs">
              <table className="table"><thead><tr><th>Program</th><th className="num">Aircraft at risk</th><th className="num">€m (list)</th><th>Bottleneck part</th><th className="num">First impact</th></tr></thead>
                <tbody>{res.programs.map((p) => (
                  <tr key={p.program}><td><span className="dot" style={{ background: PROGRAM_COLOR[p.program], marginRight: 8 }} />{p.program}</td>
                    <td className="num">{fmt(p.aircraft_at_risk)}</td><td className="num">{fmt(p.value_at_risk_meur)}</td>
                    <td>{p.bottleneck_part ? <Link to={`/parts/${p.bottleneck_part}`}>{p.bottleneck_name}</Link> : "—"}</td>
                    <td className="num">{p.first_impact_day != null ? `day ${p.first_impact_day}` : "—"}</td></tr>))}</tbody></table>
            </Card>
          </div>
          <Card title="Affected parts" subtitle="Dependence = share of the part's supply lost; surge = what unaffected alternative sources can absorb after 60 days.">
            <table className="table"><thead><tr><th>Part</th><th className="num">Supply lost</th><th className="num">Surge capacity</th><th className="num">Units short (peak)</th><th className="num">First short day</th></tr></thead>
              <tbody>{res.parts.slice(0, 40).map((p) => (
                <tr key={p.part_id} onClick={() => nav(`/parts/${p.part_id}`)}><td><span className="strong">{p.name}</span> <span className="muted small">{p.part_id}</span></td>
                  <td className="num">{Math.round(p.supply_lost_share * 100)} %</td><td className="num">{Math.round(p.surge_share * 100)} %</td>
                  <td className="num">{fmt(p.units_short_max)}</td><td className="num">{p.first_short_day ?? "—"}</td></tr>))}</tbody></table>
          </Card>
        </>
      )}
      {saved.data?.length > 0 && (
        <Card title="Saved scenarios" subtitle="Stored as documents in MongoDB.">
          <table className="table"><thead><tr><th>Scenario</th><th>Saved</th><th className="num">Aircraft at risk</th><th className="num">Parts short</th></tr></thead>
            <tbody>{saved.data.map((s, i) => (
              <tr key={i} onClick={() => { setSid(s.supplier_id); }}><td>{s.label}</td><td className="small muted">{s.created_at}</td>
                <td className="num">{fmt(s.aircraft_at_risk)}</td><td className="num">{s.parts_short}</td></tr>))}</tbody></table>
        </Card>
      )}
    </>
  );
}
