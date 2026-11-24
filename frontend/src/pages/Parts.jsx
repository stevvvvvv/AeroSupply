import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import { Card, Loading, Meter, Stat, fmt, label } from "../components/ui.jsx";

export default function Parts() {
  const [params] = useSearchParams();
  const [program, setProgram] = useState(params.get("program") || "");
  const [onlyRisk, setOnlyRisk] = useState(true);
  const { data, error, loading } = useFetch(() => api.parts(), []);
  const nav = useNavigate();
  const rows = useMemo(() => (data || []).filter((p) => (!program || p.programs.includes(program)) && (!onlyRisk || p.p_stockout_90 >= 0.05)),
    [data, program, onlyRisk]);
  if (loading || error) return <Loading error={error} />;
  const n = (f) => data.filter(f).length;
  return (
    <>
      <div className="page-head"><div><div className="crumbs">Parts</div><h1>Parts & stock-out forecast</h1></div></div>
      <div className="stats">
        <Stat label="Parts already short" value={n((p) => p.in_shortage_now)} sub="missing on the assembly line today" />
        <Stat label="P(stock-out ≤ 30 d) ≥ 50 %" value={n((p) => !p.in_shortage_now && p.p_stockout_30 >= 0.5)} />
        <Stat label="P(stock-out ≤ 60 d) ≥ 50 %" value={n((p) => !p.in_shortage_now && p.p_stockout_60 >= 0.5)} />
        <Stat label="Flagged by the ERP (MRP) projection" value={n((p) => !p.in_shortage_now && p.mrp_stockout_60)}
              sub="same parts, POs assumed on time" />
      </div>
      <Card title={`${rows.length} parts`}
            subtitle="Monte-Carlo simulation of the open order book (300 scenarios, arrival dates sampled from the lead-time model), probabilities recalibrated on the backtest."
            actions={
              <div className="filters">
                <select value={program} onChange={(e) => setProgram(e.target.value)}>
                  <option value="">All programs</option>{["NB-200", "WB-350", "RJ-72", "BJ-6"].map((p) => <option key={p}>{p}</option>)}
                </select>
                <label className="check"><input type="checkbox" checked={onlyRisk} onChange={(e) => setOnlyRisk(e.target.checked)} /> P(90 d) ≥ 5 % only</label>
              </div>
            }>
        <table className="table">
          <thead><tr><th>Part</th><th>Commodity</th><th>Programs</th><th className="num">Days of cover</th>
            <th className="num">Open POs (overdue)</th><th>P(≤ 30 d)</th><th>P(≤ 60 d)</th><th>P(≤ 90 d)</th><th className="num">ERP flag</th></tr></thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.part_id} onClick={() => nav(`/parts/${p.part_id}`)}>
                <td><span className="strong">{p.name}</span> <span className="muted small">{p.part_id}</span></td>
                <td>{label(p.commodity)}</td>
                <td>{p.programs}</td>
                <td className="num">{p.in_shortage_now ? <span style={{ color: "#d03b3b", fontWeight: 600 }}>■ short</span> : fmt(p.days_of_cover)}</td>
                <td className="num">{p.open_pos} ({p.overdue_pos})</td>
                <td><Meter value={p.p_stockout_30} /></td>
                <td><Meter value={p.p_stockout_60} /></td>
                <td><Meter value={p.p_stockout_90} /></td>
                <td className="num">{p.mrp_stockout_60 ? "yes" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}
