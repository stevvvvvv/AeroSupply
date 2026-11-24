import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Meter, Stat, StatusBadge, fmt, label, pct } from "../components/ui.jsx";
import { C, PROGRAM_COLOR, STATUS, baseLayout } from "../theme.js";

const CRISES = [
  ["2020-03", "2020-12", "COVID-19"],
  ["2021-06", "2023-06", "Electronics shortage"],
  ["2022-06", "2024-03", "Castings & forgings crunch"],
];

function riskText(c) {
  if (!c.critical && !c.warning) return "no supplier flagged";
  return [c.critical && `${c.critical} critical`, c.warning && `${c.warning} warning`].filter(Boolean).join(" · ");
}

export default function Overview() {
  const { data, error, loading } = useFetch(api.overview);
  const nav = useNavigate();
  if (loading || error) return <Loading error={error} />;
  const k = data.kpis;
  const lv = k.tier1_by_level;
  const otdDelta = k.otd_90d - k.otd_prev_90d;
  const ch = Object.fromEntries(data.chain.map((c) => [c.stage, c]));
  const atRisk = (lv.CRITICAL || 0) + (lv.WARNING || 0);
  const stages = [
    { key: "tier3", name: "Raw materials", to: "/suppliers?tier=3", c: ch.tier3, risk: (c) => riskText(c) },
    { key: "tier2", name: "Sub-tier suppliers", to: "/suppliers?tier=2", c: ch.tier2, risk: (c) => riskText(c) },
    { key: "tier1", name: "Tier-1 suppliers", to: "/suppliers?tier=1", c: ch.tier1, risk: (c) => riskText(c) },
    { key: "parts", name: "Parts", to: "/parts", c: ch.parts, hot: ch.parts.short_now > 0,
      risk: (c) => `${c.short_now} missing now · ${c.at_risk} likely short in 60 days` },
    { key: "programs", name: "Aircraft programs", to: "/simulate", c: ch.programs, hot: ch.programs.at_risk > 0,
      risk: (c) => `${c.at_risk} of ${c.n} waiting on a missing part` },
  ];
  return (
    <>
      <section className="hero">
        <h1>{atRisk} of {k.tier1} tier-1 suppliers need attention, and {k.parts_short_now} parts are already missing on the line.</h1>
        <p className="lede">
          On-time delivery fell to {pct(k.otd_90d)} over the last 90 days ({otdDelta >= 0 ? "+" : "−"}{Math.abs(Math.round(otdDelta * 100))} points).
          Follow the chain from raw materials to aircraft to see where the risk sits.
        </p>
        <div className="chain">
          {stages.map((st) => (
            <Link key={st.key} to={st.to} className={`stage${st.hot ?? (st.c.critical > 0) ? " hot" : ""}`}>
              <div className="node"><i /></div>
              <div className="count">{st.c.n}</div>
              <div className="name">{st.name}</div>
              <div className="risk">{st.risk(st.c)}</div>
            </Link>
          ))}
        </div>
      </section>

      <div className="stats">
        <Stat label="Open orders overdue" value={fmt(k.open_pos_overdue)} sub={`of ${fmt(k.open_pos)} open purchase orders`} />
        <Stat label="Hidden single points of failure" value={k.hidden_spof_parts} sub="dual-sourced parts that share one sub-tier supplier" />
        <Stat label="Ongoing delivery anomalies" value={k.ongoing_anomalies} sub="tier-1 suppliers, compared with their peers" />
        <Stat label="Mean delay, last 90 days" value={`${fmt(k.delay_90d, 1)} d`} sub="all tier-1 deliveries" />
      </div>

      <div className="grid-2">
        <Card title="On-time delivery, all tier-1 suppliers"
              subtitle="Monthly share of deliveries within 3 days of the promised date. Shaded: market-wide crises.">
          <Plot height={280}
            data={[{ x: data.otd_monthly.map((d) => d.month), y: data.otd_monthly.map((d) => d.otd), type: "scatter",
                     mode: "lines", line: { color: C.series[0], width: 2 }, name: "OTD",
                     hovertemplate: "%{x}<br>OTD %{y:.0%}<extra></extra>" }]}
            layout={baseLayout({
              showlegend: false, yaxis: { tickformat: ".0%", range: [0, 1] },
              shapes: CRISES.map(([a, b]) => ({ type: "rect", xref: "x", yref: "paper", x0: a, x1: b, y0: 0, y1: 1,
                                                fillcolor: "rgba(15,27,45,0.05)", line: { width: 0 } })),
              annotations: CRISES.map(([a, , t], i) => ({ x: a, y: 1 - i * 0.07, yref: "paper", text: t, showarrow: false,
                                                         xanchor: "left", font: { size: 11, color: C.muted } })),
            })} />
        </Card>
        <Card title="Programs" subtitle="Parts with P(stock-out within 60 days) ≥ 50 %, and parts already missing on the line.">
          <table className="table">
            <thead><tr><th>Program</th><th className="num">Parts</th><th className="num">At risk (60 d)</th><th className="num">Short now</th></tr></thead>
            <tbody>
              {data.programs.map((p) => (
                <tr key={p.program} onClick={() => nav(`/parts?program=${p.program}`)}>
                  <td><span className="dot" style={{ background: PROGRAM_COLOR[p.program], marginRight: 8 }} />{p.program}</td>
                  <td className="num">{p.parts}</td>
                  <td className="num">{p.parts_p60_over_50}</td>
                  <td className="num">{p.parts_short_now}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.tense_commodities.length > 0 && (
            <p className="callout small" style={{ marginTop: 14 }}>
              Market-wide tension this week: {data.tense_commodities.map(label).join(", ")} — delays well above their
              1-year baseline across all suppliers of the commodity.
            </p>
          )}
        </Card>
      </div>

      <Card title="Highest-exposure tier-1 suppliers"
            subtitle="Exposure = P(severe delivery failure in 90 days) × aircraft at risk if the supplier stopped for 90 days.">
        <table className="table">
          <thead><tr><th>Supplier</th><th>Commodity</th><th>Status</th><th>P(failure 90 d)</th><th className="num">OTD 90 d</th>
            <th className="num">Aircraft at risk if lost</th><th className="num">Exposure</th></tr></thead>
          <tbody>
            {data.top_suppliers.map((s) => (
              <tr key={s.supplier_id} onClick={() => nav(`/suppliers/${s.supplier_id}`)}>
                <td><span className="strong">{s.name}</span> <span className="muted small">{s.supplier_id} · {s.country}</span></td>
                <td>{label(s.commodity)}</td>
                <td><StatusBadge status={s.risk_level} /></td>
                <td><Meter value={s.risk_prob} /></td>
                <td className="num">{pct(s.otd_90d)}</td>
                <td className="num">{fmt(s.impact_aircraft_90d)}</td>
                <td className="num">{fmt(s.exposure, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <div className="grid-2">
        <Card title="Ongoing delivery anomalies" subtitle="Isolation Forest on peer-adjusted delay, OTD and reject rate; ≥ 2 consecutive weeks.">
          {data.ongoing_episodes.length === 0 ? <p className="muted">None.</p> : (
            <table className="table">
              <thead><tr><th>Supplier</th><th>Since</th><th className="num">Weeks</th><th className="num">Peak delay</th><th className="num">Min OTD</th></tr></thead>
              <tbody>
                {data.ongoing_episodes.map((e) => (
                  <tr key={e.supplier_id} onClick={() => nav(`/suppliers/${e.supplier_id}`)}>
                    <td className="strong">{e.name}</td><td>{e.start}</td><td className="num">{e.weeks}</td>
                    <td className="num">{fmt(e.peak_delay, 1)} d</td><td className="num">{pct(e.min_otd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <Card title="Recent negative signals" subtitle="News, audits, quality notifications and supplier surveys (MongoDB).">
          <div className="list">
            {data.recent_signals.map((s) => (
              <div className="signal" key={s.signal_id}>
                <span className="muted small mono">{s.date}</span>
                <div>
                  <div className="title">{s.title}</div>
                  <div className="small muted"><Link to={`/suppliers/${s.supplier_id}`}>{s.supplier_name}</Link> · {s.source}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </>
  );
}
