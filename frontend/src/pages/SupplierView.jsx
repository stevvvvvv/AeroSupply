import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, Meter, Stat, StatusBadge, TierTag, fmt, label, pct } from "../components/ui.jsx";
import { C, STATUS, baseLayout } from "../theme.js";

function anomalyShapes(weeks) {
  const shapes = [];
  let start = null;
  weeks.forEach((w, i) => {
    if (w.anomaly && start === null) start = w.week;
    if (start !== null && (!w.anomaly || i === weeks.length - 1)) {
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: start, x1: w.week, y0: 0, y1: 1,
                    fillcolor: "rgba(208,59,59,0.16)", line: { width: 0 } });
      start = null;
    }
  });
  return shapes;
}

function Drivers({ drivers }) {
  const max = Math.max(0.05, ...drivers.map((d) => Math.abs(d.contribution)));
  return (
    <div>
      {drivers.map((d) => {
        const w = (Math.abs(d.contribution) / max) * 50;
        const up = d.contribution > 0;
        return (
          <div className="driver" key={d.feature}>
            <span>{d.label} <span className="muted small mono">{fmt(d.value, 2)} (fleet {fmt(d.reference, 2)})</span></span>
            <div className="dbar"><div className="mid" />
              <div className="fill" style={{ left: up ? "50%" : `${50 - w}%`, width: `${w}%`, background: up ? C.neg : C.pos }} /></div>
            <span className="mono small" style={{ textAlign: "right" }}>{up ? "+" : "−"}{Math.abs(d.contribution * 100).toFixed(0)} pts</span>
          </div>
        );
      })}
      <p className="note" style={{ marginTop: 8 }}>Change in probability if this factor were at the fleet median (red raises risk, blue lowers it).</p>
    </div>
  );
}

export default function SupplierView() {
  const { id } = useParams();
  const nav = useNavigate();
  const { data: s, error, loading } = useFetch(() => api.supplier(id), [id]);
  const { data: weekly } = useFetch(() => api.supplierWeekly(id), [id]);
  if (loading || error) return <Loading error={error} />;
  const t1 = s.tier === 1;
  const st = STATUS[s.risk_level];
  return (
    <>
      <div className="page-head">
        <div>
          <div className="crumbs"><Link to="/suppliers">Suppliers</Link> / {s.supplier_id}</div>
          <h1>{s.name} <TierTag tier={s.tier} /> <StatusBadge status={s.risk_level} /></h1>
          <p className="muted">{s.commodity_label} · {s.country_name} · {s.size} · {fmt(s.employees)} employees{s.en9100 ? " · EN 9100" : ""}</p>
        </div>
        <button className="btn primary" onClick={() => nav(`/simulate?supplier=${s.supplier_id}`)}>Simulate loss of this supplier</button>
      </div>

      <div className="stats">
        {t1 ? <Stat label="P(severe failure, 90 d)" value={pct(s.risk_prob)} accent={st.color} sub="OTD < 60 % or mean delay > 30 d" />
            : <Stat label="Screening score" value={fmt(s.screening_score)} accent={st.color} sub="rule-based, 0–100 (no delivery data)" />}
        {t1 && <Stat label="On-time delivery 90 d" value={pct(s.otd_90d)} sub={`12 months: ${pct(s.otd_12m)}`} />}
        {t1 && <Stat label="Open POs overdue" value={`${s.open_pos_overdue} / ${s.open_pos}`} sub={`mean delay 90 d: ${fmt(s.delay_90d, 1)} d`} />}
        {!t1 && <Stat label="Tier-1 customers" value={s.tier1_customers} sub="direct customers in the network" />}
        <Stat label="Parts depending on it" value={s.parts_dependent} sub={`${s.parts_fully_dependent} fully · ${s.hidden_spof_parts} hidden SPOF`} />
        <Stat label="Aircraft at risk if lost 90 d" value={fmt(s.impact_aircraft_90d)} sub={`≈ €${fmt(s.impact_value_meur_90d / 1000, 1)} bn of deliveries (list price)`} />
        <Stat label="Financial score" value={fmt(s.fin_score)} sub="latest published quarter" />
      </div>

      {t1 && weekly && weekly.length > 0 && (
        <Card title="Delivery delay vs commodity peers"
              subtitle="Mean delay of deliveries over a rolling 5-week window. Shaded weeks: anomaly episodes (relative to its own baseline, after removing what peers experience).">
          <Plot height={300}
            data={[
              { x: weekly.map((w) => w.week), y: weekly.map((w) => w.c_delay), name: "Commodity peers", type: "scatter", mode: "lines",
                line: { color: C.muted, width: 1.5 }, hovertemplate: "%{x}<br>peers %{y:.1f} d<extra></extra>" },
              { x: weekly.map((w) => w.week), y: weekly.map((w) => w.delay), name: s.name, type: "scatter", mode: "lines",
                line: { color: C.series[0], width: 2 }, connectgaps: false, hovertemplate: "%{x}<br>%{y:.1f} d<extra></extra>" },
            ]}
            layout={baseLayout({ shapes: anomalyShapes(weekly), yaxis: { title: { text: "Delay (days)" } } })} />
        </Card>
      )}

      <div className="grid-2">
        <Card title={t1 ? "Why this score" : "Screening inputs"}
              subtitle={t1 ? "Gradient-boosting model, per-supplier attribution." : "Sub-tier suppliers have no delivery history at the OEM."}>
          {t1 ? <Drivers drivers={s.drivers || []} /> : (
            <dl className="kv">{(s.drivers || []).flatMap((d) => [<dt key={`${d.label}-k`}>{d.label}</dt>, <dd key={`${d.label}-v`}>{fmt(d.value, 2)}</dd>])}</dl>
          )}
        </Card>
        <Card title="Financial score" subtitle="Quarterly credit-style score (published 45 days after quarter end).">
          <Plot height={240}
            data={[{ x: s.financials.map((f) => f.period_end), y: s.financials.map((f) => f.financial_score), type: "scatter",
                     mode: "lines+markers", line: { color: C.series[0], width: 2 }, marker: { size: 8, line: { color: C.surface, width: 2 } },
                     hovertemplate: "%{x}<br>%{y:.0f}<extra></extra>" }]}
            layout={baseLayout({ showlegend: false, yaxis: { range: [0, 100] } })} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Position in the supply network">
          <div className="list">
            <div>
              <div className="muted small">Supplied by (sub-tier)</div>
              {s.upstream.length === 0 ? <span className="muted">— raw material / no mapped sub-tier</span> : (
                <div className="chain-tags">{s.upstream.map((u) => (
                  <Link key={u.upstream_id} to={`/suppliers/${u.upstream_id}`} className="tag">
                    {u.name} · {label(u.input_commodity)} · {Math.round(u.share * 100)}% <span style={{ color: STATUS[u.level]?.color }}>{STATUS[u.level]?.icon}</span>
                  </Link>))}</div>)}
            </div>
            <div>
              <div className="muted small">Customers</div>
              {s.customers.length === 0 ? <span className="muted">OEM final assembly lines</span> : (
                <div className="chain-tags">{s.customers.map((u) => (
                  <Link key={u.customer_id} to={`/suppliers/${u.customer_id}`} className="tag">
                    {u.name} <span style={{ color: STATUS[u.level]?.color }}>{STATUS[u.level]?.icon}</span>
                  </Link>))}</div>)}
            </div>
            {s.hidden_spof_part_ids.length > 0 && (
              <p className="callout small">Hidden single point of failure for {s.hidden_spof_part_ids.length} part(s): they are
                dual-sourced at tier 1, but every source depends on {s.name}.</p>
            )}
          </div>
        </Card>
        <Card title="Signals" subtitle="Unstructured events from the document store.">
          <div className="list" style={{ maxHeight: 320, overflow: "auto" }}>
            {s.signals.length === 0 && <span className="muted">No signal.</span>}
            {s.signals.map((g) => (
              <div className="signal" key={g.signal_id}>
                <span className="muted small mono">{g.date}</span>
                <div>
                  <div className="title" style={{ color: g.polarity < 0 ? C.text : C.text2 }}>{g.polarity < 0 ? "▲ " : ""}{g.title}</div>
                  <div className="small muted">{g.source} · {g.body}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title={t1 ? "Parts supplied" : "Parts depending on this supplier"}
            subtitle={t1 ? "Share = this supplier's share of the part's volume." : "Dependence = share of the part's supply lost if this supplier stopped (through the tier-1 sources)."}>
        <table className="table">
          <thead><tr><th>Part</th><th>Programs</th><th className="num">{t1 ? "Share" : "Dependence"}</th>
            {t1 && <th className="num">Days of cover</th>}<th>P(stock-out 60 d)</th></tr></thead>
          <tbody>
            {(t1 ? s.parts_supplied : s.dependent_parts).slice(0, 40).map((p) => (
              <tr key={p.part_id} onClick={() => nav(`/parts/${p.part_id}`)}>
                <td><span className="strong">{p.name}</span> <span className="muted small">{p.part_id}</span></td>
                <td>{p.programs}</td>
                <td className="num">{pct(t1 ? p.share : p.dependence)}</td>
                {t1 && <td className="num">{fmt(p.days_of_cover)}</td>}
                <td><Meter value={p.p_stockout_60} /></td>
              </tr>
            ))}
          </tbody>
        </table>
        {!t1 && s.dependent_parts.length > 40 && <p className="note" style={{ marginTop: 8 }}>Top 40 of {s.dependent_parts.length} parts, by dependence.</p>}
      </Card>
    </>
  );
}
