import { api } from "../api.js";
import { useFetch } from "../useFetch.js";
import Plot from "../components/Plot.jsx";
import { Card, Loading, fmt, pct } from "../components/ui.jsx";
import { C, baseLayout } from "../theme.js";

function Calib({ rows, xTitle }) {
  return (
    <Plot height={260}
      data={[
        { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", line: { color: C.axis, dash: "dot", width: 1 }, name: "Perfect", hoverinfo: "skip" },
        { x: rows.map((r) => r.predicted), y: rows.map((r) => r.observed), type: "scatter", mode: "lines+markers", name: "Model",
          line: { color: C.series[0], width: 2 }, marker: { size: rows.map((r) => 7 + Math.sqrt(r.n) / 2), line: { color: C.surface, width: 2 } },
          customdata: rows.map((r) => r.n), hovertemplate: "predicted %{x:.0%}<br>observed %{y:.0%}<br>n = %{customdata}<extra></extra>" },
      ]}
      layout={baseLayout({ xaxis: { title: { text: xTitle }, tickformat: ".0%", range: [0, 1] }, yaxis: { title: { text: "Observed rate" }, tickformat: ".0%", range: [0, 1] } })} />
  );
}

export default function Model() {
  const { data, error, loading } = useFetch(api.model);
  if (loading || error) return <Loading error={error} />;
  const m = data.metrics;
  const lt = m.lead_time, rk = m.risk, so = m.stockout, an = m.anomaly;
  const H = Object.entries(lt.mae_by_horizon);
  return (
    <>
      <div className="page-head"><div><div className="crumbs">Transparency</div><h1>Model card</h1>
        <p className="muted">Trained {m.trained_at?.slice(0, 10)} · data as of {m.as_of} · every metric on a time-ordered hold-out (no shuffling)</p></div></div>

      <Card title="1 · Delay of open purchase orders" subtitle={`Quantile gradient boosting (P10/P50/P90) on the residual over the supplier's recent mean delay, conformal tails. Test: ${fmt(lt.n_test_snapshots)} open-PO snapshots, cutoffs ${lt.test_cutoffs.join(" → ")}.`}>
        <div className="grid-2">
          <table className="table metric-table"><thead><tr><th>Method</th><th className="num">MAE (days)</th></tr></thead><tbody>
            <tr><td>ERP: arrives on promised date</td><td className="num">{fmt(lt.mae_promised_date, 2)}</td></tr>
            <tr><td>Supplier's 90-day mean delay</td><td className="num">{fmt(lt.mae_supplier_trailing_mean, 2)}</td></tr>
            <tr><td className="best">Model (P50)</td><td className="num best">{fmt(lt.mae_model_p50, 2)}</td></tr>
            <tr><td>80 % interval coverage (target 80 %)</td><td className="num">{pct(lt.interval_80_coverage)}</td></tr>
          </tbody></table>
          <Plot height={220}
            data={["promised", "trailing", "model"].map((k, i) => ({ x: H.map(([h]) => h), y: H.map(([, v]) => v[k]), type: "bar",
              name: { promised: "ERP promised date", trailing: "Supplier mean", model: "Model" }[k], marker: { color: C.series[i] } }))}
            layout={baseLayout({ barmode: "group", bargap: 0.3, bargroupgap: 0.1, yaxis: { title: { text: "MAE (days)" } } })} />
        </div>
      </Card>

      <Card title="2 · Anomaly detection in delivery histories" subtitle="Scored against the generator's hidden disruption log (supplier-specific disruptions with ≥ 20 days of delay impact).">
        <table className="table metric-table"><thead><tr><th>Detector</th><th className="num">Disruptions detected</th><th className="num">Median lag</th>
          <th className="num">Episode precision</th><th className="num">Weeks flagged</th></tr></thead><tbody>
          {[["Isolation Forest (served)", an.isolation_forest], ["Rule |z| > 3", an.z_score_rule]].map(([n, a]) => (
            <tr key={n}><td>{n}</td><td className="num">{a.detected} / {a.disruptions_evaluated} ({pct(a.recall)})</td>
              <td className="num">{fmt(a.median_detection_lag_days)} d</td><td className="num">{pct(a.episode_precision)}</td><td className="num">{pct(a.flagged_week_rate)}</td></tr>))}
        </tbody></table>
        <p className="note" style={{ marginTop: 10 }}>Common-cause layer (graph): {an.isolation_forest.upstream_pointed_by_common_cause} of {an.isolation_forest.upstream_disruptions_evaluated} sub-tier
          disruptions pointed at through simultaneous anomalies of their tier-1 customers; {pct(an.isolation_forest.common_cause_precision)} of its {an.isolation_forest.common_cause_alerts} alerts had a real upstream cause — it is a lead to investigate, not a verdict.</p>
      </Card>

      <Card title="3 · Supplier risk score" subtitle={`P(on-time delivery < 60 % or mean delay > 30 d over the next 90 days). Served: ${rk.served_model.replaceAll("_", " ")} (chosen on validation), refit quarterly on a 36-month window; walk-forward test ${rk.test_period.join(" → ")}.`}>
        <div className="grid-2">
          <table className="table metric-table"><thead><tr><th></th><th className="num">Model</th><th className="num">Baseline: last-quarter OTD</th></tr></thead><tbody>
            <tr><td>ROC-AUC</td><td className="num best">{fmt(rk.test_walk_forward.roc_auc, 3)}</td><td className="num">{fmt(rk.baseline_last_quarter_otd.roc_auc, 3)}</td></tr>
            <tr><td>PR-AUC (positive rate {pct(rk.test_walk_forward.positive_rate)})</td><td className="num best">{fmt(rk.test_walk_forward.pr_auc, 3)}</td><td className="num">{fmt(rk.baseline_last_quarter_otd.pr_auc, 3)}</td></tr>
            <tr><td>Early warning: suppliers healthy today (AUC)</td><td className="num best">{fmt(rk.early_warning.model_roc_auc, 3)}</td><td className="num">{fmt(rk.early_warning.baseline_roc_auc, 3)}</td></tr>
            <tr><td>Precision of the monthly top-10</td><td className="num best">{pct(rk.top10_precision.model)}</td><td className="num">{pct(rk.top10_precision.baseline)}</td></tr>
            <tr><td>Brier score</td><td className="num best">{fmt(rk.test_walk_forward.brier, 3)}</td><td className="num">—</td></tr>
          </tbody></table>
          <Calib rows={rk.calibration} xTitle="Predicted probability" />
        </div>
        <p className="note">Most influential inputs (permutation, ROC-AUC drop): {rk.global_drivers.slice(0, 6).map((d) => d.label).join(" · ")}.</p>
      </Card>

      <Card title="4 · Stock-out forecast" subtitle={`Monte-Carlo backtest at monthly cutoffs ${so.cutoffs.join(" → ")}; isotonic recalibration fitted on ${so.calibration_fit.join(" → ")}, scored on ${so.holdout.join(" → ")}.`}>
        <div className="grid-2">
          <table className="table metric-table"><thead><tr><th>Horizon</th><th className="num">Positive rate</th><th className="num">ROC-AUC</th>
            <th className="num">Recall @ P ≥ 0.2</th><th className="num">Precision @ 0.2</th><th className="num">ERP recall</th></tr></thead><tbody>
            {["h30", "h60", "h90"].map((h) => (<tr key={h}><td>{h.slice(1)} days</td><td className="num">{pct(so[h].positive_rate)}</td>
              <td className="num">{fmt(so[h].roc_auc, 3)}</td><td className="num best">{pct(so[h]["mc_recall_at_0.2"])}</td>
              <td className="num">{pct(so[h]["mc_precision_at_0.2"])}</td><td className="num">{pct(so[h].mrp_recall)}</td></tr>))}
          </tbody></table>
          <Calib rows={so.calibration_60_holdout} xTitle="Predicted P(stock-out ≤ 60 d)" />
        </div>
        <p className="note">The ERP projection assumes every open PO arrives on its promised date: it is rarely wrong when it raises a flag, but it misses almost every stock-out caused by late deliveries.</p>
      </Card>

      {data.last_run && (
        <Card title="Last pipeline run" subtitle={`${data.last_run.run_id} — same steps as the Airflow DAG aerosupply_daily_risk`}>
          <table className="table"><thead><tr><th>Step</th><th className="num">Seconds</th><th>Result</th></tr></thead>
            <tbody>{Object.entries(data.last_run.steps || {}).map(([k, v]) => (
              <tr key={k}><td className="mono">{k}</td><td className="num">{fmt(v.seconds, 1)}</td><td className="small mono" style={{ wordBreak: "break-all" }}>{JSON.stringify(v.result).slice(0, 160)}</td></tr>))}</tbody></table>
          {data.last_run.data_quality && (
            <p className="note" style={{ marginTop: 10 }}>Data-quality checks at ingestion: {data.last_run.data_quality.filter((c) => c.passed).length} / {data.last_run.data_quality.length} passed
              ({data.last_run.data_quality.map((c) => c.check).join(", ")}).</p>)}
        </Card>
      )}
    </>
  );
}
