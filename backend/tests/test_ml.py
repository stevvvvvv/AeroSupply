import numpy as np
import pandas as pd

from aerosupply.config import AS_OF
from aerosupply.ml.features import FeatureBuilder, Trailing
from aerosupply.ml.leadtime import sample_delays
from aerosupply.ml.optimize import allocate


def test_trailing_window_is_causal():
    df = pd.DataFrame({"k": ["a"] * 3, "t": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"]), "v": [1.0, 2.0, 4.0]})
    tr = Trailing(df, "k", "t", ["v"])
    s = tr.sums(["a", "a"], pd.to_datetime(["2024-01-10", "2024-01-21"]), 15)
    assert s[0].tolist() == [1, 1]          # the row dated on the query date is excluded
    assert s[1].tolist() == [2, 6]


def test_features_do_not_change_when_future_is_removed(D):
    """Features at date t computed on the full data == computed on data truncated at t."""
    t = pd.Timestamp("2024-03-01")
    full = FeatureBuilder(D)
    cut = dict(D)
    po = D["purchase_orders"].copy()
    po = po[po.order_date < t]
    po.loc[po.delivered_date >= t, ["delivered_date", "qty_accepted", "qty_rejected"]] = np.nan
    cut["purchase_orders"] = po
    cut["signals"] = D["signals"][D["signals"].date < t]
    cut["financials"] = D["financials"][D["financials"].published_date < t]
    past = FeatureBuilder(cut)
    sids = D["suppliers"].query("tier == 1").supplier_id.to_numpy()[:15]
    a = full.supplier_features(sids, [t] * len(sids))
    b = past.supplier_features(sids, [t] * len(sids))
    pd.testing.assert_frame_equal(a, b)


def test_sampled_delays_follow_quantiles():
    P = np.array([[0.0, 5.0, 20.0]])
    s = sample_delays(P, 20000, np.random.default_rng(0))[:, 0]
    assert abs(np.quantile(s, 0.5) - 5) < 0.5
    assert abs(np.quantile(s, 0.9) - 20) < 1.0


def test_network_loss_leontief(pipeline):
    from aerosupply.graph.network import Network
    net = Network(pipeline.ctx().D)
    L = net.loss
    assert ((L >= -1e-9) & (L <= 1 + 1e-9)).all().all()
    src = pipeline.ctx().D["part_sources"]
    for pid, g in src.groupby("part_id"):
        for s, sh in zip(g.supplier_id, g.share):
            assert L.loc[pid, s] >= sh - 1e-9     # losing a direct source costs at least its share


def test_hidden_spofs_are_really_hidden(pipeline):
    from aerosupply.graph.network import Network
    net = Network(pipeline.ctx().D)
    hs = net.hidden_spofs()
    assert len(hs) > 0
    assert (hs.tier1_sources >= 2).all()
    for _, r in hs.iterrows():
        assert net.loss.loc[r.part_id, r.upstream_id] > 0.999


def test_optimizer_respects_budget_and_is_monotone():
    rng = np.random.default_rng(0)
    samples = {f"P{i}": np.clip(rng.normal(50 * i, 30, 500), 0, None) for i in range(1, 5)}
    parts = pd.DataFrame({"part_id": list(samples), "unit_cost": [100.0, 200.0, 50.0, 400.0]})
    q = pd.Series(1.0, index=list(samples))
    small = allocate(samples, parts, q, 5_000)
    big = allocate(samples, parts, q, 50_000)
    assert small["spent_eur"] <= 5_000
    assert big["expected_aircraft_sets_protected"] >= small["expected_aircraft_sets_protected"]
