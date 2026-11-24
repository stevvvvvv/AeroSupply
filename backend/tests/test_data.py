import numpy as np
import pandas as pd

from aerosupply.config import AS_OF, TRUTH_DIR
from aerosupply.data.inventory import day_idx, on_hand_series, part_daily_demand, receipts_daily


def test_tiers_and_network(D):
    sup = D["suppliers"]
    assert set(sup.tier) == {1, 2, 3}
    links = D["supplier_links"]
    tier = sup.set_index("supplier_id").tier
    assert (links.upstream_id.map(tier) == links.customer_id.map(tier) + 1).all()   # edges go one tier down
    assert D["part_sources"].groupby("part_id").share.sum().round(6).eq(1).all()


def test_no_delivery_after_as_of(D):
    po = D["purchase_orders"]
    assert (po.delivered_date.dropna() <= pd.Timestamp(AS_OF)).all()
    assert (po.order_date < pd.Timestamp(AS_OF)).all()
    assert po[po.delivered_date.isna()].qty_accepted.isna().all()


def test_inventory_is_derived_consistently(D):
    """Stock = initial + receipts - consumption; the generator's ordering policy keeps it bounded."""
    dem = part_daily_demand(D["bom"], D["production_plan"])
    rec = receipts_daily(D["purchase_orders"], D["parts"].part_id, AS_OF)
    p = D["parts"].iloc[0]
    oh = on_hand_series(p.initial_stock, rec[p.part_id], dem[p.part_id])[: day_idx(AS_OF)]
    assert np.isfinite(oh).all()
    assert oh.max() < 20 * dem[p.part_id].mean() * 30   # never wildly over-stocked


def test_truth_kept_out_of_raw_exports(D):
    assert not any(c.startswith("_") for c in D["suppliers"].columns)
    assert "_truth_disruption_id" not in D["signals"].columns
    assert (TRUTH_DIR / "disruptions.csv").exists()
