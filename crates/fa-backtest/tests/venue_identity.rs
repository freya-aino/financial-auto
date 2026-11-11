//! Replay-vs-live identity (spec 41 §9, spec 40 §6.7): a recorded live paper window — the tape
//! the broker-gateway saw from its start plus every approved intent and fill on `ORDERS` — is
//! replayed through `fa_backtest::venue` (the gateway's estimators) and `fa_backtest::cost` (its
//! cost model); every recorded market fill must be reproduced.
//!
//! Fixture: `testdata/venue_fixture.jsonl`, a 25 min live capture on 2026-11-11 (03:54–04:19 UTC)
//! across a broker-gateway restart (`reset`), recorded with
//! `cargo run -p fa-backtest --example record_venue_window -- live 1500 paper-test 15 \
//!   850b1098-d844-59e3-a7eb-1fb84638c1bf=0.001,01900000-0000-7000-8000-000000000001=1,9daa8cea-67a6-5f6b-8347-f9e5a30f7463=1`
//! — market probes every 15 s on Kraken BTC (quotes, σ / ADV impact, 1 EUR minimum fee), the
//! synthetic `sim` instrument (costless venue) and a Yahoo equity, on the throwaway account
//! `paper-test`; the gateway priced them with the repo `costs.yaml`.
use fa_backtest::{
    cost::CostModel,
    venue::{VenueRecord, replay_window},
};

fn costs() -> CostModel {
    CostModel::from_yaml(include_str!("../../../go/services/broker-gateway/costs.yaml")).expect("costs.yaml")
}

fn fixture() -> Vec<VenueRecord> {
    include_str!("../testdata/venue_fixture.jsonl")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("fixture line"))
        .collect()
}

#[test]
fn a_recorded_live_window_reproduces_the_paper_fills() {
    let recs = fixture();
    let r = replay_window(costs(), &recs);
    let reset =
        recs.iter().filter(|&r| matches!(r, VenueRecord::Reset { .. })).map(VenueRecord::t).max().unwrap_or(i64::MIN);
    let recorded_fills = recs.iter().filter(|r| matches!(r, VenueRecord::Fill { .. }) && r.t() > reset).count();
    eprintln!(
        "intents {} compared {} arrival_equal {} max_fill_diff_bps {:.6} mean {:.6} max_fee_diff {:.6} max_slippage_diff_bps {:.6}",
        r.intents,
        r.compared,
        r.arrival_equal,
        r.max_fill_diff_bps,
        r.mean_fill_diff_bps,
        r.max_fee_diff,
        r.max_slippage_diff_bps
    );
    for c in r.comparisons.iter().filter(|c| c.fill_diff_bps > 0.05).take(10) {
        eprintln!("{c:?}");
    }
    assert!(r.intents >= 100, "the window holds live intents");
    assert_eq!(r.compared, recorded_fills, "every recorded fill after the gateway start is replayed");
    // every arrival price identical; fill prices within 0.05 bps (σ / ADV are recomputed from the
    // tape in receive order, and the gateway's ADV clock is its own wall clock at the intent)
    assert_eq!(r.arrival_equal, r.compared, "arrival price identical for every fill");
    assert!(r.max_fill_diff_bps < 0.05, "max fill difference {} bps", r.max_fill_diff_bps);
    assert!(r.max_fee_diff < 1e-6, "fee difference {} EUR", r.max_fee_diff);
}

#[test]
fn the_identity_check_detects_a_different_cost_model() {
    let recs = fixture();
    let doubled = costs().scaled(2.0);
    let r = replay_window(doubled, &recs);
    assert!(r.mean_fill_diff_bps > 0.5, "2× costs must not reproduce the fills: {}", r.mean_fill_diff_bps);
}
