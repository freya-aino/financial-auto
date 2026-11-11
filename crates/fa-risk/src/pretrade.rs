//! Pre-trade checks for orders that did not come out of the allocator's own risk chain
//! (manual orders from the UI, any foreign `order.intent.*`; spec 40 §6.1 step 5, §9 "no order
//! can reach a venue without passing the full risk chain"). The allocator's targets are
//! already capped by [`crate::RiskEngine::cap_targets`]; a single foreign order is checked
//! here against the same caps applied to the post-trade book:
//!
//! - sanity: finite positive quantity, a known mark, a limit price inside
//!   `limit_price_band` of the mark (fat-finger prices);
//! - order size: notional ≤ `max_order_frac` × deployable (fat-finger quantities);
//! - post-trade position ≤ the position cap (stricter for provisional instruments) and the
//!   liquidity cap `k × horizon volume`;
//! - post-trade gross per asset class, per venue and for the portfolio ≤ their caps;
//! - shorts (spec 40 §10 Q5): opening or adding to a short needs a shortable instrument
//!   (class in `short_margin`, `ref.contract_spec.shortable` not false) and margin room:
//!   Σ |short| × margin after the order ≤ `short_margin_budget_frac` × deployable.
//!
//! An order that only reduces an existing position (same sign, smaller magnitude) always
//! passes the size and cap checks — closing risk is never blocked by a cap.

use crate::{MarketInfo, RiskConfig};

/// One order to check, in EUR at the current mark.
#[derive(Debug, Clone)]
pub struct OrderCheck<'a> {
    /// Signed quantity (buy > 0).
    pub quantity: f64,
    pub limit_price: Option<f64>,
    /// Mark in EUR per unit (`None` = unknown instrument price).
    pub mark_eur: Option<f64>,
    /// Quote-currency → EUR factor of the limit price (1.0 for EUR instruments).
    pub fx_to_eur: f64,
    /// Current signed position of the instrument in EUR (book + pending).
    pub position_eur: f64,
    pub market: Option<&'a MarketInfo>,
    pub deployable_eur: f64,
    /// Current gross exposure of the instrument's asset class, venue and portfolio (EUR).
    pub class_gross_eur: f64,
    pub venue_gross_eur: f64,
    pub portfolio_gross_eur: f64,
    /// Margin currently tied up by shorts (Σ |short| × margin, EUR).
    pub short_margin_used_eur: f64,
}

/// `Ok(())` or the rejection reason.
pub fn check_order(cfg: &RiskConfig, o: &OrderCheck<'_>) -> Result<(), String> {
    if !o.quantity.is_finite() || o.quantity == 0.0 {
        return Err(format!("pre-trade: quantity {} is not a finite non-zero number", o.quantity));
    }
    let Some(mark) = o.mark_eur.filter(|m| m.is_finite() && *m > 0.0) else {
        return Err("pre-trade: no mark price for the instrument".into());
    };
    if let Some(lp) = o.limit_price {
        let lp_eur = lp * o.fx_to_eur;
        if !lp_eur.is_finite() || lp_eur <= 0.0 {
            return Err(format!("pre-trade: limit price {lp} is not positive"));
        }
        let dev = (lp_eur / mark - 1.0).abs();
        if dev > cfg.limit_price_band {
            return Err(format!(
                "pre-trade: limit price {lp} is {:.1}% away from the mark (band {:.1}%)",
                dev * 100.0,
                cfg.limit_price_band * 100.0
            ));
        }
    }
    let delta = o.quantity * mark;
    let post = o.position_eur + delta;
    let reducing = o.position_eur != 0.0
        && post.signum() * o.position_eur.signum() >= 0.0
        && post.abs() <= o.position_eur.abs() + 1e-9;
    if reducing {
        return Ok(());
    }
    let deployable = o.deployable_eur.max(0.0);
    if delta.abs() > cfg.max_order_frac * deployable {
        return Err(format!(
            "pre-trade: order notional {:.0} EUR > max_order_frac {} × deployable {:.0}",
            delta.abs(),
            cfg.max_order_frac,
            deployable
        ));
    }
    let Some(m) = o.market else {
        return Err("pre-trade: unknown market (no reference data or volume) for a risk-increasing order".into());
    };
    let pos_cap = if m.provisional { cfg.provisional_position_frac() } else { cfg.max_position_frac } * deployable;
    if post.abs() > pos_cap {
        return Err(format!("pre-trade: post-trade position {:.0} EUR > position cap {pos_cap:.0}", post.abs()));
    }
    let liq_cap = cfg.liquidity_k * m.horizon_volume_eur;
    if post.abs() > liq_cap {
        return Err(format!(
            "pre-trade: post-trade position {:.0} EUR > liquidity cap {liq_cap:.0} (k {} × horizon volume {:.0})",
            post.abs(),
            cfg.liquidity_k,
            m.horizon_volume_eur
        ));
    }
    let added_short = (-post).max(0.0) - (-o.position_eur).max(0.0);
    if added_short > 0.0 {
        let Some(margin) = m.short_margin(cfg) else {
            return Err(format!("pre-trade: {:?} instrument is not shortable (no borrow)", m.asset_class));
        };
        let budget = cfg.short_margin_budget_frac * deployable;
        let used = o.short_margin_used_eur + added_short * margin;
        if used > budget + 1e-9 {
            return Err(format!("pre-trade: short margin {used:.0} EUR > short margin budget {budget:.0}"));
        }
    }
    let added = post.abs() - o.position_eur.abs();
    let class_cap = cfg.leverage_cap.get(&m.asset_class).copied().unwrap_or(0.0).min(cfg.max_class_frac) * deployable;
    if o.class_gross_eur + added > class_cap {
        return Err(format!(
            "pre-trade: {:?} gross {:.0} EUR > class cap {class_cap:.0}",
            m.asset_class,
            o.class_gross_eur + added
        ));
    }
    let venue_cap = cfg.max_venue_frac * deployable;
    if o.venue_gross_eur + added > venue_cap {
        return Err(format!("pre-trade: venue gross {:.0} EUR > venue cap {venue_cap:.0}", o.venue_gross_eur + added));
    }
    let gross_cap = cfg.portfolio_gross_cap * deployable;
    if o.portfolio_gross_eur + added > gross_cap {
        return Err(format!(
            "pre-trade: portfolio gross {:.0} EUR > gross cap {gross_cap:.0}",
            o.portfolio_gross_eur + added
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use fa_core::AssetClass;

    use super::*;

    fn cfg() -> RiskConfig {
        RiskConfig { max_position_frac: 0.05, max_order_frac: 0.10, limit_price_band: 0.10, ..RiskConfig::default() }
    }

    fn order(q: f64, m: &MarketInfo) -> OrderCheck<'_> {
        OrderCheck {
            quantity: q,
            limit_price: None,
            mark_eur: Some(100.0),
            fx_to_eur: 1.0,
            position_eur: 0.0,
            market: Some(m),
            deployable_eur: 100_000.0,
            class_gross_eur: 0.0,
            venue_gross_eur: 0.0,
            portfolio_gross_eur: 0.0,
            short_margin_used_eur: 0.0,
        }
    }

    #[test]
    fn sane_orders_within_caps_pass() {
        let m = MarketInfo::new(AssetClass::Crypto, "binance", 10_000_000.0);
        assert_eq!(check_order(&cfg(), &order(40.0, &m)), Ok(())); // 4 000 EUR
        assert_eq!(check_order(&cfg(), &order(-40.0, &m)), Ok(()), "shorts too");
        let lim = OrderCheck { limit_price: Some(105.0), ..order(10.0, &m) };
        assert_eq!(check_order(&cfg(), &lim), Ok(()));
    }

    #[test]
    fn fat_fingers_and_unknowns_are_rejected() {
        let m = MarketInfo::new(AssetClass::Crypto, "binance", 10_000_000.0);
        let c = cfg();
        for (o, needle) in [
            (OrderCheck { quantity: f64::NAN, ..order(1.0, &m) }, "finite"),
            (OrderCheck { mark_eur: None, ..order(1.0, &m) }, "no mark"),
            (OrderCheck { limit_price: Some(150.0), ..order(1.0, &m) }, "away from the mark"),
            (OrderCheck { limit_price: Some(-1.0), ..order(1.0, &m) }, "not positive"),
            (order(200.0, &m), "max_order_frac"), // 20 000 EUR > 10 %
            (OrderCheck { market: None, ..order(1.0, &m) }, "unknown market"),
        ] {
            let e = check_order(&c, &o).unwrap_err();
            assert!(e.contains(needle), "{e}");
        }
    }

    #[test]
    fn post_trade_caps_bind_but_reductions_always_pass() {
        let c = cfg();
        let m = MarketInfo::new(AssetClass::Crypto, "binance", 10_000_000.0);
        // 4 000 held + 2 000 → 6 000 > 5 % of 100 000
        let o = OrderCheck { position_eur: 4_000.0, ..order(20.0, &m) };
        assert!(check_order(&c, &o).unwrap_err().contains("position cap"));
        // reducing a position above every cap is fine, flipping through zero is not a reduction
        let o = OrderCheck { position_eur: 9_000.0, ..order(-50.0, &m) };
        assert_eq!(check_order(&c, &o), Ok(()));
        let o = OrderCheck { position_eur: 1_000.0, ..order(-80.0, &m) };
        assert!(check_order(&c, &o).unwrap_err().contains("position cap"), "a flip through zero is risk-increasing");
        // liquidity: thin market
        let thin = MarketInfo::new(AssetClass::Crypto, "x", 100_000.0);
        assert!(check_order(&c, &order(20.0, &thin)).unwrap_err().contains("liquidity cap"));
        // provisional instruments get the stricter cap
        let prov = MarketInfo { provisional: true, ..MarketInfo::new(AssetClass::Crypto, "x", 10_000_000.0) };
        assert!(check_order(&c, &order(30.0, &prov)).unwrap_err().contains("position cap"));
        // class / venue / portfolio gross
        let o = OrderCheck { class_gross_eur: 39_000.0, ..order(20.0, &m) };
        assert!(check_order(&c, &o).unwrap_err().contains("class cap"));
        let o = OrderCheck { venue_gross_eur: 49_000.0, ..order(20.0, &m) };
        assert!(check_order(&c, &o).unwrap_err().contains("venue cap"));
        let o = OrderCheck { portfolio_gross_eur: 199_000.0, ..order(20.0, &m) };
        assert!(check_order(&c, &o).unwrap_err().contains("gross cap"));
    }

    #[test]
    fn shorts_need_borrow_and_margin_room() {
        let c = cfg();
        let eq = MarketInfo::new(AssetClass::Equity, "xetr", 10_000_000.0);
        assert_eq!(check_order(&c, &order(-40.0, &eq)), Ok(()), "4 000 short ties up 2 000 margin");
        let no_borrow = MarketInfo { shortable: Some(false), ..eq.clone() };
        assert!(check_order(&c, &order(-40.0, &no_borrow)).unwrap_err().contains("not shortable"));
        assert_eq!(check_order(&c, &order(40.0, &no_borrow)), Ok(()), "longs are unaffected");
        let rates = MarketInfo::new(AssetClass::Rates, "x", 10_000_000.0);
        assert!(check_order(&c, &order(-10.0, &rates)).unwrap_err().contains("not shortable"));
        // 49 000 margin used of the 50 000 budget: another 4 000 short at 50 % does not fit
        let o = OrderCheck { short_margin_used_eur: 49_000.0, ..order(-40.0, &eq) };
        assert!(check_order(&c, &o).unwrap_err().contains("short margin"));
        // covering a short always passes, even on a non-shortable instrument
        let o = OrderCheck { position_eur: -4_000.0, short_margin_used_eur: 60_000.0, ..order(20.0, &no_borrow) };
        assert_eq!(check_order(&c, &o), Ok(()));
    }
}
