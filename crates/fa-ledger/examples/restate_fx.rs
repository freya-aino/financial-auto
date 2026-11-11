//! FX restatement tool (spec 42 §6; `fa_ledger::restate`). Dry run by default.
//!
//! ```text
//! DATABASE_URL=postgres://fa:fa@127.0.0.1:5432/fa \
//!   cargo run -p fa-ledger --example restate_fx -- --instrument <uuid> [--book paper] [--ccy USD]
//!   [--from-ccy EUR] [--check-hours 24] [--json plan.json] [--fills 10]
//!   [--apply --reason "<why>" --actor ui:<name>]
//! ```
//! Without `--apply` nothing is written. `--apply` restates exactly the planned fills in one
//! transaction; restart the risk-engine right after so its in-memory book reloads.
use fa_core::InstrumentId;
use fa_ledger::{
    money,
    pg::PgLedger,
    restate::{BookFigures, FxRestatementRequest},
};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn row(label: &str, before: money::Decimal, after: money::Decimal) {
    println!("  {label:<34} {:>18.6} {:>18.6} {:>16.6}", money::f(before), money::f(after), money::f(after - before));
}

fn figures(b: &BookFigures, a: &BookFigures) {
    println!("  {:<34} {:>18} {:>18} {:>16}", "", "before", "after", "change");
    row("cash_eur", b.cash_eur, a.cash_eur);
    row("equity_eur", b.equity_eur, a.equity_eur);
    row("pnl.realised_eur (inception)", b.pnl.realised_eur, a.pnl.realised_eur);
    row("pnl.unrealised_eur", b.pnl.unrealised_eur, a.pnl.unrealised_eur);
    row("pnl.fx_realised_eur", b.pnl.fx_realised_eur, a.pnl.fx_realised_eur);
    row("pnl.fx_unrealised_eur", b.pnl.fx_unrealised_eur, a.pnl.fx_unrealised_eur);
    row("pnl.net_eur", b.pnl.net_eur, a.pnl.net_eur);
    row("window book net_eur", b.window_book_net_eur, a.window_book_net_eur);
    row("window strategy sum net_eur", b.window_strategy_sum_net_eur, a.window_strategy_sum_net_eur);
    row("window strategy − book gap", b.window_strategy_gap_eur, a.window_strategy_gap_eur);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL is required")?;
    let instrument = arg(&args, "--instrument").ok_or("--instrument <uuid> is required")?;
    let req = FxRestatementRequest {
        book: arg(&args, "--book").unwrap_or_else(|| "paper".into()),
        instrument: InstrumentId(instrument.parse()?),
        ccy: arg(&args, "--ccy"),
        from_ccy: arg(&args, "--from-ccy"),
        check_hours: arg(&args, "--check-hours").map(|h| h.parse()).transpose()?,
    };
    let pg = PgLedger::connect(&url).await?.with_stored_settlement().await;
    let plan = pg.plan_fx_restatement(&req).await?;
    println!(
        "FX restatement plan — book {} instrument {} {} → {} (planned {})",
        plan.book, plan.instrument, plan.from_ccy, plan.ccy, plan.planned_at
    );
    println!(
        "  fills to restate: {} {:?}; without a rate: {}",
        plan.fills.len(),
        plan.by_old_source,
        plan.missing_rate.len()
    );
    if let (Some(first), Some(last)) = (plan.fills.first(), plan.fills.last()) {
        println!(
            "  filled {} … {}; new rate(s): {:?}",
            first.filled_at,
            last.filled_at,
            plan.fills
                .iter()
                .map(|f| (f.new_fx.normalize().to_string(), f.new_source.as_str(), f.rate_ts))
                .collect::<std::collections::BTreeSet<_>>()
        );
    }
    println!("  cash delta: {} EUR", plan.cash_delta_eur.round_dp(6));
    let n: usize = arg(&args, "--fills").map(|n| n.parse()).transpose()?.unwrap_or(5);
    println!("  sample fills (fill, filled_at, qty, price, old ccy/fx/source → new ccy/fx/source, cash Δ EUR):");
    for f in plan.fills.iter().take(n) {
        println!(
            "    {} {} {} {} {}/{}/{} → {}/{}/{} {}",
            f.venue_fill_id,
            f.filled_at,
            f.qty.normalize(),
            f.price.normalize(),
            f.old_ccy,
            f.old_fx.normalize(),
            f.old_source,
            f.new_ccy,
            f.new_fx.normalize(),
            f.new_source,
            f.cash_delta_eur.round_dp(6)
        );
    }
    println!("  window: {} → {}", plan.check_from, plan.planned_at);
    figures(&plan.before, &plan.after);
    for n in &plan.notes {
        println!("  note: {n}");
    }
    if let Some(path) = arg(&args, "--json") {
        std::fs::write(&path, serde_json::to_vec_pretty(&plan)?)?;
        println!("  plan written to {path}");
    }
    if args.iter().any(|a| a == "--apply") {
        let reason = arg(&args, "--reason").ok_or("--apply needs --reason")?;
        let actor = arg(&args, "--actor").unwrap_or_else(|| "ui:operator".into());
        let applied = pg.apply_fx_restatement(&plan, &actor, &reason).await?;
        println!(
            "APPLIED batch {}: {} fills, cash delta {} EUR — restart the risk-engine now",
            applied.batch_id, applied.fills, applied.cash_delta_eur
        );
    } else {
        println!("DRY RUN — nothing written (pass --apply --reason \"…\" to apply)");
    }
    Ok(())
}
