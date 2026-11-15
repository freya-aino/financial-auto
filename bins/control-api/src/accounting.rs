//! Daily close of the organisational books (D-104, `fa-accounting`), run by control-api like the
//! other storage maintenance: it is off the trading path and already holds Postgres, the
//! VictoriaMetrics URL and the config.
//!
//! Every `FA_ACCT_CLOSE_SECS` (default 86 400 = daily; `0` disables) the previous UTC day is
//! closed: the `paper` book's net P&L becomes trading revenue, the metered costs of the day are
//! accrued at the `[accounting]` prices (flat host day rate; GPU ms from `ops.job_run`; stored
//! bytes in VictoriaMetrics + Postgres at 1/30 of the GiB-month price), and the split policy
//! allocates any positive net income. Entries are idempotent by ref, so a re-run (or a manual
//! `POST /api/accounting/close`) posts nothing twice. On the first ever run the trading-book
//! mirror is seeded with the book's external capital (deposits − withdrawals) as contributed
//! capital, so the mirror tracks equity, not just P&L since the close job existed.
//!
//! Each run is appended to `audit.log` as `ops.accounting_close`.
use chrono::{DateTime, NaiveDate, Utc};
use fa_accounting::{
    Chart, Decimal, accounts,
    close::{OpexItem, close_period},
    dec, money,
    pg::PgAccounting,
    split::SplitPolicy,
};
use fa_ledger::pg::PgLedger;
use serde_json::{Value, json};
use sqlx::Row;

use crate::state::AppState;

const ACTOR: &str = "system:control-api";
const FIRST_RUN_DELAY: std::time::Duration = std::time::Duration::from_secs(300);
const BOOK: &str = "paper";

pub fn spawn(state: &AppState) {
    let every = std::env::var("FA_ACCT_CLOSE_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(86_400u64);
    if every == 0 {
        tracing::info!("accounting close disabled (FA_ACCT_CLOSE_SECS=0)");
        return;
    }
    let s = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(FIRST_RUN_DELAY).await;
        loop {
            let day = (Utc::now() - chrono::Duration::days(1)).date_naive();
            let report = close_day(&s, day).await;
            record(&s, report).await;
            tokio::time::sleep(std::time::Duration::from_secs(every)).await;
        }
    });
}

async fn record(s: &AppState, report: Result<Value, String>) {
    let payload = match &report {
        Ok(v) => {
            tracing::info!(report = %v, "accounting close");
            json!({ "ok": true, "report": v })
        }
        Err(e) => {
            tracing::warn!(error = %e, "accounting close failed (retried next interval)");
            json!({ "ok": false, "error": e })
        }
    };
    if let Some(pool) = s.pg.get()
        && let Err(e) = crate::audit::append(pool, ACTOR, "ops.accounting_close", None, payload).await
    {
        tracing::warn!(error = %e, "audit append failed");
    }
}

/// GPU, storage and flat-rate opex of `[from, to)`, priced by the config; zero-priced or
/// zero-usage meters produce no item.
async fn opex_items(
    s: &AppState,
    pool: &sqlx::PgPool,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> Result<Vec<OpexItem>, String> {
    let cfg = &s.cfg.accounting;
    let mut items = Vec::new();
    if cfg.base_eur_per_day > 0.0 {
        items.push(OpexItem {
            account: accounts::OPERATIONS_EXPENSE.into(),
            amount: dec(cfg.base_eur_per_day),
            memo: "host day rate".into(),
        });
    }
    if cfg.gpu_eur_per_hour > 0.0 {
        let ms: i64 = sqlx::query(
            "SELECT COALESCE(SUM(used_gpu_ms), 0) FROM ops.job_run WHERE finished_at >= $1 AND finished_at < $2",
        )
        .bind(from)
        .bind(to)
        .fetch_one(pool)
        .await
        .map_err(|e| format!("ops.job_run: {e}"))?
        .get(0);
        if ms > 0 {
            let eur = money::mul(money::div(dec(ms as f64), dec(3_600_000.0)), dec(cfg.gpu_eur_per_hour));
            items.push(OpexItem { account: accounts::INFRA_EXPENSE.into(), amount: eur, memo: format!("gpu {ms} ms") });
        }
    }
    if cfg.storage_eur_per_gib_month > 0.0 {
        let vm = fa_tsdb::Tsdb::new(s.cfg.victoriametrics.url.clone());
        let vm_bytes = vm.data_size_bytes().await.unwrap_or(0);
        let pg_bytes: i64 = sqlx::query("SELECT pg_database_size(current_database())")
            .fetch_one(pool)
            .await
            .map_err(|e| format!("pg_database_size: {e}"))?
            .get(0);
        let gib = (vm_bytes as f64 + pg_bytes.max(0) as f64) / f64::from(1u32 << 30);
        let eur = money::mul(money::div(dec(gib), dec(30.0)), dec(cfg.storage_eur_per_gib_month));
        if !eur.is_zero() {
            items.push(OpexItem {
                account: accounts::INFRA_EXPENSE.into(),
                amount: eur,
                memo: format!("storage {gib:.2} GiB day share"),
            });
        }
    }
    Ok(items)
}

/// Closes one UTC day of the `paper` book into the organisational journal. Idempotent.
pub async fn close_day(s: &AppState, day: NaiveDate) -> Result<Value, String> {
    let pool = s.pg.get().cloned().ok_or("postgres not ready")?;
    let acct = PgAccounting::from_pool(pool.clone());
    acct.sync_chart(&Chart::default_chart()).await.map_err(|e| e.to_string())?;

    let from: DateTime<Utc> = day.and_hms_opt(0, 0, 0).expect("midnight exists").and_utc();
    let to = from + chrono::Duration::days(1);
    let ledger = PgLedger::from_pool(pool.clone());

    // First run: seed the trading-book mirror with the book's external capital to date.
    if acct.raw_balances().await.map_err(|e| e.to_string())?.is_empty() {
        let contributed: Option<Decimal> = sqlx::query(
            "SELECT SUM(amount * COALESCE(fx_rate_eur, 1)) FROM ledger.cash_movement
             WHERE book = $1 AND kind IN ('deposit', 'withdrawal')",
        )
        .bind(BOOK)
        .fetch_one(&pool)
        .await
        .map_err(|e| format!("cash_movement: {e}"))?
        .get(0);
        if let Some(c) = contributed.map(money::round).filter(|c| !c.is_zero()) {
            let seed = fa_accounting::Entry::new(
                from,
                "seed:contributed",
                "trading book external capital at first close",
                vec![(accounts::TRADING_BOOK, c), (accounts::CONTRIBUTED, -c)],
            )
            .map_err(|e| e.to_string())?;
            acct.post(&seed).await.map_err(|e| e.to_string())?;
        }
    }

    let pnl = ledger.book_pnl(BOOK, Some(from), Some(to), None).await.map_err(|e| e.to_string())?;
    let opex = opex_items(s, &pool, from, to).await?;
    let a = &s.cfg.accounting;
    let policy = SplitPolicy::new(a.owner_frac, a.infra_frac, a.reserve_frac)?;
    let close = close_period(&day.to_string(), to, pnl.net_eur, &opex, &policy).map_err(|e| e.to_string())?;

    let (mut posted, mut skipped) = (0, 0);
    for e in &close.entries {
        if acct.post(e).await.map_err(|e| e.to_string())? { posted += 1 } else { skipped += 1 }
    }
    Ok(json!({
        "period": day.to_string(), "book": BOOK,
        "revenue_eur": money::f(pnl.net_eur),
        "opex": opex, "allocation": close.allocation,
        "entries_posted": posted, "entries_already": skipped,
    }))
}
