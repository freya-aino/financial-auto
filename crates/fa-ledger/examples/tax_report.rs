//! Realised-gains report (spec 42 §10 Q2; `fa_ledger::tax`). Read-only.
//!
//! ```text
//! DATABASE_URL=postgres://fa:fa@127.0.0.1:5432/fa \
//!   cargo run -p fa-ledger --example tax_report -- [--book paper] [--year 2026]
//!   [--method fifo|lifo|average] [--year-start 01-01] [--format summary|csv|json] [--out file]
//! ```
//! `summary` (default) prints the per-year and per-instrument totals; `csv` / `json` print (or
//! write to `--out`) the full export.
use fa_ledger::{
    money,
    pg::PgLedger,
    tax::{TaxLotMethod, TaxReportOptions, TaxYearStart},
};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL is required")?;
    let book = arg(&args, "--book").unwrap_or_else(|| "paper".into());
    let opts = TaxReportOptions {
        method: arg(&args, "--method").map(|m| m.parse::<TaxLotMethod>()).transpose()?.unwrap_or_default(),
        year_start: arg(&args, "--year-start").map(|s| s.parse::<TaxYearStart>()).transpose()?.unwrap_or_default(),
        year: arg(&args, "--year").map(|y| y.parse()).transpose()?,
    };
    let pg = PgLedger::connect(&url).await?.with_stored_settlement().await;
    let r = pg.realised_gains(&book, &opts).await?;
    let body = match arg(&args, "--format").as_deref().unwrap_or("summary") {
        "csv" => r.to_csv(),
        "json" => serde_json::to_string_pretty(&r)?,
        "summary" => {
            let mut s = format!(
                "Realised gains — book {} method {} tax year start {:02}-{:02}{}\n",
                r.book,
                r.method.as_str(),
                r.year_start.month,
                r.year_start.day,
                r.year.map(|y| format!(" year {y}")).unwrap_or_default()
            );
            s += &format!(
                "  {:<10} {:>9} {:>18} {:>18} {:>12} {:>16}\n",
                "tax year", "disposals", "proceeds EUR", "cost EUR", "fees EUR", "gain EUR"
            );
            for y in &r.by_year {
                let t = &y.totals;
                s += &format!(
                    "  {:<10} {:>9} {:>18.2} {:>18.2} {:>12.2} {:>16.2}\n",
                    y.tax_year_label,
                    t.disposals,
                    money::f(t.proceeds_eur),
                    money::f(t.cost_eur),
                    money::f(t.fees_eur),
                    money::f(t.gain_eur)
                );
            }
            s += "  per instrument:\n";
            for i in &r.by_instrument {
                s += &format!(
                    "    {:<8} {:<28} {:>7} disposals  gain {:>14.2} EUR (gains {:.2}, losses {:.2})\n",
                    i.tax_year_label,
                    i.instrument_name.clone().unwrap_or_else(|| i.instrument.0.to_string()),
                    i.totals.disposals,
                    money::f(i.totals.gain_eur),
                    money::f(i.totals.gains_eur),
                    money::f(i.totals.losses_eur)
                );
            }
            if !r.income.is_empty() {
                s += "  income / charges (cash movements):\n";
                for i in &r.income {
                    s += &format!(
                        "    {:<8} {:<16} {:>5}× {:>14.2} EUR\n",
                        i.tax_year_label,
                        i.kind,
                        i.count,
                        money::f(i.amount_eur)
                    );
                }
            }
            s += &format!("  open lots after the period: {}\n", r.open_lots);
            for n in &r.notes {
                s += &format!("  note: {n}\n");
            }
            s
        }
        other => return Err(format!("--format {other:?}: use summary, csv or json").into()),
    };
    match arg(&args, "--out") {
        Some(path) => {
            std::fs::write(&path, body)?;
            eprintln!("wrote {path}");
        }
        None => print!("{body}"),
    }
    Ok(())
}
