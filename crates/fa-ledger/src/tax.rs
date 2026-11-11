//! Realised-gains report (spec 42 §10 Q2): jurisdiction-neutral facts a tax return is built
//! from, not a tax computation.
//!
//! Per disposal (a lot closing): tax year, venue, instrument, side, quantity, acquisition and
//! disposal time, holding period, proceeds, cost basis, fees and gain in EUR at the booking FX
//! rates (the rate stored on each fill, `ledger.fill.fx_rate_eur`). Aggregated per tax year and
//! instrument and per tax year, plus non-trade income and charges per tax year from cash
//! movements (funding, borrow, venue fees, dividends/interest/other); capital flows (deposits,
//! withdrawals, payouts) are not income and are left out.
//!
//! Lots are matched per `(venue, instrument)` by a configurable [`TaxLotMethod`] (FIFO default,
//! LIFO, or pooled average cost) over the whole history, so a disposal in a reported year uses
//! its real cost basis from earlier years. Fees are allocated pro rata by quantity: a fill's fee
//! per unit goes into the cost basis of the lot it opens (`open_fee_eur`) and reduces the
//! proceeds of the units it closes (`close_fee_eur`); `gain = proceeds − cost − fees`.
//!
//! Long lots: proceeds = `q·P₁·X₁` (sale), cost = `q·P₀·X₀` (purchase). Short lots are realised
//! when bought back: proceeds = `q·P₀·X₀` (the opening sale), cost = `q·P₁·X₁` (the buy-back).
//! Before fees the gain equals the ledger's realised price + FX parts
//! ([`crate::accounting::LotClosing`]) under the same lot method.
//!
//! Jurisdiction-specific rules — loss pools and carry-forward, holding-period exemptions,
//! wash-sale / bed-and-breakfast matching, pooling across venues, deemed-return regimes (e.g.
//! Dutch box 3), currency of the return other than EUR — are not applied: they are the user's
//! decision (spec 42 §10 Q2). The report carries the inputs they need (holding days, per-venue
//! lots, both fee parts).
use std::collections::{BTreeMap, VecDeque};

use chrono::{DateTime, Datelike, NaiveDate, Utc};
use fa_core::InstrumentId;
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::{
    accounting::{CashClass, CashEvent, LotMethod, TradeEvent, classify_cash},
    money::{self, Decimal, div, mul},
    pg::{PgError, PgLedger},
};

const ZERO: Decimal = Decimal::ZERO;

/// Lot matching for the realised-gains report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaxLotMethod {
    /// Oldest lot first.
    #[default]
    Fifo,
    /// Newest lot first.
    Lifo,
    /// One pooled lot per `(venue, instrument)` at the average EUR cost per unit.
    Average,
}

impl TaxLotMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fifo => "fifo",
            Self::Lifo => "lifo",
            Self::Average => "average",
        }
    }
}

impl From<LotMethod> for TaxLotMethod {
    fn from(m: LotMethod) -> Self {
        match m {
            LotMethod::Fifo => Self::Fifo,
            LotMethod::Average => Self::Average,
        }
    }
}

impl std::str::FromStr for TaxLotMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fifo" => Ok(Self::Fifo),
            "lifo" => Ok(Self::Lifo),
            "average" | "avg" | "average_cost" | "average-cost" => Ok(Self::Average),
            other => Err(format!("unknown lot method {other:?}: use fifo, lifo or average")),
        }
    }
}

/// First day of the tax year (month, day), e.g. `01-01` (default) or `04-06`. Dates are UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaxYearStart {
    pub month: u32,
    pub day: u32,
}

impl Default for TaxYearStart {
    fn default() -> Self {
        Self { month: 1, day: 1 }
    }
}

impl std::str::FromStr for TaxYearStart {
    type Err = String;
    /// `MM-DD`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (m, d) = s.trim().split_once('-').ok_or_else(|| format!("tax year start {s:?}: use MM-DD"))?;
        let (month, day) = (
            m.parse::<u32>().map_err(|e| format!("tax year start {s:?}: {e}"))?,
            d.parse::<u32>().map_err(|e| format!("tax year start {s:?}: {e}"))?,
        );
        // validate against a leap year so 02-29 is accepted (it falls back to 03-01 elsewhere)
        NaiveDate::from_ymd_opt(2024, month, day).ok_or_else(|| format!("tax year start {s:?}: not a date"))?;
        Ok(Self { month, day })
    }
}

impl TaxYearStart {
    fn start_in(self, year: i32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, self.month, self.day)
            .or_else(|| NaiveDate::from_ymd_opt(year, 3, 1))
            .expect("valid date")
    }

    /// The tax year (the calendar year in which it starts) containing `t`.
    pub fn year_of(self, t: DateTime<Utc>) -> i32 {
        let d = t.date_naive();
        if d >= self.start_in(d.year()) { d.year() } else { d.year() - 1 }
    }

    /// `2026` for calendar tax years, else `2026/27`.
    pub fn label(self, year: i32) -> String {
        if (self.month, self.day) == (1, 1) { year.to_string() } else { format!("{year}/{:02}", (year + 1) % 100) }
    }

    /// `[start, end)` of tax year `year` as UTC dates.
    pub fn bounds(self, year: i32) -> (NaiveDate, NaiveDate) {
        (self.start_in(year), self.start_in(year + 1))
    }
}

/// What to report.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TaxReportOptions {
    pub method: TaxLotMethod,
    pub year_start: TaxYearStart,
    /// Only this tax year (all years with a disposal or income item when `None`).
    pub year: Option<i32>,
}

/// One disposal (a lot closing) with EUR amounts at the booking rates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Disposal {
    pub tax_year: i32,
    pub tax_year_label: String,
    pub venue: Uuid,
    pub instrument: InstrumentId,
    /// `ref.instrument.name` when known.
    #[serde(default)]
    pub instrument_name: Option<String>,
    /// `long` (bought, then sold) or `short` (sold, then bought back).
    pub side: String,
    /// Unsigned quantity disposed of.
    #[serde(with = "money::json")]
    pub qty: Decimal,
    /// Opening fill of the lot (`ledger.fill.id`; for an average pool the first fill of the pool).
    pub lot_id: Uuid,
    pub close_fill: Uuid,
    pub close_fill_id: String,
    pub acquired_at: DateTime<Utc>,
    pub disposed_at: DateTime<Utc>,
    pub holding_days: i64,
    /// Native currency of the closing fill.
    pub ccy: String,
    /// Native prices per unit at open and close.
    #[serde(with = "money::json")]
    pub open_price: Decimal,
    #[serde(with = "money::json")]
    pub close_price: Decimal,
    /// EUR per unit of the fill currency at open (cost-weighted for a pool) and at close.
    #[serde(with = "money::json")]
    pub open_fx: Decimal,
    #[serde(with = "money::json")]
    pub close_fx: Decimal,
    #[serde(with = "money::json")]
    pub proceeds_eur: Decimal,
    #[serde(with = "money::json")]
    pub cost_eur: Decimal,
    #[serde(with = "money::json")]
    pub open_fee_eur: Decimal,
    #[serde(with = "money::json")]
    pub close_fee_eur: Decimal,
    /// `open_fee_eur + close_fee_eur`.
    #[serde(with = "money::json")]
    pub fees_eur: Decimal,
    /// `proceeds − cost − fees`.
    #[serde(with = "money::json")]
    pub gain_eur: Decimal,
}

/// Sums of disposals (per tax year and instrument, or per tax year).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GainTotals {
    pub disposals: usize,
    #[serde(with = "money::json")]
    pub qty: Decimal,
    #[serde(with = "money::json")]
    pub proceeds_eur: Decimal,
    #[serde(with = "money::json")]
    pub cost_eur: Decimal,
    #[serde(with = "money::json")]
    pub fees_eur: Decimal,
    #[serde(with = "money::json")]
    pub gain_eur: Decimal,
    /// Sum of the positive gains.
    #[serde(with = "money::json")]
    pub gains_eur: Decimal,
    /// Sum of the negative gains (≤ 0).
    #[serde(with = "money::json")]
    pub losses_eur: Decimal,
}

impl GainTotals {
    fn add(&mut self, d: &Disposal) {
        self.disposals += 1;
        self.qty += d.qty;
        self.proceeds_eur += d.proceeds_eur;
        self.cost_eur += d.cost_eur;
        self.fees_eur += d.fees_eur;
        self.gain_eur += d.gain_eur;
        if d.gain_eur.is_sign_negative() {
            self.losses_eur += d.gain_eur;
        } else {
            self.gains_eur += d.gain_eur;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentYear {
    pub tax_year: i32,
    pub tax_year_label: String,
    pub instrument: InstrumentId,
    #[serde(default)]
    pub instrument_name: Option<String>,
    #[serde(flatten)]
    pub totals: GainTotals,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YearTotal {
    pub tax_year: i32,
    pub tax_year_label: String,
    #[serde(flatten)]
    pub totals: GainTotals,
}

/// Non-trade income or charge of one cash-movement kind in a tax year (signed EUR at the
/// booking rate: + received, − paid).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IncomeItem {
    pub tax_year: i32,
    pub tax_year_label: String,
    pub kind: String,
    pub class: CashClass,
    pub count: usize,
    #[serde(with = "money::json")]
    pub amount_eur: Decimal,
}

/// The realised-gains report of one book.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaxReport {
    pub book: String,
    pub method: TaxLotMethod,
    pub year_start: TaxYearStart,
    pub year: Option<i32>,
    pub base_currency: String,
    pub disposals: Vec<Disposal>,
    pub by_instrument: Vec<InstrumentYear>,
    pub by_year: Vec<YearTotal>,
    pub income: Vec<IncomeItem>,
    /// Lots still open after the last reported year (or at the end of the data).
    pub open_lots: usize,
    /// What the report does and does not do (printed with the CSV as well).
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
struct TaxLot {
    lot_id: Uuid,
    opened_at: DateTime<Utc>,
    /// Remaining signed quantity.
    qty: Decimal,
    price: Decimal,
    fx: Decimal,
    /// EUR cost (signed like `qty`: `qty · price · fx`) and fee of the remaining quantity.
    cost_eur: Decimal,
    fee_eur: Decimal,
}

fn fee_share(fee: Decimal, part: Decimal, whole: Decimal) -> Decimal {
    if whole.is_zero() { ZERO } else { div(mul(fee, part), whole) }
}

/// Builds the report from a book's fills and cash movements (any order; de-duplicated by id).
pub fn realised_gains(
    book: &str,
    trades: &[TradeEvent],
    cash: &[CashEvent],
    opts: &TaxReportOptions,
    names: &BTreeMap<InstrumentId, String>,
) -> TaxReport {
    let ys = opts.year_start;
    let mut ts: Vec<&TradeEvent> = trades.iter().collect();
    ts.sort_by(|a, b| a.filled_at.cmp(&b.filled_at).then(a.fill_uuid.cmp(&b.fill_uuid)));
    let mut seen_fills = std::collections::HashSet::new();
    ts.retain(|t| seen_fills.insert(t.fill_uuid));

    let mut lots: BTreeMap<(Uuid, InstrumentId), VecDeque<TaxLot>> = BTreeMap::new();
    let mut disposals = Vec::new();
    for t in ts {
        if t.qty.is_zero() {
            continue;
        }
        let fx = t.fx();
        let ccy = if t.ccy.is_empty() { fa_core::BASE_CURRENCY.to_owned() } else { t.ccy.to_ascii_uppercase() };
        let book_lots = lots.entry((t.venue, t.instrument)).or_default();
        let mut rem = t.qty;
        while !rem.is_zero() {
            let lot = match opts.method {
                TaxLotMethod::Lifo => book_lots.back_mut(),
                _ => book_lots.front_mut(),
            };
            let Some(lot) = lot else { break };
            if lot.qty.is_sign_negative() == rem.is_sign_negative() {
                break;
            }
            let take = rem.abs().min(lot.qty.abs());
            let c = if lot.qty.is_sign_negative() { -take } else { take };
            let cost_part =
                if take == lot.qty.abs() { lot.cost_eur } else { fee_share(lot.cost_eur, take, lot.qty.abs()) };
            let open_fee =
                if take == lot.qty.abs() { lot.fee_eur } else { fee_share(lot.fee_eur, take, lot.qty.abs()) };
            let close_fee = fee_share(t.fee_eur, take, t.qty.abs());
            let close_value = mul(c, mul(t.price, fx));
            // long: sold now for close_value, bought for cost_part; short: sold for −cost_part,
            // bought back now for −close_value
            let (proceeds, cost) =
                if c.is_sign_negative() { (-cost_part, -close_value) } else { (close_value, cost_part) };
            let year = ys.year_of(t.filled_at);
            disposals.push(Disposal {
                tax_year: year,
                tax_year_label: ys.label(year),
                venue: t.venue,
                instrument: t.instrument,
                instrument_name: names.get(&t.instrument).cloned(),
                side: if c.is_sign_negative() { "short" } else { "long" }.into(),
                qty: take,
                lot_id: lot.lot_id,
                close_fill: t.fill_uuid,
                close_fill_id: t.fill_id.clone(),
                acquired_at: lot.opened_at,
                disposed_at: t.filled_at,
                holding_days: (t.filled_at - lot.opened_at).num_days(),
                ccy: ccy.clone(),
                open_price: lot.price,
                close_price: t.price,
                open_fx: lot.fx,
                close_fx: fx,
                proceeds_eur: proceeds,
                cost_eur: cost,
                open_fee_eur: open_fee,
                close_fee_eur: close_fee,
                fees_eur: open_fee + close_fee,
                gain_eur: proceeds - cost - open_fee - close_fee,
            });
            lot.qty -= c;
            lot.cost_eur -= cost_part;
            lot.fee_eur -= open_fee;
            rem += c;
            if lot.qty.is_zero() {
                match opts.method {
                    TaxLotMethod::Lifo => book_lots.pop_back(),
                    _ => book_lots.pop_front(),
                };
            }
        }
        if !rem.is_zero() {
            let fee = fee_share(t.fee_eur, rem.abs(), t.qty.abs());
            let cost = mul(rem, mul(t.price, fx));
            match (opts.method, book_lots.back_mut()) {
                (TaxLotMethod::Average, Some(pool)) => {
                    let q = pool.qty + rem;
                    let native = mul(pool.qty, pool.price) + mul(rem, t.price);
                    pool.cost_eur += cost;
                    pool.fee_eur += fee;
                    pool.price = div(native, q);
                    pool.fx = if native.is_zero() { fx } else { div(pool.cost_eur, native) };
                    pool.qty = q;
                }
                _ => book_lots.push_back(TaxLot {
                    lot_id: t.fill_uuid,
                    opened_at: t.filled_at,
                    qty: rem,
                    price: t.price,
                    fx,
                    cost_eur: cost,
                    fee_eur: fee,
                }),
            }
        }
    }
    let open_lots = lots.values().map(VecDeque::len).sum();
    disposals.retain(|d| opts.year.is_none_or(|y| d.tax_year == y));

    let mut by_inst: BTreeMap<(i32, InstrumentId), GainTotals> = BTreeMap::new();
    let mut by_year: BTreeMap<i32, GainTotals> = BTreeMap::new();
    for d in &disposals {
        by_inst.entry((d.tax_year, d.instrument)).or_default().add(d);
        by_year.entry(d.tax_year).or_default().add(d);
    }

    let mut seen = std::collections::HashSet::new();
    let mut income: BTreeMap<(i32, String), (CashClass, usize, Decimal)> = BTreeMap::new();
    for c in cash {
        let class = classify_cash(&c.kind);
        let year = ys.year_of(c.occurred_at);
        if class == CashClass::External
            || opts.year.is_some_and(|y| y != year)
            || !seen.insert((c.kind.clone(), c.reference.clone()))
        {
            continue;
        }
        let e = income.entry((year, c.kind.trim().to_ascii_lowercase())).or_insert((class, 0, ZERO));
        e.1 += 1;
        e.2 += c.amount_eur();
    }

    TaxReport {
        book: book.to_owned(),
        method: opts.method,
        year_start: ys,
        year: opts.year,
        base_currency: fa_core::BASE_CURRENCY.to_owned(),
        by_instrument: by_inst
            .into_iter()
            .map(|((y, i), totals)| InstrumentYear {
                tax_year: y,
                tax_year_label: ys.label(y),
                instrument: i,
                instrument_name: names.get(&i).cloned(),
                totals,
            })
            .collect(),
        by_year: by_year
            .into_iter()
            .map(|(y, totals)| YearTotal { tax_year: y, tax_year_label: ys.label(y), totals })
            .collect(),
        income: income
            .into_iter()
            .map(|((y, kind), (class, count, amount_eur))| IncomeItem {
                tax_year: y,
                tax_year_label: ys.label(y),
                kind,
                class,
                count,
                amount_eur,
            })
            .collect(),
        disposals,
        open_lots,
        notes: vec![
            format!(
                "Lots matched per (venue, instrument) by {} over the whole history; EUR at the rate booked on each fill.",
                opts.method.as_str()
            ),
            "gain = proceeds - cost - fees; fees allocated pro rata by quantity (open fee into the cost basis, close fee against the proceeds).".into(),
            "Short lots are realised when bought back (proceeds = opening sale, cost = buy-back).".into(),
            "Not applied (jurisdiction-specific, user decision): loss pools/carry-forward, holding-period exemptions, wash-sale/bed-and-breakfast matching, cross-venue pooling, deemed-return regimes.".into(),
            "Income lists non-capital cash movements (funding, borrow, fees, dividends/interest/other); deposits, withdrawals and payouts are excluded.".into(),
        ],
    }
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_owned() }
}

fn n(d: Decimal) -> String {
    d.normalize().to_string()
}

impl TaxReport {
    pub const CSV_HEADER: &'static str = "section,tax_year,venue,instrument,instrument_name,side,qty,acquired_at,disposed_at,holding_days,ccy,open_price,close_price,open_fx,close_fx,proceeds_eur,cost_eur,open_fee_eur,close_fee_eur,fees_eur,gain_eur,ref";

    /// Flat CSV in exact decimals; `section` is `disposal`, `instrument_year`, `year_total`,
    /// `income` or `note`. Columns that do not apply to a section are empty (for `income` the
    /// kind is in `side` and the amount in `gain_eur`; for totals the disposal count is in `ref`).
    pub fn to_csv(&self) -> String {
        let mut out = vec![Self::CSV_HEADER.to_owned()];
        let mut push = |cols: [String; 22]| out.push(cols.iter().map(|c| csv_field(c)).collect::<Vec<_>>().join(","));
        let e = String::new;
        for d in &self.disposals {
            push([
                "disposal".into(),
                d.tax_year_label.clone(),
                d.venue.to_string(),
                d.instrument.0.to_string(),
                d.instrument_name.clone().unwrap_or_default(),
                d.side.clone(),
                n(d.qty),
                d.acquired_at.to_rfc3339(),
                d.disposed_at.to_rfc3339(),
                d.holding_days.to_string(),
                d.ccy.clone(),
                n(d.open_price),
                n(d.close_price),
                n(d.open_fx),
                n(d.close_fx),
                n(d.proceeds_eur),
                n(d.cost_eur),
                n(d.open_fee_eur),
                n(d.close_fee_eur),
                n(d.fees_eur),
                n(d.gain_eur),
                d.close_fill_id.clone(),
            ]);
        }
        let totals = |section: &str, label: &str, inst: String, name: String, t: &GainTotals| {
            [
                section.into(),
                label.into(),
                e(),
                inst,
                name,
                e(),
                n(t.qty),
                e(),
                e(),
                e(),
                e(),
                e(),
                e(),
                e(),
                e(),
                n(t.proceeds_eur),
                n(t.cost_eur),
                e(),
                e(),
                n(t.fees_eur),
                n(t.gain_eur),
                format!("disposals={} gains={} losses={}", t.disposals, n(t.gains_eur), n(t.losses_eur)),
            ]
        };
        for r in &self.by_instrument {
            push(totals(
                "instrument_year",
                &r.tax_year_label,
                r.instrument.0.to_string(),
                r.instrument_name.clone().unwrap_or_default(),
                &r.totals,
            ));
        }
        for r in &self.by_year {
            push(totals("year_total", &r.tax_year_label, e(), e(), &r.totals));
        }
        for i in &self.income {
            let mut cols: [String; 22] = Default::default();
            cols[0] = "income".into();
            cols[1] = i.tax_year_label.clone();
            cols[5] = i.kind.clone();
            cols[20] = n(i.amount_eur);
            cols[21] = format!("count={} class={:?}", i.count, i.class).to_lowercase();
            push(cols);
        }
        for note in &self.notes {
            let mut cols: [String; 22] = Default::default();
            cols[0] = "note".into();
            cols[21] = note.clone();
            push(cols);
        }
        out.join("\n") + "\n"
    }

    /// JSON (numbers as in the other ledger reads: decimals serialised as their nearest `f64`).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

impl PgLedger {
    /// `ref.instrument.name` per instrument.
    pub async fn instrument_names(&self, ids: &[InstrumentId]) -> Result<BTreeMap<InstrumentId, String>, PgError> {
        let ids: Vec<Uuid> = ids.iter().map(|i| i.0).collect();
        let rows = sqlx::query("SELECT id, name FROM ref.instrument WHERE id = ANY($1)")
            .bind(&ids)
            .fetch_all(self.pool())
            .await?;
        Ok(rows.iter().map(|r| (InstrumentId(r.get::<Uuid, _>("id")), r.get::<String, _>("name"))).collect())
    }

    /// The realised-gains report of `book` ([`realised_gains`]); read-only.
    pub async fn realised_gains(&self, book: &str, opts: &TaxReportOptions) -> Result<TaxReport, PgError> {
        // a year's report needs all earlier fills for the cost basis, none after the year
        let before = opts.year.map(|y| opts.year_start.bounds(y).1.and_hms_opt(0, 0, 0).expect("midnight").and_utc());
        let trades = self.trade_events(book, before).await?;
        let cash = self.cash_events(book, before).await?;
        let ids: Vec<InstrumentId> =
            trades.iter().map(|t| t.instrument).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let names = self.instrument_names(&ids).await?;
        Ok(realised_gains(book, &trades, &cash, opts, &names))
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone};

    use super::*;
    use crate::accounting::replay;

    fn d(s: &str) -> Decimal {
        s.parse().unwrap()
    }

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2025, 12, 1, 10, 0, 0).unwrap()
    }

    fn trade(n: u128, days: i64, qty: &str, price: &str, ccy: &str, fx: &str, fee: &str) -> TradeEvent {
        TradeEvent {
            fill_uuid: Uuid::from_u128(n),
            fill_id: format!("f{n}"),
            intent_id: format!("i{n}"),
            venue: Uuid::from_u128(7),
            instrument: InstrumentId(Uuid::from_u128(42)),
            qty: d(qty),
            price: d(price),
            ccy: ccy.into(),
            fx_to_eur: d(fx),
            fx_source: "test".into(),
            fee_eur: d(fee),
            filled_at: t0() + Duration::days(days),
            attribution: Vec::new(),
            settle_native: false,
        }
    }

    fn opts(method: TaxLotMethod) -> TaxReportOptions {
        TaxReportOptions { method, ..Default::default() }
    }

    #[test]
    fn fifo_lifo_average_gains_with_fees_and_years() {
        // buy 10 @100 USD (fx 0.9, fee 1), buy 10 @120 (fx 0.8, fee 2), sell 15 @130 (fx 0.85, fee 3)
        let ts = vec![
            trade(1, 0, "10", "100", "USD", "0.9", "1"),
            trade(2, 10, "10", "120", "USD", "0.8", "2"),
            trade(3, 40, "-15", "130", "USD", "0.85", "3"),
        ];
        let names = BTreeMap::from([(InstrumentId(Uuid::from_u128(42)), "TEST".to_owned())]);
        let r = realised_gains("paper", &ts, &[], &opts(TaxLotMethod::Fifo), &names);
        assert_eq!(r.disposals.len(), 2);
        let (a, b) = (&r.disposals[0], &r.disposals[1]);
        assert_eq!((a.qty, a.lot_id, a.tax_year), (d("10"), Uuid::from_u128(1), 2026));
        assert_eq!(a.proceeds_eur, d("1105")); // 10·130·0.85
        assert_eq!(a.cost_eur, d("900"));
        assert_eq!(a.open_fee_eur, d("1"));
        assert_eq!(a.close_fee_eur, d("2")); // 3 · 10/15
        assert_eq!(a.gain_eur, d("202"));
        assert_eq!(a.holding_days, 40);
        assert_eq!(b.qty, d("5"));
        assert_eq!(b.cost_eur, d("480")); // 5·120·0.8
        assert_eq!(b.open_fee_eur, d("1"));
        assert_eq!(b.close_fee_eur, d("1"));
        assert_eq!(b.gain_eur, d("552.5") - d("480") - d("2"));
        assert_eq!(r.by_year[0].totals.gain_eur, a.gain_eur + b.gain_eur);
        assert_eq!(r.by_instrument[0].instrument_name.as_deref(), Some("TEST"));
        assert_eq!(r.open_lots, 1);

        let l = realised_gains("paper", &ts, &[], &opts(TaxLotMethod::Lifo), &names);
        assert_eq!(l.disposals[0].lot_id, Uuid::from_u128(2));
        assert_eq!(l.disposals[0].cost_eur, d("960"));
        assert_eq!(l.disposals[1].cost_eur, d("450"));

        let avg = realised_gains("paper", &ts, &[], &opts(TaxLotMethod::Average), &names);
        assert_eq!(avg.disposals.len(), 1);
        // pooled EUR cost (900 + 960) / 20 · 15 = 1395, pooled fee 3 · 15/20 = 2.25
        assert_eq!(avg.disposals[0].cost_eur, d("1395"));
        assert_eq!(avg.disposals[0].open_fee_eur, d("2.25"));
        assert_eq!(avg.disposals[0].gain_eur, d("1657.5") - d("1395") - d("2.25") - d("3"));

        // every method realises the same total once the position is flat
        let mut flat = ts.clone();
        flat.push(trade(4, 400, "-5", "90", "USD", "0.95", "0.5"));
        let totals: Vec<Decimal> = [TaxLotMethod::Fifo, TaxLotMethod::Lifo, TaxLotMethod::Average]
            .into_iter()
            .map(|m| realised_gains("paper", &flat, &[], &opts(m), &names).disposals.iter().map(|x| x.gain_eur).sum())
            .collect();
        assert!(totals.iter().all(|x| (*x - totals[0]).abs() < d("0.000000000001")), "{totals:?}");
        // the year filter keeps the cost basis of earlier years
        let y27 = realised_gains(
            "paper",
            &flat,
            &[],
            &TaxReportOptions { year: Some(2027), ..opts(TaxLotMethod::Fifo) },
            &names,
        );
        assert_eq!(y27.disposals.len(), 1);
        assert_eq!(y27.disposals[0].cost_eur, d("480"));
        assert_eq!(y27.by_year.len(), 1);
    }

    #[test]
    fn shorts_and_tax_year_start() {
        // sell short 4 @50 EUR on 2026-03-01, buy back 4 @40 on 2026-05-01
        let mut s = trade(1, 0, "-4", "50", "EUR", "1", "0.4");
        s.filled_at = Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap();
        let mut b = trade(2, 0, "4", "40", "EUR", "1", "0.4");
        b.filled_at = Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap();
        let o = TaxReportOptions { year_start: "04-06".parse().unwrap(), ..Default::default() };
        let r = realised_gains("paper", &[b.clone(), s.clone()], &[], &o, &BTreeMap::new());
        let x = &r.disposals[0];
        assert_eq!((x.side.as_str(), x.qty), ("short", d("4")));
        assert_eq!((x.proceeds_eur, x.cost_eur, x.fees_eur, x.gain_eur), (d("200"), d("160"), d("0.8"), d("39.2")));
        assert_eq!((x.tax_year, x.tax_year_label.as_str()), (2026, "2026/27"));
        assert_eq!(o.year_start.year_of(s.filled_at), 2025, "March is still the 2025/26 year");
        assert!("13-01".parse::<TaxYearStart>().is_err());
    }

    #[test]
    fn matches_ledger_realised_parts_and_reports_income() {
        let ts = vec![
            trade(1, 0, "3", "10", "USD", "0.9", "0.1"),
            trade(2, 1, "-5", "12", "USD", "0.92", "0.2"),
            trade(3, 2, "4", "11", "USD", "0.91", "0.3"),
            trade(4, 3, "-2", "13", "USD", "0.93", "0.05"),
            trade(4, 3, "-2", "13", "USD", "0.93", "0.05"), // duplicate
        ];
        for (m, tm) in [(LotMethod::Fifo, TaxLotMethod::Fifo), (LotMethod::Average, TaxLotMethod::Average)] {
            let st = replay(m, &ts, &[], None);
            let ledger: Decimal = st.closings.iter().map(|c| c.realised_eur + c.fx_eur).sum();
            let fees: Decimal = st.fees_eur;
            let r = realised_gains("paper", &ts, &[], &opts(tm), &BTreeMap::new());
            let before_fees: Decimal = r.disposals.iter().map(|x| x.gain_eur + x.fees_eur).sum();
            assert!((before_fees - ledger).abs() < d("0.000000000001"), "{m:?}: {before_fees} vs {ledger}");
            // the position is flat: every fee has been allocated to a disposal
            let allocated: Decimal = r.disposals.iter().map(|x| x.fees_eur).sum();
            assert!((allocated - fees).abs() < d("0.000000000001"), "{allocated} vs {fees}");
            assert_eq!(r.open_lots, 0);
        }
        let cash = |kind: &str, r: &str, amt: &str| CashEvent {
            kind: kind.into(),
            reference: r.into(),
            amount: d(amt),
            ccy: "USD".into(),
            fx_to_eur: d("0.9"),
            occurred_at: t0(),
        };
        let cs = vec![
            cash("deposit", "a", "1000"),
            cash("funding", "b", "-2"),
            cash("funding", "c", "1"),
            cash("dividend", "d", "10"),
            cash("dividend", "d", "10"),
        ];
        let r = realised_gains("paper", &[], &cs, &opts(TaxLotMethod::Fifo), &BTreeMap::new());
        let get = |k: &str| r.income.iter().find(|i| i.kind == k).unwrap();
        assert_eq!((get("funding").count, get("funding").amount_eur), (2, d("-0.9")));
        assert_eq!((get("dividend").count, get("dividend").amount_eur), (1, d("9")));
        assert!(r.income.iter().all(|i| i.kind != "deposit"));
        assert_eq!(r.income[0].tax_year, 2025);
    }

    #[test]
    fn csv_and_json_exports() {
        let ts = vec![trade(1, 0, "2", "10", "EUR", "1", "0.2"), trade(2, 5, "-2", "11", "EUR", "1", "0.2")];
        let r = realised_gains("paper", &ts, &[], &opts(TaxLotMethod::Fifo), &BTreeMap::new());
        let csv = r.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], TaxReport::CSV_HEADER);
        let cols = TaxReport::CSV_HEADER.split(',').count();
        assert!(lines.iter().all(|l| l.split(',').count() == cols || l.contains('"')), "{csv}");
        assert!(lines[1].starts_with("disposal,2025,"));
        assert!(lines[1].contains(",1.6,f2"), "gain 22 − 20 − 0.4 = 1.6: {}", lines[1]);
        assert!(csv.contains("\nyear_total,2025,"));
        let j = r.to_json();
        assert_eq!(j["method"], "fifo");
        assert_eq!(j["disposals"][0]["gain_eur"], 1.6);
        assert_eq!(j["by_year"][0]["disposals"], 1);
        let back: TaxReport = serde_json::from_value(j).unwrap();
        assert_eq!(back.disposals[0].gain_eur, d("1.6"));
    }

    /// Read-only against the live book (needs `DATABASE_URL`).
    #[tokio::test]
    async fn live_report_reads_and_reconciles_with_lots() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pg = PgLedger::connect(&url).await.unwrap();
        let r = pg.realised_gains("paper", &TaxReportOptions::default()).await.unwrap();
        let trades = pg.trade_events("paper", None).await.unwrap();
        let st = replay(LotMethod::Fifo, &trades, &[], None);
        let ledger: Decimal = st.closings.iter().map(|c| c.realised_eur + c.fx_eur).sum();
        let before_fees: Decimal = r.disposals.iter().map(|x| x.gain_eur + x.fees_eur).sum();
        assert!((before_fees - ledger).abs() < d("0.000001"), "{before_fees} vs {ledger}");
        assert_eq!(r.disposals.len(), st.closings.len());
        assert!(r.to_csv().lines().count() > r.disposals.len());
    }
}
