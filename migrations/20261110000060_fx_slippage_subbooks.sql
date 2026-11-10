-- FX for accounting, realised slippage per fill and restart-safe strategy sub-books (D-061;
-- specs 12 §6, 40 §6.3/§6.5, 42 §6). Idempotent, so a manually applied copy does not break
-- sqlx::migrate on the next control-api start.

-- ledger.fill already carries arrival_price (spec 42 §5); the signed slippage of the fill against
-- it (basis points of the arrival price, positive = paid) comes from OrderEvent.slippage_bps (D-051).
alter table ledger.fill add column if not exists slippage_bps numeric(38, 18);

-- Each strategy's attributed sub-book (active: positions, cash, realised; shadow: virtual
-- exposure), its bounded P&L history and realised slippage, written by the risk-engine on every
-- flush and restored at start so the per-strategy drawdown tracker survives restarts.
alter table ledger.strategy_state add column if not exists book jsonb;

-- ref.fx_rate rows are keyed (ccy, ts, source_id) = (quote currency, as_of, source) with EUR as
-- the implicit base (rate_to_eur): that primary key is the idempotency key of the ECB upsert.
-- source_id references catalog.source, so the accounting source gets its catalog row here
-- (spec 12 §7 fx.accounting_source = ecb_daily; the `ecb` SDMX source stays a separate entry).
insert into catalog.source (id, name, category, capabilities, asset_classes, transport, auth, status, notes)
values ('ecb_daily', 'ECB euro foreign exchange reference rates (daily XML)', 'fx', '{data}', '{fx}', '{rest}', 'none', 'live',
        'eurofxref-daily.xml fetched by risk-engine into ref.fx_rate (spec 12 §6, spec 42 §6); official EUR conversion for accounting.')
on conflict (id) do nothing;

-- as-of lookups: latest rate with ts <= t per currency
create index if not exists fx_rate_ccy_ts_idx on ref.fx_rate (ccy, ts desc);
