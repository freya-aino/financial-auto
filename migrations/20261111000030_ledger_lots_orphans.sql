-- Ledger derived accounting and orphan fills (spec 42 §5, §6, §8). Idempotent, so a manually
-- applied copy does not break sqlx::migrate on the next control-api start.

-- Fills whose client_order_id is unknown (spec 42 §8): stored here instead of failing, until an
-- operator books them against the (now known) order or dismisses them. qty is signed (+ buy).
create table if not exists ledger.orphan_fill (
    id              uuid primary key,
    book            text,
    venue_id        uuid not null,
    client_order_id text not null,
    venue_fill_id   text not null,
    instrument_id   uuid not null,
    qty             numeric(38, 18) not null,
    price           numeric(38, 18) not null,
    ccy             text not null,
    fee             numeric(38, 18) not null,
    fee_ccy         text not null,
    fx_rate_eur     numeric(38, 18) not null,
    fx_source       text not null,
    arrival_price   numeric(38, 18),
    slippage_bps    numeric(38, 18),
    strategy_ids    jsonb not null default '[]'::jsonb,
    filled_at       timestamptz not null,
    received_at     timestamptz not null default now(),
    status          text not null default 'open' check (status in ('open', 'booked', 'dismissed')),
    resolved_at     timestamptz,
    resolved_by     text,
    resolution      text,
    unique (client_order_id, venue_fill_id)
);
create index if not exists orphan_fill_open_idx on ledger.orphan_fill (venue_id, received_at) where status = 'open';

-- Tax lots (ledger.lot_method = fifo | average), a cache derived from ledger.fill by
-- fa_ledger::pg::PgLedger::rebuild_lots: each rebuild replaces a (book, method) wholesale, so the
-- tables can always be dropped and rebuilt from events. qty is signed (+ long, − short); prices
-- are in the lot currency, fx_rate_eur is EUR per unit of it at the opening fill (cost-weighted
-- for average-cost lots). lot_id is the ledger.fill id of the opening fill.
create table if not exists ledger.lot (
    book          text not null,
    method        text not null check (method in ('fifo', 'average')),
    lot_id        uuid not null,
    venue_id      uuid not null,
    instrument_id uuid not null,
    opened_at     timestamptz not null,
    open_qty      numeric(38, 18) not null,
    remaining_qty numeric(38, 18) not null,
    price         numeric(38, 18) not null,
    ccy           text not null,
    fx_rate_eur   numeric(38, 18) not null,
    primary key (book, method, lot_id)
);
create index if not exists lot_open_idx on ledger.lot (book, method, venue_id, instrument_id) where remaining_qty <> 0;

-- One row per (lot, closing fill): realised P&L in the lot currency and in EUR, the latter split
-- into the price part (at the closing FX rate) and the FX part (cost basis revalued).
create table if not exists ledger.lot_closing (
    book            text not null,
    method          text not null check (method in ('fifo', 'average')),
    lot_id          uuid not null,
    close_fill_id   uuid not null,
    venue_id        uuid not null,
    instrument_id   uuid not null,
    opened_at       timestamptz not null,
    closed_at       timestamptz not null,
    qty             numeric(38, 18) not null,
    open_price      numeric(38, 18) not null,
    close_price     numeric(38, 18) not null,
    ccy             text not null,
    open_fx         numeric(38, 18) not null,
    close_fx        numeric(38, 18) not null,
    realised_native numeric(38, 18) not null,
    realised_eur    numeric(38, 18) not null,
    fx_eur          numeric(38, 18) not null,
    primary key (book, method, lot_id, close_fill_id)
);
create index if not exists lot_closing_closed_at_idx on ledger.lot_closing (book, method, closed_at);

-- Bookkeeping of the last rebuild per (book, method).
create table if not exists ledger.lot_rebuild (
    book       text not null,
    method     text not null,
    rebuilt_at timestamptz not null default now(),
    fills      bigint not null,
    lots       bigint not null,
    closings   bigint not null,
    primary key (book, method)
);

-- Every fill with its native and EUR values, FX source and the validity time of the rate used
-- (spec 42 §9): the rate's own timestamp from ref.fx_rate when the source is a stored series,
-- the fill time for EUR / fixed conversions.
create or replace view ledger.fill_valued as
select f.id,
       oi.book,
       f.client_order_id,
       f.venue_fill_id,
       vo.venue_id,
       oi.instrument_id,
       case when oi.side = 'sell' then -f.qty else f.qty end as signed_qty,
       f.price,
       f.ccy,
       f.qty * f.price                                       as notional_native,
       f.qty * f.price * f.fx_rate_eur                       as notional_eur,
       f.fee                                                 as fee_eur,
       f.fx_rate_eur,
       f.fx_source,
       coalesce(r.ts, case when upper(f.ccy) = 'EUR' or f.fx_source in ('eur', 'fixed') then f.filled_at end) as fx_as_of,
       f.filled_at
from ledger.fill f
join ledger.venue_order vo on vo.client_order_id = f.client_order_id
join ledger.order_intent oi on oi.id = vo.intent_id
left join lateral (
    select x.ts from ref.fx_rate x
    where x.ccy = f.ccy and x.source_id = f.fx_source and x.ts <= f.filled_at
    order by x.ts desc limit 1
) r on true;

-- As-of reads by venue / book.
create index if not exists venue_order_intent_idx on ledger.venue_order (intent_id);
create index if not exists cash_movement_book_occurred_idx on ledger.cash_movement (book, occurred_at);
