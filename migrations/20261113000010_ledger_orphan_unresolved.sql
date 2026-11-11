-- Orphan fills without an order anywhere (spec 42 §8, round 3): never store a made-up
-- instrument or assume the fee is in EUR. Idempotent (safe to apply by hand before control-api's
-- sqlx migrator runs it).
--
-- instrument_status: resolved   — instrument_id known (from the fill or its venue symbol)
--                    unresolved — no listing matches the venue symbol (or none was given)
--                    ambiguous  — several instruments list that symbol at the fill time
-- fee / fee_ccy are the venue's fee in its own currency; fee_eur is its EUR value when a rate
-- was known (NULL otherwise, converted when the orphan is booked). ccy / fx_rate_eur are NULL
-- while the fill currency is unknown.

alter table ledger.orphan_fill alter column instrument_id drop not null;
alter table ledger.orphan_fill alter column ccy drop not null;
alter table ledger.orphan_fill alter column fx_rate_eur drop not null;
alter table ledger.orphan_fill add column if not exists venue_symbol text;
alter table ledger.orphan_fill add column if not exists instrument_status text not null default 'resolved';
alter table ledger.orphan_fill add column if not exists fee_eur numeric(38, 18);

-- earlier rows: fee was booked as EUR; a nil instrument meant "unknown"
update ledger.orphan_fill set fee_eur = fee where fee_eur is null and upper(fee_ccy) = 'EUR';
update ledger.orphan_fill set instrument_id = null, instrument_status = 'unresolved'
 where instrument_id = '00000000-0000-0000-0000-000000000000';

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'orphan_fill_instrument_status_chk') then
        alter table ledger.orphan_fill add constraint orphan_fill_instrument_status_chk check (
            instrument_status in ('resolved', 'unresolved', 'ambiguous')
            and ((instrument_status = 'resolved') = (instrument_id is not null)));
    end if;
end $$;

create index if not exists orphan_fill_unresolved_idx on ledger.orphan_fill (venue_symbol)
    where instrument_status <> 'resolved';
