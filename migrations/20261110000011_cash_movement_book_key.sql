-- D-047 fix: the idempotency key of a cash movement must include the book, otherwise a second
-- book can never record its own "initial" deposit. NULLS NOT DISTINCT so a missing venue still dedups.
-- Idempotent, so a manually applied copy does not break sqlx::migrate on the next control-api start.
alter table ledger.cash_movement drop constraint if exists cash_movement_venue_id_kind_ref_key;
do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'cash_movement_book_venue_kind_ref_key') then
        alter table ledger.cash_movement
            add constraint cash_movement_book_venue_kind_ref_key unique nulls not distinct (book, venue_id, kind, ref);
    end if;
end $$;
