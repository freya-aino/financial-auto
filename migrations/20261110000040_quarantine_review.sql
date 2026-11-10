-- Quarantine review loop (spec 10 §5.3, spec 12 §6; D-054). The provisional instrument a
-- quarantine row stands for was only recoverable through ref.listing; record it directly so the
-- UI can address a row by instrument id, and keep the resolution for audit. Idempotent.
alter table ingest.quarantine add column if not exists instrument_id uuid references ref.instrument (id);
alter table ingest.quarantine add column if not exists resolved_at timestamptz;
alter table ingest.quarantine add column if not exists resolution jsonb;

update ingest.quarantine q
   set instrument_id = l.instrument_id
  from ref.listing l
 where q.instrument_id is null
   and l.source_id = q.source_id and l.venue_symbol = q.venue_symbol
   and now() >= l.valid_from and now() < l.valid_to;

create index if not exists quarantine_status_idx on ingest.quarantine (status, last_seen desc);
create index if not exists quarantine_instrument_idx on ingest.quarantine (instrument_id);

-- Instrument search (GET /api/instruments?q=) hits names, identifiers and venue symbols.
create index if not exists instrument_name_lower_idx on ref.instrument (lower(name));
create index if not exists identifier_value_lower_idx on ref.identifier (lower(value));
create index if not exists listing_symbol_lower_idx on ref.listing (lower(venue_symbol));
