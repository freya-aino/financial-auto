-- Repair Kraken instruments provisioned before `go/pkg/refdata` had a `kraken` hint: they were
-- created with asset_class/kind `unknown` and no currencies. Same rule as
-- `refdata.KrakenPair`/`Classify`: wsname `XBT/EUR`, legacy `XXBTZEUR`/`XETHXXBT`, altname
-- `XBTEUR`/`SOLUSDT`; Kraken codes XBT → BTC, XDG → DOGE; two fiat legs → fx pair.
--
-- Point in time (D-067): `ref.instrument` holds one row per id (primary key `id`, no version
-- table), and the class is an inference from the symbol, not a real-world change, so this is a
-- correction in place, like the Yahoo repair in 20261111000010 §6. `valid_from` and
-- `recorded_at` stay as they are; bumping `recorded_at` would hide the instrument from
-- `fa_warehouse::asof::instrument` reads with an earlier `known_at`. Every corrected row gets an
-- `audit.log` entry with the old and new values. Only provisional rows still `unknown` are
-- touched (an operator-activated or already classified instrument wins), so a rerun is a no-op.
--
-- A migration cannot publish `ref.changed.instrument`: risk-engine reloads its RefTable at start,
-- every `refdata_refresh_secs` and on the next `ref.changed.*` from any source; the ingestor's
-- resolver caches ids only, so it needs nothing.
do $$
declare
    r     record;
    m     text[];
    b     text;
    q     text;
    sfx   text;
    cls   text;
    knd   text;
    fiat  text[] := array['EUR', 'USD', 'GBP', 'JPY', 'CAD', 'CHF', 'AUD'];
begin
    for r in
        select distinct on (i.id) i.id, upper(btrim(l.venue_symbol)) as sym, i.asset_class, i.kind, i.base_ccy, i.quote_ccy
          from ref.instrument i
          join ref.listing l on l.instrument_id = i.id
         where l.source_id = 'kraken' and i.status = 'provisional'
           and i.asset_class = 'unknown' and i.kind = 'unknown'
         order by i.id, l.valid_from desc
    loop
        b := null;
        q := null;
        m := regexp_match(r.sym, '^([A-Z0-9]{2,10})/([A-Z0-9]{2,10})$');
        if m is null then
            m := regexp_match(r.sym, '^[XZ]([A-Z0-9]{3})[XZ]([A-Z0-9]{3})$');
        end if;
        if m is not null then
            b := m[1];
            q := m[2];
        else
            foreach sfx in array array['USDT', 'USDC', 'EUR', 'USD', 'GBP', 'JPY', 'CAD', 'CHF', 'AUD', 'XBT', 'ETH', 'DAI'] loop
                if right(r.sym, length(sfx)) = sfx
                   and left(r.sym, length(r.sym) - length(sfx)) ~ '^[A-Z0-9]{2,10}$' then
                    b := left(r.sym, length(r.sym) - length(sfx));
                    q := sfx;
                    exit;
                end if;
            end loop;
        end if;
        if b is null then
            -- unrecognised shape: still a Kraken spot instrument (the hint), currencies unknown
            cls := 'crypto';
            knd := 'crypto_spot';
        else
            b := case b when 'XBT' then 'BTC' when 'XDG' then 'DOGE' else b end;
            q := case q when 'XBT' then 'BTC' when 'XDG' then 'DOGE' else q end;
            if b = any (fiat) and q = any (fiat) then
                cls := 'fx';
                knd := 'fx_pair';
            else
                cls := 'crypto';
                knd := 'crypto_spot';
            end if;
        end if;
        update ref.instrument
           set asset_class = cls, kind = knd,
               base_ccy = coalesce(base_ccy, b), quote_ccy = coalesce(quote_ccy, q)
         where id = r.id;
        insert into audit.log (actor, action, subject, payload)
        values ('system:migration', 'refdata.reclassify', r.id::text,
                jsonb_build_object(
                    'migration', '20261113000020_refdata_kraken_class',
                    'source_id', 'kraken', 'venue_symbol', r.sym,
                    'old', jsonb_build_object('asset_class', r.asset_class, 'kind', r.kind,
                                              'base_ccy', r.base_ccy, 'quote_ccy', r.quote_ccy),
                    'new', jsonb_build_object('asset_class', cls, 'kind', knd,
                                              'base_ccy', coalesce(r.base_ccy, b), 'quote_ccy', coalesce(r.quote_ccy, q))));
    end loop;
end $$;
