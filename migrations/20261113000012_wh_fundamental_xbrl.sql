-- wh.fundamental fed by SEC EDGAR XBRL company facts (spec 20 §5.1, §6.1; spec 10/11 sec_edgar).
-- Facts are keyed by the issuer's CIK (no ref.entity/ref.instrument row exists for most filers;
-- instrument_id is filled when ref.identifier has scheme 'cik' for it). Idempotent.

alter table wh.fundamental add column if not exists issuer_cik    text;  -- 10 digits, zero padded
alter table wh.fundamental add column if not exists period_start  date;  -- null for instant facts
alter table wh.fundamental add column if not exists period_end    date;
alter table wh.fundamental add column if not exists form          text;  -- 10-K, 10-Q, ...
alter table wh.fundamental add column if not exists accession     text;
alter table wh.fundamental add column if not exists fiscal_year   text;  -- fy of the reporting filing
alter table wh.fundamental add column if not exists fiscal_period text;  -- fp of the reporting filing

alter table wh.fundamental drop constraint if exists fundamental_check;
alter table wh.fundamental drop constraint if exists fundamental_subject_check;
alter table wh.fundamental add constraint fundamental_subject_check
    check (instrument_id is not null or entity_id is not null or issuer_cik is not null);

-- Natural key: subject (CIK first, so a later cik → instrument mapping does not duplicate rows),
-- field, unit, period, vintage, source. A revision is a new vintage, never an overwrite.
drop index if exists wh.fundamental_natural_key;
create unique index if not exists fundamental_natural_key_v2
    on wh.fundamental (coalesce('cik:' || issuer_cik, instrument_id::text, entity_id::text),
                       field, coalesce(unit, ''), period, vintage, source_id);

-- As-of reads: latest vintage <= t per subject, field and period.
create index if not exists fundamental_asof_idx
    on wh.fundamental (issuer_cik, field, period, vintage desc);
