-- Review 2026-11-15 §3.2 item 6: the payout quota is per book, not global. Existing rows (all
-- written for the single paper book so far) keep their data under book = 'paper'. Idempotent:
-- it may be applied by hand before control-api records it.

alter table ledger.payout_quota add column if not exists book text not null default 'paper';
alter table ledger.payout_quota drop constraint if exists payout_quota_pkey;
alter table ledger.payout_quota add constraint payout_quota_pkey primary key (book, day);
