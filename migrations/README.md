# migrations

PostgreSQL warehouse + ledger schema, in sqlx format (`<timestamp>_<name>.sql`, D-013).

**Status:** 33 migrations (schemas `catalog`, `ref`, `ingest`, `wh`, `ledger`, `audit`, `validation`, `ops`, `acct`, `graph`). control-api applies them at startup (`sqlx::migrate!`); later files are idempotent (`if not exists`) so a manually applied copy does not break that run. The initial six were verified 2026-11-06 on a scratch `postgres:17-alpine` (23 tables).

| File | Schema | Spec |
|---|---|---|
| `…01_schemas.sql` | `catalog`, `ref`, `ingest`, `wh`, `ledger`, `audit` | 20 §5.4 |
| `…02_catalog.sql` | `catalog.source` | 11 |
| `…03_reference.sql` | `ref.entity/instrument/identifier/venue/listing/contract_spec/calendar_session/corporate_action/fx_rate` | 12 |
| `…04_ingest.sql` | `ingest.quarantine/backfill_cursor/connector_run`, `wh.macro_observation` | 10, 20 |
| `…05_ledger.sql` | `ledger.order_intent/venue_order/fill/fill_attribution/cash_movement/position_snapshot/payout_quota/reconciliation_run` | 42 |
| `…06_audit.sql` | `audit.log` | 42 |
| `20261110000010_ledger_indexes.sql`, `…11_cash_movement_book_key.sql` | ledger read-path indexes, cash movement idempotency key | 42, D-047 |
| `20261110000020_strategy_state.sql` | `ledger.strategy_state` (strategy lifecycle `active/paused/flattening/shadow`, `prev_status`, shadow progress, last gate) | 40 §3/§6.6, D-051 |
| `20261110000030_validation.sql` | schema `validation`: `holdout` (immutable by trigger), `trial` (trial registry, block returns), `generation_stats` | 41 §5/§6.4/§6.5, D-052 |
| `20261110000040_quarantine_review.sql` | `ingest.quarantine.instrument_id/resolved_at/resolution` (+ backfill from `ref.listing`), search indexes on `ref.instrument.name`, `ref.identifier.value`, `ref.listing.venue_symbol` | 10 §5.3, 12 §6, D-054 |
| `20261110000050_holdout_lock.sql` | stable `validation.holdout.id`, `validation.trial.holdout_id` + kind index (holdout evaluations) | 41 §6.4, D-059 |
| `20261110000060_fx_slippage_subbooks.sql` | `ledger.fill.slippage_bps`, `ledger.strategy_state.book` (restart-safe sub-books), `ecb_daily` catalog row, `ref.fx_rate` as-of index | 12 §6, 40, 42 §6, D-061 |
| `20261111000010_refdata_asof.sql` | bitemporal-lite columns on `ref.entity/venue/calendar_session/fx_rate`, corporate-action lifecycle + symbol changes + natural key, `ref.instrument_relation`, as-of indexes, reference venues, repair of Yahoo fx/crypto classification, `wh.fundamental`, macro as-of index | 12, 20 |
| `20261111000030_ledger_lots_orphans.sql` | `ledger.orphan_fill`, tax-lot cache `ledger.lot/lot_closing/lot_rebuild`, view `ledger.fill_valued` (native + EUR + FX source/time per fill), as-of indexes | 42 |
| `20261111000045_wh_filings.sql` | `wh.filing` (FilingEvent metadata, idempotent on source + accession) | 10, 20 |
| `20261111000050_strategy_pipeline.sql` | `ledger.strategy_state`: statuses `candidate/holdout_passed/retired`, `pipeline` JSONB, `last_reason` | 35 §6.4, 40 §6.6 |
| `20261111000051_risk_state.sql` | `ledger.risk_state` (restart-safe halt state machine, calibration history, tournament records) | 40 §8 |
| `20261112000010_ledger_settlement_payout.sql` | `ledger.venue_settlement` (settlement `eur`/`native` + margin model per venue; paper venue seeded `eur`), `ledger.payout_quota.rule` (rule + inputs that set the quota); idempotent | 42 §10 Q1/Q5 |
| `20261112000020_generation_stats_effective_trials.sql` | `validation.generation_stats.effective_trials` (N_eff behind the deflated Sharpe) | 41 §6.5, D-071 |
| `20261113000010_ledger_orphan_unresolved.sql` | `ledger.orphan_fill`: nullable `instrument_id`/`ccy`/`fx_rate_eur`, `instrument_status` (`resolved`/`unresolved`/`ambiguous`), `venue_symbol`, `fee_eur` (fee kept in `fee_ccy`); nil instruments migrated to `unresolved`; idempotent | 42 §8 |
| `20261113000011_ledger_fill_restatement.sql` | `ledger.fill_restatement` (old/new currency, rate and source per restated fill, batch, actor, reason; `fa_ledger::restate`); idempotent | 42 §6 |
| `20261113000012_wh_fundamental_xbrl.sql` | `wh.fundamental` keyed by issuer CIK (`issuer_cik`, `period_start/end`, `form`, `accession`, `fiscal_year/period`), natural key incl. unit, as-of index; written by the ingestor from SEC XBRL company facts; idempotent | 20 §5.1 |
| `20261113000013_ops.sql` | schema `ops`: `alert_event` (alert transitions), `job_run` (JOB_RESULTS history), `config_snapshot` (redacted config per digest); written by control-api via `fa_warehouse::ops`; idempotent | 20 §5.4, 60 |
| `20261113000020_refdata_kraken_class.sql` | corrects Kraken instruments provisioned as `unknown` to `crypto`/`crypto_spot` (or `fx_pair` for two fiat legs) with base/quote currencies (XBT→BTC, XDG→DOGE), in place with one `audit.log` row each; only provisional `unknown/unknown` rows of `source_id = 'kraken'`; idempotent | 12 §6 |
| `20261113000040_holdout_embargo.sql` | `validation.holdout.embargo_ms` (embargo the holdout was locked with; NULL = timescale default); idempotent | 41 §6.4 |
| `20261114030000_ingest_runtime_catalog_adjusted.sql` | `catalog.source.rate_limit` / `config` (JSONB runtime overrides, `ctl.ingest.configure` / `PATCH /api/sources/{id}`), `status_note` / `status_changed_at` (auto-disable reason); `wh.adjusted_bar_1d` (materialised adjusted daily bars, both modes, action-set digest; `fa_warehouse::adjusted`); idempotent; verified on a scratch `postgres:17-alpine` with every migration applied | 11 §4/§7, 12 §7 (retired → catalog / fa-warehouse READMEs) |
| `20261114050000_encoder_generation.sql` | `validation.encoder_generation` (one row per trainer generation run: `GenPhase`, status, selected snapshot, full `record` JSONB) and `validation.data_window` (train / validation boundaries and excluded locked holdouts per generation); written by the trainer, read by control-api `/api/generations`; idempotent | 36 §4.1/§5, D-091 |
| `20261114070000_risk_history.sql` | `ledger.allocation_snapshot` (active allocator weights / sleeves per book), `ledger.portfolio_target` (netted targets raw → capped → gated per decision tick), `ledger.promotion_review` (promotion-gate decisions, kept as audit trail); written by the risk-engine (`history.rs`), snapshots pruned after `[history] retention_days`; idempotent | 40 §5, D-094 |
| `20261114090000_decoder_output.sql` | `wh.decoder_output` (every `DecoderOutput` control-api receives: decoder, genome, family, snapshot, timescale, nullable `instrument_id`, `as_of`, `output` JSONB with refs / quantiles / analogs; index on snapshot × timescale × instrument × `as_of`; 30-day retention by control-api); idempotent | 51 §6.3, D-098 |
| `20261115100000_accounting.sql` | schema `acct` (fa-accounting, D-104): `account` (chart of accounts with organisational segments and external refs), `entry` (append-only journal, unique `entry_ref`), `posting` (`NUMERIC(38,18)` amounts); balance checks live in the crate | 00, D-104 |
| `20261115110000_payout_quota_book.sql` | `ledger.payout_quota.book` (quota per book, not global; existing rows become `paper`), primary key → (book, day); idempotent | 42 §10 (review 2026-11-15 §3.2) |
| `20261115120000_graph.sql` | schema `graph` (ADR 0010 second half, D-108): 25 typed tables replacing the Helix model-state graph — reference mirror, snapshots/spaces/adapters, regions/transitions + `TransitionVersion`/`RegionStatusVersion`, lineage/hierarchy, region sets/kernels/cross priors/episodes, string-keyed `TrainedOn`/`FeedbackFrom` links; ids from the shared `graph.node_id` sequence; written through `fa_hub::graph::PgGraph` | 20, ADR 0010, D-108 |

Apply with `just migrate`, which needs `sqlx-cli`: `cargo install sqlx-cli --no-default-features --features rustls,postgres`.
