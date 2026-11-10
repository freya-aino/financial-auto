-- Ledger read paths added with persistence (D-047): book rebuilds replay fills in time order,
-- P&L filters by range and the UI lists recent reconciliation runs.
create index if not exists fill_filled_at_idx on ledger.fill (filled_at);
create index if not exists order_intent_book_idx on ledger.order_intent (book);
create index if not exists cash_movement_book_idx on ledger.cash_movement (book);
create index if not exists position_snapshot_book_as_of_idx on ledger.position_snapshot (book, as_of desc);
create index if not exists reconciliation_run_started_at_idx on ledger.reconciliation_run (started_at desc);
