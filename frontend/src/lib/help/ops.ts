// Ops pages: / (overview), /alerts, /history, /refdata, /infra, /growth.
export const ops: Record<string, string> = {
	// ---- Ops overview (/) ----
	'ops.services':
		'Every service announces itself on the bus (svc.heartbeat.>) every few seconds with its version and counters; a row goes stale after 10 s without one. The buttons start, stop or restart the service’s Docker container through control-api.',
	'ops.jobs':
		'Background jobs (evolution, ingest backfills) and the GPU arbiter that schedules them: jobs lease GPU memory in priority-class order, and a higher class can preempt a running lower-class job, which then shows as preempted until it resumes.',
	'ops.bustap':
		'A live sample of raw bus traffic matching the subject patterns in the box, with per-subject rates over the last 60 s. For watching, not history: the server drops frames beyond 200 msg/s per socket and only the newest 200 rows are kept.',
	'ops.streams':
		'The bus’s JetStream streams: which subjects each one captures, its retention, how full it is, and every consumer’s backlog. Streams are bounded transport, not a store of record — old messages are discarded by age or byte cap.',
	'ops.streams.consumers':
		'pending is how many stream messages a consumer has not received yet; ack is delivered but not yet acknowledged. Both at zero means the consumer is caught up; growing numbers mean its service is slow, down or failing and being redelivered.',
	'ops.streams.cap':
		'Usage of the stream’s byte cap; at the cap the oldest messages are discarded. “planned … not applied” means the stream already held more than a newly planned cap, so the old limit was kept — shrinking it would purge messages at once.',
	'ops.sources':
		'Every data and execution connector in the catalog. Status (live, stub, disabled) comes from the Postgres catalog; health is published by each running connector every 5 s. Enable/disable persists the catalog status; start/stop controls the running connector.',
	'ops.sources.limiter':
		'How much of the source’s rate limit the connector is using, from its request limiter. Near 100% the connector throttles itself to stay inside the provider’s quota, so requests queue up and ingest for this source lags.',
	'ops.instruments':
		'Search instruments by name, identifier or venue symbol, straight from the reference data in Postgres. Point-in-time views and manual corrections live on the Refdata page.',
	'ops.quarantine':
		'Data for a (source, symbol) pair the reference data does not know creates a provisional instrument here instead of guessing. Resolve each row: activate it with real identifiers, merge it into the instrument it duplicates, or ignore it. Resolutions are audited.',
	'ops.audit':
		'Every mutation made through control-api — by an operator (actor ui) or by the system — read back from the audit.log table in Postgres. Click a row to expand its payload.',
	'ops.runtime-config':
		'Settings that apply without a restart, each bounded by the server: the UI’s poll interval and the WebSocket flush rates. A change takes effect in-process, is persisted to config/runtime.toml and is audited.',

	// ---- /alerts ----
	'ops.alerts.pipeline':
		'Health of the alerting chain: vmalert evaluates rules over VictoriaMetrics and posts transitions to control-api, which publishes them on ops.alert. The Watchdog rule fires forever by design — if it goes stale, the chain is broken and alerts may be missing.',
	'ops.alerts.active':
		'Alerts firing right now, most severe first. vmalert re-sends an active alert on every evaluation (counted as refreshes); one that stops being re-sent expires and drops off this list.',
	'ops.alerts.recent':
		'The latest firing and resolved transitions, held in control-api memory — this list starts empty on every control-api restart. The History panel is the durable record.',
	'ops.alerts.history':
		'Alert transitions stored in Postgres (ops.alert_event), so they survive restarts. Pick a window, filter by alert name, or click a name in the table to filter on it.',
	'ops.alerts.live':
		'The ops.alert bus subject as it arrives over the WebSocket. Only state changes are published — an alert starting to fire or resolving — never repeats of an alert that is still firing.',

	// ---- /history ----
	'ops.history.summary':
		'Finished jobs per kind and status over the chosen window, from the ops.job_run table control-api fills from every jobs.result message on the bus. Click a kind to filter the runs panel.',
	'ops.history.runs':
		'One row per stored job result: outcome, worker, GPU time and outputs. Unlike the live Jobs & GPU panel on the Ops overview, these rows are in Postgres and survive restarts. Click a row for the full result payload.',
	'ops.history.snapshots':
		'Each service’s config is checked every 5 minutes but a row is only written when its digest changes, so this is the history of configuration changes and who made them. Values are redacted; tick “include config” and open a row to see a section.',

	// ---- /refdata ----
	'ops.refdata.bitemporal':
		'Reference data is bitemporal: “at” is the real-world instant a fact was true, “known at” is when the system knew it — so you can reproduce exactly what was believed at any past moment. Empty means now; every panel answers as of these two instants.',
	'ops.refdata.resolve':
		'Which instrument a venue or source symbol pointed to at the chosen instants: one match, none, or ambiguous with candidates. Symbols get reused, so the same ticker can resolve to different instruments at different times.',
	'ops.refdata.asof':
		'The instrument’s record as it stood at the chosen instants — status, listings, identifiers and the contract specs valid then. Search by name or load a UUID directly.',
	'ops.refdata.calendar':
		'Trading sessions per venue: basis “table” rows were materialised into Postgres, otherwise sessions are computed from the venue’s built-in rule. Materialise writes the rule’s sessions into the table for the chosen dates; the write is audited.',
	'ops.refdata.actions':
		'Splits, dividends and other corporate actions; prices are adjusted on read from the ex date on. Cancelling records a new fact rather than deleting — queries as of an earlier knowledge instant still see the action. Entry and cancel are confirmed and audited.',
	'ops.refdata.overrides':
		'Manual corrections for the loaded instrument: add an identifier (rejected if it belongs to another instrument) or write a new contract-spec version, which the risk engine’s caps and the short check read. Empty spec fields become NULL; every write is audited.',

	// ---- /infra ----
	'ops.infra.connections':
		'Every external service that needs credentials: keyed catalog sources, the Garage S3 key and the infra endpoints. Secret values never leave the server — rows only show set/missing plus a masked tail, and name the config/secrets.toml entry to edit.',
	'ops.infra.postgres':
		'PostgreSQL holds the records: reference data, warehouse, ledger, audit and ops history. Shown here: database size and connections in use, probed live — a down store reports an error instead of numbers.',
	'ops.infra.vm':
		'VictoriaMetrics holds every time series: ingested bars and the metrics scraped from services. The bar is on-disk size against the disk budget (FA_VM_DISK_BUDGET_GIB); the nightly VM retention job tightens retention against the same budget.',
	'ops.infra.garage':
		'Garage is the local S3 blob store: raw payload archive, media, model weights and checkpoints live in its buckets. Objects and bytes per bucket, probed live.',
	'ops.infra.qdrant':
		'Qdrant holds vector embeddings — latents, news and media — for similarity search. Points and health per collection, probed live.',
	'ops.infra.helix':
		'HelixDB is the graph store: instruments, latent regions and transitions, genomes with their lineage and fitness, snapshot metadata. Total node and edge counts, probed live.',
	'ops.infra.backup':
		'The nightly Postgres backup, the WAL archive (a segment every few minutes, so point-in-time restore loses little) and the weekly restore drill that proves a backup actually restores. Badges warn once an age outruns its schedule.',
	'ops.infra.maintenance':
		'The last run of each housekeeping job, read from audit.log: VM retention keeps VictoriaMetrics inside its disk budget, genome GC prunes old model weights, media downsample drops full video once frames are sampled, and the accounting close books the day.',
	'ops.infra.dlq':
		'Messages the bus gave up on: after repeated failed deliveries a message is moved here with its original subject and the failure reason, so it stops blocking its consumer. A growing count means some consumer cannot process what it is being sent.',
	'ops.infra.dlq-actions':
		'Replay publishes a dead letter back to its original subject and deletes it here — use it after fixing whatever made the consumer fail. Purge deletes all dead letters without redelivering anything. Both ask for confirmation and are audited.',

	// ---- /growth ----
	'ops.growth.range':
		'One range picker drives every chart; each runs its PromQL query against VictoriaMetrics over this window at the shown step. A chart with no data prints the exact query it ran, so a quiet system looks different from a wrong query.',
	'ops.growth.data':
		'How much data the system takes in and keeps: ingest events (every record a connector ingested, counted per source), bars written to the time-series store, VictoriaMetrics size and rows, and stream bytes against their caps.',
	'ops.growth.learning':
		'What the system has learned so far, from service heartbeat counters: live latent regions per timescale, regions born since start, encoder snapshots in the lineage, strategies loaded, and evolution activity once those services report it.',
	'ops.growth.money':
		'What funds the growth: each segment’s funds from the accounting books, and the running cost of operating. “all retained” means no profit split is configured yet, so nothing has been allocated to the segments.',
	'ops.growth.opex':
		'Every expense account in the books with its cumulative total — what running the system has cost so far. The trial balance underneath should be zero; anything else means the journal does not balance.'
};
