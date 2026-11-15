// Infra observability (spec 51 §6 round 9): client for the infra/DLQ routes plus the pure
// logic of the /infra page — age formatting and ok/stale/dead thresholds for the backup / WAL /
// restore-drill card, VM usage against the disk budget, one-line maintenance summaries and the
// DLQ confirm texts. Replay and purge keep the confirm-then-API-result pattern (never optimistic).
import { bytes, postJson, q } from './api';
import { getApi } from './console';

// ---------------------------------------------------------------- types (shapes of infra_routes.rs / dlq_routes.rs)

/** A store probe of `GET /api/infra/stores`: the store's numbers, or `{error}` when it is down. */
export type StoreResult<T> = Partial<T> & { error?: string };

export interface PgStore {
	size_bytes: number;
	connections: number;
	max_connections: number;
}

export interface VmStore {
	data_size_bytes: number;
	active_series: number;
	series: number;
	disk_budget_gib: number;
	used_ratio: number;
}

export interface GarageBucket {
	bucket: string;
	objects?: number;
	bytes?: number;
	error?: string;
}

export interface QdrantCollection {
	name: string;
	points?: number | null;
	status?: string | null;
	error?: string;
}

export interface StoresView {
	stores: {
		postgres: StoreResult<PgStore>;
		victoriametrics: StoreResult<VmStore>;
		garage: StoreResult<{ buckets: GarageBucket[] }>;
		qdrant: StoreResult<{ collections: QdrantCollection[] }>;
		helix: StoreResult<{ nodes: number; edges: number }>;
	};
	vm_disk_budget_gib: number;
}

/** One maintenance job of `GET /api/infra/maintenance` (nulls when it never ran). */
export interface MaintJob {
	action: string;
	at: string | null;
	ok: boolean | null;
	report: unknown;
	error: unknown;
}

export interface BackupView {
	backup: {
		last_success_ts: number | null;
		age_s: number | null;
		last_run_ts: number | null;
		ok: boolean | null;
		last_bytes: number | null;
		last_duration_s: number | null;
		kept: number | null;
	};
	wal: {
		up: boolean | null;
		mode_on: boolean | null;
		files: number | null;
		bytes: number | null;
		archived_total: number | null;
		failed_total: number | null;
		last_archived_ts: number | null;
		last_archived_age_s: number | null;
		last_failed_ts: number | null;
	};
	drill: { ok: boolean | null; last_success_ts: number | null; age_s: number | null };
	metrics: Record<string, number>;
}

export interface DlqMessage {
	seq: number;
	dlq_subject: string;
	original_stream: string | null;
	original_subject: string;
	original_seq: string | null;
	consumer: string | null;
	reason: string | null;
	deliveries: string | null;
	at: number;
	age_s: number;
	size: number;
	preview: string;
	lossy: boolean;
	truncated: boolean;
}

export interface DlqView {
	stream: { messages: number; bytes: number; first_seq: number; last_seq: number };
	messages: DlqMessage[];
}

export const infraApi = {
	stores: () => getApi<StoresView>('/api/infra/stores'),
	maintenance: () => getApi<{ jobs: MaintJob[] }>('/api/infra/maintenance'),
	backup: () => getApi<BackupView>('/api/infra/backup'),
	dlq: (limit = 50) => getApi<DlqView>(`/api/dlq${q({ limit })}`),
	/** Only called after the operator confirmed `replayConfirmText`. */
	dlqReplay: (seq: number) => postJson<{ ok: boolean; seq: number; subject: string; deleted: boolean }>('/api/dlq/replay', { seq }),
	/** Only called after the operator confirmed `purgeConfirmText`. */
	dlqPurge: (subject?: string) =>
		postJson<{ ok: boolean; purged: number }>('/api/dlq/purge', subject ? { confirm: true, subject } : { confirm: true })
};

// ---------------------------------------------------------------- pure logic

/** `3720` → `'1.0 h'`; null/undefined → `'never'`. */
export function fmtAge(s: number | null | undefined): string {
	if (s === null || s === undefined || !Number.isFinite(s)) return 'never';
	const v = Math.max(0, s);
	if (v < 90) return `${Math.round(v)} s`;
	if (v < 90 * 60) return `${Math.round(v / 60)} min`;
	if (v < 48 * 3600) return `${(v / 3600).toFixed(1)} h`;
	return `${(v / 86400).toFixed(1)} d`;
}

/** Age against ok/stale thresholds → badge class; an unknown age is dead. */
export function ageClass(ageS: number | null | undefined, okS: number, staleS: number): 'ok' | 'warn' | 'danger' {
	if (ageS === null || ageS === undefined || !Number.isFinite(ageS)) return 'danger';
	return ageS <= okS ? 'ok' : ageS <= staleS ? 'warn' : 'danger';
}

/** Daily backups: ok within 26 h, stale within 52 h, dead beyond. */
export const BACKUP_OK_S = 26 * 3600;
export const BACKUP_STALE_S = 52 * 3600;
/** WAL archiving is continuous: ok within 1 h, stale within 6 h. */
export const WAL_OK_S = 3600;
export const WAL_STALE_S = 6 * 3600;
/** Restore drills are weekly-ish: ok within 8 d, stale within 15 d. */
export const DRILL_OK_S = 8 * 86400;
export const DRILL_STALE_S = 15 * 86400;

export interface BackupRow {
	id: 'backup' | 'wal' | 'drill';
	label: string;
	cls: 'ok' | 'warn' | 'danger';
	age: string;
	detail: string;
}

/** The three age-colored rows of the backup card. A failed last run is danger whatever the age. */
export function backupRows(v: BackupView): BackupRow[] {
	const worst = (a: 'ok' | 'warn' | 'danger', bad: boolean) => (bad ? 'danger' : a);
	const backup: BackupRow = {
		id: 'backup',
		label: 'Postgres backup',
		cls: worst(ageClass(v.backup.age_s, BACKUP_OK_S, BACKUP_STALE_S), v.backup.ok === false),
		age: fmtAge(v.backup.age_s),
		detail:
			v.backup.ok === false
				? 'last run failed'
				: v.backup.last_bytes != null
					? `${bytes(v.backup.last_bytes)} in ${Math.round(v.backup.last_duration_s ?? 0)} s${v.backup.kept != null ? ` · ${v.backup.kept} kept` : ''}`
					: 'no backup series in VictoriaMetrics'
	};
	const walBad = v.wal.up === false || v.wal.mode_on === false;
	const wal: BackupRow = {
		id: 'wal',
		label: 'WAL archive',
		cls: worst(ageClass(v.wal.last_archived_age_s, WAL_OK_S, WAL_STALE_S), walBad),
		age: fmtAge(v.wal.last_archived_age_s),
		detail: walBad
			? v.wal.mode_on === false
				? 'archive_mode off'
				: 'archiver down'
			: v.wal.files != null
				? `${v.wal.files} files · ${bytes(v.wal.bytes ?? 0)}${v.wal.failed_total ? ` · ${v.wal.failed_total} failed` : ''}`
				: 'no WAL series in VictoriaMetrics'
	};
	const drill: BackupRow = {
		id: 'drill',
		label: 'Restore drill',
		cls: worst(ageClass(v.drill.age_s, DRILL_OK_S, DRILL_STALE_S), v.drill.ok === false),
		age: fmtAge(v.drill.age_s),
		detail: v.drill.ok === false ? 'last drill failed' : v.drill.ok ? 'last drill passed' : 'no drill recorded'
	};
	return [backup, wal, drill];
}

/** VM disk usage vs `FA_VM_DISK_BUDGET_GIB` for the usage bar. */
export function vmBudget(vm: StoreResult<VmStore>): { ratio: number; pct: string; cls: '' | 'warn' | 'danger' } {
	const ratio = vm.used_ratio ?? 0;
	return {
		ratio,
		pct: `${(100 * ratio).toFixed(1)}%`,
		cls: ratio > 1 ? 'danger' : ratio > 0.8 ? 'warn' : ''
	};
}

/** Key numbers of a maintenance report in one line (reclaimed bytes first, spec 51 round 9). */
export function maintSummary(report: unknown): string {
	if (report === null || typeof report !== 'object') return '';
	const r = report as Record<string, unknown>;
	const parts: string[] = [];
	const num = (k: string) => (typeof r[k] === 'number' ? (r[k] as number) : null);
	if (num('deleted_bytes') != null) parts.push(`freed ${bytes(num('deleted_bytes') ?? 0)}`);
	if (num('deleted') != null) parts.push(`${num('deleted')} deleted`);
	if (num('data_bytes') != null) parts.push(`VM at ${bytes(num('data_bytes') ?? 0)}`);
	if (r['tightened'] === true) parts.push('policy tightened');
	if (num('entries_posted') != null) parts.push(`${num('entries_posted')} entries posted`);
	if (r['dry_run'] === true) parts.push('dry run');
	return parts.join(' · ');
}

/** Short job names for the maintenance table. */
export const MAINT_LABELS: Record<string, string> = {
	'ops.vm_retention': 'VM retention',
	'ops.genome_gc': 'Genome GC',
	'ops.media_downsample': 'Media downsample',
	'ops.accounting_close': 'Accounting close'
};

export const replayConfirmText = (m: DlqMessage) =>
	`Replay DLQ seq ${m.seq} (${bytes(m.size)}) to ${m.original_subject}? The message is republished to its original subject and removed from the DLQ stream.`;

export const purgeConfirmText = (count: number, filter?: string) =>
	`Purge ${filter ? `subjects matching ${filter} from` : `all ${count} message${count === 1 ? '' : 's'} of`} the DLQ stream? Purged dead letters cannot be replayed.`;
