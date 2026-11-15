import { describe, expect, it } from 'vitest';
import {
	ageClass,
	backupRows,
	BACKUP_OK_S,
	DRILL_OK_S,
	fmtAge,
	MAINT_LABELS,
	maintSummary,
	purgeConfirmText,
	replayConfirmText,
	vmBudget,
	WAL_STALE_S,
	type BackupView,
	type DlqMessage
} from './infra';

const emptyBackup: BackupView = {
	backup: { last_success_ts: null, age_s: null, last_run_ts: null, ok: null, last_bytes: null, last_duration_s: null, kept: null },
	wal: {
		up: null, mode_on: null, files: null, bytes: null, archived_total: null, failed_total: null,
		last_archived_ts: null, last_archived_age_s: null, last_failed_ts: null
	},
	drill: { ok: null, last_success_ts: null, age_s: null },
	metrics: {}
};

describe('ages', () => {
	it('formats seconds, minutes, hours, days and never', () => {
		expect(fmtAge(45)).toBe('45 s');
		expect(fmtAge(300)).toBe('5 min');
		expect(fmtAge(3.5 * 3600)).toBe('3.5 h');
		expect(fmtAge(3 * 86400)).toBe('3.0 d');
		expect(fmtAge(null)).toBe('never');
		expect(fmtAge(undefined)).toBe('never');
		expect(fmtAge(-5)).toBe('0 s');
	});

	it('colors ok / stale / dead against thresholds', () => {
		expect(ageClass(100, 3600, 7200)).toBe('ok');
		expect(ageClass(5000, 3600, 7200)).toBe('warn');
		expect(ageClass(8000, 3600, 7200)).toBe('danger');
		expect(ageClass(null, 3600, 7200)).toBe('danger');
	});
});

describe('backup card', () => {
	it('is all danger when nothing was ever recorded', () => {
		const rows = backupRows(emptyBackup);
		expect(rows.map((r) => r.id)).toEqual(['backup', 'wal', 'drill']);
		expect(rows.every((r) => r.cls === 'danger')).toBe(true);
		expect(rows[0].age).toBe('never');
	});

	it('a fresh successful backup is ok with its size, a failed one is danger whatever the age', () => {
		const v = structuredClone(emptyBackup);
		v.backup = { ...v.backup, age_s: 3600, ok: true, last_bytes: 2 * 1024 ** 3, last_duration_s: 42.4, kept: 7 };
		expect(backupRows(v)[0]).toMatchObject({ cls: 'ok', detail: '2.00 GB in 42 s · 7 kept' });
		v.backup.ok = false;
		const row = backupRows(v)[0];
		expect(row.cls).toBe('danger');
		expect(row.detail).toBe('last run failed');
		expect(3600).toBeLessThan(BACKUP_OK_S);
	});

	it('WAL: archiver down or archive_mode off is danger even when recent; stale age is warn', () => {
		const v = structuredClone(emptyBackup);
		v.wal = { ...v.wal, up: true, mode_on: true, files: 12, bytes: 1024, last_archived_age_s: 60 };
		expect(backupRows(v)[1]).toMatchObject({ cls: 'ok', detail: '12 files · 1.0 KB' });
		v.wal.last_archived_age_s = WAL_STALE_S - 1;
		expect(backupRows(v)[1].cls).toBe('warn');
		v.wal.mode_on = false;
		expect(backupRows(v)[1]).toMatchObject({ cls: 'danger', detail: 'archive_mode off' });
	});

	it('drill passes within its weekly window', () => {
		const v = structuredClone(emptyBackup);
		v.drill = { ok: true, last_success_ts: 1, age_s: DRILL_OK_S - 10 };
		expect(backupRows(v)[2]).toMatchObject({ cls: 'ok', detail: 'last drill passed' });
		v.drill.ok = false;
		expect(backupRows(v)[2].cls).toBe('danger');
	});
});

describe('VM budget and maintenance summaries', () => {
	it('turns used_ratio into a bar class', () => {
		expect(vmBudget({ used_ratio: 0.5 })).toMatchObject({ pct: '50.0%', cls: '' });
		expect(vmBudget({ used_ratio: 0.9 }).cls).toBe('warn');
		expect(vmBudget({ used_ratio: 1.2 }).cls).toBe('danger');
		expect(vmBudget({ error: 'down' })).toMatchObject({ ratio: 0, cls: '' });
	});

	it('summarises the known report fields, reclaimed bytes first', () => {
		expect(maintSummary({ deleted_bytes: 5 * 1024 ** 2, deleted: 12, dry_run: false })).toBe('freed 5.0 MB · 12 deleted');
		expect(maintSummary({ data_bytes: 1024 ** 3, tightened: true, dry_run: true })).toBe('VM at 1.00 GB · policy tightened · dry run');
		expect(maintSummary({ entries_posted: 4 })).toBe('4 entries posted');
		expect(maintSummary(null)).toBe('');
		expect(maintSummary('text')).toBe('');
		expect(Object.keys(MAINT_LABELS)).toHaveLength(4);
	});
});

describe('DLQ confirm texts name the consequence', () => {
	const msg: DlqMessage = {
		seq: 9, dlq_subject: 'dlq.LATENT.latent.state.1m.x', original_stream: 'LATENT',
		original_subject: 'latent.state.1m.x', original_seq: '41', consumer: 'model-runtime',
		reason: 'max_deliveries', deliveries: '5', at: 0, age_s: 60, size: 2048,
		preview: '…', lossy: false, truncated: false
	};
	it('replay names seq, size, subject and the removal', () => {
		const t = replayConfirmText(msg);
		expect(t).toContain('seq 9');
		expect(t).toContain('2.0 KB');
		expect(t).toContain('latent.state.1m.x');
		expect(t).toContain('removed from the DLQ stream');
	});
	it('purge names the count or the filter and the irreversibility', () => {
		expect(purgeConfirmText(3)).toContain('all 3 messages');
		expect(purgeConfirmText(1)).toContain('all 1 message of');
		expect(purgeConfirmText(3, 'dlq.LATENT.>')).toContain('subjects matching dlq.LATENT.>');
		expect(purgeConfirmText(0)).toContain('cannot be replayed');
	});
});
