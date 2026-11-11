// Alert banner + alerts view logic (D-066, spec 60 §6.5): `GET /api/alerts` is the snapshot,
// `ops.alert` frames on the bus tap are transitions; `pipeline.ok = false` (no Watchdog) is
// itself an alert, because then nothing else can be trusted to fire.
import type { ActiveAlert, AlertEvent, AlertsView } from './console';

const RANK: Record<string, number> = { critical: 0, page: 0, error: 1, warning: 2, info: 3 };

export const severityRank = (s: string | undefined) => RANK[(s ?? '').toLowerCase()] ?? 2;

export const severityClass = (s: string | undefined) => (severityRank(s) <= 1 ? 'danger' : severityRank(s) === 2 ? 'warn' : '');

/** Most severe first, then most recent start first. */
export function sortAlerts<T extends AlertEvent>(alerts: T[]): T[] {
	return [...alerts].sort(
		(a, b) => severityRank(a.severity) - severityRank(b.severity) || (b.starts_at ?? b.at).localeCompare(a.starts_at ?? a.at) || a.alertname.localeCompare(b.alertname)
	);
}

/** Applies one `ops.alert` transition to the active list (firing upserts, resolved removes). */
export function applyAlertEvent(active: ActiveAlert[], ev: AlertEvent): ActiveAlert[] {
	if (!ev || typeof ev.fingerprint !== 'string' || !ev.alertname) return active;
	const rest = active.filter((a) => a.fingerprint !== ev.fingerprint);
	if (ev.status !== 'firing') return rest;
	const prev = active.find((a) => a.fingerprint === ev.fingerprint);
	return sortAlerts([...rest, { ...ev, last_seen: ev.at, refreshes: (prev?.refreshes ?? 0) + 1 }]);
}

export interface BannerItem {
	key: string;
	level: 'danger' | 'warn';
	text: string;
	title?: string;
}

export interface Banner {
	level: 'ok' | 'warn' | 'danger';
	items: BannerItem[];
	critical: number;
	warning: number;
}

/**
 * What the global banner shows. `fetchError` = `/api/alerts` unreachable (the banner must not
 * pretend all is well then). At most `max` alert items; the rest are counted.
 */
export function alertBanner(view: AlertsView | null, fetchError = '', now = Date.now(), max = 3): Banner {
	const items: BannerItem[] = [];
	if (fetchError) items.push({ key: 'fetch', level: 'warn', text: 'alerts unavailable', title: fetchError });
	if (view && !view.pipeline.ok) {
		const seen = view.pipeline.watchdog_seen_at;
		const age = seen ? Math.round((now - Date.parse(seen)) / 60000) : null;
		items.push({
			key: 'pipeline',
			level: 'danger',
			text: `alert pipeline down: ${age === null ? 'no Watchdog received yet' : `no Watchdog for ${age} min`}`,
			title: `vmalert → control-api (${view.pipeline.posts} posts, last ${view.pipeline.last_post_at ?? 'never'}); other alerts may be missing`
		});
	}
	const active = sortAlerts(view?.active ?? []);
	const critical = active.filter((a) => severityRank(a.severity) <= 1).length;
	const warning = active.length - critical;
	for (const a of active.slice(0, max)) {
		items.push({ key: a.fingerprint, level: severityRank(a.severity) <= 1 ? 'danger' : 'warn', text: `${a.alertname}: ${a.summary || a.annotations?.summary || ''}`.replace(/: $/, ''), title: labelText(a.labels) });
	}
	if (active.length > max) items.push({ key: 'more', level: critical > 0 ? 'danger' : 'warn', text: `+${active.length - max} more` });
	const level = items.some((i) => i.level === 'danger') ? 'danger' : items.length ? 'warn' : 'ok';
	return { level, items, critical, warning };
}

export const labelText = (labels: Record<string, string> | undefined) =>
	Object.entries(labels ?? {})
		.filter(([k]) => k !== 'alertname')
		.map(([k, v]) => `${k}=${v}`)
		.join(' ');

/** Duration since `iso` as "3m" / "2h 5m" / "1d 4h". */
export function since(iso: string | null | undefined, now = Date.now()): string {
	if (!iso) return '—';
	const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
	if (s < 60) return `${s}s`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m`;
	const h = Math.floor(m / 60);
	if (h < 24) return `${h}h ${m % 60}m`;
	return `${Math.floor(h / 24)}d ${h % 24}h`;
}
