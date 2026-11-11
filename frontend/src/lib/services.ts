// Heartbeat counter grouping for the Ops services table (D-053/D-055 gauges): prefixes like
// `primitives.*`, `validation.*`, `pareto.*`, `hub.*`, `gpu.*` become small gauge groups,
// per-strategy and per-subject counters are folded so the table stays readable.
export interface GaugeGroup {
	prefix: string;
	items: { key: string; value: number }[];
}

/** Prefixes rendered as named gauge groups (order = display order). */
export const GAUGE_PREFIXES = ['primitives', 'validation', 'pareto', 'hub', 'gpu', 'connectors', 'archive', 'tsdb'];

/** Per-item counters that would flood the table; folded into `<prefix>.* (n)`. */
const FOLD_PREFIXES = ['strategy.', 'trades.md.', 'states_published.', 'region_events.', 'regions_live.', 'transition_rows.', 'transitions_published.', 'bars_closed.', 'unresolved.', 'events.'];

export interface GroupedCounters {
	gauges: GaugeGroup[];
	/** Everything else, sorted by key. */
	other: { key: string; value: number }[];
	/** Folded families: prefix → number of counters hidden. */
	folded: { prefix: string; count: number; sum: number }[];
}

export function groupCounters(counters: Record<string, number>): GroupedCounters {
	const groups = new Map<string, { key: string; value: number }[]>();
	const other: { key: string; value: number }[] = [];
	const folded = new Map<string, { count: number; sum: number }>();
	for (const [key, value] of Object.entries(counters).sort(([a], [b]) => a.localeCompare(b))) {
		const fold = FOLD_PREFIXES.find((p) => key.startsWith(p) && key.length > p.length);
		if (fold) {
			const f = folded.get(fold) ?? { count: 0, sum: 0 };
			f.count++;
			f.sum += value;
			folded.set(fold, f);
			continue;
		}
		const dot = key.indexOf('.');
		const prefix = dot > 0 ? key.slice(0, dot) : '';
		if (prefix && GAUGE_PREFIXES.includes(prefix)) {
			const g = groups.get(prefix) ?? [];
			g.push({ key: key.slice(dot + 1), value });
			groups.set(prefix, g);
		} else other.push({ key, value });
	}
	return {
		gauges: GAUGE_PREFIXES.filter((p) => groups.has(p)).map((p) => ({ prefix: p, items: groups.get(p)! })),
		other,
		folded: [...folded].map(([prefix, f]) => ({ prefix: prefix.replace(/\.$/, ''), ...f }))
	};
}

/** Share of primitive calls answered from the graph / decoder cache vs the price fallback (D-053). */
export function primitiveShares(counters: Record<string, number>): { graph: number; decoder: number; fallback: number } | null {
	const g = counters['primitives.graph'] ?? 0;
	const d = counters['primitives.decoder'] ?? 0;
	const f = counters['primitives.fallback'] ?? 0;
	const t = g + d + f;
	if (!t) return null;
	return { graph: g / t, decoder: d / t, fallback: f / t };
}

export const fmtCounter = (v: number) => (Number.isInteger(v) ? String(v) : Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1));

// ---------------------------------------------------------------- service control (D-060)

export type ServiceAction = 'start' | 'stop' | 'restart';
export const SERVICE_ACTIONS: ServiceAction[] = ['start', 'stop', 'restart'];

/** Mirrors control-api `services_routes::REFUSED` / `CONFIRM` (the API stays the authority:
 *  a refusal is still answered 409 and shown). */
export const REFUSED_SERVICES = ['control-api', 'nats', 'postgres', 'helix', 'victoriametrics', 'garage'];
export const CONFIRM_SERVICES = ['risk-engine'];

export interface ContainerSummary {
	name: string;
	id: string;
	image: string;
	/** Docker state: running | exited | restarting | paused | created | dead. */
	state: string;
	status: string;
}

export interface Supervisor {
	available: boolean;
	project?: string;
	socket?: string;
	services?: string[];
	detail?: string;
}

/** Whether an action button is enabled, with the reason when it is not. */
export function actionGate(service: string, action: ServiceAction, container: ContainerSummary | null | undefined, supervisor: Supervisor | null): { enabled: boolean; why: string } {
	if (REFUSED_SERVICES.includes(service))
		return { enabled: false, why: service === 'control-api' ? 'control-api cannot supervise itself' : 'infrastructure: use docker compose' };
	if (!supervisor?.available) return { enabled: false, why: `no Docker socket in control-api${supervisor?.detail ? `: ${supervisor.detail}` : ''}` };
	if (!container) return { enabled: false, why: 'no container for this service in the compose project' };
	const running = container.state === 'running';
	if (action === 'start' && running) return { enabled: false, why: 'already running' };
	if (action === 'stop' && !running) return { enabled: false, why: 'already stopped' };
	return { enabled: true, why: '' };
}

/** Confirmation text; risk-engine stop/restart names the trading halt (sent with `confirm: true`). */
export function confirmText(service: string, action: ServiceAction): string {
	const base = `${action[0].toUpperCase()}${action.slice(1)} ${service}? Audited as service.${action}.`;
	if (CONFIRM_SERVICES.includes(service) && action !== 'start')
		return `${base}\n\n${action} of ${service} halts trading (spec 40): no orders pass the risk chain until it is back.`;
	return action === 'stop' ? `${base} Its heartbeat and bus output stop until started again.` : base;
}

/** Body of `POST /api/services/{name}/{action}`. */
export const actionBody = (service: string, action: ServiceAction) => (CONFIRM_SERVICES.includes(service) && action !== 'start' ? { confirm: true } : {});

/** Operator text for the API's answer (`{ok, state, noop, audited}`) or its error body. */
export function describeResult(service: string, action: ServiceAction, r: { ok?: boolean; state?: string; noop?: boolean; audited?: boolean } | null, err?: { status: number; body: unknown; message: string }): string {
	if (err) {
		const b = (err.body ?? {}) as { error?: string; message?: string; detail?: string; hint?: string };
		const what = b.message ?? b.detail ?? b.error ?? err.message;
		return `${service} ${action} failed (${err.status}): ${what}${b.hint ? ` — ${b.hint}` : ''}`;
	}
	if (!r) return '';
	return `${service} ${action}: ${r.noop ? 'no change, ' : ''}state ${r.state ?? '?'}${r.audited ? ' · audited' : ' · NOT audited'}`;
}
