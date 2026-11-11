// Risk desk + strategy pipeline logic (D-071, spec 35 §6.4, spec 40 §6): halt scopes and their
// re-arm, pipeline stages, PBO/DSR evidence, allocator tournament ranking, risk-event grouping.
import type { AllocatorEntry, RiskEventRow } from './console';
import type { StrategyView } from './api';

/** Scope families the risk-engine re-arms (mirrors control-api `REARM_FAMILIES`). */
export const REARM_FAMILIES = ['portfolio', 'daily_loss', 'source', 'feed', 'model', 'snapshot', 'broker', 'venue', 'fx'] as const;

export const haltFamily = (scope: string) => scope.split(':')[0];

export function isRearmable(scope: string): boolean {
	const s = scope.trim();
	if (!s || s.length > 128 || /[.*> ]/.test(s)) return false;
	return (REARM_FAMILIES as readonly string[]).includes(haltFamily(s));
}

/** What each family's re-arm means, for the confirm dialog. */
export function rearmConsequence(scope: string): string {
	switch (haltFamily(scope)) {
		case 'portfolio':
			return 'Trading resumes for the whole portfolio (kill switch released; drawdown peak re-based).';
		case 'daily_loss':
			return 'Risk-increasing orders are allowed again today despite the daily-loss limit having tripped.';
		case 'source':
		case 'feed':
			return scope.includes(':') ? `Signals depending on feed ${scope.split(':')[1]} are trusted again.` : 'Every halted feed is trusted again.';
		case 'model':
		case 'snapshot':
			return 'Model-driven strategies may trade again (model health halt cleared).';
		case 'broker':
		case 'venue':
			return 'Orders route to the venue again. A reconciliation diff halt should only be re-armed after the diff is understood.';
		case 'fx':
			return 'The stale-FX reduce-only halt is cleared (it re-trips if rates are still stale).';
		default:
			return '';
	}
}

/** Confirm text: names scope, consequence and that it is audited. */
export function rearmConfirmText(scope: string, reason: string): string {
	return `RE-ARM halt scope "${scope}"?\n\nCurrent reason: ${reason || '—'}\n${rearmConsequence(scope)}\n\nPublishes fa.v1.Resume on risk.resume.${scope}; audited as halt.rearm (actor ui).`;
}

export const haltClass = (scope: string) => {
	const f = haltFamily(scope);
	return f === 'portfolio' || f === 'daily_loss' ? 'danger' : 'warn';
};

// ------------------------------------------------------------------ strategy pipeline

export const PIPELINE_STAGES = ['candidate', 'holdout_passed', 'shadow', 'active', 'retired'] as const;
export type PipelineStage = (typeof PIPELINE_STAGES)[number] | 'paused' | 'other';

export function stageOf(status: string): PipelineStage {
	return (PIPELINE_STAGES as readonly string[]).includes(status) ? (status as PipelineStage) : status === 'paused' ? 'paused' : 'other';
}

export function stageCounts(rows: { status: string }[]): Record<string, number> {
	const out: Record<string, number> = Object.fromEntries([...PIPELINE_STAGES, 'paused'].map((s) => [s, 0]));
	for (const r of rows) {
		const s = stageOf(r.status);
		out[s] = (out[s] ?? 0) + 1;
	}
	return out;
}

export interface Selection {
	pbo: number | null;
	dsr: number | null;
	sharpe: number | null;
	n_obs: number | null;
}

interface PipelineRec {
	selection?: { pbo?: number | null; deflated_sharpe?: number | null; sharpe?: number | null; n_obs?: number | null } | null;
	holdout?: { deflated_sharpe?: number | null; sharpe?: number | null; n_obs?: number | null; holdout_id?: string } | null;
	[k: string]: unknown;
}

export interface PipelinePolicy {
	pbo_max?: number;
	dsr_min_candidate?: number;
	holdout_min_dsr?: number;
	holdout_min_obs?: number;
	[k: string]: unknown;
}

export type StrategyWithPipeline = StrategyView & { pipeline?: PipelineRec | null; evolved?: boolean };

/**
 * PBO / DSR of the strategy's selection and its holdout outcome (null = unknown, which blocks).
 * Holdout pass follows spec 41 §6.4 / D-071: DSR ≥ `holdout_min_dsr`, Sharpe > 0, n ≥ `holdout_min_obs`.
 */
export function evidence(s: StrategyWithPipeline, policy: PipelinePolicy = {}): { selection: Selection; holdout: Selection & { pass: boolean | null } } {
	const sel = s.pipeline?.selection ?? null;
	const ho = s.pipeline?.holdout ?? null;
	const n = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
	const dsr = n(ho?.deflated_sharpe);
	const sharpe = n(ho?.sharpe);
	const obs = n(ho?.n_obs);
	let pass: boolean | null = null;
	if (ho && policy.holdout_min_dsr !== undefined && policy.holdout_min_obs !== undefined) {
		pass = dsr !== null && sharpe !== null && obs !== null && dsr >= policy.holdout_min_dsr && sharpe > 0 && obs >= policy.holdout_min_obs;
	}
	return {
		selection: { pbo: n(sel?.pbo), dsr: n(sel?.deflated_sharpe), sharpe: n(sel?.sharpe), n_obs: n(sel?.n_obs) },
		holdout: { pbo: null, dsr, sharpe, n_obs: obs, pass }
	};
}

/** Pass/fail of a value against a threshold; null when the value is unknown. */
export function verdict(value: number | null, threshold: number | undefined, dir: 'min' | 'max'): boolean | null {
	if (value === null || threshold === undefined) return null;
	return dir === 'min' ? value >= threshold : value <= threshold;
}

// ------------------------------------------------------------------ allocator tournament

/** Ranked by cumulative return (ties: Sharpe); `active` marks the serving allocator. */
export function rankAllocators(entries: AllocatorEntry[], active: string | null): (AllocatorEntry & { rank: number; serving: boolean; lead: number })[] {
	const sorted = [...entries].sort((a, b) => b.cumulative_return - a.cumulative_return || (b.sharpe ?? -Infinity) - (a.sharpe ?? -Infinity));
	const best = sorted[0]?.cumulative_return ?? 0;
	return sorted.map((e, i) => ({ ...e, rank: i + 1, serving: e.name === 'active' || e.name === active, lead: e.cumulative_return - best }));
}

// ------------------------------------------------------------------ risk events

/** Groups audit-backed risk events per kind, newest first per group, for the summary row. */
export function eventKinds(events: RiskEventRow[]): { kind: string; count: number; last: string }[] {
	const m = new Map<string, { kind: string; count: number; last: string }>();
	for (const e of events) {
		const g = m.get(e.kind) ?? { kind: e.kind, count: 0, last: e.at };
		g.count++;
		if (e.at > g.last) g.last = e.at;
		m.set(e.kind, g);
	}
	return [...m.values()].sort((a, b) => b.count - a.count || a.kind.localeCompare(b.kind));
}

/** Normalises a live `risk.event.*` / `risk.halt.*` / `risk.resume.*` bus frame into an event row. */
export function busRiskEvent(subject: string, decoded: Record<string, unknown> | null, seq: number): RiskEventRow | null {
	if (!decoded) return null;
	const type = decoded.type as string | undefined;
	const at = (decoded.at ?? decoded.since ?? new Date().toISOString()) as string;
	if (type === 'risk_event') {
		// risk-engine puts the sub-kind (`limit.clip`, `feed.frozen`, …) in `scope` and audits it as
		// `risk.event.<sub-kind>` with the instrument / strategy as subject: mirror that here
		const sub = typeof decoded.scope === 'string' && decoded.scope ? decoded.scope : '';
		const kind = sub || String(decoded.kind ?? 'risk_event').replace(/^KIND_/, '').toLowerCase();
		const target = (decoded.instrument_id as string) || (decoded.strategy_id as string) || null;
		return { id: -seq, at, actor: 'bus', action: `risk.event.${kind}`, kind, scope: target ?? (sub ? null : (decoded.scope as string) || null), detail: decoded };
	}
	if (type === 'halt' || type === 'resume') {
		return { id: -seq, at, actor: (decoded.actor as string) || 'bus', action: type, kind: type, scope: (decoded.scope as string) ?? subject.split('.').slice(2).join('.'), detail: decoded };
	}
	return null;
}
