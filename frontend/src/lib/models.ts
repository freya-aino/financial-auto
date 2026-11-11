// Model-side logic for the console (D-068, D-069, D-070): encoder snapshot expert health and
// promotion checklist, neural decoder slots and online promotion progress, MCMC answers.
import type { CheckItem, DecoderCardRow, DecoderSlot, Dist, SnapshotDetail } from './console';

// ------------------------------------------------------------------ encoder snapshots (D-070)

export const NEURAL_FAMILIES = ['fsq_hybrid', 'jepa_vae'] as const;
export const isNeuralFamily = (f: string) => (NEURAL_FAMILIES as readonly string[]).includes(f);

export interface ExpertHealth {
	k_eff: number | null;
	pre_prune_k_eff: number | null;
	resid_corr: number[][] | null;
	resid_corr_max: number | null;
	skill: number[];
	usage: number[];
	mix_skill: number | null;
	pruned: number | null;
	n_windows: number | null;
	cka: unknown;
}

const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const nums = (v: unknown) => (Array.isArray(v) ? v.map((x) => (typeof x === 'number' ? x : NaN)) : []);

/** Expert health from `manifest.params.health` (matrices) and `manifest.metrics` (per-expert keys). */
export function expertHealth(manifest: Record<string, unknown> | null): ExpertHealth | null {
	if (!manifest) return null;
	const metrics = (manifest.metrics ?? {}) as Record<string, unknown>;
	const health = (((manifest.params ?? {}) as Record<string, unknown>).health ?? {}) as Record<string, unknown>;
	const perExpert = (prefix: string) =>
		Object.keys(metrics)
			.map((k) => /^(\w+?)_e(\d+)$/.exec(k))
			.filter((m): m is RegExpExecArray => !!m && m[1] === prefix)
			.sort((a, b) => Number(a[2]) - Number(b[2]))
			.map((m) => metrics[m[0]] as number);
	const skill = nums(health.skill).length ? nums(health.skill) : perExpert('skill');
	const usage = nums(health.usage).length ? nums(health.usage) : perExpert('usage');
	const k_eff = num(health.k_eff) ?? num(metrics.k_eff);
	if (k_eff === null && !skill.length && !Array.isArray(health.resid_corr)) return null;
	return {
		k_eff,
		pre_prune_k_eff: num(metrics.pre_prune_k_eff),
		resid_corr: Array.isArray(health.resid_corr) ? (health.resid_corr as number[][]) : null,
		resid_corr_max: num(health.resid_corr_max) ?? num(metrics.resid_corr_max),
		skill,
		usage,
		mix_skill: num(metrics.mix_skill),
		pruned: num(metrics.experts_pruned),
		n_windows: num(health.n_windows) ?? num(metrics.n_val_windows),
		cka: health.cka ?? metrics.cka ?? null
	};
}

/** Checklist rows plus the items not evaluated (spec 36 §6 checklist, D-070 additions). */
export function checklistRows(d: SnapshotDetail | null): { rows: (CheckItem & { state: 'pass' | 'fail' | 'unknown' })[]; passed: boolean; notEvaluated: string[] } {
	const rows = (d?.checklist?.checks ?? []).map((c) => ({ ...c, state: (c.ok ? 'pass' : c.value === null ? 'unknown' : 'fail') as 'pass' | 'fail' | 'unknown' }));
	return { rows, passed: !!d?.checklist?.passed, notEvaluated: d?.not_evaluated ?? [] };
}

/** The D-070 checklist items a neural snapshot must carry; missing ones are shown as gaps. */
export const NEURAL_CHECKS = ['ssl_regression', 'expert_k_eff', 'expert_resid_corr', 'expert_min_skill'] as const;

export function missingNeuralChecks(d: SnapshotDetail | null): string[] {
	if (!d || !isNeuralFamily(d.snapshot.encoder_family)) return [];
	const have = new Set((d.checklist?.checks ?? []).map((c) => c.name));
	return NEURAL_CHECKS.filter((n) => !have.has(n));
}

// ------------------------------------------------------------------ neural decoders (D-068)

export const PROMOTE_MIN_OUTPUTS = 200;
export const PROMOTE_MARGIN = 0.02;

export interface SlotVerdict {
	/** Shadow's relative loss improvement over serving (positive = shadow better). */
	improvement: number | null;
	/** Outputs scored, min(serving, shadow), over the minimum. */
	progress: number;
	ready: boolean;
	wouldPromote: boolean;
	text: string;
}

/** Online promotion progress of a slot (serving vs shadow). The 30-min shadow age is not visible here. */
export function slotVerdict(slot: DecoderSlot, minN = PROMOTE_MIN_OUTPUTS, margin = PROMOTE_MARGIN): SlotVerdict {
	const s = slot.serving;
	const h = slot.shadow;
	if (slot.disabled) return { improvement: null, progress: 0, ready: false, wouldPromote: false, text: 'slot off' };
	if (!s) return { improvement: null, progress: 0, ready: false, wouldPromote: false, text: 'no serving decoder' };
	if (!h) return { improvement: null, progress: 0, ready: false, wouldPromote: false, text: 'no shadow' };
	const n = Math.min(s.online_n, h.online_n);
	const progress = Math.min(1, n / minN);
	const improvement = s.online_loss !== null && h.online_loss !== null && s.online_loss > 0 ? (s.online_loss - h.online_loss) / s.online_loss : null;
	const ready = n >= minN;
	const wouldPromote = ready && improvement !== null && improvement > margin;
	const text = !ready
		? `${n}/${minN} scored`
		: improvement === null
			? 'losses unknown'
			: wouldPromote
				? `shadow better by ${(improvement * 100).toFixed(1)}% (promotes after 30 min shadow)`
				: `shadow ${improvement >= 0 ? 'better' : 'worse'} by ${(Math.abs(improvement) * 100).toFixed(1)}% (needs > ${(margin * 100).toFixed(0)}%)`;
	return { improvement, progress, ready, wouldPromote, text };
}

export interface CardSummary {
	key: string;
	family: string;
	output_family: string;
	timescale: string;
	generation: string;
	genome: string;
	score: number | null;
	/** Skill vs the naive baseline (see `cardSummary`). */
	skill: number | null;
	objectives: Record<string, number>;
	created: string | null;
	beats_baseline: boolean | null;
}

/** Flattens a decoder card (`.card.json`: `{meta, bucket, key, created_ms}`). */
export function cardSummary(row: DecoderCardRow): CardSummary {
	const c = (row.card ?? {}) as Record<string, unknown>;
	const meta = (c.meta ?? {}) as Record<string, unknown>;
	const genome = String(meta.genome ?? '');
	const ts = /ts=([0-9a-z]+)/.exec(genome)?.[1] ?? String(meta.timescale ?? '');
	const objectives = (meta.objectives ?? {}) as Record<string, number>;
	const created = typeof c.created_ms === 'number' ? new Date(c.created_ms).toISOString() : row.last_modified;
	const score = num(meta.score);
	// skill vs the naive baseline: CRPS skill for forecasts, reconstruction skill for rasters,
	// else the family score (news / narrative / chart report 0 skills and a signed score); > 0 beats it
	const skill = [num(objectives.crps_skill), num(objectives.recon_skill)].find((x) => x !== null && x !== 0) ?? score;
	return {
		key: String(c.key ?? row.key.replace(/\.card\.json$/, '.safetensors')),
		family: String(meta.family ?? ''),
		output_family: String(meta.output_family ?? meta.family ?? ''),
		timescale: ts,
		generation: String(meta.generation ?? ''),
		genome,
		score,
		objectives,
		created,
		skill,
		beats_baseline: skill === null ? null : skill > 0
	};
}

/** `genome` string `fam:k=v;k=v` → `[fam, {k: v}]` for compact display. */
export function genomeParams(genome: string): [string, Record<string, string>] {
	const [fam, rest = ''] = genome.split(/:(.*)/s);
	const params: Record<string, string> = {};
	for (const kv of rest.split(';')) {
		const [k, v] = kv.split('=');
		if (k && v !== undefined) params[k.trim()] = v.trim();
	}
	return [fam, params];
}

// ------------------------------------------------------------------ MCMC (D-069)

export const fmtDist = (d: Dist | null | undefined, digits = 3) =>
	d ? `${d.mean.toFixed(digits)} [${d.lower.toFixed(digits)}, ${d.upper.toFixed(digits)}]` : '—';

/** "a, b ,c" / "a b" → ["a", "b", "c"]. */
export const parseList = (s: string) =>
	s
		.split(/[\s,]+/)
		.map((x) => x.trim())
		.filter(Boolean);

/** Cumulative hitting probability from a pmf (P(hit by step t)). */
export function cumulative(pmf: number[]): number[] {
	let acc = 0;
	return pmf.map((p) => (acc += p));
}

/** Membership top region per expert with its weight (soft membership frame from the bus). */
export function topMemberships(m: { experts: { expert_index: number; novelty: number; out_of_region: boolean; weights: { region_id: string; weight: number }[] }[] }) {
	return m.experts.map((e) => {
		const top = [...e.weights].sort((a, b) => b.weight - a.weight)[0];
		const entropy = -e.weights.reduce((s, w) => s + (w.weight > 0 ? w.weight * Math.log(w.weight) : 0), 0);
		return { expert_index: e.expert_index, region_id: top?.region_id ?? '—', weight: top?.weight ?? 0, entropy, novelty: e.novelty, out: e.out_of_region, k: e.weights.length };
	});
}

export const LIFECYCLE_CLASS: Record<string, string> = { born: 'ok', carried: '', split: 'warn', merged: 'warn', died: 'danger' };
