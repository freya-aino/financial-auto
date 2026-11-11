import { describe, expect, it } from 'vitest';
import { alertBanner, applyAlertEvent, severityRank, since, sortAlerts } from './alerts';
import type { ActiveAlert, AlertEvent, AlertsView, DecoderSlot, SnapshotDetail } from './console';
import { cardSummary, checklistRows, cumulative, expertHealth, genomeParams, isNeuralFamily, liveExpertHealth, missingNeuralChecks, parseList, shadowAge, slotVerdict, topMemberships } from './models';
import { busRiskEvent, eventKinds, evidence, haltFamily, isRearmable, rankAllocators, rearmConfirmText, stageCounts, verdict } from './riskdesk';

const ev = (p: Partial<AlertEvent>): AlertEvent => ({
	status: 'firing', alertname: 'A', severity: 'warning', summary: 's', fingerprint: 'f', labels: {}, annotations: {},
	starts_at: '2026-11-11T00:00:00Z', ends_at: null, generator_url: null, at: '2026-11-11T00:00:00Z', ...p
});
const view = (active: ActiveAlert[], ok = true): AlertsView => ({
	active, recent: [], pipeline: { ok, watchdog_seen_at: '2026-11-11T00:00:00Z', posts: 3, last_post_at: null, subject: 'ops.alert' }
});

describe('alerts', () => {
	it('ranks severities and sorts critical first', () => {
		expect(severityRank('critical')).toBe(0);
		expect(severityRank('nonsense')).toBe(2);
		const s = sortAlerts([ev({ alertname: 'w' }), ev({ alertname: 'c', severity: 'critical' })]);
		expect(s.map((a) => a.alertname)).toEqual(['c', 'w']);
	});

	it('applies firing and resolved transitions by fingerprint', () => {
		let a: ActiveAlert[] = [];
		a = applyAlertEvent(a, ev({ fingerprint: 'x' }));
		a = applyAlertEvent(a, ev({ fingerprint: 'x', at: '2026-11-11T00:01:00Z' }));
		expect(a).toHaveLength(1);
		expect(a[0].refreshes).toBe(2);
		a = applyAlertEvent(a, ev({ fingerprint: 'y', severity: 'critical' }));
		expect(a[0].fingerprint).toBe('y');
		a = applyAlertEvent(a, ev({ fingerprint: 'x', status: 'resolved' }));
		expect(a.map((x) => x.fingerprint)).toEqual(['y']);
		expect(applyAlertEvent(a, null as unknown as AlertEvent)).toBe(a);
	});

	it('banner: ok, warn, danger, pipeline down is itself an alert', () => {
		expect(alertBanner(view([])).level).toBe('ok');
		const w = alertBanner(view([{ ...ev({}), last_seen: '', refreshes: 1 }]));
		expect(w.level).toBe('warn');
		expect(w.items[0].text).toBe('A: s');
		const down = alertBanner(view([], false), '', Date.parse('2026-11-11T00:10:00Z'));
		expect(down.level).toBe('danger');
		expect(down.items[0].text).toContain('no Watchdog for 10 min');
		expect(alertBanner(null, 'boom').items[0].text).toBe('alerts unavailable');
		const many = alertBanner(view(['a', 'b', 'c', 'd', 'e'].map((f) => ({ ...ev({ fingerprint: f, severity: 'critical' }), last_seen: '', refreshes: 1 }))));
		expect(many.items.at(-1)?.text).toBe('+2 more');
		expect(many.critical).toBe(5);
	});

	it('formats durations', () => {
		const t0 = Date.parse('2026-11-11T00:00:00Z');
		expect(since('2026-11-11T00:00:00Z', t0 + 30_000)).toBe('30s');
		expect(since('2026-11-11T00:00:00Z', t0 + 125 * 60_000)).toBe('2h 5m');
		expect(since('2026-11-11T00:00:00Z', t0 + 28 * 3_600_000)).toBe('1d 4h');
		expect(since(null)).toBe('—');
	});
});

describe('risk desk', () => {
	it('re-arm is allowed for every halt family the risk-engine knows, not only portfolio', () => {
		for (const s of ['portfolio', 'daily_loss', 'broker', 'source:yahoo', 'fx', 'model:stat-v1-1m', 'venue:xetr']) expect(isRearmable(s)).toBe(true);
		for (const s of ['strategy:x', 'risk.halt.portfolio', 'source.>', '', 'a b']) expect(isRearmable(s)).toBe(false);
		expect(haltFamily('source:kraken')).toBe('source');
		const t = rearmConfirmText('broker', 'reconciliation diff');
		expect(t).toContain('risk.resume.broker');
		expect(t).toContain('reconciliation diff');
		expect(t).toContain('audited');
	});

	it('counts pipeline stages and reads PBO / DSR evidence', () => {
		const c = stageCounts([{ status: 'candidate' }, { status: 'retired' }, { status: 'retired' }, { status: 'paused' }, { status: 'flattening' }]);
		expect(c).toMatchObject({ candidate: 1, retired: 2, paused: 1, other: 1, active: 0 });
		const s = {
			strategy_id: 'gp', status: 'candidate', pipeline: { selection: { pbo: 0.86, deflated_sharpe: 0.97, sharpe: 0.1, n_obs: 500 }, holdout: { deflated_sharpe: 0.96, sharpe: 0.02, n_obs: 150 } }
		} as unknown as Parameters<typeof evidence>[0];
		const e = evidence(s, { holdout_min_dsr: 0.95, holdout_min_obs: 100 });
		expect(e.selection.pbo).toBe(0.86);
		expect(e.holdout.pass).toBe(true);
		expect(evidence(s).holdout.pass).toBeNull();
		expect(verdict(0.86, 0.2, 'max')).toBe(false);
		expect(verdict(null, 0.2, 'max')).toBeNull();
	});

	it('ranks allocators by cumulative return and marks the serving one', () => {
		const r = rankAllocators(
			[
				{ name: 'risk_parity', cumulative_return: -0.14, sharpe: 0.01, max_drawdown: 0.14, steps: 1, turnover: 1, costs: 0, weights: {} },
				{ name: 'equal_weight', cumulative_return: -0.003, sharpe: 0.02, max_drawdown: 0.004, steps: 1, turnover: 1, costs: 0, weights: {} }
			],
			'equal_weight'
		);
		expect(r[0]).toMatchObject({ name: 'equal_weight', rank: 1, serving: true, lead: 0 });
		expect(r[1].lead).toBeCloseTo(-0.137);
	});

	it('groups events and normalises bus frames', () => {
		const rows = [
			{ id: 1, at: '2026-11-11T00:00:00Z', actor: 'a', action: 'risk.event.limit.clip', kind: 'limit.clip', scope: 'i', detail: null },
			{ id: 2, at: '2026-11-11T00:01:00Z', actor: 'a', action: 'risk.event.limit.clip', kind: 'limit.clip', scope: 'i', detail: null },
			{ id: 3, at: '2026-11-11T00:00:30Z', actor: 'a', action: 'halt', kind: 'halt', scope: 'broker', detail: null }
		];
		expect(eventKinds(rows)).toEqual([
			{ kind: 'limit.clip', count: 2, last: '2026-11-11T00:01:00Z' },
			{ kind: 'halt', count: 1, last: '2026-11-11T00:00:30Z' }
		]);
		const b = busRiskEvent('risk.event.paper', { type: 'risk_event', kind: 'KIND_LIMIT_BREACH', at: 't', scope: '' }, 7);
		expect(b).toMatchObject({ id: -7, kind: 'limit_breach', scope: null });
		const c = busRiskEvent('risk.event.paper', { type: 'risk_event', kind: 'LIMIT_BREACH', at: 't', scope: 'limit.clip', instrument_id: 'i1' }, 8);
		expect(c).toMatchObject({ kind: 'limit.clip', action: 'risk.event.limit.clip', scope: 'i1' });
		expect(busRiskEvent('risk.halt.broker', { type: 'halt', scope: 'broker', since: 't' }, 1)?.kind).toBe('halt');
		expect(busRiskEvent('x', null, 1)).toBeNull();
	});
});

describe('models', () => {
	const manifest = {
		metrics: { k_eff: 1.18, resid_corr_max: 0.95, skill_e0: 0.01, skill_e1: -2.2, skill_e10: 0.5, usage_e0: 0.2, mix_skill: 0.12, experts_pruned: 1 },
		params: { health: { k_eff: 1.186, resid_corr: [[1, 0.8], [0.8, 1]], resid_corr_max: 0.95, skill: [0.018, -2.2], usage: [0.3, 0.7], n_windows: 661 } }
	};

	it('reads expert health from the manifest (health block first, metrics fallback)', () => {
		const h = expertHealth(manifest)!;
		expect(h.k_eff).toBe(1.186);
		expect(h.skill).toEqual([0.018, -2.2]);
		expect(h.resid_corr?.[0][1]).toBe(0.8);
		const m = expertHealth({ metrics: manifest.metrics })!;
		expect(m.skill).toEqual([0.01, -2.2, 0.5]); // numeric order, e10 last
		expect(expertHealth({ metrics: { carried_mass: 1 } })).toBeNull();
		expect(expertHealth(null)).toBeNull();
	});

	it('checklist rows and missing neural checks', () => {
		const d = {
			snapshot: { encoder_family: 'jepa_vae' },
			checklist: { passed: false, checks: [{ name: 'expert_k_eff', ok: false, value: 1.1, threshold: 2, detail: '' }, { name: 'shadow_minutes', ok: false, value: null, threshold: 30, detail: '' }] },
			not_evaluated: ['4 mcmc_ece']
		} as unknown as SnapshotDetail;
		const c = checklistRows(d);
		expect(c.rows.map((r) => r.state)).toEqual(['fail', 'unknown']);
		expect(c.notEvaluated).toEqual(['4 mcmc_ece']);
		expect(missingNeuralChecks(d)).toEqual(['ssl_regression', 'expert_resid_corr', 'expert_min_skill']);
		expect(missingNeuralChecks({ ...d, snapshot: { ...d.snapshot, encoder_family: 'statistical' } })).toEqual([]);
	});

	it('slot promotion progress mirrors model-runtime (200 outputs, 2 % margin)', () => {
		const slot = (sl: number, sh: number, n = 300): DecoderSlot => ({
			family: 'retrieval', timescale: '1s', disabled: false, pinned: null,
			serving: { key: 'a', genome: null, score: 0.2, online_n: n, online_loss: sl },
			shadow: { key: 'b', genome: null, score: 0.1, online_n: n, online_loss: sh }
		});
		expect(slotVerdict(slot(1, 0.9)).wouldPromote).toBe(true);
		expect(slotVerdict(slot(1, 0.99)).wouldPromote).toBe(false);
		expect(slotVerdict(slot(1, 0.5, 100))).toMatchObject({ ready: false, progress: 0.5, text: '100/200 scored' });
		expect(slotVerdict({ ...slot(1, 1), shadow: null }).text).toBe('no shadow');
		expect(slotVerdict({ ...slot(1, 1), disabled: true }).text).toBe('slot off');
	});

	it('shows the shadow age against the promotion minimum', () => {
		const base: DecoderSlot = {
			family: 'rnn_prior', timescale: '1m', disabled: false, pinned: null,
			serving: { key: 'a', genome: null, score: 0.2, online_n: 0, online_loss: null },
			shadow: { key: 'b', genome: null, score: 0.1, online_n: 0, online_loss: null }
		};
		expect(shadowAge(base)).toBeNull();
		expect(shadowAge({ ...base, shadow_age_s: 720, promote_min_s: 1800 })).toEqual({ text: '12 min / 30 min', ripe: false });
		expect(shadowAge({ ...base, shadow_age_s: 9000, promote_min_s: 1800 })?.ripe).toBe(true);
		expect(shadowAge({ ...base, shadow: null, shadow_age_s: 5 })).toBeNull();
	});

	it('reads live expert health from latent.expert.health', () => {
		expect(liveExpertHealth(null)).toBeNull();
		const h = liveExpertHealth({
			snapshot_id: 's', timescale: '1m', as_of: '2026-11-11T10:00:00Z', source: 'live', expert_ids: [0, 3],
			usage: [0.7, 0.3], skill: [0.05, -0.1], resid_corr: [[1, 0.4], [0.4, 1]], k_eff: 1.7, resid_corr_max: 0.4, n_windows: 300
		});
		expect(h?.labels).toEqual(['E0', 'E3']);
		expect(h?.resid_corr?.[0][1]).toBe(0.4);
		expect(h?.k_eff).toBe(1.7);
		expect(isNeuralFamily('hier_vae_flow') && isNeuralFamily('latent_diffusion')).toBe(true);
	});

	it('summarises decoder cards and genomes', () => {
		const c = cardSummary({
			key: 'decoders/G/nn_forecast-x-J.card.json', size: 1, last_modified: null,
			card: { key: 'decoders/G/nn_forecast-x-J.safetensors', created_ms: 0, meta: { family: 'nn_forecast', output_family: 'rnn_prior', genome: 'nn_forecast:ts=1s;h=5', score: 0.18, generation: 'G', objectives: { crps_skill: 0.18, recon_skill: 0 } } }
		});
		expect(c).toMatchObject({ family: 'nn_forecast', output_family: 'rnn_prior', timescale: '1s', beats_baseline: true, created: '1970-01-01T00:00:00.000Z' });
		const news = cardSummary({ key: 'k.card.json', size: 1, last_modified: null, card: { meta: { family: 'nn_news', output_family: 'text.news', genome: 'nn_news:ts=1s', score: -0.18, objectives: { crps_skill: 0, recon_skill: 0 } } } });
		expect(news).toMatchObject({ skill: -0.18, beats_baseline: false, key: 'k.safetensors' });
		expect(genomeParams('nn_forecast:ts=1s;h=5')).toEqual(['nn_forecast', { ts: '1s', h: '5' }]);
	});

	it('MCMC helpers', () => {
		expect(parseList('a, b  c,')).toEqual(['a', 'b', 'c']);
		expect(cumulative([0.1, 0.2, 0.3]).map((x) => +x.toFixed(2))).toEqual([0.1, 0.3, 0.6]);
		const t = topMemberships({ experts: [{ expert_index: 0, novelty: 0.4, out_of_region: false, weights: [{ region_id: 'r1', weight: 0.2 }, { region_id: 'r2', weight: 0.8 }] }] });
		expect(t[0]).toMatchObject({ region_id: 'r2', weight: 0.8, k: 2 });
		expect(t[0].entropy).toBeGreaterThan(0);
	});
});
