import { afterEach, describe, expect, it, vi } from 'vitest';
import { ago, ApiError, api, eur, live, pct, q, short } from './api';

describe('formatters', () => {
	it('eur formats EUR and dashes for missing values', () => {
		expect(eur(1234.5)).toMatch(/€1,234\.50/);
		expect(eur(null)).toBe('—');
		expect(eur(undefined)).toBe('—');
		expect(eur(NaN)).toBe('—');
		expect(eur(-0.004)).toMatch(/€0\.00/);
	});

	it('pct scales fractions', () => {
		expect(pct(0.1234)).toBe('12.34%');
		expect(pct(0.1234, 0)).toBe('12%');
		expect(pct(null)).toBe('—');
	});

	it('ago buckets into s/m/h and never goes negative', () => {
		const now = Date.parse('2026-11-08T12:00:00Z');
		expect(ago('2026-11-08T11:59:50Z', now)).toBe('10s');
		expect(ago('2026-11-08T11:50:00Z', now)).toBe('10m');
		expect(ago('2026-11-08T09:00:00Z', now)).toBe('3h');
		expect(ago('2026-11-08T12:00:30Z', now)).toBe('0s');
		expect(ago(null, now)).toBe('—');
		expect(ago('', now)).toBe('—');
	});

	it('short strips dashes and truncates long ids only', () => {
		expect(short('0192a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b')).toBe('0192a1b2');
		expect(short('0192a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b', 12)).toBe('0192a1b2c3d4');
		expect(short('abc')).toBe('abc');
	});
});

describe('query building', () => {
	it('drops undefined and empty values and encodes the rest', () => {
		expect(q({})).toBe('');
		expect(q({ family: undefined, limit: 30 })).toBe('?limit=30');
		expect(q({ family: '', kind: 'evo.eval.decoder', fills: true })).toBe('?kind=evo.eval.decoder&fills=true');
		expect(q({ s: 'a b&c' })).toBe('?s=a+b%26c');
	});
});

describe('fetch wrappers', () => {
	const fetchMock = vi.fn();
	vi.stubGlobal('fetch', fetchMock);
	afterEach(() => fetchMock.mockReset());

	const response = (status: number, body: unknown, ok = status < 400) => ({
		ok,
		status,
		statusText: status === 501 ? 'Not Implemented' : 'Bad Request',
		json: () => Promise.resolve(body)
	});

	it('postJson throws ApiError with the server reason (501 explanation)', async () => {
		fetchMock.mockResolvedValue(response(501, { error: 'not_implemented', reason: 'ingestor has no ctl.ingest handler' }));
		const err = await api.ingest('heartbeat', 'start').catch((e) => e);
		expect(err).toBeInstanceOf(ApiError);
		expect(err.status).toBe(501);
		expect(err.message).toBe('ingestor has no ctl.ingest handler');
		expect(fetchMock).toHaveBeenCalledWith('/api/ingest/heartbeat/start', expect.objectContaining({ method: 'POST' }));
	});

	it('postJson falls back to `error` then status text', async () => {
		fetchMock.mockResolvedValueOnce(response(400, { error: 'bad mode' }));
		await expect(live.halt('x')).rejects.toMatchObject({ status: 400, message: 'bad mode' });
		fetchMock.mockResolvedValueOnce({ ok: false, status: 501, statusText: 'Not Implemented', json: () => Promise.reject(new Error('no body')) });
		await expect(live.resume()).rejects.toMatchObject({ status: 501, message: '501 Not Implemented' });
	});

	it('paperOrder always sends mode=paper and the halt sends its reason', async () => {
		fetchMock.mockResolvedValue(response(200, { ok: true, intent_id: 'i1' }));
		await live.paperOrder({ instrument_id: 'abc', side: 'buy', quantity: 2 });
		const body = JSON.parse(fetchMock.mock.calls[0][1].body);
		expect(body).toEqual({ instrument_id: 'abc', side: 'buy', quantity: 2, mode: 'paper' });
		await live.halt('because');
		expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ reason: 'because' });
	});

	it('getJson builds query strings and throws on non-2xx', async () => {
		fetchMock.mockResolvedValueOnce(response(200, { outputs: [], families: [] }));
		await live.decoderOutputs('text', 5);
		expect(fetchMock.mock.calls[0][0]).toBe('/api/decoders/outputs?family=text&limit=5');
		fetchMock.mockResolvedValueOnce(response(503, {}));
		await expect(live.risk()).rejects.toThrow(/\/api\/trading\/risk: 503/);
	});
});
