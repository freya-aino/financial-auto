import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BusFrame } from './api';
import { backoffMs, BusManager, debounce, frameBatcher, matchSubject } from './ws.svelte';

/** Minimal WebSocket stand-in the manager can drive; tests open/close it by hand. */
class FakeSocket {
	static instances: FakeSocket[] = [];
	onopen: (() => void) | null = null;
	onclose: (() => void) | null = null;
	onerror: (() => void) | null = null;
	onmessage: ((ev: { data: string }) => void) | null = null;
	closed = false;
	constructor(public url: string) {
		FakeSocket.instances.push(this);
	}
	close() {
		this.closed = true;
	}
	open() {
		this.onopen?.();
	}
	drop() {
		this.onclose?.();
	}
	send(frame: unknown) {
		this.onmessage?.({ data: JSON.stringify(frame) });
	}
}

const frame = (subject: string, seq = 1, decoded: unknown = null): BusFrame => ({
	stream: 'bus',
	seq,
	ts: '2026-11-08T12:00:00.000Z',
	data: { subject, size: 1, decoded }
});

function manager() {
	return new BusManager({
		urlFor: (p) => `ws://test/ws/bus?subjects=${p.join(',')}`,
		socket: (url) => new FakeSocket(url) as unknown as WebSocket
	});
}

const flushMicrotasks = () => new Promise<void>((r) => queueMicrotask(r));

describe('backoffMs', () => {
	it('doubles from 0.5 s and caps at 10 s', () => {
		expect([1, 2, 3, 4, 5, 6, 7].map((n) => backoffMs(n))).toEqual([500, 1000, 2000, 4000, 8000, 10000, 10000]);
		expect(backoffMs(0)).toBe(500);
	});
});

describe('matchSubject', () => {
	it('implements NATS wildcards', () => {
		expect(matchSubject('order.>', 'order.event.paper')).toBe(true);
		expect(matchSubject('order.>', 'order')).toBe(false);
		expect(matchSubject('signal.exposure.*', 'signal.exposure.s1')).toBe(true);
		expect(matchSubject('signal.exposure.*', 'signal.exposure.s1.x')).toBe(false);
		expect(matchSubject('md.trade.sim.abc', 'md.trade.sim.abc')).toBe(true);
		expect(matchSubject('md.trade.sim.abc', 'md.trade.sim')).toBe(false);
		expect(matchSubject('>', 'anything.at.all')).toBe(true);
		expect(matchSubject('*.heartbeat', 'svc.heartbeat')).toBe(true);
	});
});

describe('BusManager', () => {
	beforeEach(() => {
		vi.useFakeTimers();
		FakeSocket.instances = [];
	});
	afterEach(() => vi.useRealTimers());

	it('opens one socket for all subscribers and reconnects when the pattern set changes', async () => {
		const m = manager();
		const a = vi.fn();
		const b = vi.fn();
		m.subscribe('order.>', a);
		m.subscribe(['risk.>', 'order.>'], b);
		expect(m.status).toBe('closed');
		await flushMicrotasks();
		expect(FakeSocket.instances).toHaveLength(1);
		expect(FakeSocket.instances[0].url).toBe('ws://test/ws/bus?subjects=order.>,risk.>');
		expect(m.status).toBe('connecting');
		FakeSocket.instances[0].open();
		expect(m.status).toBe('open');

		// a new pattern → one reconnect carrying the merged list
		const off = m.subscribe('latent.>', vi.fn());
		await flushMicrotasks();
		expect(FakeSocket.instances).toHaveLength(2);
		expect(FakeSocket.instances[0].closed).toBe(true);
		expect(FakeSocket.instances[1].url).toContain('latent.>,order.>,risk.>');

		// removing it reconnects again; removing everything closes the socket
		off();
		await flushMicrotasks();
		expect(FakeSocket.instances).toHaveLength(3);
		expect(m.patterns).toEqual(['order.>', 'risk.>']);
		m.destroy();
		expect(FakeSocket.instances[2].closed).toBe(true);
		expect(m.status).toBe('closed');
	});

	it('fans frames out by subject match, once per listener, and counts lag', async () => {
		const m = manager();
		const orders = vi.fn();
		const both = vi.fn();
		m.subscribe('order.>', orders);
		m.subscribe(['order.>', 'risk.>'], both);
		await flushMicrotasks();
		const s = FakeSocket.instances[0];
		s.open();
		s.send(frame('order.event.paper', 1));
		s.send(frame('risk.halt.portfolio', 2));
		s.send(frame('md.trade.sim.x', 3));
		s.send({ op: 'lagged', dropped: 7 });
		s.send({ op: 'error', error: 'nats unavailable' });
		expect(orders).toHaveBeenCalledTimes(1);
		expect(both).toHaveBeenCalledTimes(2);
		expect(m.frames).toBe(3);
		expect(m.dropped).toBe(7);
		expect(m.error).toBe('nats unavailable');
		expect(() => m.handle('not json')).not.toThrow();
	});

	it('reconnects with exponential backoff and resets after a successful open', async () => {
		const m = manager();
		m.subscribe('jobs.>', vi.fn());
		await flushMicrotasks();
		FakeSocket.instances[0].drop();
		expect(m.status).toBe('closed');
		expect(m.attempt).toBe(1);
		expect(m.stale).toBe(true);
		vi.advanceTimersByTime(499);
		expect(FakeSocket.instances).toHaveLength(1);
		vi.advanceTimersByTime(1);
		expect(FakeSocket.instances).toHaveLength(2);
		FakeSocket.instances[1].drop();
		expect(m.attempt).toBe(2);
		vi.advanceTimersByTime(1000);
		expect(FakeSocket.instances).toHaveLength(3);
		FakeSocket.instances[2].drop();
		vi.advanceTimersByTime(2000);
		expect(FakeSocket.instances).toHaveLength(4);
		// keep failing: the delay saturates at 10 s
		for (let i = 0; i < 6; i++) {
			FakeSocket.instances.at(-1)!.drop();
			vi.advanceTimersByTime(10_000);
		}
		expect(m.attempt).toBeGreaterThanOrEqual(6);
		expect(m.nextRetryAt).toBeNull();
		FakeSocket.instances.at(-1)!.open();
		expect(m.attempt).toBe(0);
		expect(m.stale).toBe(false);
		// the resubscribed socket still carries the wanted patterns
		expect(FakeSocket.instances.at(-1)!.url).toContain('jobs.>');
		m.destroy();
	});

	it('reconnectNow skips the pending backoff', async () => {
		const m = manager();
		m.subscribe('x.>', vi.fn());
		await flushMicrotasks();
		FakeSocket.instances[0].drop();
		expect(m.nextRetryAt).not.toBeNull();
		m.reconnectNow();
		expect(FakeSocket.instances).toHaveLength(2);
		expect(m.nextRetryAt).toBeNull();
		m.destroy();
	});

	it('does not reconnect once every subscriber is gone', async () => {
		const m = manager();
		const off = m.subscribe('x.>', vi.fn());
		await flushMicrotasks();
		off();
		await flushMicrotasks();
		expect(FakeSocket.instances[0].closed).toBe(true);
		vi.advanceTimersByTime(60_000);
		expect(FakeSocket.instances).toHaveLength(1);
		expect(m.stale).toBe(false);
	});
});

describe('coalescing', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('frameBatcher delivers everything from one frame in a single flush', () => {
		const flush = vi.fn();
		const push = frameBatcher<number>(flush);
		push(1);
		push(2);
		push(3);
		expect(flush).not.toHaveBeenCalled();
		vi.advanceTimersByTime(16); // node fallback for requestAnimationFrame
		expect(flush).toHaveBeenCalledTimes(1);
		expect(flush).toHaveBeenCalledWith([1, 2, 3]);
		push(4);
		vi.advanceTimersByTime(16);
		expect(flush).toHaveBeenLastCalledWith([4]);
	});

	it('topic() keeps only the latest frame per animation frame', async () => {
		const m = manager();
		const t = m.topic<{ n: number }>('md.>');
		await flushMicrotasks();
		const s = FakeSocket.instances.at(-1)!;
		s.open();
		for (let i = 1; i <= 50; i++) s.send(frame('md.trade.sim.x', i, { n: i }));
		expect(t.latest).toBeNull();
		vi.advanceTimersByTime(16);
		expect(t.latest?.seq).toBe(50);
		expect(t.data).toEqual({ n: 50 });
		expect(t.count).toBe(50);
		t.close();
		m.destroy();
	});

	it('debounce runs once after the burst and can be cancelled', () => {
		const fn = vi.fn();
		const d = debounce(fn, 250);
		d();
		d();
		vi.advanceTimersByTime(249);
		d();
		vi.advanceTimersByTime(249);
		expect(fn).not.toHaveBeenCalled();
		vi.advanceTimersByTime(1);
		expect(fn).toHaveBeenCalledTimes(1);
		d();
		d.cancel();
		vi.advanceTimersByTime(1000);
		expect(fn).toHaveBeenCalledTimes(1);
	});
});

// ------------------------------------------------------------ state streams (D-055)
import { FrameRouter, StreamSocket, TopicCache, topicMatches, type StreamFrame } from './ws.svelte';

const sframe = (topic: string, data: unknown = null, seq = 1): StreamFrame => ({ stream: 'trading', seq, ts: '2026-11-10T06:00:00Z', topic, data });

describe('topicMatches (server semantics)', () => {
	it('matches equal, dotted continuation and > prefixes only', () => {
		expect(topicMatches('risk', 'risk')).toBe(true);
		expect(topicMatches('order', 'order.abc')).toBe(true);
		expect(topicMatches('order', 'orders')).toBe(false);
		expect(topicMatches('order.>', 'order.abc')).toBe(true);
		expect(topicMatches('order.>', 'order.')).toBe(true);
		expect(topicMatches('order.>', 'ord')).toBe(false);
		expect(topicMatches('>', 'anything.at.all')).toBe(true);
		expect(topicMatches('generation.g1', 'generation.g10')).toBe(false);
	});
});

describe('FrameRouter', () => {
	it('routes data frames once per listener and reports which filters need (un)subscribe ops', () => {
		const got: string[] = [];
		const r = new FrameRouter();
		const l = (f: StreamFrame) => got.push(f.topic);
		expect(r.add('order.>', l)).toBe(true);
		expect(r.add('order.>', () => got.push('second'))).toBe(false);
		expect(r.add('risk', l)).toBe(true);
		expect(r.filters).toEqual(['order.>', 'risk']);
		r.handle(JSON.stringify(sframe('order.x1')));
		r.handle(JSON.stringify(sframe('risk')));
		r.handle(JSON.stringify(sframe('positions')));
		expect(got).toEqual(['order.x1', 'second', 'risk']);
		expect(r.frames).toBe(3);
		expect(r.remove('order.>', l)).toBe(false);
		expect(r.remove('risk', l)).toBe(true);
		expect(FrameRouter.op('subscribe', ['a'])).toBe('{"op":"subscribe","topics":["a"]}');
	});

	it('handles hello / subscribed / lagged / error ops and ignores garbage', () => {
		const ev: unknown[] = [];
		const r = new FrameRouter({
			hello: (s, ms) => ev.push(['hello', s, ms]),
			subscribed: (t) => ev.push(['sub', t]),
			lagged: (n) => ev.push(['lag', n]),
			error: (m) => ev.push(['err', m])
		});
		expect(r.handle('{"op":"hello","stream":"trading","flush_ms":200}')).toBeNull();
		r.handle('{"op":"subscribed","topics":["risk"]}');
		r.handle('{"op":"lagged","dropped":4}');
		r.handle('{"op":"lagged","dropped":2}');
		r.handle('{"op":"error","error":"bad op"}');
		r.handle('not json');
		r.handle('{"stream":"trading"}');
		expect(ev).toEqual([['hello', 'trading', 200], ['sub', ['risk']], ['lag', 4], ['lag', 2], ['err', 'bad op']]);
		expect(r.flushMs).toBe(200);
		expect(r.dropped).toBe(6);
		expect(r.acked).toEqual(['risk']);
		expect(r.frames).toBe(0);
	});
});

describe('TopicCache', () => {
	it('keeps the latest frame per topic and reports changed topics per flush', () => {
		const c = new TopicCache<number>();
		c.push(sframe('order.a', 1, 1) as StreamFrame<number>);
		c.push(sframe('order.b', 2, 2) as StreamFrame<number>);
		c.push(sframe('order.a', 3, 3) as StreamFrame<number>);
		expect(c.size).toBe(2);
		expect(c.values.get('order.a')?.data).toBe(3);
		expect(c.flush()).toEqual(['order.a', 'order.b']);
		expect(c.flush()).toEqual([]);
		c.delete('order.a');
		expect(c.flush()).toEqual(['order.a']);
		expect(c.size).toBe(1);
	});
});

describe('StreamSocket', () => {
	class StreamFake extends FakeSocket {
		sent: string[] = [];
		send(frame: unknown) {
			if (typeof frame === 'string') this.sent.push(frame);
			else super.send(frame);
		}
		push(obj: unknown) {
			this.onmessage?.({ data: JSON.stringify(obj) });
		}
	}
	const make = () =>
		new StreamSocket('trading', {
			urlFor: (q) => `ws://test/ws/trading${q.snapshot ? `?snapshot=${q.snapshot}` : ''}`,
			socket: (url) => new StreamFake(url) as unknown as WebSocket
		});
	const last = () => FakeSocket.instances[FakeSocket.instances.length - 1] as StreamFake;

	beforeEach(() => {
		FakeSocket.instances = [];
		vi.useFakeTimers();
	});
	afterEach(() => vi.useRealTimers());

	it('connects on first subscriber, sends the filter set on open and in-band ops afterwards', () => {
		const ws = make();
		const got: string[] = [];
		const off1 = ws.subscribe('risk', (f) => got.push(f.topic));
		expect(FakeSocket.instances).toHaveLength(1);
		expect(ws.status).toBe('connecting');
		const s = last();
		s.open();
		expect(ws.status).toBe('open');
		expect(s.sent).toEqual(['{"op":"subscribe","topics":["risk"]}']);
		const off2 = ws.subscribe('order.>', (f) => got.push(f.topic));
		expect(s.sent[1]).toBe('{"op":"subscribe","topics":["order.>"]}');
		s.push({ op: 'hello', stream: 'trading', flush_ms: 200 });
		s.push({ op: 'subscribed', topics: ['risk', 'order.>'] });
		s.push(sframe('order.i1'));
		s.push(sframe('positions'));
		s.push({ op: 'lagged', dropped: 3 });
		expect(got).toEqual(['order.i1']);
		expect(ws.frames).toBe(2); // every data frame counts, matched or not
		expect(ws.lagged).toBe(3);
		expect(ws.flushMs).toBe(200);
		expect(ws.subscribed).toEqual(['risk', 'order.>']);
		off2();
		expect(s.sent[2]).toBe('{"op":"unsubscribe","topics":["order.>"]}');
		off1();
		expect(ws.status).toBe('closed');
		expect(s.closed).toBe(true);
		expect(FakeSocket.instances).toHaveLength(1);
	});

	it('reconnects with backoff and resubscribes every wanted filter', () => {
		const ws = make();
		ws.subscribe('risk', () => {});
		ws.subscribe('halts', () => {});
		last().open();
		expect(last().sent).toEqual(['{"op":"subscribe","topics":["halts","risk"]}']);
		last().drop();
		expect(ws.status).toBe('closed');
		expect(ws.attempt).toBe(1);
		expect(ws.stale).toBe(true);
		vi.advanceTimersByTime(499);
		expect(FakeSocket.instances).toHaveLength(1);
		vi.advanceTimersByTime(1);
		expect(FakeSocket.instances).toHaveLength(2);
		last().drop();
		vi.advanceTimersByTime(1000);
		expect(FakeSocket.instances).toHaveLength(3);
		last().open();
		expect(ws.attempt).toBe(0);
		expect(ws.stale).toBe(false);
		expect(last().sent).toEqual(['{"op":"subscribe","topics":["halts","risk"]}']);
		ws.destroy();
	});

	it('setQuery reconnects only when the query changes and something listens', () => {
		const ws = make();
		ws.setQuery({ snapshot: 'a' });
		expect(FakeSocket.instances).toHaveLength(0);
		ws.subscribe('state.>', () => {});
		expect(last().url).toBe('ws://test/ws/trading?snapshot=a');
		ws.setQuery({ snapshot: 'a' });
		expect(FakeSocket.instances).toHaveLength(1);
		ws.setQuery({ snapshot: 'b', instruments: '' });
		expect(FakeSocket.instances).toHaveLength(2);
		expect(last().url).toBe('ws://test/ws/trading?snapshot=b');
		ws.destroy();
	});

	it('topics() keeps one entry per topic, replaced once per animation frame', async () => {
		const ws = make();
		const orders = ws.topics<{ n: number }>('order.>');
		last().open();
		last().push(sframe('order.a', { n: 1 }, 1));
		last().push(sframe('order.b', { n: 2 }, 2));
		last().push(sframe('order.a', { n: 3 }, 3));
		expect(orders.entries.size).toBe(0);
		await vi.advanceTimersByTimeAsync(20);
		expect(orders.entries.size).toBe(2);
		expect(orders.entries.get('order.a')?.data.n).toBe(3);
		expect(orders.count).toBe(3);
		orders.close();
		expect(ws.status).toBe('closed');
	});
});
