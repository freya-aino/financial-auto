// Shared WebSocket managers (spec 51 §4, §6.6, §8; D-055 protocol).
//
// `BusManager` (`bus`) owns the one `/ws/bus` connection: NATS subject subscriptions are
// multiplexed client-side, reconnect backs off 0.5 s → 10 s and resubscribes, and high-rate
// topics coalesce to one update per animation frame. The bus tap only takes its NATS subject
// list in the URL (`?subjects=a,b`; the in-band ops filter what the tap already receives), so
// whenever the wanted pattern set changes the manager reconnects with the merged list; changes
// in one tick are batched into one reconnect.
//
// `StreamSocket` serves the state streams `/ws/latent`, `/ws/evolution` and `/ws/trading`:
// frames `{stream, seq, ts, topic, data}` carry the latest value per topic, clients filter with
// `{op: subscribe|unsubscribe, topics}` (prefix match on `.` boundaries, `>` wildcard), the
// server says `{op: "hello", flush_ms}` on connect, `{op: "subscribed", topics}` after each op
// and `{op: "lagged", dropped}` after superseded frames. The wanted topic set is resent on every
// (re)connect. `FrameRouter` is the pure part (parsing, routing, op handling) under unit test.
import { busTapUrl, type BusFrame } from './api';

export type BusStatus = 'closed' | 'connecting' | 'open';
export type FrameListener = (frame: BusFrame) => void;

export const BACKOFF_MIN_MS = 500;
export const BACKOFF_MAX_MS = 10_000;

/** Reconnect delay for the n-th consecutive failed attempt (n >= 1): 0.5 s doubling, capped at 10 s. */
export function backoffMs(attempt: number, min = BACKOFF_MIN_MS, max = BACKOFF_MAX_MS): number {
	return Math.min(max, min * 2 ** Math.max(0, attempt - 1));
}

/** NATS subject matching: `*` matches one token, `>` matches one or more trailing tokens. */
export function matchSubject(pattern: string, subject: string): boolean {
	const p = pattern.split('.');
	const s = subject.split('.');
	for (let i = 0; i < p.length; i++) {
		if (p[i] === '>') return s.length > i;
		if (i >= s.length) return false;
		if (p[i] !== '*' && p[i] !== s[i]) return false;
	}
	return p.length === s.length;
}

const raf: (cb: () => void) => void =
	typeof requestAnimationFrame === 'function' ? (cb) => requestAnimationFrame(() => cb()) : (cb) => setTimeout(cb, 16);

/** Collects items and hands them to `flush` at most once per animation frame. */
export function frameBatcher<T>(flush: (items: T[]) => void): (item: T) => void {
	let pending: T[] = [];
	let scheduled = false;
	return (item) => {
		pending.push(item);
		if (scheduled) return;
		scheduled = true;
		raf(() => {
			scheduled = false;
			const items = pending;
			pending = [];
			flush(items);
		});
	};
}

/** Trailing debounce; `fn` runs once `ms` after the last call. */
export function debounce(fn: () => void, ms: number): (() => void) & { cancel(): void } {
	let t: ReturnType<typeof setTimeout> | undefined;
	const d = () => {
		clearTimeout(t);
		t = setTimeout(fn, ms);
	};
	d.cancel = () => clearTimeout(t);
	return d;
}

/** Reactive view of one subject pattern: latest frame, coalesced to one update per animation frame. */
export interface Topic<T = unknown> {
	readonly pattern: string;
	readonly latest: BusFrame | null;
	readonly data: T | null;
	/** Frames received (including coalesced ones). */
	readonly count: number;
	/** Stops the subscription. */
	close(): void;
}

export interface BusManagerOptions {
	urlFor?: (patterns: string[]) => string;
	socket?: (url: string) => WebSocket;
	minBackoffMs?: number;
	maxBackoffMs?: number;
}

export class BusManager {
	status = $state<BusStatus>('closed');
	/** Consecutive failed connection attempts; 0 once connected. */
	attempt = $state(0);
	/** Epoch ms of the next reconnect attempt, or null. */
	nextRetryAt = $state<number | null>(null);
	/** Frames the server dropped because of its per-socket rate limit. */
	dropped = $state(0);
	frames = $state(0);
	lastFrameAt = $state<number | null>(null);
	error = $state('');
	/** Subject patterns the current socket was opened with. */
	patterns = $state<string[]>([]);

	private listeners = new Map<string, Set<FrameListener>>();
	private sock: WebSocket | null = null;
	private retryTimer: ReturnType<typeof setTimeout> | undefined;
	private syncScheduled = false;
	private readonly opts: Required<BusManagerOptions>;

	constructor(opts: BusManagerOptions = {}) {
		this.opts = {
			urlFor: opts.urlFor ?? ((p) => busTapUrl(p.join(','))),
			socket: opts.socket ?? ((url) => new WebSocket(url)),
			minBackoffMs: opts.minBackoffMs ?? BACKOFF_MIN_MS,
			maxBackoffMs: opts.maxBackoffMs ?? BACKOFF_MAX_MS
		};
	}

	/** Patterns wanted by current subscribers, sorted. */
	get wanted(): string[] {
		return [...this.listeners.keys()].sort();
	}

	/** True when live updates are expected but not arriving (spec 51 §8 banner). */
	get stale(): boolean {
		return this.listeners.size > 0 && this.status !== 'open';
	}

	/**
	 * Delivers every frame whose subject matches one of `patterns` to `listener`.
	 * Returns the unsubscribe function. (Re)connects as needed.
	 */
	subscribe(patterns: string | string[], listener: FrameListener): () => void {
		const list = (Array.isArray(patterns) ? patterns : patterns.split(','))
			.map((p) => p.trim())
			.filter(Boolean);
		for (const p of list) {
			let set = this.listeners.get(p);
			if (!set) this.listeners.set(p, (set = new Set()));
			set.add(listener);
		}
		this.scheduleSync();
		return () => {
			for (const p of list) {
				const set = this.listeners.get(p);
				set?.delete(listener);
				if (set && set.size === 0) this.listeners.delete(p);
			}
			this.scheduleSync();
		};
	}

	/** Reactive latest-value store for one pattern, coalesced per animation frame. */
	topic<T = unknown>(pattern: string): Topic<T> {
		let latest = $state<BusFrame | null>(null);
		let count = $state(0);
		const push = frameBatcher<BusFrame>((frames) => {
			latest = frames[frames.length - 1];
			count += frames.length;
		});
		const off = this.subscribe(pattern, push);
		return {
			pattern,
			get latest() {
				return latest;
			},
			get data() {
				return (latest?.data.decoded as T | undefined) ?? null;
			},
			get count() {
				return count;
			},
			close: off
		};
	}

	/** Closes the socket and forgets every subscriber (tests / hot reload). */
	destroy(): void {
		this.listeners.clear();
		clearTimeout(this.retryTimer);
		this.retryTimer = undefined;
		this.nextRetryAt = null;
		this.closeSocket();
		this.status = 'closed';
	}

	/** Reconnects now, dropping any pending backoff (manual "retry" button). */
	reconnectNow(): void {
		clearTimeout(this.retryTimer);
		this.retryTimer = undefined;
		this.nextRetryAt = null;
		this.connect();
	}

	private scheduleSync() {
		if (this.syncScheduled) return;
		this.syncScheduled = true;
		queueMicrotask(() => {
			this.syncScheduled = false;
			this.sync();
		});
	}

	/** Brings the socket in line with the wanted pattern set. */
	private sync() {
		const wanted = this.wanted;
		if (wanted.length === 0) {
			clearTimeout(this.retryTimer);
			this.retryTimer = undefined;
			this.nextRetryAt = null;
			this.closeSocket();
			this.status = 'closed';
			this.attempt = 0;
			return;
		}
		const same = wanted.length === this.patterns.length && wanted.every((p, i) => p === this.patterns[i]);
		if (same && (this.sock || this.retryTimer)) return;
		clearTimeout(this.retryTimer);
		this.retryTimer = undefined;
		this.nextRetryAt = null;
		this.connect();
	}

	private closeSocket() {
		const s = this.sock;
		this.sock = null;
		if (s) {
			s.onopen = s.onclose = s.onerror = s.onmessage = null;
			try {
				s.close();
			} catch {
				// already closed
			}
		}
	}

	private connect() {
		this.closeSocket();
		const patterns = this.wanted;
		if (!patterns.length) return;
		this.patterns = patterns;
		this.status = 'connecting';
		let sock: WebSocket;
		try {
			sock = this.opts.socket(this.opts.urlFor(patterns));
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
			this.scheduleRetry();
			return;
		}
		this.sock = sock;
		sock.onopen = () => {
			if (this.sock !== sock) return;
			this.status = 'open';
			this.attempt = 0;
			this.error = '';
			this.nextRetryAt = null;
		};
		sock.onerror = () => {
			if (this.sock === sock) this.error = 'websocket error';
		};
		sock.onclose = () => {
			if (this.sock !== sock) return;
			this.sock = null;
			this.status = 'closed';
			this.scheduleRetry();
		};
		sock.onmessage = (ev) => {
			if (this.sock !== sock) return;
			this.handle(typeof ev.data === 'string' ? ev.data : String(ev.data));
		};
	}

	private scheduleRetry() {
		if (this.retryTimer || !this.listeners.size) return;
		this.attempt += 1;
		const delay = backoffMs(this.attempt, this.opts.minBackoffMs, this.opts.maxBackoffMs);
		this.nextRetryAt = Date.now() + delay;
		this.retryTimer = setTimeout(() => {
			this.retryTimer = undefined;
			this.nextRetryAt = null;
			this.connect();
		}, delay);
	}

	/** Parses one server message and fans it out to matching listeners. Exposed for tests. */
	handle(raw: string): void {
		let msg: Record<string, unknown>;
		try {
			msg = JSON.parse(raw);
		} catch {
			return;
		}
		if (msg.op === 'lagged') {
			this.dropped += Number(msg.dropped) || 0;
			return;
		}
		if (msg.op === 'error') {
			this.error = String(msg.error ?? 'bus error');
			return;
		}
		if (msg.stream !== 'bus' || !msg.data) return;
		const frame = msg as unknown as BusFrame;
		this.frames += 1;
		this.lastFrameAt = Date.now();
		const subject = frame.data.subject;
		const seen = new Set<FrameListener>();
		for (const [pattern, set] of this.listeners) {
			if (!matchSubject(pattern, subject)) continue;
			for (const l of set) {
				if (seen.has(l)) continue;
				seen.add(l);
				try {
					l(frame);
				} catch (e) {
					console.error('bus listener failed', e);
				}
			}
		}
	}
}

/** The app-wide connection. Components subscribe through it; the layout shows its status. */
export const bus = new BusManager();

// ------------------------------------------------------------------ state streams (D-055)

/** One frame of a state stream. `data` is the same JSON the matching REST route returns. */
export interface StreamFrame<T = unknown> {
	stream: string;
	seq: number;
	ts: string;
	topic: string;
	data: T;
}

/**
 * Server-side topic filter semantics (`ws_routes::topic_matches`): equal, or `topic` continues
 * `filter` after a `.`, or `filter` ends in `>` and `topic` starts with the part before it.
 */
export function topicMatches(filter: string, topic: string): boolean {
	if (filter === '>' || filter === '*' || filter === topic) return true;
	if (filter.endsWith('>')) return topic.startsWith(filter.slice(0, -1));
	return topic.length > filter.length && topic.startsWith(filter) && topic[filter.length] === '.';
}

export type StreamListener<T = unknown> = (frame: StreamFrame<T>) => void;

export interface RouterEvents {
	hello?: (stream: string, flushMs: number) => void;
	subscribed?: (topics: string[]) => void;
	lagged?: (dropped: number) => void;
	error?: (message: string) => void;
}

/**
 * Pure frame router for a state stream: parses server messages, dispatches control ops to
 * `events`, delivers data frames to every listener whose filter matches the topic (once per
 * listener), and tracks the filter set that must be (re)sent to the server.
 */
export class FrameRouter {
	private listeners = new Map<string, Set<StreamListener>>();
	frames = 0;
	dropped = 0;
	flushMs: number | null = null;
	/** Filters the server last acknowledged. */
	acked: string[] = [];

	constructor(private events: RouterEvents = {}) {}

	/** Distinct filters wanted by current listeners, sorted. */
	get filters(): string[] {
		return [...this.listeners.keys()].sort();
	}

	get size(): number {
		return this.listeners.size;
	}

	/** Adds a listener; returns `true` when `filter` is new (needs a subscribe op). */
	add(filter: string, listener: StreamListener): boolean {
		let set = this.listeners.get(filter);
		const fresh = !set;
		if (!set) this.listeners.set(filter, (set = new Set()));
		set.add(listener);
		return fresh;
	}

	/** Removes a listener; returns `true` when `filter` has no listener left (needs an unsubscribe op). */
	remove(filter: string, listener: StreamListener): boolean {
		const set = this.listeners.get(filter);
		if (!set) return false;
		set.delete(listener);
		if (set.size) return false;
		this.listeners.delete(filter);
		return true;
	}

	/** Client op for the server. */
	static op(op: 'subscribe' | 'unsubscribe', topics: string[]): string {
		return JSON.stringify({ op, topics });
	}

	/** Handles one raw server message. Returns the parsed data frame, or null for ops/garbage. */
	handle(raw: string): StreamFrame | null {
		let msg: Record<string, unknown>;
		try {
			msg = JSON.parse(raw);
		} catch {
			return null;
		}
		if (!msg || typeof msg !== 'object') return null;
		if (typeof msg.op === 'string') {
			switch (msg.op) {
				case 'hello':
					this.flushMs = Number(msg.flush_ms) || null;
					this.events.hello?.(String(msg.stream ?? ''), this.flushMs ?? 0);
					break;
				case 'subscribed':
					this.acked = Array.isArray(msg.topics) ? msg.topics.map(String) : [];
					this.events.subscribed?.(this.acked);
					break;
				case 'lagged': {
					const n = Number(msg.dropped) || 0;
					this.dropped += n;
					this.events.lagged?.(n);
					break;
				}
				case 'error':
					this.events.error?.(String(msg.error ?? 'stream error'));
					break;
			}
			return null;
		}
		if (typeof msg.topic !== 'string' || !('data' in msg)) return null;
		const frame = msg as unknown as StreamFrame;
		this.frames += 1;
		const seen = new Set<StreamListener>();
		for (const [filter, set] of this.listeners) {
			if (!topicMatches(filter, frame.topic)) continue;
			for (const l of set) {
				if (seen.has(l)) continue;
				seen.add(l);
				try {
					l(frame);
				} catch (e) {
					console.error('stream listener failed', e);
				}
			}
		}
		return frame;
	}
}

/**
 * Latest value per exact topic, coalesced: `push` records frames, `flush` returns the topics
 * that changed since the previous flush (in first-seen order) and the current map. Pure.
 */
export class TopicCache<T = unknown> {
	readonly values = new Map<string, StreamFrame<T>>();
	private changed = new Set<string>();

	push(frame: StreamFrame<T>): void {
		this.values.set(frame.topic, frame);
		this.changed.add(frame.topic);
	}

	delete(topic: string): void {
		this.values.delete(topic);
		this.changed.add(topic);
	}

	flush(): string[] {
		const out = [...this.changed];
		this.changed.clear();
		return out;
	}

	get size(): number {
		return this.values.size;
	}
}

export type StreamName = 'latent' | 'evolution' | 'trading';

/** Reactive latest value of one topic filter (the newest frame of any matching topic). */
export interface TopicStore<T = unknown> {
	readonly filter: string;
	readonly latest: StreamFrame<T> | null;
	readonly data: T | null;
	readonly count: number;
	close(): void;
}

/** Reactive map of every topic matching a filter (e.g. `order.>` → one entry per intent). */
export interface TopicMapStore<T = unknown> {
	readonly filter: string;
	/** Topic → latest frame, replaced (new Map) at most once per animation frame. */
	readonly entries: Map<string, StreamFrame<T>>;
	readonly count: number;
	close(): void;
}

export interface StreamSocketOptions {
	/** Builds the socket URL from the current query parameters. */
	urlFor?: (query: Record<string, string>) => string;
	socket?: (url: string) => WebSocket;
	minBackoffMs?: number;
	maxBackoffMs?: number;
}

/** URL of a state stream relative to the current origin. */
export function streamUrl(stream: StreamName, query: Record<string, string> = {}): string {
	const proto = location.protocol === 'https:' ? 'wss' : 'ws';
	const s = new URLSearchParams();
	for (const [k, v] of Object.entries(query)) if (v) s.set(k, v);
	const qs = s.toString();
	return `${proto}://${location.host}/ws/${stream}${qs ? `?${qs}` : ''}`;
}

/**
 * One state-stream connection (`/ws/latent`, `/ws/evolution`, `/ws/trading`): connects while
 * anything listens, (re)subscribes the wanted filters on every open, reconnects with the bus
 * backoff schedule, and exposes reactive status plus per-topic stores coalesced per frame.
 */
export class StreamSocket {
	status = $state<BusStatus>('closed');
	attempt = $state(0);
	nextRetryAt = $state<number | null>(null);
	/** Frames the server superseded or dropped (latest-value coalescing) since connect. */
	lagged = $state(0);
	frames = $state(0);
	lastFrameAt = $state<number | null>(null);
	flushMs = $state<number | null>(null);
	error = $state('');
	/** Filters the server acknowledged last. */
	subscribed = $state<string[]>([]);
	/** Query parameters of the current connection (e.g. `{snapshot}` for `/ws/latent`). */
	query = $state<Record<string, string>>({});

	readonly router: FrameRouter;
	private sock: WebSocket | null = null;
	private retryTimer: ReturnType<typeof setTimeout> | undefined;
	private readonly opts: Required<StreamSocketOptions>;

	constructor(
		public readonly stream: StreamName,
		opts: StreamSocketOptions = {}
	) {
		this.opts = {
			urlFor: opts.urlFor ?? ((q) => streamUrl(stream, q)),
			socket: opts.socket ?? ((url) => new WebSocket(url)),
			minBackoffMs: opts.minBackoffMs ?? BACKOFF_MIN_MS,
			maxBackoffMs: opts.maxBackoffMs ?? BACKOFF_MAX_MS
		};
		this.router = new FrameRouter({
			hello: (_s, ms) => (this.flushMs = ms || null),
			subscribed: (t) => (this.subscribed = t),
			lagged: (n) => (this.lagged += n),
			error: (m) => (this.error = m)
		});
	}

	/** True when live data is wanted but the socket is not open (polling fallback active). */
	get stale(): boolean {
		return this.router.size > 0 && this.status !== 'open';
	}

	get live(): boolean {
		return this.status === 'open';
	}

	/** Changes the connection query (e.g. the latent snapshot); reconnects when it differs. */
	setQuery(query: Record<string, string>): void {
		const clean = Object.fromEntries(Object.entries(query).filter(([, v]) => v));
		if (JSON.stringify(clean) === JSON.stringify(this.query)) return;
		this.query = clean;
		if (this.router.size) this.reconnectNow();
	}

	/** Delivers frames whose topic matches `filter` (server semantics) to `listener`. */
	subscribe<T = unknown>(filter: string, listener: StreamListener<T>): () => void {
		const l = listener as StreamListener;
		if (this.router.add(filter, l)) this.send(FrameRouter.op('subscribe', [filter]));
		if (!this.sock && !this.retryTimer) this.connect();
		return () => {
			if (this.router.remove(filter, l)) this.send(FrameRouter.op('unsubscribe', [filter]));
			if (this.router.size === 0) this.shutdown();
		};
	}

	/** Reactive newest frame of any topic matching `filter`, one update per animation frame. */
	topic<T = unknown>(filter: string): TopicStore<T> {
		let latest = $state<StreamFrame<T> | null>(null);
		let count = $state(0);
		const push = frameBatcher<StreamFrame<T>>((frames) => {
			latest = frames[frames.length - 1];
			count += frames.length;
		});
		const off = this.subscribe<T>(filter, push);
		return {
			filter,
			get latest() {
				return latest;
			},
			get data() {
				return latest?.data ?? null;
			},
			get count() {
				return count;
			},
			close: off
		};
	}

	/** Reactive map topic → latest frame for every topic matching `filter`. */
	topics<T = unknown>(filter: string): TopicMapStore<T> {
		const cache = new TopicCache<T>();
		let entries = $state<Map<string, StreamFrame<T>>>(new Map());
		let count = $state(0);
		const push = frameBatcher<StreamFrame<T>>((frames) => {
			for (const f of frames) cache.push(f);
			cache.flush();
			entries = new Map(cache.values);
			count += frames.length;
		});
		const off = this.subscribe<T>(filter, push);
		return {
			filter,
			get entries() {
				return entries;
			},
			get count() {
				return count;
			},
			close: off
		};
	}

	reconnectNow(): void {
		clearTimeout(this.retryTimer);
		this.retryTimer = undefined;
		this.nextRetryAt = null;
		this.connect();
	}

	destroy(): void {
		this.shutdown();
	}

	private shutdown() {
		clearTimeout(this.retryTimer);
		this.retryTimer = undefined;
		this.nextRetryAt = null;
		this.attempt = 0;
		this.closeSocket();
		this.status = 'closed';
	}

	private send(text: string) {
		const s = this.sock;
		if (!s || this.status !== 'open') return;
		try {
			s.send(text);
		} catch {
			// socket closing; the filters are resent on the next open
		}
	}

	private closeSocket() {
		const s = this.sock;
		this.sock = null;
		if (s) {
			s.onopen = s.onclose = s.onerror = s.onmessage = null;
			try {
				s.close();
			} catch {
				// already closed
			}
		}
	}

	private connect() {
		this.closeSocket();
		if (!this.router.size) return;
		this.status = 'connecting';
		let sock: WebSocket;
		try {
			sock = this.opts.socket(this.opts.urlFor(this.query));
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
			this.scheduleRetry();
			return;
		}
		this.sock = sock;
		sock.onopen = () => {
			if (this.sock !== sock) return;
			this.status = 'open';
			this.attempt = 0;
			this.error = '';
			this.nextRetryAt = null;
			const filters = this.router.filters;
			if (filters.length) this.send(FrameRouter.op('subscribe', filters));
		};
		sock.onerror = () => {
			if (this.sock === sock) this.error = 'websocket error';
		};
		sock.onclose = () => {
			if (this.sock !== sock) return;
			this.sock = null;
			this.status = 'closed';
			this.scheduleRetry();
		};
		sock.onmessage = (ev) => {
			if (this.sock !== sock) return;
			this.handle(typeof ev.data === 'string' ? ev.data : String(ev.data));
		};
	}

	private scheduleRetry() {
		if (this.retryTimer || !this.router.size) return;
		this.attempt += 1;
		const delay = backoffMs(this.attempt, this.opts.minBackoffMs, this.opts.maxBackoffMs);
		this.nextRetryAt = Date.now() + delay;
		this.retryTimer = setTimeout(() => {
			this.retryTimer = undefined;
			this.nextRetryAt = null;
			this.connect();
		}, delay);
	}

	/** Parses one server message (exposed for tests). */
	handle(raw: string): void {
		const frame = this.router.handle(raw);
		if (frame) {
			this.frames += 1;
			this.lastFrameAt = Date.now();
		}
	}
}

/** The app-wide state-stream connections (one per stream, opened while something listens). */
export const tradingWs = new StreamSocket('trading');
export const evolutionWs = new StreamSocket('evolution');
export const latentWs = new StreamSocket('latent');

/** All sockets the stale banner reports on. */
export const sockets = { bus, trading: tradingWs, evolution: evolutionWs, latent: latentWs } as const;

/**
 * Runs `fn` (debounced by `ms`) whenever a frame matching `patterns` arrives: the "refetch
 * on bus activity" pattern used by the views, with polling as the fallback. Returns cleanup.
 */
export function onBus(patterns: string | string[], fn: () => void | Promise<void>, ms = 250): () => void {
	const run = debounce(() => void fn(), ms);
	const off = bus.subscribe(patterns, run);
	return () => {
		run.cancel();
		off();
	};
}
