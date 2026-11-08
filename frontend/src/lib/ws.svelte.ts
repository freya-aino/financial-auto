// Shared WebSocket manager (spec 51 §4, §6.6, §8): one connection to /ws/bus for the whole
// app, multiplexing NATS subject subscriptions, with reactive status (Svelte 5 runes),
// auto-reconnect with 0.5 s → 10 s backoff and resubscribe, and per-animation-frame
// coalescing for high-rate topics.
//
// The server only takes the subject list in the URL (`?subjects=a,b`; in-band
// subscribe/unsubscribe ops of spec 50 §4.8 are not implemented yet), so whenever the set of
// wanted patterns changes the manager reconnects with the new list. Changes made in the same
// tick (a page mounting several subscriptions) are batched into one reconnect.
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
