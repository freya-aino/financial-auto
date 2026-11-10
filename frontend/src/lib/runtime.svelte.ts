// Runtime-tunable UI settings from `GET /api/config/runtime` (D-055): the views read
// `runtime.pollMs` for their REST fallback interval. Loaded once by the layout, refreshed when the
// Ops Runtime Config panel saves. Defaults match `fa_core::config::RuntimeConfig`.
import { api, type RuntimeConfig } from './api';

export const DEFAULT_POLL_MS = 2000;

class Runtime {
	values = $state<RuntimeConfig | null>(null);
	error = $state('');
	loadedAt = $state<number | null>(null);

	get pollMs(): number {
		const v = this.values?.ui.poll_ms;
		return typeof v === 'number' && v >= 250 ? v : DEFAULT_POLL_MS;
	}

	async load(): Promise<void> {
		try {
			this.values = (await api.runtimeConfig()).values;
			this.loadedAt = Date.now();
			this.error = '';
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
		}
	}

	set(values: RuntimeConfig): void {
		this.values = values;
		this.loadedAt = Date.now();
	}
}

export const runtime = new Runtime();

/**
 * Poll interval for a view whose live data arrives on `socketOpen` sockets: the runtime
 * `poll_ms` while the socket is down, `slowFactor × poll_ms` (a safety refetch) while it is up.
 */
export function fallbackInterval(socketOpen: boolean, slowFactor = 15): number {
	return socketOpen ? runtime.pollMs * slowFactor : runtime.pollMs;
}
