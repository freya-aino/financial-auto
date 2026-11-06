// Typed client for control-api (spec 50). Paths are relative; vite proxies them in dev.

export interface ServiceStatus {
	service: string;
	instance_id: string;
	version: string;
	last_seen: string;
	counters: Record<string, number>;
}

export interface Source {
	id: string;
	name: string;
	category: string;
	capabilities: string[];
	asset_classes: string[];
	transport: string[];
	auth: string;
	status: 'planned' | 'stub' | 'live' | 'disabled';
	tos_notes: string;
	notes: string;
}

export interface Health {
	service: string;
	version: string;
	nats: { connected: boolean; url: string };
	dependencies: Record<string, boolean>;
}

export interface BusFrame {
	stream: string;
	seq: number;
	ts: string;
	data: { subject: string; size: number; decoded: unknown };
}

async function getJson<T>(path: string): Promise<T> {
	const res = await fetch(path);
	if (!res.ok) throw new Error(`${path}: ${res.status} ${res.statusText}`);
	return (await res.json()) as T;
}

export const api = {
	health: () => getJson<Health>('/api/health'),
	services: () => getJson<{ services: ServiceStatus[]; nats_connected: boolean }>('/api/services'),
	sources: () => getJson<{ version: number; sources: Source[] }>('/api/sources')
};

/** WebSocket URL for the bus tap, relative to the current origin. */
export function busTapUrl(subjects: string): string {
	const proto = location.protocol === 'https:' ? 'wss' : 'ws';
	return `${proto}://${location.host}/ws/bus?subjects=${encodeURIComponent(subjects)}`;
}
