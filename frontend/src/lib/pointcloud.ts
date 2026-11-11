// Camera, projection and colour maths for the canvas point cloud (spec 51 §6.3, Q1 decided:
// dependency-free Canvas 2D, D-085 proposal). Pure so the pan / zoom / orbit behaviour and the
// colour mappings are unit-tested without a DOM.

/** Points normalised into the unit cube around their centre; `xyz` holds x, y, z per point. */
export interface Cloud {
	n: number;
	dims: 2 | 3;
	xyz: Float32Array;
}

/** View state: orbit angles (3-D only), zoom factor and screen-pixel pan. */
export interface Camera {
	yaw: number;
	pitch: number;
	zoom: number;
	panX: number;
	panY: number;
}

export const HOME: Camera = { yaw: 0.6, pitch: 0.35, zoom: 1, panX: 0, panY: 0 };
export const HOME_2D: Camera = { yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0 };
export const ZOOM_MIN = 0.2;
export const ZOOM_MAX = 400;

/** The normalisation of a cloud: centre on the mean, divide by the 99th percentile of the
 *  per-point max |coordinate| (one far outlier does not shrink the cloud). Returned as a function
 *  so later points (live markers, centroids) map consistently. */
export function normaliser(coords: number[][], dims: 2 | 3): (c: number[]) => [number, number, number] {
	const n = coords.length;
	const mean = [0, 0, 0];
	for (const c of coords) for (let d = 0; d < dims; d++) mean[d] += (c[d] ?? 0) / Math.max(1, n);
	const mags = coords.map((c) => {
		let m = 0;
		for (let d = 0; d < dims; d++) m = Math.max(m, Math.abs((c[d] ?? 0) - mean[d]));
		return m;
	});
	mags.sort((a, b) => a - b);
	const s = (mags.length && (mags[Math.min(mags.length - 1, Math.floor(mags.length * 0.99))] || mags[mags.length - 1])) || 1;
	return (c) => [((c[0] ?? 0) - mean[0]) / s, ((c[1] ?? 0) - mean[1]) / s, dims === 3 ? ((c[2] ?? 0) - mean[2]) / s : 0];
}

/** Points normalised into roughly the unit cube (see [`normaliser`]). */
export function normalise(coords: number[][], dims: 2 | 3): Cloud {
	const f = normaliser(coords, dims);
	const xyz = new Float32Array(coords.length * 3);
	coords.forEach((c, i) => xyz.set(f(c), i * 3));
	return { n: coords.length, dims, xyz };
}

/** Pixels per normalised unit at zoom 1 (the unit cube fills ~90 % of the shorter side). */
export const unitPx = (w: number, h: number) => (Math.min(w, h) / 2) * 0.9;

/** Rotates (x, y, z) by yaw (about y) then pitch (about x); returns screen u, v and depth. */
export function rotate(cam: Camera, x: number, y: number, z: number): [number, number, number] {
	const cy = Math.cos(cam.yaw);
	const sy = Math.sin(cam.yaw);
	const cp = Math.cos(cam.pitch);
	const sp = Math.sin(cam.pitch);
	const x1 = x * cy + z * sy;
	const z1 = -x * sy + z * cy;
	const y2 = y * cp - z1 * sp;
	const z2 = y * sp + z1 * cp;
	return [x1, y2, z2];
}

/** One point onto the canvas (orthographic). */
export function toScreen(cam: Camera, w: number, h: number, x: number, y: number, z: number): [number, number, number] {
	const [u, v, d] = rotate(cam, x, y, z);
	const s = unitPx(w, h) * cam.zoom;
	return [w / 2 + u * s + cam.panX, h / 2 - v * s + cam.panY, d];
}

/** Projects every point into `out` (x, y pairs); returns `out`. Hot path: no allocation. */
export function projectAll(cloud: Cloud, cam: Camera, w: number, h: number, out: Float32Array): Float32Array {
	const cy = Math.cos(cam.yaw);
	const sy = Math.sin(cam.yaw);
	const cp = Math.cos(cam.pitch);
	const sp = Math.sin(cam.pitch);
	const s = unitPx(w, h) * cam.zoom;
	const ox = w / 2 + cam.panX;
	const oy = h / 2 + cam.panY;
	const xyz = cloud.xyz;
	for (let i = 0; i < cloud.n; i++) {
		const x = xyz[i * 3];
		const y = xyz[i * 3 + 1];
		const z = xyz[i * 3 + 2];
		const x1 = x * cy + z * sy;
		const z1 = -x * sy + z * cy;
		const y2 = y * cp - z1 * sp;
		out[i * 2] = ox + x1 * s;
		out[i * 2 + 1] = oy - y2 * s;
	}
	return out;
}

/** Zooms by `factor` keeping the point under the cursor (sx, sy) fixed on screen. */
export function zoomAt(cam: Camera, factor: number, sx: number, sy: number, w: number, h: number): Camera {
	const zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, cam.zoom * factor));
	const s0 = unitPx(w, h) * cam.zoom;
	const s1 = unitPx(w, h) * zoom;
	const u = (sx - w / 2 - cam.panX) / s0;
	const v = (sy - h / 2 - cam.panY) / s0;
	return { ...cam, zoom, panX: sx - w / 2 - u * s1, panY: sy - h / 2 - v * s1 };
}

export const pan = (cam: Camera, dx: number, dy: number): Camera => ({ ...cam, panX: cam.panX + dx, panY: cam.panY + dy });

/** Orbit by a pointer drag of (dx, dy) pixels; pitch is clamped to ±89°. */
export function orbit(cam: Camera, dx: number, dy: number): Camera {
	const lim = Math.PI / 2 - 0.02;
	return { ...cam, yaw: cam.yaw + dx * 0.01, pitch: Math.max(-lim, Math.min(lim, cam.pitch + dy * 0.01)) };
}

/** Index of the projected point nearest to (sx, sy) within `maxPx`, or −1. Linear scan: 50 000
 *  points take well under a millisecond, cheaper than keeping a spatial index current. */
export function nearest(screen: Float32Array, n: number, sx: number, sy: number, maxPx = 6, skip?: (i: number) => boolean): number {
	let best = -1;
	let bestD = maxPx * maxPx;
	for (let i = 0; i < n; i++) {
		if (skip?.(i)) continue;
		const dx = screen[i * 2] - sx;
		const dy = screen[i * 2 + 1] - sy;
		const d = dx * dx + dy * dy;
		if (d <= bestD) {
			bestD = d;
			best = i;
		}
	}
	return best;
}

// ---------------------------------------------------------------- colour

function hexRgb(hex: string): [number, number, number] | null {
	const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
	if (!m) return null;
	const v = parseInt(m[1], 16);
	return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** `#rrggbb` + alpha in [0, 1] → one RGBA pixel as a little-endian Uint32 (ImageData order). */
export function packRgba(hex: string, alpha: number): number {
	const [r, g, b] = hexRgb(hex) ?? [136, 136, 136];
	const a = Math.round(Math.max(0, Math.min(1, alpha)) * 255);
	return ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

/** Linear interpolation along hex colour stops for t in [0, 1] (time colouring). */
export function rampColor(t: number, stops: string[]): string {
	const rgb = stops.map(hexRgb).filter((x): x is [number, number, number] => x !== null);
	if (!rgb.length) return '#888888';
	if (rgb.length === 1) return stops[0];
	const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * (rgb.length - 1);
	const i = Math.min(rgb.length - 2, Math.floor(x));
	const f = x - i;
	const c = rgb[i].map((a, k) => Math.round(a + (rgb[i + 1][k] - a) * f));
	return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

/** Palette index per point plus the palette, so the renderer fills one path per colour. */
export interface Coloring {
	index: Uint16Array;
	palette: string[];
}

/** A labelled live position drawn over the cloud (normalised coordinates). */
export interface Marker {
	pos: [number, number, number];
	label: string;
	color: string;
	/** Previous positions, oldest first (drawn as a trail). */
	trail?: [number, number, number][];
}

/** Point indices ordered by palette slot: slot `c` owns `order[start[c] .. start[c + 1]]`. */
export function groupByColor(c: Coloring): { order: Uint32Array; start: Uint32Array } {
	const k = c.palette.length;
	const start = new Uint32Array(k + 1);
	for (let i = 0; i < c.index.length; i++) start[c.index[i] + 1]++;
	for (let s = 0; s < k; s++) start[s + 1] += start[s];
	const fill = start.slice(0, k);
	const order = new Uint32Array(c.index.length);
	for (let i = 0; i < c.index.length; i++) order[fill[c.index[i]]++] = i;
	return { order, start };
}

/** Categorical colouring by key (region / instrument) with a fixed key → colour map. */
export function categorical(keys: string[], colors: Map<string, string>, fallback = '#888888'): Coloring {
	const palette: string[] = [];
	const slot = new Map<string, number>();
	const index = new Uint16Array(keys.length);
	keys.forEach((k, i) => {
		const c = colors.get(k) ?? fallback;
		let s = slot.get(c);
		if (s === undefined) {
			s = palette.length;
			slot.set(c, s);
			palette.push(c);
		}
		index[i] = s;
	});
	return { index, palette };
}

/** Sequential colouring of timestamps (oldest → newest) in `steps` quantised ramp steps. */
export function byTime(ms: number[], stops: string[], steps = 24): Coloring & { lo: number; hi: number } {
	let lo = Infinity;
	let hi = -Infinity;
	for (const t of ms) {
		if (!Number.isFinite(t)) continue;
		if (t < lo) lo = t;
		if (t > hi) hi = t;
	}
	if (!Number.isFinite(lo)) lo = hi = 0;
	const palette = Array.from({ length: steps }, (_, i) => rampColor(steps === 1 ? 1 : i / (steps - 1), stops));
	const index = new Uint16Array(ms.length);
	const span = hi - lo || 1;
	ms.forEach((t, i) => (index[i] = Math.min(steps - 1, Math.max(0, Math.floor(((t - lo) / span) * steps)))));
	return { index, palette, lo, hi };
}

// ---------------------------------------------------------------- occupancy

export interface Occupancy {
	/** Bucket centres (ms). */
	times: number[];
	/** Points per bucket, all regions. */
	totals: number[];
	/** Per region: share of the bucket's points (0 when the bucket is empty). */
	share: Map<string, number[]>;
	count: Map<string, number>;
}

/** Region occupancy over time from a latent sample: per time bucket, the share of points that
 *  fall in each region (spec 51 §6.3 "occupancy over time"). */
export function occupancy(points: { region_id: string; ts: string }[], buckets = 30): Occupancy {
	const ms = points.map((p) => Date.parse(p.ts));
	const finite = ms.filter(Number.isFinite);
	if (!finite.length) return { times: [], totals: [], share: new Map(), count: new Map() };
	const lo = Math.min(...finite);
	const hi = Math.max(...finite);
	const nb = hi === lo ? 1 : buckets;
	const width = (hi - lo) / nb || 1;
	const totals = new Array<number>(nb).fill(0);
	const counts = new Map<string, number[]>();
	const count = new Map<string, number>();
	points.forEach((p, i) => {
		if (!Number.isFinite(ms[i])) return;
		const b = Math.min(nb - 1, Math.floor((ms[i] - lo) / width));
		totals[b]++;
		let row = counts.get(p.region_id);
		if (!row) counts.set(p.region_id, (row = new Array<number>(nb).fill(0)));
		row[b]++;
		count.set(p.region_id, (count.get(p.region_id) ?? 0) + 1);
	});
	const share = new Map<string, number[]>();
	for (const [r, row] of counts) share.set(r, row.map((c, b) => (totals[b] ? c / totals[b] : 0)));
	const times = totals.map((_, b) => lo + (b + 0.5) * width);
	return { times, totals, share, count };
}

/** Mean coordinates per region (the projected centroid when no `centroid` row was fetched). */
export function regionCentroids(points: { region_id: string; kind?: string; coords: number[] }[]): Map<string, number[]> {
	const explicit = new Map<string, number[]>();
	const sum = new Map<string, { c: number[]; n: number }>();
	for (const p of points) {
		if (p.kind === 'centroid') explicit.set(p.region_id, p.coords);
		const s = sum.get(p.region_id) ?? { c: p.coords.map(() => 0), n: 0 };
		p.coords.forEach((v, d) => (s.c[d] = (s.c[d] ?? 0) + v));
		s.n++;
		sum.set(p.region_id, s);
	}
	const out = new Map<string, number[]>();
	for (const [r, s] of sum) out.set(r, explicit.get(r) ?? s.c.map((v) => v / s.n));
	return out;
}

/** Smoothstep tween between two positions, t in [0, 1]. */
export function tween(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
	const k = t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t);
	return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
}

/** PromQL for the per-region share of membership weight of one expert space over time
 *  (`fa_region_weight{expert, region, inst, tf}`, written per closed bar by model-runtime). */
export function regionWeightQuery(expertSpaceId: string): string {
	const sel = `fa_region_weight{expert="${expertSpaceId.replace(/["\\]/g, '')}"}`;
	return `sum by (region) (${sel}) / scalar(sum(${sel}))`;
}

/** A Prometheus `query_range` matrix (as VictoriaMetrics answers it) → occupancy shares on a
 *  shared time axis (missing samples = 0). */
export type VmMatrix = { data?: { result?: { metric: Record<string, string>; values: [number, string][] }[] } };
export function occupancyFromMatrix(resp: VmMatrix): Occupancy {
	const result = resp.data?.result ?? [];
	const ts = [...new Set(result.flatMap((r) => r.values.map((v) => v[0])))].sort((a, b) => a - b);
	const at = new Map(ts.map((t, i) => [t, i]));
	const share = new Map<string, number[]>();
	const count = new Map<string, number>();
	for (const r of result) {
		const key = r.metric.region ?? JSON.stringify(r.metric);
		const row = new Array<number>(ts.length).fill(0);
		for (const [t, v] of r.values) {
			const x = Number(v);
			row[at.get(t)!] = Number.isFinite(x) ? x : 0;
		}
		share.set(key, row);
		count.set(key, r.values.length);
	}
	return { times: ts.map((t) => t * 1000), totals: ts.map(() => 1), share, count };
}
