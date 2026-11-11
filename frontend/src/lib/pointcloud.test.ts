import { describe, expect, it } from 'vitest';
import { HOME_2D, byTime, categorical, groupByColor, nearest, normalise, normaliser, occupancy, orbit, projectAll, rampColor, regionCentroids, toScreen, tween, zoomAt } from './pointcloud';

describe('point cloud camera', () => {
	it('normalises into the unit box around the mean', () => {
		const c = normalise([[0, 0], [2, 0], [4, 0]], 2);
		expect(Array.from(c.xyz)).toEqual([-1, 0, 0, 0, 0, 0, 1, 0, 0]);
		const f = normaliser([[0, 0], [2, 0], [4, 0]], 2);
		expect(f([3, 0])).toEqual([0.5, 0, 0]);
	});

	it('projects the origin to the centre and zooms about the cursor', () => {
		const [x, y] = toScreen(HOME_2D, 400, 200, 0, 0, 0);
		expect([x, y]).toEqual([200, 100]);
		// the point under the cursor stays put
		const before = toScreen(HOME_2D, 400, 200, 0.5, 0.25, 0);
		const cam = zoomAt(HOME_2D, 3, before[0], before[1], 400, 200);
		const after = toScreen(cam, 400, 200, 0.5, 0.25, 0);
		expect(after[0]).toBeCloseTo(before[0], 6);
		expect(after[1]).toBeCloseTo(before[1], 6);
		expect(cam.zoom).toBe(3);
	});

	it('projectAll matches toScreen in 3-D and orbit clamps the pitch', () => {
		const cloud = normalise([[1, 2, 3], [-1, 0, 2], [0, -2, -5]], 3);
		const cam = orbit({ ...HOME_2D }, 50, 400);
		expect(cam.pitch).toBeLessThan(Math.PI / 2);
		const out = projectAll(cloud, cam, 300, 300, new Float32Array(6));
		for (let i = 0; i < 3; i++) {
			const [x, y] = toScreen(cam, 300, 300, cloud.xyz[i * 3], cloud.xyz[i * 3 + 1], cloud.xyz[i * 3 + 2]);
			expect(out[i * 2]).toBeCloseTo(x, 3);
			expect(out[i * 2 + 1]).toBeCloseTo(y, 3);
		}
	});

	it('picks the nearest point within the radius', () => {
		const s = Float32Array.from([10, 10, 50, 50, 52, 52]);
		expect(nearest(s, 3, 51, 51, 5)).toBeGreaterThanOrEqual(1);
		expect(nearest(s, 3, 100, 100, 5)).toBe(-1);
		expect(nearest(s, 3, 11, 10, 5, (i) => i === 0)).toBe(-1);
	});

	it('handles 50 000 points', () => {
		const coords = Array.from({ length: 50_000 }, (_, i) => [Math.sin(i), Math.cos(i * 0.7), Math.sin(i * 0.3)]);
		const cloud = normalise(coords, 3);
		const out = new Float32Array(cloud.n * 2);
		const t0 = performance.now();
		projectAll(cloud, { yaw: 0.6, pitch: 0.35, zoom: 1, panX: 0, panY: 0 }, 800, 420, out);
		nearest(out, cloud.n, 400, 210, 6);
		expect(performance.now() - t0).toBeLessThan(200);
		expect(groupByColor(byTime(coords.map((_, i) => i), ['#000000', '#ffffff'])).order.length).toBe(50_000);
	});
});

describe('point cloud colours and occupancy', () => {
	it('packs pixels in ImageData byte order', async () => {
		const { packRgba } = await import('./pointcloud');
		const px = new Uint8ClampedArray(new Uint32Array([packRgba('#102030', 1)]).buffer);
		expect(Array.from(px)).toEqual([0x10, 0x20, 0x30, 255]);
	});

	it('interpolates ramps and quantises time', () => {
		expect(rampColor(0, ['#000000', '#ffffff'])).toBe('#000000');
		expect(rampColor(0.5, ['#000000', '#ffffff'])).toBe('#808080');
		const t = byTime([0, 50, 100], ['#000000', '#ffffff'], 3);
		expect(Array.from(t.index)).toEqual([0, 1, 2]);
		expect([t.lo, t.hi]).toEqual([0, 100]);
	});

	it('groups indices by palette slot', () => {
		const c = categorical(['a', 'b', 'a'], new Map([['a', '#111111'], ['b', '#222222']]));
		expect(c.palette).toEqual(['#111111', '#222222']);
		const g = groupByColor(c);
		expect(Array.from(g.order)).toEqual([0, 2, 1]);
		expect(Array.from(g.start)).toEqual([0, 2, 3]);
	});

	it('computes per-bucket region shares', () => {
		const pts = [
			{ region_id: 'r1', ts: '2026-11-11T00:00:00Z' },
			{ region_id: 'r2', ts: '2026-11-11T00:00:00Z' },
			{ region_id: 'r1', ts: '2026-11-11T01:00:00Z' }
		];
		const o = occupancy(pts, 2);
		expect(o.totals).toEqual([2, 1]);
		expect(o.share.get('r1')).toEqual([0.5, 1]);
		expect(o.share.get('r2')).toEqual([0.5, 0]);
		expect(o.count.get('r1')).toBe(2);
	});

	it('prefers explicit centroid rows and tweens smoothly', () => {
		const c = regionCentroids([
			{ region_id: 'r', coords: [0, 0] },
			{ region_id: 'r', coords: [2, 2] },
			{ region_id: 'q', coords: [1, 1] },
			{ region_id: 'q', kind: 'centroid', coords: [5, 5] }
		]);
		expect(c.get('r')).toEqual([1, 1]);
		expect(c.get('q')).toEqual([5, 5]);
		expect(tween([0, 0, 0], [2, 0, 0], 0.5)).toEqual([1, 0, 0]);
		expect(tween([0, 0, 0], [2, 0, 0], 2)).toEqual([2, 0, 0]);
	});
});

describe('occupancy from VictoriaMetrics', () => {
	it('builds the query and aligns series on one axis', async () => {
		const { regionWeightQuery, occupancyFromMatrix } = await import('./pointcloud');
		expect(regionWeightQuery('s-e0')).toBe('sum by (region) (fa_region_weight{expert="s-e0"}) / scalar(sum(fa_region_weight{expert="s-e0"}))');
		const o = occupancyFromMatrix({ data: { result: [
			{ metric: { region: 'a' }, values: [[1, '0.25'], [2, '0.5']] },
			{ metric: { region: 'b' }, values: [[2, '0.5']] }
		] } });
		expect(o.times).toEqual([1000, 2000]);
		expect(o.share.get('a')).toEqual([0.25, 0.5]);
		expect(o.share.get('b')).toEqual([0, 0.5]);
	});
});
