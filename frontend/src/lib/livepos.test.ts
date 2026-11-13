import { describe, expect, it } from 'vitest';
import { livePosition, projectMean } from './livepos';

const pca = { mean: [1, 2, 3], components: [[1, 0, 0], [0, 0.6, 0.8]] };

describe('projectMean', () => {
	it('projects (x − mean)·components like control-api', () => {
		const p = projectMean([2, 4, 7], pca)!;
		expect(p[0]).toBeCloseTo(1);
		expect(p[1]).toBeCloseTo(0.6 * 2 + 0.8 * 4);
	});
	it('maps the PCA mean to the origin', () => {
		expect(projectMean([1, 2, 3], pca)).toEqual([0, 0]);
	});
	it('refuses missing or mismatched inputs', () => {
		expect(projectMean(undefined, pca)).toBeNull();
		expect(projectMean([], pca)).toBeNull();
		expect(projectMean([1, 2], pca)).toBeNull();
		expect(projectMean([1, 2, 3], null)).toBeNull();
		expect(projectMean([1, 2, 3], { mean: [0, 0, 0], components: [] })).toBeNull();
		expect(projectMean([1, 2, 3], { mean: [0, 0, 0], components: [[1, 0]] })).toBeNull();
		expect(projectMean([1, NaN, 3], pca)).toBeNull();
	});
});

describe('livePosition', () => {
	it('prefers the exact projection and falls back to the centroid', () => {
		expect(livePosition([2, 2, 3], pca, [9, 9])).toEqual({ coords: [1, 0], exact: true });
		expect(livePosition(undefined, pca, [9, 9])).toEqual({ coords: [9, 9], exact: false });
		expect(livePosition([1, 2], pca, [9, 9])).toEqual({ coords: [9, 9], exact: false });
		expect(livePosition(undefined, pca, undefined)).toBeNull();
	});
});
