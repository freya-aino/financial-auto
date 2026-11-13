// Exact live position of an instrument in the projection (spec 51 §6.3): the live latent mean
// (`ExpertStateView.mean` on `/ws/latent`) projected with the PCA the projection was drawn with
// (`pca{mean, components}` of `GET /api/latent/{snapshot}/projection`), i.e. the same
// `(x − mean)·components` control-api applies to the sample points. Pure so it is unit-tested.

export interface PcaBasis {
	mean: number[];
	components: number[][];
}

/**
 * `(x − pca.mean)·components[j]` for every component, or `null` when there is nothing to
 * project (no mean, no PCA, no components) or the dimensions do not match — the caller then
 * falls back to the region centroid.
 */
export function projectMean(mean: readonly number[] | null | undefined, pca: PcaBasis | null | undefined): number[] | null {
	if (!mean?.length || !pca?.components?.length || pca.mean?.length !== mean.length) return null;
	if (pca.components.some((c) => c.length !== mean.length)) return null;
	if (mean.some((v) => !Number.isFinite(v))) return null;
	return pca.components.map((c) => {
		let s = 0;
		for (let i = 0; i < mean.length; i++) s += c[i] * (mean[i] - pca.mean[i]);
		return s;
	});
}

/** Where to draw a tracked instrument: its projected mean, else its region's centroid. */
export function livePosition(
	mean: readonly number[] | null | undefined,
	pca: PcaBasis | null | undefined,
	centroid: number[] | undefined
): { coords: number[]; exact: boolean } | null {
	const p = projectMean(mean, pca);
	if (p) return { coords: p, exact: true };
	return centroid ? { coords: centroid, exact: false } : null;
}
