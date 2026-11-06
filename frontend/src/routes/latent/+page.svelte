<script lang="ts">
	import Planned from '../../lib/Planned.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
</script>

<ViewHeader title="Latent & Graph" spec="51 §6.3" note="waits on snapshots from trainer (spec 36)" />

<div class="grid cols">
	<Planned
		title="Snapshots & experts"
		items={[
			'Snapshot picker: timescale × snapshot (active / candidate / retired)',
			'Expert list with routing share',
			'Residual covariance heatmap between experts (target ≈ 0, D-008)'
		]}
		endpoints={['GET /api/latent/snapshots', 'GET /api/latent/{snapshot}/experts']}
	/>
	<Planned
		title="Expert space projection"
		items={[
			'2D/3D scatter of latent points over a time range (WebGL, server-side downsampled)',
			'Colour by region, instrument or time',
			'Live position of selected instruments animated'
		]}
		endpoints={['GET /api/latent/{snapshot}/projection', 'WS /ws/latent']}
	/>
	<Planned
		title="Region detail"
		items={[
			'Occupancy over time, lineage across generations (D-007)',
			'Typical decoder reconstructions: chart renderings, retrieved analogs',
			'Outgoing transitions with credible intervals'
		]}
		endpoints={['GET /api/latent/{snapshot}/regions']}
	/>
	<Planned
		title="MCMC region graph"
		items={[
			'Regions as nodes, transition posteriors as edges (width = mean, opacity = certainty)',
			'Cross-timescale hierarchy toggle',
			'Probability threshold filter, sampler comparison'
		]}
		endpoints={['GET /api/mcmc/{snapshot}/graph', 'GET /api/graph/neighborhood']}
	/>
</div>

<style>
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
	}
</style>
