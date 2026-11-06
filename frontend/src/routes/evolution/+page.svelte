<script lang="ts">
	import Planned from '../../lib/Planned.svelte';
	import ViewHeader from '../../lib/ViewHeader.svelte';
</script>

<ViewHeader title="Evolution Lab" spec="51 §6.4" note="waits on evo-orchestrator (spec 34)" />

<div class="grid cols">
	<Planned
		title="Populations"
		items={[
			'Decoders by family (visual, retrieval, text, RNN-prior)',
			'Strategies: GP programs vs neural policies',
			'Allocators, samplers; generation and GPU-hour budget used'
		]}
		endpoints={['GET /api/evolution/populations']}
	/>
	<Planned
		title="Pareto front"
		items={[
			'Pick 2–3 objectives (reconstruction, forecast, trading value, cost); rest as colour/size',
			'Selectable points open the genome inspector'
		]}
		endpoints={['GET /api/evolution/populations/{id}/pareto']}
	/>
	<Planned
		title="Runs"
		items={[
			'Successive-halving progress: survivors per round',
			'Start / pause / stop with GPU-hour budget'
		]}
		endpoints={['POST /api/evolution/runs', 'WS /ws/evolution']}
	/>
	<Planned
		title="Genome inspector & lineage"
		items={[
			'Genome definition: GP program pretty-printed or neural architecture',
			'Fitness breakdown, eval report (equity curve, deflated Sharpe, PBO, costs)',
			'Decoder outputs: images, text, forecast fans, analogs with time distance',
			'Lineage tree (mutate / crossover / finetune)'
		]}
		endpoints={['GET /api/evolution/genomes/{id}', 'GET /api/graph/lineage/{genome_id}']}
	/>
</div>

<style>
	.cols {
		grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
	}
</style>
