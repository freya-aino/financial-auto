<script lang="ts">
	// Ops view (spec 51 §6.2): services, jobs & GPU, bus tap, JetStream streams, sources with
	// health + backfill, instruments & quarantine, audit log, runtime config.
	import AuditLog from '../lib/AuditLog.svelte';
	import BusTap from '../lib/BusTap.svelte';
	import Instruments from '../lib/Instruments.svelte';
	import JobsGpu from '../lib/JobsGpu.svelte';
	import RuntimeConfig from '../lib/RuntimeConfig.svelte';
	import Services from '../lib/Services.svelte';
	import Sources from '../lib/Sources.svelte';
	import Streams from '../lib/Streams.svelte';
	import ViewHeader from '../lib/ViewHeader.svelte';
</script>

<ViewHeader title="Ops" spec="51 §6.2" note="services · jobs & GPU · bus · streams · sources · instruments · audit · runtime config" />

<div class="grid top">
	<Services />
	<JobsGpu />
</div>

<BusTap />

<div class="grid mid">
	<Streams />
</div>

<Sources />

<div class="grid bottom">
	<Instruments />
	<AuditLog />
	<RuntimeConfig />
</div>

<style>
	.top {
		grid-template-columns: 2fr 1fr;
		margin-bottom: 12px;
	}
	.mid {
		margin-top: 12px;
	}
	.bottom {
		grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
		margin-top: 12px;
	}
	@media (max-width: 900px) {
		.top {
			grid-template-columns: 1fr;
		}
	}
</style>
