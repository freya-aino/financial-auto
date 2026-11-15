// Latent & Graph pages: /latent, /latent/mcmc, /latent/snapshots.
export const latent: Record<string, string> = {
	// /latent toolbar
	'latent.timescale':
		'Each timescale (1m, 1h, …) has its own encoder, regions and transition graph; this picks which resolution the whole page shows.',
	'latent.snapshot':
		'A snapshot is one frozen, trained encoder generation for a timescale. The active one serves live; a candidate can be promoted and an active one rolled back — both ask for confirmation and are audited.',
	// /latent panels
	'latent.graph':
		'Nodes are latent regions (recurring market states of one expert space); an edge is the learned probability that an instrument moves from one region to the other at the next bar. Edge width is the posterior mean; node size is current occupancy.',
	'latent.regions':
		'Every region of this timescale with its owning expert, status and occupancy. The lineage id is an identity that survives retraining: matched regions keep it, and splits or merges record their ancestry.',
	'latent.states':
		'The latest encoder output per instrument: the router’s weight on each expert, and per expert its current region, the size of its latent mean ‖μ‖ and its uncertainty σ̄.',
	'latent.decoderOutputs':
		'Live decoder outputs by family: forecast quantiles and retrieved analogs (similar past windows with their similarity and age). Decoders see only the latents, so this feed shows what the spaces actually encode.',
	// lib/Experts.svelte
	'latent.experts':
		'The snapshot’s experts, each encoding into its own latent space. Gate mean is the average routing weight an expert receives; top share is how often it is the highest-weighted expert for a state.',
	'latent.expertCorrelation':
		'Correlation between the experts’ latent activity on recent live states. Experts are trained to encode different things, so off-diagonal values near ±1 mean two experts are redundant.',
	// lib/Projection.svelte
	'latent.projection':
		'A PCA view of one expert’s latent space: each point is one encoded market moment, pressed into 2-D or 3-D. Nearby points are states the encoder considers similar, and clusters usually line up with regions.',
	// lib/RegionDetail.svelte
	'latent.regionDetail':
		'What this region meant in market terms: stored episodes of instruments sitting in it, each with the price path over the stay and the decoder outputs produced inside that window.',
	// lib/GraphExplorer.svelte
	'latent.graphExplorer':
		'Browse the graph database behind these pages from any id (snapshot, region, expert space, genome, instrument): its neighbouring nodes and edges, or one of the whitelisted named queries.',
	// /latent/mcmc
	'latent.kernels':
		'Per expert, several transition-model samplers compete on live data, scored before each answer is known: log score, Brier and ECE/PIT measure accuracy and calibration. The serving one comes from the Pareto front and is only swapped for a clear gain.',
	'latent.mcmcQuery':
		'Questions answered from the posterior of the serving kernel, as mean [lower, upper] over draws. Hitting time: bars until any target region is first reached. Stationary: the long-run share of time in each region.',
	'latent.lineageTrace':
		'How a region’s identity survived retraining: DescendsFrom edges record which older regions it was carried, split or merged from, with their overlap. A region with no ancestors was born in this snapshot.',
	'latent.hierarchy':
		'Links fine-timescale regions to the coarse regions they occur inside (e.g. 1m regions within a 1h region): conditional probabilities both ways and PMI, bits of co-occurrence above chance.',
	'latent.crossPriors':
		'Instrument pairs where one appears to lead the other: correlation, lead correlation, and ΔCRPS — the forecast improvement from using the source. Ranked proposals seed cross-asset decoders.',
	'latent.memberships':
		'Per closed bar each expert spreads soft weights over nearby regions instead of one hard pick. H is the entropy of those weights, ν the novelty of the state; an orange entry is outside every known region.',
	'latent.lifecycle':
		'Region births, splits, merges and deaths as they stream in. A regime break means the new region could not be matched to the old set, so its transition kernel restarted from the prior.',
	// /latent/snapshots
	'latent.snapshotList':
		'Every persisted encoder snapshot with its family (statistical or neural), lifecycle status (candidate → shadow → active → retired) and age. Select a row to inspect its checklist and health.',
	'latent.checklist':
		'The gates a candidate must pass before promotion: enough effective experts (K_eff), expert diversity (latent CKA — similarity of the experts’ latents, lower is better), SSL/reconstruction skill, per-expert skill. The API refuses promote while an item fails.',
	'latent.expertHealthPanel':
		'Expert-space health on the held-out 15% time-ordered validation split: effective experts, residual correlation, and each expert’s usage and skill against a naive baseline.',
	'latent.kEff':
		'Effective number of experts the router actually uses. Even routing across K experts gives K; a collapsed router that leans on one expert gives 1. Promotion wants at least 2.',
	'latent.residCorr':
		'The largest correlation between two experts’ prediction residuals. Since D-099 the spaces are trained end to end on reconstruction, so this is a diagnostic to watch, not a promotion gate; diversity is gated by latent CKA instead.',
	'latent.liveHealth':
		'The same health numbers computed on the windows the snapshot actually serves (active or shadow), published on closed bars — a check that validation-split health holds up live.',
	'latent.training':
		'Hyper-parameters and metrics from the snapshot’s training manifest: architecture, validation losses (SSL = the self-supervised reconstruction part), region mass carried from the parent snapshot, and training time.',
	'latent.shadowGauges':
		'Counters from serving this candidate in shadow next to the active encoder on the same bars: shadow minutes and region agreement, the share of steps where the shadow’s region continues the active one.',
};
