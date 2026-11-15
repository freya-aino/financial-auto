// Evolution Lab pages: /evolution, /evolution/decoders.
export const evolution: Record<string, string> = {
	// /evolution
	'evolution.generations':
		'Each row is one evolution run: a population of genomes evaluated under successive halving — everyone gets a short cheap evaluation, only the best advance to longer ones — looping variation → evaluation → selection for up to max generations.',
	'evolution.rung':
		'The successive-halving stage within a loop: rung 0 evaluates every candidate on the minimum budget, then each rung keeps the top 1/eta and gives survivors eta× the budget. ✓ marks the final rung.',
	'evolution.dsr':
		'Deflated Sharpe ratio: the best trial’s Sharpe corrected for how many candidates were tried and the spread, skew and kurtosis of their results. A high Sharpe found by testing many genomes deflates toward zero; gates use DSR, never raw Sharpe.',
	'evolution.pbo':
		'Probability of Backtest Overfitting: how often the candidate that won in-sample lands in the worse half out-of-sample, across many block splits of the same trials. Near 50% means selection was luck; promotion wants it low.',
	'evolution.start':
		'Queues an evolution run: the genomes are evaluated on the chosen encoder snapshot’s latents under successive halving, and each loop’s elites are varied (mutation and crossover) into the next population.',
	'evolution.eta':
		'The halving factor: each rung keeps the best 1/eta of its candidates and multiplies the survivors’ evaluation budget by eta. 3 means a third survive each rung.',
	'evolution.gpu':
		'The single GPU is leased to jobs in strict priority order, preempting lower classes when memory runs short. Evaluation jobs run against GPU-millisecond budgets; a rung that would overshoot the run’s total budget is skipped.',
	'evolution.pareto':
		'Candidates on two chosen objectives. With several objectives there is no single winner, so selection keeps the Pareto front: the candidates no other one beats on every objective at once (highlighted, rank 0).',
	'evolution.jobs':
		'The individual evaluation jobs behind the generations: each scores one genome at one rung within its GPU-ms budget. The score is the genome’s family skill; the full objectives feed the Pareto ranking.',
	'evolution.lifecycle':
		'Where each evolved strategy stands on the way to trading: selection statistics (PBO, DSR) → locked holdout → paper shadow → active. Badges show the evidence for each gate; a demoted strategy shows the reason.',
	// lib/GenomeDrawer.svelte / lib/LineageTree.svelte
	'evolution.fitness':
		'Every recorded evaluation of this genome: its objective values per generation and rung it survived, so you can see whether it improves as budgets grow.',
	'evolution.genomeLineage':
		'The genome’s ancestry through the variation operators: which parents it was crossed and mutated from, traced back through the generations.',
	// /evolution/decoders
	'evolution.slots':
		'One slot per (output family, timescale): the serving decoder publishes the outputs strategies consume, while the runner-up card runs in shadow on the same inputs. Both are scored online on the same losses.',
	'evolution.crps':
		'CRPS scores a forecast distribution against the realised value once the horizon closes: low when the predicted quantiles sit tight around what happened. Like an error, lower is better.',
	'evolution.promotionRule':
		'The shadow replaces the serving decoder automatically once it has ≥ 200 online-scored outputs, ≥ 30 minutes of runtime, and a mean online loss better by more than 2%.',
	'evolution.cards':
		'Every trained decoder stored as weights plus a card: its genome, evaluation objectives and score. Pinning a loaded card forces it to serve its slot regardless of the online ranking.',
	'evolution.skillColumn':
		'Skill = 1 − loss/naive-baseline loss at evaluation time: positive (green) means the decoder beat a naive forecast; negative means it did not.',
};
