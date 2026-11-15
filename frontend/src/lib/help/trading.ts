// Trading Desk pages: /trading, /trading/pipeline, /trading/risk, /trading/validation, /trading/ledger.
export const trading: Record<string, string> = {
	// /trading — tiles and charts
	'trading.equity':
		'Equity as the risk service reports it, or from the paper book while that feed is down. This is a live in-memory view; the exact, authoritative figures come from the ledger (Trading → Ledger).',
	'trading.deployable':
		'Equity minus the cash reserved for today’s payout quota. Strategy sleeves are allocated out of the deployable part only.',
	'trading.equityChart':
		'Equity and P&L sampled in the browser from the live risk frames (or polling); the newest 300 samples survive reloads. Exact history lives in the ledger, not here.',
	'trading.drawdown':
		'Fall from the equity peak. The reported line is the risk-engine’s own figure — at the cap it halts the portfolio until re-armed; the client line is recomputed from sampled equity as a cross-check.',
	'trading.exposureCaps':
		'Each gauge shows a live figure against its cap from the risk limits; the gap between them is the headroom left before the risk-engine rejects or trims orders. Position gauges use the per-position cap (max_position_frac × equity).',

	// /trading — panels
	'trading.positions':
		'Positions of control-api’s in-memory paper book, marked live; non-EUR fills are converted at the latest known rate (a missing rate falls back to 1.0, warned and counted). The authoritative book is the ledger.',
	'trading.ticket':
		'Hand-built order that walks the same path as strategy orders: order.intent.paper → risk-engine pre-trade checks → paper venue. Check risk runs those checks as a dry run first; nothing is sent until the exact ticket has passed.',
	'trading.haltScopes':
		'Each halt blocks new orders in its scope: portfolio, broker and model halts flatten, daily_loss and fx are reduce-only, source halts flatten that source’s instruments. A halt stays until re-armed.',
	'trading.orders':
		'Order intents with their fills, costs and the strategies behind them, newest first. A resting row is an acknowledged limit order waiting on the paper venue until the price crosses it.',
	'trading.signals':
		'Each strategy’s latest wanted exposure per instrument in [−1, 1] of its sleeve (band = mean ± stddev). The allocator and risk caps turn these into orders — a signal alone trades nothing.',

	// /trading — strategies table and side panels
	'trading.strategies':
		'Lifecycle, sleeve, attributed P&L, shadow progress and promotion gate per strategy, from the risk-engine’s persisted state. Every action asks for confirmation and is audited.',
	'trading.sleeve':
		'The slice of deployable equity a strategy may use: allocator weight × deployable. Its positions, caps and drawdown are all measured against this sleeve.',
	'trading.strategyDrawdown':
		'Drawdown of the strategy’s own attributed book relative to its sleeve. Reaching strategy_max_drawdown trips a per-strategy kill switch: the risk-engine flattens and pauses just this strategy.',
	'trading.lifecycle':
		'Evolved strategies climb candidate → holdout_passed → shadow → active and end retired, moved by the pipeline on evidence. paused and flattening are operator or safety side-states, not pipeline stages.',
	'trading.limits':
		'The hard limits the risk-engine enforces. From here they can only be tightened — loosening one needs an edit of risk.toml and a restart, so limits never loosen in the heat of the moment.',
	'trading.pnlRange':
		'Realised P&L rebuilt from the fills persisted in the ledger, for one range and optionally one strategy. Unrealised P&L and the FX split are in Trading → Ledger.',
	'trading.reconciliation':
		'The risk-engine regularly compares its in-memory book with the Postgres ledger at one cut, in exact decimals. Any difference halts scope broker (flatten, no orders) until the books agree and broker is re-armed.',

	// /trading/pipeline
	'trading.pipelineStrategies':
		'Every evolved strategy with the evidence the pipeline decides on: selection PBO and deflated Sharpe, the single holdout outcome and the gate. Click a row to open its trial reports in the Validation card.',
	'trading.dsrPbo':
		'The round’s thresholds. PBO is the probability the backtest winner is overfit (estimated by cross-validation over the trials); the deflated Sharpe discounts a Sharpe for how many candidates were tried. Unknown values block — they never pass.',
	'trading.pipelineRound':
		'What the last automatic round did (risk-engine, every 60 s): holdout coverage per timescale, admissions, promotions and retirements, and the managed live set written for strategy-runtime.',
	'trading.rejected':
		'Candidates the round turned away at selection, newest first, with the recorded reason — typically PBO or deflated-Sharpe thresholds not met, or missing evidence.',
	'trading.tournament':
		'Fixed allocators race the serving one on the same strategy statistics, each in a virtual book with the same cost assumption. The ranking is advisory — switching the serving allocator is a limits change, not a button here.',

	// /trading/validation (holdouts & shadow)
	'trading.holdouts':
		'A holdout is a recent data segment locked away per timescale and never touched by selection. Locked rows are immutable; a candidate gets one evaluation on it, because repeated tries would just overfit the holdout too.',
	'trading.holdoutEvaluate':
		'Queues a single promotion replay of the candidate inside the chosen locked holdout. The outcome lands in the candidate’s trial reports and is the holdout evidence the pipeline and gate use.',
	'trading.shadowConsistency':
		'Compares what the strategy actually earned in shadow (hourly Sharpe of its fill-marked virtual book) with its backtest Sharpe scaled to the hour. The gate blocks promotion when shadow falls more than 2 standard errors below.',

	// /trading/risk
	'trading.killVsHalts':
		'The kill switch is the manual portfolio halt: flatten everything, no orders, surviving restarts until re-armed. Every other scope is raised by the risk-engine itself when one of its rules trips.',
	'trading.haltsPanel':
		'Active halts from ledger.risk_state (so they survive restarts) merged with what control-api saw on the bus. Each one is blocking order flow in its scope right now.',
	'trading.rearm':
		'Re-arming a halt publishes risk.resume for its scope and records your reason in the audit log; re-arming the portfolio also resets the drawdown peak. Positions are not re-opened.',
	'trading.riskEvents':
		'The risk-engine’s decisions as they happened: halts, resumes, feed-guard and FX actions, reconciliation diffs. History comes from the audit log; rows marked live just arrived on the bus.',

	// /trading/ledger
	'trading.pnlBreakdown':
		'The book’s P&L from the ledger, in parts that sum exactly to net. The FX lines split each gain into the price move (at the closing rate) and the EUR exchange-rate move, plus the revaluation of foreign cash.',
	'trading.pnlPerStrategy':
		'Each strategy’s attributed share of the book P&L, from fill attribution plus marked sub-positions. With unattributed trades and book-level cash items the rows sum exactly to the book figure.',
	'trading.lots':
		'The open tax lots behind the positions, matched FIFO per venue and instrument. When a lot closes, its realised gain is computed from exactly these recorded prices and FX rates.',
	'trading.statement':
		'Opening and closing cash, equity and positions for the range, replayed purely from recorded fills and cash movements and valued at ECB daily rates. The CSV download lists every fill.',
	'trading.orphans':
		'A fill whose order the ledger does not recognise is parked here instead of booked — and any open orphan halts scope broker. Book it once the instrument is resolved, or dismiss it with a reason; both are audited.',
	'trading.payout':
		'Books a cash movement out of the book. Each book and day has a payout quota (reserved out of equity, so it is not deployable); booking is refused when the amount exceeds the quota minus what that day already paid.',
	'trading.orgBooks':
		'Double-entry books of the organisation around the trading system: each account belongs to a segment (trading, infrastructure, operations, owner, treasury). The trial balance must be exactly zero — anything else means the books are broken.',
	'trading.journal':
		'Append-only accounting entries, newest first. Postings of an entry sum to zero (debit positive, credit negative), and an entry_ref that was booked once is never booked again.',
	'trading.closeDay':
		'Books one finished UTC day into the organisational books: the trading book’s net P&L as revenue, the opex accrual and the revenue split, as deterministic entries. Closing the same day again posts nothing.',
	'trading.fxRestatement':
		'Rebooks one instrument’s fills from the wrong currency at the proper rate — only the FX leg changes, never quantities, prices or fees. The dry run shows the exact plan first, because applying rewrites booked history; restart the risk-engine afterwards.',
};
