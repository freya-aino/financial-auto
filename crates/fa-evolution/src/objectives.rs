//! Named objectives → minimisation vectors shared by the worker result protocol and the
//! orchestrator's NSGA-II ranking (spec 34 §6.3, D-044).
//!
//! `JobResult` carries `objectives: {name → value}` and `directions: {name → "min" | "max"}`.
//! Ranking uses the sorted set of names that have a known direction; `max` objectives are
//! negated; a candidate missing a ranked objective gets `+∞` for it (worst). Names without a
//! direction (`n_eval`, `coverage`) are informational and never ranked.

use std::collections::BTreeMap;

/// Direction of one objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Min,
    Max,
}

impl Direction {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "min" | "minimise" | "minimize" | "lower" => Some(Self::Min),
            "max" | "maximise" | "maximize" | "higher" => Some(Self::Max),
            _ => None,
        }
    }
}

/// Sorted objective names that have a known direction in `directions`.
pub fn ranked_keys(directions: &BTreeMap<String, String>) -> Vec<String> {
    directions.iter().filter(|(_, d)| Direction::parse(d).is_some()).map(|(k, _)| k.clone()).collect()
}

/// Minimisation vector of one candidate over `keys` (see module docs for the rules).
pub fn minimised(
    objectives: &BTreeMap<String, f64>,
    directions: &BTreeMap<String, String>,
    keys: &[String],
) -> Vec<f64> {
    keys.iter()
        .map(|k| {
            let dir = directions.get(k).and_then(|d| Direction::parse(d)).unwrap_or(Direction::Min);
            match objectives.get(k) {
                Some(v) if v.is_finite() => match dir {
                    Direction::Min => *v,
                    Direction::Max => -*v,
                },
                _ => f64::INFINITY,
            }
        })
        .collect()
}

/// Pareto rank (0 = non-dominated) of every candidate.
pub fn pareto_ranks(objs: &[Vec<f64>]) -> Vec<usize> {
    let mut rank = vec![0usize; objs.len()];
    for (r, front) in crate::non_dominated_sort(objs).into_iter().enumerate() {
        for i in front {
            rank[i] = r;
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimised_negates_max_and_fills_missing() {
        let dirs: BTreeMap<String, String> = [("crps", "min"), ("crps_skill", "max"), ("n_eval", "count")]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let keys = ranked_keys(&dirs);
        assert_eq!(keys, vec!["crps".to_string(), "crps_skill".to_string()]);
        let o: BTreeMap<String, f64> =
            [("crps", 0.5), ("crps_skill", 0.2)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
        assert_eq!(minimised(&o, &dirs, &keys), vec![0.5, -0.2]);
        let partial: BTreeMap<String, f64> = [("crps", 0.5)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
        assert_eq!(minimised(&partial, &dirs, &keys), vec![0.5, f64::INFINITY]);
        assert_eq!(pareto_ranks(&[vec![1.0, 1.0], vec![2.0, 2.0], vec![0.5, 3.0]]), vec![0, 1, 0]);
    }
}
