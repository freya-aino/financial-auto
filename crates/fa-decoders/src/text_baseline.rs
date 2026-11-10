//! Baseline text decoder (spec 34 §6.1 *Text*, provisional): a deterministic lexical scorer over
//! `news.item.>` items that stands in for the news / summary decoders until a learned text
//! model exists. Pure: the caller feeds [`NewsDoc`]s into a [`TextWindow`] per (instrument,
//! timescale) and asks for scores at a time; nothing here touches the bus or a clock.
//!
//! - **Instrument mapping** ([`InstrumentIndex`]): a document is attributed to every instrument
//!   whose ticker symbol appears as a case-sensitive upper-case token (`SPY`, `$SPY`) in the title
//!   or body, whose cleaned name (legal suffixes dropped, lower-cased) appears as a phrase, or
//!   whose symbol / name appears in the item's `entities_raw` (`query:Apple stock`, matched
//!   case-insensitively after the `<kind>:` prefix).
//! - **`text.news` score** in `[-1, 1]`: per item, the embedded finance lexicon ([`LEXICON_POS`],
//!   [`LEXICON_NEG`]) counts positive and negative hits with negation handling (a negator flips the
//!   polarity of the next [`NEGATION_SCOPE`] tokens); item polarity `s = (pos − neg) / (pos + neg)`,
//!   confidence `w = 1 − exp(−hits / 2)`. Over a window of length `W` ending at `now`, items are
//!   recency-decayed `d = exp(−age / (W / 2))` and volume-normalised with a shrinkage constant:
//!   `score = Σ d·w·s / (Σ d·w + SHRINK)`, so a single weak item cannot saturate the score.
//! - **`text.summary` score** in `[0, 1]`: novelty of the current window's vocabulary against the
//!   trailing window `[now − 2W, now − W)`: `1 − cos(tf_current, tf_trailing)` over hashed term
//!   frequencies ([`HASH_DIM`] buckets, FNV-1a). `0` when either window is empty.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

/// Decoder id published with every output of this module.
pub const DECODER_ID: &str = "text-stat-v1";
/// Term-frequency hash buckets for the novelty score.
pub const HASH_DIM: usize = 1024;
/// Tokens after a negator whose polarity is flipped.
pub const NEGATION_SCOPE: usize = 3;
/// Volume-normalisation constant of the window score (pseudo-weight of "no evidence").
pub const SHRINK: f64 = 0.5;

/// Positive finance lexicon (lower-case tokens).
pub const LEXICON_POS: &[&str] = &[
    "gain",
    "gains",
    "gained",
    "rally",
    "rallies",
    "rallied",
    "surge",
    "surges",
    "surged",
    "soar",
    "soars",
    "soared",
    "jump",
    "jumps",
    "jumped",
    "climb",
    "climbs",
    "climbed",
    "rise",
    "rises",
    "rose",
    "rising",
    "record",
    "high",
    "highs",
    "beat",
    "beats",
    "upgrade",
    "upgrades",
    "upgraded",
    "outperform",
    "outperforms",
    "bullish",
    "bull",
    "strong",
    "strength",
    "growth",
    "grow",
    "grows",
    "grew",
    "profit",
    "profits",
    "profitable",
    "boost",
    "boosts",
    "boosted",
    "recover",
    "recovers",
    "recovery",
    "rebound",
    "rebounds",
    "rebounded",
    "optimism",
    "optimistic",
    "buy",
    "buyback",
    "dividend",
    "expand",
    "expands",
    "expansion",
    "exceed",
    "exceeds",
    "exceeded",
    "success",
    "successful",
    "upside",
    "momentum",
    "approve",
    "approved",
    "approval",
    "win",
    "wins",
    "won",
    "robust",
    "resilient",
    "improve",
    "improves",
    "improved",
    "positive",
    "upbeat",
    "accelerate",
    "accelerates",
    "accelerating",
    "raised",
    "raises",
];

/// Negative finance lexicon (lower-case tokens).
pub const LEXICON_NEG: &[&str] = &[
    "loss",
    "losses",
    "lose",
    "loses",
    "lost",
    "fall",
    "falls",
    "fell",
    "falling",
    "drop",
    "drops",
    "dropped",
    "plunge",
    "plunges",
    "plunged",
    "slump",
    "slumps",
    "slumped",
    "crash",
    "crashes",
    "crashed",
    "tumble",
    "tumbles",
    "tumbled",
    "decline",
    "declines",
    "declined",
    "sink",
    "sinks",
    "sank",
    "slide",
    "slides",
    "slid",
    "miss",
    "misses",
    "missed",
    "downgrade",
    "downgrades",
    "downgraded",
    "underperform",
    "underperforms",
    "bearish",
    "bear",
    "weak",
    "weakness",
    "recession",
    "layoff",
    "layoffs",
    "bankruptcy",
    "bankrupt",
    "default",
    "defaults",
    "fraud",
    "lawsuit",
    "sued",
    "probe",
    "investigation",
    "fine",
    "fines",
    "penalty",
    "warn",
    "warns",
    "warned",
    "warning",
    "cut",
    "cuts",
    "risk",
    "risks",
    "risky",
    "fear",
    "fears",
    "panic",
    "selloff",
    "volatile",
    "volatility",
    "concern",
    "concerns",
    "worry",
    "worries",
    "worst",
    "low",
    "lows",
    "negative",
    "inflation",
    "tariff",
    "tariffs",
    "sanction",
    "sanctions",
    "halt",
    "halted",
    "suspend",
    "suspended",
    "delay",
    "delays",
    "delayed",
    "debt",
    "downturn",
    "deficit",
    "shortfall",
    "disappointing",
    "disappoint",
    "disappoints",
    "turmoil",
    "crisis",
    "lowered",
    "lowers",
];

/// Tokens that flip the polarity of the next [`NEGATION_SCOPE`] lexicon hits.
pub const NEGATORS: &[&str] = &["not", "no", "never", "without", "hardly", "barely", "nor", "neither", "cannot"];

/// Legal / fund suffixes dropped from instrument names before phrase matching.
const NAME_SUFFIXES: &[&str] = &[
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "plc",
    "sa",
    "ag",
    "nv",
    "se",
    "etf",
    "trust",
    "fund",
    "holdings",
    "holding",
    "group",
    "class",
    "a",
    "b",
    "c",
    "the",
    "shares",
    "common",
    "stock",
    "ordinary",
    "adr",
    "ads",
    "llc",
    "lp",
];

/// One news item as the decoder needs it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct NewsDoc {
    /// Event time (ms).
    pub ts_ms: i64,
    /// `news.item.<source>` token.
    pub source: String,
    pub title: String,
    pub body: String,
    /// `NewsItem.entities_raw` (`query:Apple stock`, `author:…`, bare tickers / names).
    pub entities: Vec<String>,
}

/// Lower-case alphanumeric tokens (`'` and `&` kept inside words, `n't` → `not`), at least 2 chars.
pub fn tokenize(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase().replace("n't", " not");
    lowered
        .split(|c: char| !(c.is_alphanumeric() || c == '&' || c == '\''))
        .map(|t| t.trim_matches('\''))
        .filter(|t| t.len() >= 2)
        .map(str::to_owned)
        .collect()
}

/// Lexical polarity of one text.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ItemScore {
    /// `(pos − neg) / (pos + neg)` after negation, 0 without hits.
    pub polarity: f64,
    pub pos: u32,
    pub neg: u32,
    pub tokens: usize,
}

impl ItemScore {
    pub fn hits(&self) -> u32 {
        self.pos + self.neg
    }

    /// `1 − exp(−hits / 2)`: one hit ≈ 0.39, three hits ≈ 0.78.
    pub fn confidence(&self) -> f64 {
        1.0 - (-(f64::from(self.hits())) / 2.0).exp()
    }
}

fn lexicon() -> (&'static HashSet<&'static str>, &'static HashSet<&'static str>, &'static HashSet<&'static str>) {
    use std::sync::OnceLock;
    static POS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    static NEG: OnceLock<HashSet<&'static str>> = OnceLock::new();
    static NEGATORS_SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    (
        POS.get_or_init(|| LEXICON_POS.iter().copied().collect()),
        NEG.get_or_init(|| LEXICON_NEG.iter().copied().collect()),
        NEGATORS_SET.get_or_init(|| NEGATORS.iter().copied().collect()),
    )
}

/// Scores one text with the embedded lexicon and negation handling.
pub fn score_text(text: &str) -> ItemScore {
    let (pos_set, neg_set, negators) = lexicon();
    let tokens = tokenize(text);
    let (mut pos, mut neg) = (0u32, 0u32);
    let mut negate_left = 0usize;
    for t in &tokens {
        if negators.contains(t.as_str()) {
            negate_left = NEGATION_SCOPE;
            continue;
        }
        let hit = if pos_set.contains(t.as_str()) {
            Some(true)
        } else if neg_set.contains(t.as_str()) {
            Some(false)
        } else {
            None
        };
        if let Some(mut positive) = hit {
            if negate_left > 0 {
                positive = !positive;
            }
            if positive {
                pos += 1;
            } else {
                neg += 1;
            }
        }
        negate_left = negate_left.saturating_sub(1);
    }
    let hits = pos + neg;
    let polarity = if hits > 0 { (f64::from(pos) - f64::from(neg)) / f64::from(hits) } else { 0.0 };
    ItemScore { polarity, pos, neg, tokens: tokens.len() }
}

/// FNV-1a over a token, reduced to a [`HASH_DIM`] bucket.
pub fn hash_bucket(token: &str) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in token.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h % HASH_DIM as u64) as usize
}

/// Hashed term frequencies of a token list (sparse: bucket → count).
pub fn hashed_tf(tokens: &[String]) -> HashMap<usize, f64> {
    let mut m = HashMap::new();
    for t in tokens {
        *m.entry(hash_bucket(t)).or_insert(0.0) += 1.0;
    }
    m
}

fn sparse_cosine(a: &HashMap<usize, f64>, b: &HashMap<usize, f64>) -> f64 {
    let na: f64 = a.values().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.values().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let dot: f64 = a.iter().filter_map(|(k, v)| b.get(k).map(|w| v * w)).sum();
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Reference row of one instrument for the matcher.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InstrumentRef {
    pub id: String,
    /// Venue symbols and ticker-like identifiers (`SPY`, `BTC-USD`).
    pub symbols: Vec<String>,
    /// Display names (`SPDR S&P 500 ETF Trust`).
    pub names: Vec<String>,
}

#[derive(Debug, Clone)]
struct Entry {
    id: String,
    /// Exact upper-case symbol tokens (length ≥ 2).
    symbols: Vec<String>,
    /// Lower-case name phrases (length ≥ 3, not a bare suffix).
    phrases: Vec<String>,
}

/// Maps documents to instruments by symbol, name and entity.
#[derive(Debug, Clone, Default)]
pub struct InstrumentIndex {
    entries: Vec<Entry>,
}

/// Lower-cases a name, drops parenthesised notes (`AAPL (yahoo)` → `aapl`), legal / fund suffix
/// words and punctuation.
pub fn clean_name(name: &str) -> String {
    let mut plain = String::with_capacity(name.len());
    let mut depth = 0usize;
    for c in name.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => plain.push(c),
            _ => {}
        }
    }
    let words: Vec<String> = plain
        .to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '&' || c == '-'))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect();
    // drop suffix words from the end, and a leading "the"
    let mut end = words.len();
    while end > 0 && NAME_SUFFIXES.contains(&words[end - 1].as_str()) {
        end -= 1;
    }
    let mut start = 0;
    while start < end && words[start] == "the" {
        start += 1;
    }
    let kept = &words[start..end];
    if kept.is_empty() { words.join(" ") } else { kept.join(" ") }
}

fn is_symbol_like(s: &str) -> bool {
    s.len() >= 2
        && s.len() <= 12
        && s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '.' || c == '-')
}

/// Word-boundary phrase containment in lower-cased text.
fn contains_phrase(haystack: &str, phrase: &str) -> bool {
    let mut from = 0;
    while let Some(i) = haystack[from..].find(phrase) {
        let start = from + i;
        let end = start + phrase.len();
        let before_ok = start == 0 || !haystack[..start].chars().next_back().is_some_and(char::is_alphanumeric);
        let after_ok = end == haystack.len() || !haystack[end..].chars().next().is_some_and(char::is_alphanumeric);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

impl InstrumentIndex {
    pub fn new(refs: &[InstrumentRef]) -> Self {
        let entries = refs
            .iter()
            .filter(|r| !r.id.is_empty())
            .map(|r| {
                let mut symbols: Vec<String> = r
                    .symbols
                    .iter()
                    .map(|s| s.trim().to_ascii_uppercase())
                    // a pair `BTC-USD` is also mentioned by its base `BTC`
                    .flat_map(|s| {
                        let base = s.split_once('-').map(|(b, _)| b.to_owned()).filter(|b| b.len() >= 3);
                        std::iter::once(s).chain(base)
                    })
                    .filter(|s| is_symbol_like(s))
                    .collect();
                symbols.sort();
                symbols.dedup();
                let mut phrases: Vec<String> = r
                    .names
                    .iter()
                    .map(|n| clean_name(n))
                    .filter(|p| p.len() >= 3 && !NAME_SUFFIXES.contains(&p.as_str()))
                    // a name that is just the ticker matches case-sensitively as a symbol only
                    .filter(|p| !symbols.iter().any(|sym| sym.to_lowercase() == *p))
                    .collect();
                phrases.sort();
                phrases.dedup();
                Entry { id: r.id.clone(), symbols, phrases }
            })
            .collect();
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Instrument ids the document mentions (sorted, unique).
    pub fn match_doc(&self, doc: &NewsDoc) -> Vec<String> {
        let text = format!("{} {}", doc.title, doc.body);
        // case-sensitive symbol tokens from the raw text
        let raw_tokens: HashSet<&str> = text
            .split(|c: char| !(c.is_alphanumeric() || c == '$' || c == '.' || c == '-'))
            .map(|t| t.trim_start_matches('$').trim_end_matches('.'))
            .filter(|t| !t.is_empty())
            .collect();
        let lowered = text.to_lowercase();
        let entities: Vec<String> = doc
            .entities
            .iter()
            .map(|e| e.split_once(':').map_or(e.as_str(), |(_, v)| v).trim().to_lowercase())
            .filter(|e| !e.is_empty())
            .collect();
        let mut out = BTreeSet::new();
        for e in &self.entries {
            let by_symbol = e.symbols.iter().any(|s| raw_tokens.contains(s.as_str()));
            let by_name = e.phrases.iter().any(|p| contains_phrase(&lowered, p));
            let by_entity = entities.iter().any(|ent| {
                e.symbols.iter().any(|s| ent == &s.to_lowercase() || contains_phrase(ent, &s.to_lowercase()))
                    || e.phrases.iter().any(|p| contains_phrase(ent, p))
            });
            if by_symbol || by_name || by_entity {
                out.insert(e.id.clone());
            }
        }
        out.into_iter().collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Scored {
    ts_ms: i64,
    source: String,
    score: ItemScore,
    tf: HashMap<usize, f64>,
}

/// Rolling set of scored items of one (instrument | market, timescale) key.
#[derive(Debug, Clone, Default)]
pub struct TextWindow {
    items: VecDeque<Scored>,
}

/// Scores of a window at one time.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct WindowScore {
    /// `text.news` score in `[-1, 1]`.
    pub news: f64,
    /// `text.summary` novelty in `[0, 1]`.
    pub novelty: f64,
    /// Items in the current window.
    pub n_items: usize,
    /// Items in the trailing window.
    pub n_trailing: usize,
    /// Distinct sources of the current window.
    pub sources: Vec<String>,
    /// Lexicon hits in the current window.
    pub hits: u32,
}

impl TextWindow {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a document (kept in time order; duplicates by `(ts_ms, title hash)` are not detected
    /// here: callers dedupe by `dedup_key` / `event_id`).
    pub fn push(&mut self, doc: &NewsDoc) {
        let text = format!("{} {}", doc.title, doc.body);
        let scored = Scored {
            ts_ms: doc.ts_ms,
            source: doc.source.clone(),
            score: score_text(&text),
            tf: hashed_tf(&tokenize(&text)),
        };
        let pos = self.items.partition_point(|x| x.ts_ms <= scored.ts_ms);
        self.items.insert(pos, scored);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Drops items older than `2 · window_ms` before `now_ms` (what [`Self::score_at`] can still use).
    pub fn prune(&mut self, now_ms: i64, window_ms: i64) {
        let cutoff = now_ms - 2 * window_ms.max(1);
        while self.items.front().is_some_and(|x| x.ts_ms < cutoff) {
            self.items.pop_front();
        }
    }

    /// Scores as of `now_ms` over `[now − W, now]` with trailing `[now − 2W, now − W)`. Items after
    /// `now_ms` are ignored (point-in-time).
    pub fn score_at(&self, now_ms: i64, window_ms: i64) -> WindowScore {
        let w = window_ms.max(1);
        let tau = w as f64 / 2.0;
        let mut num = 0.0;
        let mut den = 0.0;
        let mut n_items = 0;
        let mut hits = 0;
        let mut sources = BTreeSet::new();
        let mut cur_tf: HashMap<usize, f64> = HashMap::new();
        let mut trail_tf: HashMap<usize, f64> = HashMap::new();
        let mut n_trailing = 0;
        for it in &self.items {
            if it.ts_ms > now_ms {
                break;
            }
            let age = now_ms - it.ts_ms;
            if age <= w {
                let d = (-(age as f64) / tau).exp();
                let wgt = d * it.score.confidence();
                num += wgt * it.score.polarity;
                den += wgt;
                n_items += 1;
                hits += it.score.hits();
                sources.insert(it.source.clone());
                for (k, v) in &it.tf {
                    *cur_tf.entry(*k).or_insert(0.0) += v;
                }
            } else if age <= 2 * w {
                n_trailing += 1;
                for (k, v) in &it.tf {
                    *trail_tf.entry(*k).or_insert(0.0) += v;
                }
            }
        }
        let news = if den > 0.0 { (num / (den + SHRINK)).clamp(-1.0, 1.0) } else { 0.0 };
        let novelty =
            if n_items > 0 && n_trailing > 0 { (1.0 - sparse_cosine(&cur_tf, &trail_tf)).clamp(0.0, 1.0) } else { 0.0 };
        WindowScore { news, novelty, n_items, n_trailing, sources: sources.into_iter().collect(), hits }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(ts_ms: i64, source: &str, title: &str, entities: &[&str]) -> NewsDoc {
        NewsDoc {
            ts_ms,
            source: source.into(),
            title: title.into(),
            body: String::new(),
            entities: entities.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    #[test]
    fn lexicon_scores_polarity_with_negation() {
        let up = score_text("Shares surge to record high after strong profit beat");
        assert_eq!((up.pos, up.neg), (6, 0));
        assert_eq!(up.polarity, 1.0);
        let down = score_text("Stock plunges as lawsuit and layoffs spark fears of recession");
        assert_eq!((down.pos, down.neg), (0, 5), "{down:?}");
        assert_eq!(down.polarity, -1.0);
        // negation flips the next hits: "not strong" → negative, "no loss" → positive
        let neg = score_text("Results were not strong and the outlook is weak");
        assert_eq!((neg.pos, neg.neg), (0, 2), "{neg:?}");
        let pos = score_text("The bank did not lose money and reported no loss at all");
        assert_eq!((pos.pos, pos.neg), (2, 0), "{pos:?}");
        // "n't" counts as "not"; scope is limited
        let scoped = score_text("It isn't a loss. Afterwards the company's growth and profit rose");
        assert_eq!((scoped.pos, scoped.neg), (4, 0), "{scoped:?}");
        let none = score_text("The company announced a new chief financial officer");
        assert_eq!(none, ItemScore { polarity: 0.0, pos: 0, neg: 0, tokens: none.tokens });
        assert_eq!(none.confidence(), 0.0);
        assert!((ItemScore { pos: 1, ..none }.confidence() - (1.0 - (-0.5f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn instrument_matching_by_symbol_name_and_entity() {
        let idx = InstrumentIndex::new(&[
            InstrumentRef {
                id: "spy".into(),
                symbols: vec!["SPY".into()],
                names: vec!["SPDR S&P 500 ETF Trust".into()],
            },
            InstrumentRef { id: "aapl".into(), symbols: vec!["AAPL".into()], names: vec!["Apple Inc.".into()] },
            InstrumentRef { id: "sim".into(), symbols: vec!["SIM-1".into()], names: vec!["Heartbeat".into()] },
        ]);
        assert_eq!(idx.len(), 3);
        assert_eq!(clean_name("SPDR S&P 500 ETF Trust"), "spdr s&p 500");
        assert_eq!(clean_name("Apple Inc."), "apple");
        assert_eq!(clean_name("The Coca-Cola Company"), "coca-cola");
        assert_eq!(clean_name("AAPL (yahoo)"), "aapl");
        let pairs = InstrumentIndex::new(&[InstrumentRef {
            id: "btc".into(),
            symbols: vec!["BTC-USD".into()],
            names: vec!["BTC-USD (yahoo)".into()],
        }]);
        assert_eq!(pairs.match_doc(&doc(0, "x", "BTC slides below 90k", &[])), vec!["btc"]);
        assert_eq!(pairs.match_doc(&doc(0, "x", "BTC-USD hits a record", &[])), vec!["btc"]);
        let ticker_named = InstrumentIndex::new(&[InstrumentRef {
            id: "spy".into(),
            symbols: vec!["SPY".into()],
            names: vec!["SPY (yahoo)".into()],
        }]);
        assert!(ticker_named.match_doc(&doc(0, "x", "a spy novel", &[])).is_empty());
        assert_eq!(ticker_named.match_doc(&doc(0, "x", "SPY slips", &[])), vec!["spy"]);
        assert_eq!(ticker_named.match_doc(&doc(0, "x", "wrap", &["query:spy"])), vec!["spy"]);
        // symbol token, case-sensitive, with or without `$`
        assert_eq!(idx.match_doc(&doc(0, "x", "$SPY rallies while AAPL slides", &[])), vec!["aapl", "spy"]);
        assert!(
            idx.match_doc(&doc(0, "x", "a spy thriller about an orchard", &[])).is_empty(),
            "lower-case ticker words do not match"
        );
        // name phrase, case-insensitive, word bounded
        assert_eq!(idx.match_doc(&doc(0, "x", "Apple unveils new chips", &[])), vec!["aapl"]);
        assert!(idx.match_doc(&doc(0, "x", "Pineapples are cheap", &[])).is_empty());
        assert_eq!(idx.match_doc(&doc(0, "x", "The SPDR S&P 500 ETF saw inflows", &[])), vec!["spy"]);
        // entities carry the Google News query
        assert_eq!(idx.match_doc(&doc(0, "x", "Markets wrap", &["query:Apple stock"])), vec!["aapl"]);
        assert_eq!(idx.match_doc(&doc(0, "x", "Markets wrap", &["query:spy etf", "author:bob"])), vec!["spy"]);
        assert!(idx.match_doc(&doc(0, "x", "Markets wrap", &["author:dhouston"])).is_empty());
        assert_eq!(idx.match_doc(&doc(0, "x", "SIM-1 ticks", &[])), vec!["sim"]);
    }

    #[test]
    fn window_score_decays_shrinks_and_measures_novelty() {
        let w = 60_000; // one minute window
        let mut tw = TextWindow::new();
        tw.push(&doc(10_000, "hackernews", "Shares surge to record high after strong profit beat", &[]));
        // one strongly positive item at age 0: score = w·1 / (w + SHRINK) with w = conf(6 hits)
        let s = tw.score_at(10_000, w);
        let conf = 1.0 - (-3.0f64).exp();
        assert!((s.news - conf / (conf + SHRINK)).abs() < 1e-12, "{s:?}");
        assert_eq!((s.n_items, s.n_trailing, s.hits), (1, 0, 6));
        assert_eq!(s.sources, vec!["hackernews"]);
        assert_eq!(s.novelty, 0.0, "no trailing window yet");
        // the same item 50 s later is decayed: smaller score, same sign
        let later = tw.score_at(60_000, w);
        assert!(later.news > 0.0 && later.news < s.news, "{later:?}");
        // a negative item now outweighs the decayed positive one
        tw.push(&doc(60_000, "google_news", "Stock plunges as lawsuit and layoffs spark fears of recession", &[]));
        let mixed = tw.score_at(60_000, w);
        assert!(mixed.news < 0.0, "{mixed:?}");
        assert_eq!(mixed.sources, vec!["google_news", "hackernews"]);
        // point in time: as of 10 s the later item is invisible
        assert_eq!(tw.score_at(10_000, w).n_items, 1);
        // novelty: a current window with unrelated vocabulary vs the trailing window → high
        let mut nv = TextWindow::new();
        nv.push(&doc(0, "a", "central bank holds rates steady as inflation cools", &[]));
        nv.push(&doc(90_000, "a", "chipmaker unveils quantum photonics roadmap", &[]));
        let n = nv.score_at(120_000, w);
        assert_eq!((n.n_items, n.n_trailing), (1, 1));
        assert!(n.novelty > 0.9, "{n:?}");
        // repeating the same headline is not novel
        let mut same = TextWindow::new();
        same.push(&doc(0, "a", "central bank holds rates steady", &[]));
        same.push(&doc(90_000, "a", "central bank holds rates steady", &[]));
        assert!(same.score_at(120_000, w).novelty < 1e-9);
        // prune keeps two windows
        same.prune(300_000, w);
        assert_eq!(same.len(), 0);
        // outside both windows nothing counts
        assert_eq!(nv.score_at(400_000, w), WindowScore::default());
        // deterministic hashing
        assert_eq!(hash_bucket("inflation"), hash_bucket("inflation"));
        assert!(hash_bucket("x") < HASH_DIM);
    }

    #[test]
    fn tokenizer_shapes() {
        assert_eq!(
            tokenize("Fed's rates, S&P 500 won't fall!"),
            vec!["fed's", "rates", "s&p", "500", "wo", "not", "fall"]
        );
    }
}
