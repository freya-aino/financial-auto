//! S-expression syntax for GP programs, so hand-written baselines and evolved genomes can
//! be stored in plain text files (`strategies.toml`).
//!
//! Grammar: `atom | (op arg...)`. Atoms are numbers or primitive names. Operators:
//! `add sub mul div neg tanh min max if_gt`. Num-typed primitives: `stay_prob`, `(path_up k)`,
//! `region_entropy`, `(quantile q)`, `analog_forecast`, `(score family)`, `position`,
//! `unrealised_pnl`, `time_in_trade`, and the spec 35 §4.2 set `(p_trans a b k)`,
//! `(p_to_set r S k)`, `(hitting_time r S)`, `(dwell r)`, `(novelty expert)`,
//! `(region_feature r mass|mean_fwd_return|vol)`, `(path_value (sample_walk r k n) up|mass|mean_fwd_return|vol)`,
//! `(fc_quantile q k)`, `(fc_dist k)`, `(analog_outcome k)`, `analog_age`, `(text_score topic)`,
//! `(visual_score kind)`. Typed sub-expressions: `(cur_region expert)` and the region literal
//! `(region expert ordinal)` (spec 35 §6.5, bound to the strategy's snapshot) are `Region`,
//! `(neighbors r n)` is `RegionSet`, `(sample_walk r k n)` is `[Path]`; they are only valid as
//! arguments of the primitives above (a `Region` where a `Num` is expected is a type error).
//!
//! The rest of the §4.2 combinator set: `(clip x lo hi)`, the portfolio primitives
//! `exposure_gross` and `drawdown`, `Bool`-typed comparisons `(gt a b)` / `(lt a b)` with
//! `(and p q)`, `(or p q)`, `(not p)`, the conditional `(if p a b)` (a `Bool` condition; a `Bool`
//! where a `Num` is expected, or a `Num` where a `Bool` is expected, is a type error; a program
//! must be `Num`), and `(prob_gt P x)` over the `Prob` forms `(fc_dist k)` / `(analog_outcome k)`.
use std::fmt;

use crate::{CmpOp, DecoderOp, Expr, GraphOp, PathFn, PortfolioOp, ProbSrc, RegionFeature, RegionRef, SetRef};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected end of input")]
    Eof,
    #[error("unexpected token `{0}`")]
    Unexpected(String),
    #[error("unknown operator or primitive `{0}`")]
    Unknown(String),
    #[error("`{op}` takes {expected} argument(s), got {got}")]
    Arity { op: String, expected: usize, got: usize },
    #[error("trailing input after expression: `{0}`")]
    Trailing(String),
    #[error("type error: {0}")]
    Type(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Open,
    Close,
    Atom(String),
}

fn tokenize(src: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in src.chars() {
        if c == '(' || c == ')' || c.is_whitespace() {
            if !cur.is_empty() {
                out.push(Tok::Atom(std::mem::take(&mut cur)));
            }
            if c == '(' {
                out.push(Tok::Open);
            } else if c == ')' {
                out.push(Tok::Close);
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(Tok::Atom(cur));
    }
    out
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
}

impl Parser<'_> {
    fn next(&mut self) -> Result<&Tok, ParseError> {
        let t = self.toks.get(self.pos).ok_or(ParseError::Eof)?;
        self.pos += 1;
        Ok(t)
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        match self.next()?.clone() {
            Tok::Close => Err(ParseError::Unexpected(")".into())),
            Tok::Atom(a) => atom(&a),
            Tok::Open => {
                let Tok::Atom(op) = self.next()?.clone() else {
                    return Err(ParseError::Unexpected("(".into()));
                };
                let mut args = Vec::new();
                loop {
                    match self.toks.get(self.pos) {
                        None => return Err(ParseError::Eof),
                        Some(Tok::Close) => {
                            self.pos += 1;
                            break;
                        }
                        Some(_) => args.push(self.arg()?),
                    }
                }
                form(&op, args)
            }
        }
    }

    /// An argument is a sub-expression, a bare word (for `score`'s family name or a feature
    /// name) or a typed form (`Region`, `RegionSet`, `[Path]`).
    fn arg(&mut self) -> Result<Arg, ParseError> {
        match (self.toks.get(self.pos), self.toks.get(self.pos + 1)) {
            (Some(Tok::Atom(a)), _) if a.parse::<f64>().is_err() && atom(a).is_err() => {
                self.pos += 1;
                Ok(Arg::Word(a.clone()))
            }
            (Some(Tok::Open), Some(Tok::Atom(op))) if TYPED_FORMS.contains(&op.as_str()) => self.typed(),
            _ => Ok(Arg::Expr(self.expr()?)),
        }
    }

    /// Parses `(cur_region e)`, `(region e k)`, `(neighbors r n)` or `(sample_walk r k n)`.
    fn typed(&mut self) -> Result<Arg, ParseError> {
        self.pos += 1; // Open
        let Tok::Atom(op) = self.next()?.clone() else { unreachable!("checked by arg") };
        let mut args = Vec::new();
        loop {
            match self.toks.get(self.pos) {
                None => return Err(ParseError::Eof),
                Some(Tok::Close) => {
                    self.pos += 1;
                    break;
                }
                Some(_) => args.push(self.arg()?),
            }
        }
        let arity = |n: usize| {
            if args.len() == n {
                Ok(())
            } else {
                Err(ParseError::Arity { op: op.clone(), expected: n, got: args.len() })
            }
        };
        Ok(match op.as_str() {
            "cur_region" => {
                arity(1)?;
                Arg::Region(RegionRef::Cur(uint(&op, &args[0])?))
            }
            "region" => {
                arity(2)?;
                Arg::Region(RegionRef::Lit(uint(&op, &args[0])?, uint(&op, &args[1])?))
            }
            "neighbors" => {
                arity(2)?;
                Arg::Set(SetRef::Neighbors(region(&op, &args[0])?, uint(&op, &args[1])?))
            }
            _ => {
                arity(3)?;
                Arg::Paths(region(&op, &args[0])?, uint(&op, &args[1])?, uint(&op, &args[2])?)
            }
        })
    }
}

const TYPED_FORMS: [&str; 4] = ["cur_region", "region", "neighbors", "sample_walk"];

enum Arg {
    Expr(Expr),
    Word(String),
    Region(RegionRef),
    Set(SetRef),
    /// `(sample_walk r k n)`
    Paths(RegionRef, u32, u32),
}

fn type_name(a: &Arg) -> &'static str {
    match a {
        Arg::Expr(_) => "Num",
        Arg::Word(_) => "word",
        Arg::Region(_) => "Region",
        Arg::Set(_) => "RegionSet",
        Arg::Paths(..) => "[Path]",
    }
}

fn uint(op: &str, a: &Arg) -> Result<u32, ParseError> {
    match a {
        Arg::Expr(Expr::Const(c)) => Ok(c.max(0.0) as u32),
        Arg::Expr(_) => Err(ParseError::Unexpected("non-constant literal argument".into())),
        Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
        other => Err(ParseError::Type(format!("`{op}` expects a number, got {}", type_name(other)))),
    }
}

fn region(op: &str, a: &Arg) -> Result<RegionRef, ParseError> {
    match a {
        Arg::Region(r) => Ok(*r),
        Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
        other => Err(ParseError::Type(format!("`{op}` expects a Region, got {}", type_name(other)))),
    }
}

fn set(op: &str, a: &Arg) -> Result<SetRef, ParseError> {
    match a {
        Arg::Set(s) => Ok(*s),
        Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
        other => Err(ParseError::Type(format!("`{op}` expects a RegionSet, got {}", type_name(other)))),
    }
}

fn feature(w: &str) -> Result<RegionFeature, ParseError> {
    Ok(match w {
        "mass" => RegionFeature::Mass,
        "mean_fwd_return" => RegionFeature::MeanFwdReturn,
        "vol" => RegionFeature::Vol,
        other => return Err(ParseError::Unknown(other.into())),
    })
}

fn word<'a>(op: &str, a: &'a Arg) -> Result<&'a str, ParseError> {
    match a {
        Arg::Word(w) => Ok(w),
        other => Err(ParseError::Type(format!("`{op}` expects a name, got {}", type_name(other)))),
    }
}

fn atom(a: &str) -> Result<Expr, ParseError> {
    if let Ok(v) = a.parse::<f64>() {
        return Ok(Expr::Const(v));
    }
    Ok(match a {
        "stay_prob" => Expr::Graph(GraphOp::StayProb),
        "region_entropy" => Expr::Graph(GraphOp::RegionEntropy),
        "analog_forecast" => Expr::Decoder(DecoderOp::AnalogForecast),
        "position" => Expr::Portfolio(PortfolioOp::Position),
        "unrealised_pnl" => Expr::Portfolio(PortfolioOp::UnrealisedPnl),
        "time_in_trade" => Expr::Portfolio(PortfolioOp::TimeInTrade),
        "exposure_gross" => Expr::Portfolio(PortfolioOp::ExposureGross),
        "drawdown" => Expr::Portfolio(PortfolioOp::Drawdown),
        "analog_age" => Expr::Decoder(DecoderOp::AnalogAge),
        other => return Err(ParseError::Unknown(other.into())),
    })
}

fn form(op: &str, args: Vec<Arg>) -> Result<Expr, ParseError> {
    let arity = |n: usize| {
        if args.len() == n { Ok(()) } else { Err(ParseError::Arity { op: op.into(), expected: n, got: args.len() }) }
    };
    let b = |a: &Arg| -> Result<Box<Expr>, ParseError> {
        match a {
            Arg::Expr(e) if e.is_bool() => Err(ParseError::Type(format!("`{op}` expects a Num, got Bool"))),
            Arg::Expr(e) => Ok(Box::new(e.clone())),
            Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
            other => Err(ParseError::Type(format!("`{op}` expects a Num, got {}", type_name(other)))),
        }
    };
    let bl = |a: &Arg| -> Result<Box<Expr>, ParseError> {
        match a {
            Arg::Expr(e) if e.is_bool() => Ok(Box::new(e.clone())),
            Arg::Expr(_) => Err(ParseError::Type(format!("`{op}` expects a Bool, got Num"))),
            Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
            other => Err(ParseError::Type(format!("`{op}` expects a Bool, got {}", type_name(other)))),
        }
    };
    let num = |a: &Arg| -> Result<f64, ParseError> {
        match a {
            Arg::Expr(Expr::Const(c)) => Ok(*c),
            Arg::Expr(_) => Err(ParseError::Unexpected("non-constant literal argument".into())),
            Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
            other => Err(ParseError::Type(format!("`{op}` expects a number, got {}", type_name(other)))),
        }
    };
    Ok(match op {
        "add" | "sub" | "mul" | "div" | "min" | "max" => {
            arity(2)?;
            let (x, y) = (b(&args[0])?, b(&args[1])?);
            match op {
                "add" => Expr::Add(x, y),
                "sub" => Expr::Sub(x, y),
                "mul" => Expr::Mul(x, y),
                "div" => Expr::Div(x, y),
                "min" => Expr::Min(x, y),
                _ => Expr::Max(x, y),
            }
        }
        "neg" | "tanh" => {
            arity(1)?;
            let x = b(&args[0])?;
            if op == "neg" { Expr::Neg(x) } else { Expr::Tanh(x) }
        }
        "if_gt" => {
            arity(4)?;
            Expr::IfGt(b(&args[0])?, b(&args[1])?, b(&args[2])?, b(&args[3])?)
        }
        "clip" => {
            arity(3)?;
            Expr::Clip(b(&args[0])?, b(&args[1])?, b(&args[2])?)
        }
        "if" => {
            arity(3)?;
            Expr::If(bl(&args[0])?, b(&args[1])?, b(&args[2])?)
        }
        "gt" | "lt" => {
            arity(2)?;
            let cmp = if op == "gt" { CmpOp::Gt } else { CmpOp::Lt };
            Expr::Cmp(cmp, b(&args[0])?, b(&args[1])?)
        }
        "and" | "or" => {
            arity(2)?;
            let (p, q) = (bl(&args[0])?, bl(&args[1])?);
            if op == "and" { Expr::And(p, q) } else { Expr::Or(p, q) }
        }
        "not" => {
            arity(1)?;
            Expr::Not(bl(&args[0])?)
        }
        "prob_gt" => {
            arity(2)?;
            let src = match &args[0] {
                Arg::Expr(Expr::Decoder(DecoderOp::FcDist(k))) => ProbSrc::Fc(*k),
                Arg::Expr(Expr::Decoder(DecoderOp::AnalogOutcome(k))) => ProbSrc::Analog(*k),
                other => {
                    return Err(ParseError::Type(format!(
                        "`prob_gt` expects a Prob ((fc_dist k) or (analog_outcome k)), got {}",
                        match other {
                            Arg::Expr(e) if e.is_bool() => "Bool",
                            o => type_name(o),
                        }
                    )));
                }
            };
            Expr::ProbGt(src, b(&args[1])?)
        }
        "path_up" => {
            arity(1)?;
            Expr::Graph(GraphOp::PathUpProb(num(&args[0])?.max(0.0) as u32))
        }
        "quantile" => {
            arity(1)?;
            Expr::Decoder(DecoderOp::Quantile(num(&args[0])?.clamp(0.0, 1.0)))
        }
        "score" => {
            arity(1)?;
            match &args[0] {
                Arg::Word(w) => Expr::Decoder(DecoderOp::Score(w.clone())),
                _ => return Err(ParseError::Unexpected("score expects a family name".into())),
            }
        }
        // spec 35 §4.2 graph primitives (typed arguments)
        "p_trans" => {
            arity(3)?;
            Expr::Graph(GraphOp::PTrans(region(op, &args[0])?, region(op, &args[1])?, uint(op, &args[2])?))
        }
        "p_to_set" => {
            arity(3)?;
            Expr::Graph(GraphOp::PToSet(region(op, &args[0])?, set(op, &args[1])?, uint(op, &args[2])?))
        }
        "hitting_time" => {
            arity(2)?;
            Expr::Graph(GraphOp::HittingTime(region(op, &args[0])?, set(op, &args[1])?))
        }
        "dwell" => {
            arity(1)?;
            Expr::Graph(GraphOp::Dwell(region(op, &args[0])?))
        }
        "novelty" => {
            arity(1)?;
            Expr::Graph(GraphOp::Novelty(uint(op, &args[0])?))
        }
        "region_feature" => {
            arity(2)?;
            Expr::Graph(GraphOp::RegionFeature(region(op, &args[0])?, feature(word(op, &args[1])?)?))
        }
        "path_value" => {
            arity(2)?;
            let Arg::Paths(r, k, n) = &args[0] else {
                return Err(ParseError::Type(format!(
                    "`{op}` expects (sample_walk r k n), got {}",
                    type_name(&args[0])
                )));
            };
            let f = match word(op, &args[1])? {
                "up" => PathFn::Up,
                w => PathFn::Feature(feature(w)?),
            };
            Expr::Graph(GraphOp::PathValue(*r, *k, *n, f))
        }
        "cur_region" | "region" | "neighbors" | "sample_walk" => {
            return Err(ParseError::Type(format!("`{op}` is not Num-typed; use it as an argument")));
        }
        // spec 35 §4.2 decoder primitives (asset = the evaluated instrument)
        "fc_quantile" => {
            arity(2)?;
            Expr::Decoder(DecoderOp::FcQuantile(num(&args[0])?.clamp(0.0, 1.0), uint(op, &args[1])?))
        }
        "fc_dist" => {
            arity(1)?;
            Expr::Decoder(DecoderOp::FcDist(uint(op, &args[0])?))
        }
        "analog_outcome" => {
            arity(1)?;
            Expr::Decoder(DecoderOp::AnalogOutcome(uint(op, &args[0])?))
        }
        "text_score" => {
            arity(1)?;
            Expr::Decoder(DecoderOp::TextScore(word(op, &args[0])?.to_owned()))
        }
        "visual_score" => {
            arity(1)?;
            Expr::Decoder(DecoderOp::VisualScore(word(op, &args[0])?.to_owned()))
        }
        other => return Err(ParseError::Unknown(other.into())),
    })
}

/// Parses one program.
pub fn parse(src: &str) -> Result<Expr, ParseError> {
    let toks = tokenize(src);
    let mut p = Parser { toks: &toks, pos: 0 };
    let e = p.expr()?;
    if p.pos != toks.len() {
        return Err(ParseError::Trailing(format!("{:?}", &toks[p.pos..])));
    }
    if e.is_bool() {
        return Err(ParseError::Type("a program must be Num-typed, got Bool (use it as an `if` condition)".into()));
    }
    Ok(e)
}

impl fmt::Display for RegionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionRef::Cur(e) => write!(f, "(cur_region {e})"),
            RegionRef::Lit(e, k) => write!(f, "(region {e} {k})"),
        }
    }
}

impl fmt::Display for SetRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SetRef::Neighbors(r, n) => write!(f, "(neighbors {r} {n})"),
        }
    }
}

impl fmt::Display for ProbSrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbSrc::Fc(k) => write!(f, "(fc_dist {k})"),
            ProbSrc::Analog(k) => write!(f, "(analog_outcome {k})"),
        }
    }
}

impl fmt::Display for RegionFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RegionFeature::Mass => "mass",
            RegionFeature::MeanFwdReturn => "mean_fwd_return",
            RegionFeature::Vol => "vol",
        })
    }
}

impl fmt::Display for PathFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathFn::Up => f.write_str("up"),
            PathFn::Feature(x) => write!(f, "{x}"),
        }
    }
}

/// Pretty-prints a program in the same syntax (`parse(&format!("{e}")) == e`).
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Graph(GraphOp::StayProb) => f.write_str("stay_prob"),
            Expr::Graph(GraphOp::PathUpProb(k)) => write!(f, "(path_up {k})"),
            Expr::Graph(GraphOp::RegionEntropy) => f.write_str("region_entropy"),
            Expr::Graph(GraphOp::PTrans(a, b, k)) => write!(f, "(p_trans {a} {b} {k})"),
            Expr::Graph(GraphOp::PToSet(r, s, k)) => write!(f, "(p_to_set {r} {s} {k})"),
            Expr::Graph(GraphOp::HittingTime(r, s)) => write!(f, "(hitting_time {r} {s})"),
            Expr::Graph(GraphOp::Dwell(r)) => write!(f, "(dwell {r})"),
            Expr::Graph(GraphOp::Novelty(e)) => write!(f, "(novelty {e})"),
            Expr::Graph(GraphOp::RegionFeature(r, x)) => write!(f, "(region_feature {r} {x})"),
            Expr::Graph(GraphOp::PathValue(r, k, n, x)) => write!(f, "(path_value (sample_walk {r} {k} {n}) {x})"),
            Expr::Decoder(DecoderOp::Quantile(q)) => write!(f, "(quantile {q})"),
            Expr::Decoder(DecoderOp::AnalogForecast) => f.write_str("analog_forecast"),
            Expr::Decoder(DecoderOp::Score(s)) => write!(f, "(score {s})"),
            Expr::Decoder(DecoderOp::FcQuantile(q, k)) => write!(f, "(fc_quantile {q} {k})"),
            Expr::Decoder(DecoderOp::FcDist(k)) => write!(f, "(fc_dist {k})"),
            Expr::Decoder(DecoderOp::AnalogOutcome(k)) => write!(f, "(analog_outcome {k})"),
            Expr::Decoder(DecoderOp::AnalogAge) => f.write_str("analog_age"),
            Expr::Decoder(DecoderOp::TextScore(t)) => write!(f, "(text_score {t})"),
            Expr::Decoder(DecoderOp::VisualScore(k)) => write!(f, "(visual_score {k})"),
            Expr::Portfolio(PortfolioOp::Position) => f.write_str("position"),
            Expr::Portfolio(PortfolioOp::UnrealisedPnl) => f.write_str("unrealised_pnl"),
            Expr::Portfolio(PortfolioOp::TimeInTrade) => f.write_str("time_in_trade"),
            Expr::Portfolio(PortfolioOp::ExposureGross) => f.write_str("exposure_gross"),
            Expr::Portfolio(PortfolioOp::Drawdown) => f.write_str("drawdown"),
            Expr::Add(a, b) => write!(f, "(add {a} {b})"),
            Expr::Sub(a, b) => write!(f, "(sub {a} {b})"),
            Expr::Mul(a, b) => write!(f, "(mul {a} {b})"),
            Expr::Div(a, b) => write!(f, "(div {a} {b})"),
            Expr::Min(a, b) => write!(f, "(min {a} {b})"),
            Expr::Max(a, b) => write!(f, "(max {a} {b})"),
            Expr::Neg(a) => write!(f, "(neg {a})"),
            Expr::Tanh(a) => write!(f, "(tanh {a})"),
            Expr::IfGt(a, b, c, d) => write!(f, "(if_gt {a} {b} {c} {d})"),
            Expr::Clip(x, lo, hi) => write!(f, "(clip {x} {lo} {hi})"),
            Expr::If(c, a, b) => write!(f, "(if {c} {a} {b})"),
            Expr::ProbGt(src, x) => write!(f, "(prob_gt {src} {x})"),
            Expr::Cmp(CmpOp::Gt, a, b) => write!(f, "(gt {a} {b})"),
            Expr::Cmp(CmpOp::Lt, a, b) => write!(f, "(lt {a} {b})"),
            Expr::And(a, b) => write!(f, "(and {a} {b})"),
            Expr::Or(a, b) => write!(f, "(or {a} {b})"),
            Expr::Not(a) => write!(f, "(not {a})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_round_trips() {
        let src = "(mul 4 (sub (path_up 3) 0.5))";
        let e = parse(src).unwrap();
        assert_eq!(e.size(), 5);
        assert_eq!(e.to_string(), src);
        let e2 = parse("(if_gt position 0 (score text.news) (neg (quantile 0.25)))").unwrap();
        assert_eq!(parse(&e2.to_string()).unwrap(), e2);
        assert_eq!(parse("-0.5").unwrap(), Expr::Const(-0.5));
    }

    #[test]
    fn parses_typed_graph_and_decoder_primitives() {
        let r = RegionRef::Cur(0);
        let cases = [
            (
                "(p_to_set (cur_region 0) (neighbors (cur_region 0) 3) 1)",
                Expr::Graph(GraphOp::PToSet(r, SetRef::Neighbors(r, 3), 1)),
            ),
            ("(p_trans (cur_region 0) (cur_region 1) 2)", Expr::Graph(GraphOp::PTrans(r, RegionRef::Cur(1), 2))),
            (
                "(hitting_time (cur_region 0) (neighbors (cur_region 0) 2))",
                Expr::Graph(GraphOp::HittingTime(r, SetRef::Neighbors(r, 2))),
            ),
            ("(dwell (cur_region 0))", Expr::Graph(GraphOp::Dwell(r))),
            (
                "(p_trans (cur_region 1) (region 1 7) 3)",
                Expr::Graph(GraphOp::PTrans(RegionRef::Cur(1), RegionRef::Lit(1, 7), 3)),
            ),
            ("(dwell (region 0 2))", Expr::Graph(GraphOp::Dwell(RegionRef::Lit(0, 2)))),
            ("(novelty 2)", Expr::Graph(GraphOp::Novelty(2))),
            (
                "(region_feature (cur_region 0) mean_fwd_return)",
                Expr::Graph(GraphOp::RegionFeature(r, RegionFeature::MeanFwdReturn)),
            ),
            (
                "(path_value (sample_walk (cur_region 0) 3 64) up)",
                Expr::Graph(GraphOp::PathValue(r, 3, 64, PathFn::Up)),
            ),
            (
                "(path_value (sample_walk (cur_region 0) 2 8) vol)",
                Expr::Graph(GraphOp::PathValue(r, 2, 8, PathFn::Feature(RegionFeature::Vol))),
            ),
            ("(fc_quantile 0.9 5)", Expr::Decoder(DecoderOp::FcQuantile(0.9, 5))),
            ("(fc_dist 5)", Expr::Decoder(DecoderOp::FcDist(5))),
            ("(analog_outcome 3)", Expr::Decoder(DecoderOp::AnalogOutcome(3))),
            ("analog_age", Expr::Decoder(DecoderOp::AnalogAge)),
            ("(text_score news)", Expr::Decoder(DecoderOp::TextScore("news".into()))),
            ("(visual_score future)", Expr::Decoder(DecoderOp::VisualScore("future".into()))),
        ];
        for (src, want) in cases {
            let e = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_eq!(e, want, "{src}");
            assert_eq!(e.to_string(), src);
            assert_eq!(parse(&e.to_string()).unwrap(), e);
        }
        assert!(matches!(parse("(add 1 (region 0 1))"), Err(ParseError::Type(_))), "a Region is not Num");
        assert_eq!(parse("(dwell (region 0))"), Err(ParseError::Arity { op: "region".into(), expected: 2, got: 1 }));
        let nested = parse("(mul 4 (sub (p_to_set (cur_region 0) (neighbors (cur_region 0) 3) 1) 0.5))").unwrap();
        assert_eq!(nested.size(), 5);
    }

    #[test]
    fn text_visual_and_dotted_score_forms_round_trip() {
        let cases = [
            "(mul 3 (mul (text_score news) (visual_score future)))",
            "(sub 1 (score text.news))",
            "(max (sub 1 (score text.summary)) (sub 1 (score visual.chart)))",
            "(if_gt (mul (visual_score chart) (add (text_score narrative) (text_score summary))) 0 (mul 2 (add (text_score narrative) (text_score summary))) 0)",
            "(score rnn_prior)",
        ];
        for src in cases {
            let e = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_eq!(e.to_string(), src);
            assert_eq!(parse(&e.to_string()).unwrap(), e);
        }
        assert_eq!(parse("(score text.news)").unwrap(), Expr::Decoder(DecoderOp::Score("text.news".into())));
        assert_eq!(parse("(text_score)"), Err(ParseError::Arity { op: "text_score".into(), expected: 1, got: 0 }));
        assert!(matches!(parse("(text_score 1)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(visual_score (cur_region 0))"), Err(ParseError::Type(_))));
        // every baseline program round-trips
        for def in crate::population::StrategyFile::baselines().strategy {
            for src in [&def.signal, &def.uncertainty] {
                let e = parse(src).unwrap();
                assert_eq!(parse(&e.to_string()).unwrap(), e, "{}", def.id);
            }
        }
    }

    #[test]
    fn rejects_type_errors() {
        assert!(matches!(parse("(cur_region 0)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(add (cur_region 0) 1)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(p_trans 1 2 3)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(p_to_set (cur_region 0) (cur_region 0) 1)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(path_value (cur_region 0) up)"), Err(ParseError::Type(_))));
        assert_eq!(parse("(region_feature (cur_region 0) rsi)"), Err(ParseError::Unknown("rsi".into())));
        assert_eq!(parse("(dwell)"), Err(ParseError::Arity { op: "dwell".into(), expected: 1, got: 0 }));
        assert_eq!(
            parse("(p_to_set (cur_region 0) (neighbors (cur_region 0)) 1)"),
            Err(ParseError::Arity { op: "neighbors".into(), expected: 2, got: 1 })
        );
    }

    #[test]
    fn full_combinator_set_parses_types_and_round_trips() {
        let cases = [
            "(clip (mul 400 analog_forecast) -0.5 0.5)",
            "(if (and (gt stay_prob 0.6) (not (lt (fc_quantile 0.25 3) 0))) 1 (neg exposure_gross))",
            "(mul (sub (prob_gt (fc_dist 3) 0) 0.5) (sub 1 drawdown))",
            "(if (or (gt drawdown 0.05) (gt exposure_gross 1.5)) 0 (prob_gt (analog_outcome 2) 0.001))",
        ];
        for src in cases {
            let e = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_eq!(e.to_string(), src);
            assert_eq!(parse(&e.to_string()).unwrap(), e);
            assert!(!e.is_bool());
        }
        // Bool where Num is expected and vice versa
        assert!(matches!(parse("(gt 1 2)"), Err(ParseError::Type(_))), "a program must be Num");
        assert!(matches!(parse("(add (gt 1 2) 1)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(if 1 2 3)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(and (gt 1 2) 1)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(if_gt (gt 1 2) 0 1 2)"), Err(ParseError::Type(_))));
        assert!(matches!(parse("(prob_gt stay_prob 0)"), Err(ParseError::Type(_))), "not a Prob");
        assert!(matches!(parse("(prob_gt (fc_dist 2) (lt 1 2))"), Err(ParseError::Type(_))));
        assert_eq!(parse("(clip 1 2)"), Err(ParseError::Arity { op: "clip".into(), expected: 3, got: 2 }));
    }

    #[test]
    fn rejects_bad_programs() {
        assert_eq!(parse("(add 1)"), Err(ParseError::Arity { op: "add".into(), expected: 2, got: 1 }));
        assert_eq!(parse("(rsi 14)"), Err(ParseError::Unknown("rsi".into())));
        assert!(matches!(parse("(add 1 2"), Err(ParseError::Eof)));
        assert!(matches!(parse("1 2"), Err(ParseError::Trailing(_))));
    }
}
