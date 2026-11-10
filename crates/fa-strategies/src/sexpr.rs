//! S-expression syntax for GP programs, so hand-written baselines and evolved genomes can
//! be stored in plain text files (`strategies.toml`).
//!
//! Grammar: `atom | (op arg...)`. Atoms are numbers or primitive names. Operators:
//! `add sub mul div neg tanh min max if_gt`. Primitives: `stay_prob`, `(path_up k)`,
//! `region_entropy`, `(quantile q)`, `analog_forecast`, `(score family)`, `position`,
//! `unrealised_pnl`, `time_in_trade`.
use std::fmt;

use crate::{DecoderOp, Expr, GraphOp, PortfolioOp};

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

    /// An argument is either a sub-expression or a bare word (for `score`'s family name).
    fn arg(&mut self) -> Result<Arg, ParseError> {
        match self.toks.get(self.pos) {
            Some(Tok::Atom(a)) if a.parse::<f64>().is_err() && atom(a).is_err() => {
                self.pos += 1;
                Ok(Arg::Word(a.clone()))
            }
            _ => Ok(Arg::Expr(self.expr()?)),
        }
    }
}

enum Arg {
    Expr(Expr),
    Word(String),
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
        other => return Err(ParseError::Unknown(other.into())),
    })
}

fn form(op: &str, args: Vec<Arg>) -> Result<Expr, ParseError> {
    let arity = |n: usize| {
        if args.len() == n { Ok(()) } else { Err(ParseError::Arity { op: op.into(), expected: n, got: args.len() }) }
    };
    let b = |a: &Arg| -> Result<Box<Expr>, ParseError> {
        match a {
            Arg::Expr(e) => Ok(Box::new(e.clone())),
            Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
        }
    };
    let num = |a: &Arg| -> Result<f64, ParseError> {
        match a {
            Arg::Expr(Expr::Const(c)) => Ok(*c),
            Arg::Expr(_) => Err(ParseError::Unexpected("non-constant literal argument".into())),
            Arg::Word(w) => Err(ParseError::Unknown(w.clone())),
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
                Arg::Expr(_) => {
                    return Err(ParseError::Unexpected("score expects a family name".into()));
                }
            }
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
    Ok(e)
}

/// Pretty-prints a program in the same syntax (`parse(&format!("{e}")) == e`).
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Graph(GraphOp::StayProb) => f.write_str("stay_prob"),
            Expr::Graph(GraphOp::PathUpProb(k)) => write!(f, "(path_up {k})"),
            Expr::Graph(GraphOp::RegionEntropy) => f.write_str("region_entropy"),
            Expr::Decoder(DecoderOp::Quantile(q)) => write!(f, "(quantile {q})"),
            Expr::Decoder(DecoderOp::AnalogForecast) => f.write_str("analog_forecast"),
            Expr::Decoder(DecoderOp::Score(s)) => write!(f, "(score {s})"),
            Expr::Portfolio(PortfolioOp::Position) => f.write_str("position"),
            Expr::Portfolio(PortfolioOp::UnrealisedPnl) => f.write_str("unrealised_pnl"),
            Expr::Portfolio(PortfolioOp::TimeInTrade) => f.write_str("time_in_trade"),
            Expr::Add(a, b) => write!(f, "(add {a} {b})"),
            Expr::Sub(a, b) => write!(f, "(sub {a} {b})"),
            Expr::Mul(a, b) => write!(f, "(mul {a} {b})"),
            Expr::Div(a, b) => write!(f, "(div {a} {b})"),
            Expr::Min(a, b) => write!(f, "(min {a} {b})"),
            Expr::Max(a, b) => write!(f, "(max {a} {b})"),
            Expr::Neg(a) => write!(f, "(neg {a})"),
            Expr::Tanh(a) => write!(f, "(tanh {a})"),
            Expr::IfGt(a, b, c, d) => write!(f, "(if_gt {a} {b} {c} {d})"),
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
    fn rejects_bad_programs() {
        assert_eq!(parse("(add 1)"), Err(ParseError::Arity { op: "add".into(), expected: 2, got: 1 }));
        assert_eq!(parse("(rsi 14)"), Err(ParseError::Unknown("rsi".into())));
        assert!(matches!(parse("(add 1 2"), Err(ParseError::Eof)));
        assert!(matches!(parse("1 2"), Err(ParseError::Trailing(_))));
    }
}
