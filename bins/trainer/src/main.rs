//! trainer: offline training CLI (spec 36). Subcommands validate their arguments and
//! print the plan; model training lands with the `burn` backend (D-004).

use clap::{Parser, Subcommand};
use fa_core::Timescale;
use fa_encoders::{FamilyId, default_sizing, descriptors};

#[derive(Parser)]
#[command(about = "Encoder training: gen-0 pretraining, generations, online adapters")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Gen 0: self-supervised pretraining (reconstruction + next-step), no decoder feedback.
    Pretrain {
        #[arg(long)]
        timescale: Timescale,
        #[arg(long, value_parser = parse_family)]
        family: FamilyId,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Generation n ≥ 1: retrain with elite-decoder feedback (D-009).
    Generation {
        #[arg(long)]
        timescale: Timescale,
        #[arg(long, value_parser = parse_family)]
        family: FamilyId,
        #[arg(long)]
        generation: u32,
        /// Number of elite decoders per family used as frozen auxiliary heads.
        #[arg(long, default_value_t = 8)]
        elite_k: u32,
    },
    /// Online adapter update on top of a frozen snapshot.
    Adapter {
        #[arg(long)]
        snapshot: String,
    },
}

fn parse_family(s: &str) -> Result<FamilyId, String> {
    FamilyId::ALL
        .into_iter()
        .find(|f| f.as_str() == s)
        .ok_or_else(|| format!("unknown family `{s}`; one of: {}", FamilyId::ALL.map(FamilyId::as_str).join(", ")))
}

fn check_family(family: FamilyId, ts: Timescale) -> anyhow::Result<()> {
    let d = descriptors().into_iter().find(|d| d.id == family).expect("descriptor exists for every family");
    anyhow::ensure!(d.timescales.contains(&ts), "family {} is not enabled for timescale {ts} (D-010)", family.as_str());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    match Args::parse().cmd {
        Cmd::Pretrain { timescale, family, seed } => {
            check_family(family, timescale)?;
            println!(
                "plan: gen-0 pretrain {} @ {timescale} seed={seed} sizing={:?}",
                family.as_str(),
                default_sizing(timescale)
            );
        }
        Cmd::Generation { timescale, family, generation, elite_k } => {
            check_family(family, timescale)?;
            anyhow::ensure!(generation >= 1, "generation 0 is `pretrain`");
            println!(
                "plan: generation {generation} {} @ {timescale} with top-{elite_k} elite decoders",
                family.as_str()
            );
        }
        Cmd::Adapter { snapshot } => println!("plan: online adapter update for snapshot {snapshot}"),
    }
    println!("not yet implemented: training requires the `burn` backend (D-004)");
    Ok(())
}
