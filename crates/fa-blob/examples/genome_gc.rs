//! Genome-weight GC for `fa-models/decoders/<generation>/*.safetensors` (spec 20 §6.5,
//! `fa_blob::gc`). **Dry run by default**; `--apply` deletes.
//!
//! ```sh
//! cargo run -p fa-blob --example genome_gc                       # dry run against the compose Garage
//! cargo run -p fa-blob --example genome_gc -- --list             # + one line per decision
//! cargo run -p fa-blob --example genome_gc -- --apply            # delete (needs the serving check)
//! ```
//! Options: `--keep-generations N` (5), `--elite-per-family K` (8), `--min-age-days D` (7),
//! `--control-api URL` (`http://127.0.0.1:7700`; `none` skips — then `--apply` is refused unless
//! `--no-serving-check`), `--keep-file PATH` (extra protected keys: one per line, or any JSON —
//! every bundle key string in it counts). Protection = serving/shadow/pinned decoders from
//! `GET /api/decoders/serving` + every bundle referenced by a snapshot manifest + `--keep-file` +
//! the card elites; see `fa_blob::gc`.
use std::collections::BTreeSet;

use fa_blob::{
    BlobStore, S3BlobStore, buckets,
    gc::{self, GcPolicy, Protection},
};
use fa_core::{Secrets, config::PlatformConfig};

struct Args {
    apply: bool,
    list: bool,
    policy: GcPolicy,
    control_api: Option<String>,
    serving_check: bool,
    keep_file: Option<String>,
}

fn parse() -> Result<Args> {
    let mut a = Args {
        apply: false,
        list: false,
        policy: GcPolicy::default(),
        control_api: Some("http://127.0.0.1:7700".into()),
        serving_check: true,
        keep_file: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--apply" => a.apply = true,
            "--dry-run" => a.apply = false,
            "--list" => a.list = true,
            "--no-serving-check" => a.serving_check = false,
            "--keep-generations" => a.policy.keep_generations = val()?.parse().map_err(|e| format!("{e}"))?,
            "--elite-per-family" => a.policy.elite_per_family = val()?.parse().map_err(|e| format!("{e}"))?,
            "--min-age-days" => {
                a.policy.min_age = chrono::Duration::days(val()?.parse().map_err(|e| format!("{e}"))?);
            }
            "--control-api" => {
                let v = val()?;
                a.control_api = (v != "none").then_some(v);
            }
            "--keep-file" => a.keep_file = Some(val()?),
            "-h" | "--help" => return Err("see the doc comment of crates/fa-blob/examples/genome_gc.rs".into()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn store() -> Result<S3BlobStore> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let cfg_path = std::env::var("FA_PLATFORM").unwrap_or_else(|_| format!("{root}/config/platform.toml"));
    let cfg = PlatformConfig::load(cfg_path).map_err(|e| format!("platform.toml: {e}"))?;
    let secrets_path = std::env::var_os("FA_SECRETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| format!("{root}/config/secrets.toml").into());
    let secrets = Secrets::load_from(secrets_path).map_err(|e| format!("secrets: {e}"))?;
    S3BlobStore::from_config(&cfg, &secrets).map_err(|e| e.to_string())
}

type Result<T> = std::result::Result<T, String>;

fn keep_file_keys(path: &str) -> Result<BTreeSet<String>> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = BTreeSet::new();
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) => gc::collect_bundle_refs(&v, &mut out),
        Err(_) => {
            for l in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
                let k = gc::normalize_key(l);
                if gc::is_weight_key(&k) {
                    out.insert(k);
                }
            }
        }
    }
    Ok(out)
}

async fn serving_keys(base: &str) -> Result<BTreeSet<String>> {
    let url = format!("{}/api/decoders/serving", base.trim_end_matches('/'));
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("{url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("{url}: {}", resp.status()));
    }
    let v: serde_json::Value = resp.json().await.map_err(|e| format!("{url}: {e}"))?;
    if v.get("slots").is_none() {
        return Err(format!("{url}: no `slots` in reply (model-runtime not answering?)"));
    }
    let mut out = BTreeSet::new();
    gc::collect_bundle_refs(&v, &mut out);
    Ok(out)
}

fn mib(b: u64) -> String {
    format!("{:.1} MiB", b as f64 / 1_048_576.0)
}

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("genome_gc: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = parse()?;
    let store = store()?;
    let listing = store.list(buckets::MODELS, gc::DECODER_PREFIX).await.map_err(|e| e.to_string())?;

    let mut prot = Protection::default();
    let snaps = gc::snapshot_refs(&store).await.map_err(|e| e.to_string())?;
    prot.extend(snaps.iter().cloned());
    let serving = match &args.control_api {
        Some(base) => match serving_keys(base).await {
            Ok(k) => Some(k),
            Err(e) if args.apply && args.serving_check => {
                return Err(format!("serving check failed ({e}); refusing --apply (use --no-serving-check)"));
            }
            Err(e) => {
                eprintln!("warning: serving check failed: {e}");
                None
            }
        },
        None if args.apply && args.serving_check => {
            return Err("--control-api none with --apply needs --no-serving-check".into());
        }
        None => None,
    };
    if let Some(s) = &serving {
        prot.extend(s.iter().cloned());
    }
    if let Some(p) = &args.keep_file {
        prot.extend(keep_file_keys(p)?);
    }
    let (cards, bad_cards) = gc::load_cards(&store, &listing).await;
    let elites = gc::card_elites(&cards, args.policy.elite_per_family);

    let plan = gc::plan(&listing, &args.policy, &prot, &elites, chrono::Utc::now());
    let weights = plan.delete.len() + plan.keep.len();
    println!(
        "fa-models/decoders: {} objects, {} weight bundles in {} generations, {} cards ({} unparseable)",
        listing.len(),
        weights,
        plan.generations.len(),
        cards.len(),
        bad_cards
    );
    println!(
        "protection: {} keys ({} from snapshot manifests, {} serving/shadow/pinned{}), {} card elites",
        prot.keys.len(),
        snaps.len(),
        serving.as_ref().map_or(0, BTreeSet::len),
        if serving.is_none() { " — serving check skipped" } else { "" },
        elites.len()
    );
    println!(
        "policy: keep {} newest generations, {} elites per (output_family, step), min age {} d",
        args.policy.keep_generations,
        args.policy.elite_per_family,
        args.policy.min_age.num_days()
    );
    for (reason, (n, bytes)) in plan.by_reason() {
        println!("  {reason:<18} {n:>6}  {}", mib(bytes));
    }
    if args.list {
        for d in plan.delete.iter().chain(&plan.keep) {
            println!("{:<18} {:>10}  {}", d.reason.as_str(), d.size, d.key);
        }
    }
    let report = gc::execute(&store, &plan, !args.apply).await.map_err(|e| e.to_string())?;
    println!(
        "{}: {} bundles, {} ({} failed); kept {} bundles, {}",
        if report.dry_run { "would delete (dry run)" } else { "deleted" },
        report.deleted,
        mib(report.deleted_bytes),
        report.failed.len(),
        plan.keep.len(),
        mib(plan.keep_bytes())
    );
    for (k, e) in &report.failed {
        eprintln!("  failed {k}: {e}");
    }
    Ok(())
}
