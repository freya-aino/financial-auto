//! Compiles `proto/fa/v1/*.proto` with protox (pure Rust, no protoc needed; D-002).
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../proto");
    let dir = root.join("fa/v1");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "proto"))
        .collect();
    files.sort();

    let fds = protox::compile(&files, [&root])?;
    prost_build::Config::new().compile_fds(fds)?;

    println!("cargo:rerun-if-changed={}", dir.display());
    Ok(())
}
