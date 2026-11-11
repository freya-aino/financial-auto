//! Build provenance for the run manifest (spec 36 §6.6): git commit, `Cargo.lock` and `go.sum`
//! hashes, embedded as compile-time env vars. Containers build without `.git` and without `go/`,
//! so every value degrades to `unknown` / empty instead of failing: pass the commit as the
//! `FA_GIT_COMMIT` build env (`docker build --build-arg FA_GIT_COMMIT=$(git rev-parse HEAD)`).

use std::{path::Path, process::Command};

use sha2::{Digest, Sha256};

fn sha256_file(p: &Path) -> String {
    std::fs::read(p).map(|b| format!("{:x}", Sha256::digest(&b))).unwrap_or_default()
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lock = root.join("Cargo.lock");
    let gosum = root.join("go/go.sum");
    println!("cargo:rerun-if-changed={}", lock.display());
    println!("cargo:rerun-if-changed={}", gosum.display());
    println!("cargo:rerun-if-env-changed=FA_GIT_COMMIT");
    let git_head = root.join(".git/HEAD");
    if git_head.exists() {
        println!("cargo:rerun-if-changed={}", git_head.display());
        println!("cargo:rerun-if-changed={}", root.join(".git/index").display());
    }
    let commit = std::env::var("FA_GIT_COMMIT").ok().filter(|c| !c.trim().is_empty()).or_else(|| {
        let out = Command::new("git").args(["rev-parse", "HEAD"]).current_dir(&root).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    });
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(&root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| if o.stdout.is_empty() { "false" } else { "true" })
        .unwrap_or("unknown");
    println!("cargo:rustc-env=FA_BUILD_GIT_COMMIT={}", commit.unwrap_or_else(|| "unknown".into()));
    println!("cargo:rustc-env=FA_BUILD_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=FA_BUILD_CARGO_LOCK_SHA256={}", sha256_file(&lock));
    println!("cargo:rustc-env=FA_BUILD_GO_SUM_SHA256={}", sha256_file(&gosum));
}
