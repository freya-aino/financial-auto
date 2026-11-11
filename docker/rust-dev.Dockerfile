# Rust toolchain for `just cargo` (AGENTS.md: no builds on the host). Build output lives in the
# fa-cargo-target volume, the registry in fa-cargo-registry / fa-cargo-git.
FROM rust:1.98-slim-trixie
RUN rustup component add clippy rustfmt
