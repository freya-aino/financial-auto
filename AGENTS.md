# Agent rules

Rules for coding agents (Claude Code and others) working in this repository. They override
habits and defaults; when a rule blocks a task, stop and ask instead of working around it.

## Never build in the repository working tree

Build output must not land in this checkout or on the host disk outside Docker. A single day of
parallel agents grew `./target` past 200 GB and filled the disk.

- **Rust:** never run `cargo` on the host. Use `just cargo <args>` (e.g. `just cargo test -p fa-bus`,
  `just cargo clippy --workspace --all-targets -- -D warnings`) or `just rust-check|rust-test|rust-clippy`.
  The recipe runs the toolchain image with the build output in the `fa-cargo-target` Docker volume
  (`CARGO_TARGET_DIR=/target`), so nothing is written to `./target`. Without `just` installed, the
  equivalent is (build the toolchain image once with
  `docker build -t financial-auto/rust-dev:1.98 -f docker/rust-dev.Dockerfile docker`):
  ```sh
  docker run --rm --network host -v "$PWD":/src -w /src \
    -v fa-cargo-target:/target -e CARGO_TARGET_DIR=/target \
    -v fa-cargo-registry:/usr/local/cargo/registry -v fa-cargo-git:/usr/local/cargo/git \
    financial-auto/rust-dev:1.98 cargo test -p fa-bus
  ```
- **Never** create extra target directories: no `CARGO_TARGET_DIR` pointing into the repo, the
  scratchpad, `/tmp` or `$HOME`, no per-agent `target/agent-*`, no `git worktree` copies to build in.
  Parallel agents share the one `fa-cargo-target` volume; cargo's lock serialises them.
- **Go:** `just go <args>` (or the `golang` image with the `fa-gomod` / `fa-gocache` volumes).
  Don't run `go build -o` into the repo.
- **Frontend:** `just pnpm <args>` / the `node:22-alpine` image. `frontend/node_modules`,
  `.svelte-kit` and `build` are tolerated (gitignored, small) but nothing else.
- **Images:** `docker compose build` (or `just build`) — builds happen inside
  BuildKit with cache mounts. Compose has no profiles (D-123): `up` starts every service, and a
  subset is named (`just infra-up` for the data stores only).
- Live-infrastructure tests need `--network host`, which `just cargo` already sets; `FA_*`,
  `DATABASE_URL`, `RUST_LOG` and `UPDATE_FIXTURE` are passed through.

If an existing `./target` is present, leave it alone — deleting it is the owner's call.

## Disk and Docker hygiene

- Check `df -h /` before long builds or sweeps; stop and report when the disk is above 90 %.
- Never delete anything outside your own scratch output without asking: no `rm -rf` of build
  directories, caches or other projects' data.
- Never run `docker compose down -v` (except the scoped `just eph-down` below), `docker volume rm|prune`, `docker system prune`,
  `docker container prune` or `docker kill $(docker ps -q)`. Other projects share this Docker host.
- Scratch containers get a unique `--name`, `--rm`, tmpfs storage and no named volumes; remove only
  the containers you started, by name.
- Don't restart, wipe or re-seed the compose stack unless the task says so.

## Ephemeral test stacks (owner rule, D-096)

Test runs, including long ones, assume the platform is ephemeral: start from empty volumes and delete
the run's data afterwards, so test data never piles up on the disk.

- Start with `just eph-up <run>` (or `scripts/eph.sh up <run>`): compose project `fa-eph-<run>`,
  fresh volumes, backups in `scratch/backups-<run>/`, local connectors only (`synthetic,heartbeat`), `FA_MARKET_DATA=synthetic` so the stack paper-trades
  the synthetic market (the default everywhere else is `real`, D-103).
- Always finish with `just eph-down <run>`, which removes **only** that project's containers, volumes
  and backups (`down -v` scoped to `fa-eph-<run>`; this is the one exception to the rule above).
  Check `just eph-ls` is empty at the end of a task.
- Develop and test against local and synthetic data. Real remote sources and APIs are added and
  tested as the last step of a piece of work.

## Working conventions

- Decisions go to `specs/DECISIONS.md` (one row per decision, provisional until the owner confirms).
- Specs describe unfinished work only; a spec is retired into its component READMEs once its
  "Remaining" list is empty (see `specs/README.md`).
- Don't apply the FX restatement, re-arm halts, or promote/roll back snapshots without the owner.
- Run `scripts/secret-scan.sh --staged` before committing.
