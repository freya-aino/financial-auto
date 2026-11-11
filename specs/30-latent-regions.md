# 30 — Latent Regions

| | |
|---|---|
| **Status** | draft |
| **Maturity** | High (implemented on the baselines; see §12) |
| **Owners (code)** | `crates/fa-patterns`, `bins/model-runtime`, `bins/trainer` |
| **Depends on** | 31 Expert encoders, 32 Expert spaces, 20 Storage, 36 Training lifecycle |

## 1. Purpose
"Regional patterns" in this system are **regions of the learned latent space**: dense basins, attractors and regime areas on the expert manifolds produced by the expert encoders. Regions are the discrete vocabulary of the system. They become the nodes of the Markov state graph (spec 33), the anchors for retrieval decoders (spec 34) and the primitives that strategies traverse (spec 35). This spec says how regions are discovered, identified, stored and kept stable.

## 2. Scope
**In scope:**
- Defining a region and its descriptive statistics.
- The region-discovery contract that each encoder family implements.
- Assigning live latents to regions (online membership).
- Cross-timescale hierarchy edges between regions of different timescale models.
- Region lifecycle across encoder snapshots (birth, merge, split, death, identity carry-over).
- Persisting regions and memberships in HelixDB.

**Out of scope:**
- Transition kernels between regions (spec 33).
- How the latent space itself is learned (specs 31, 32).
- Strategy logic that consumes regions (spec 35).

## 3. Decisions
- Regions are **regions of latent space**, not time windows or asset clusters. Temporal and cross-asset structure appears only through what the encoders embed.
- The discovery method is **encoder-family-specific**. Each `EncoderFamily` (spec 31) declares its own `RegionDiscovery` implementation, e.g. codebook usage for FSQ/VQ families, density clustering for continuous families, Mapper/topological methods where suitable.
- Each **timescale** has its own encoder model (spec 31), so each timescale has its own region set. Cross-timescale relationships are explicit **hierarchy edges**.
- Regions exist **per expert space** (spec 32). A region belongs to exactly one (snapshot, timescale, expert) triple.
- Regions and memberships are stored in **HelixDB** as graph nodes with vector properties. Raw latents over time go to VictoriaMetrics only as summary features; full latent vectors for retrieval live in Helix vectors (spec 20).

## 4. Interfaces

### 4.1 Region discovery trait (proposal)
```rust
/// Implemented per encoder family; see spec 31 `EncoderFamily::region_discovery`.
pub trait RegionDiscovery: Send + Sync {
    /// Fit regions on a sample of latents from one (snapshot, timescale, expert).
    fn fit(&self, latents: &LatentSample, cfg: &RegionCfg) -> Result<RegionSet>;

    /// Assign a live latent to regions. Soft membership: (region, weight) pairs summing to 1.
    fn assign(&self, set: &RegionSet, z: &Latent) -> Membership;

    /// Map an old region set onto a new one after re-fit / new snapshot (see §6.4).
    fn align(&self, old: &RegionSet, new: &RegionSet, probe: &LatentSample) -> Alignment;
}

pub struct Region {
    pub id: RegionId,               // stable lineage id (see §6.4)
    pub local_id: u32,              // index within this RegionSet
    pub key: RegionKey,             // (snapshot_id, timescale, expert_id)
    pub centroid: Vec<f32>,
    pub scale: RegionScale,         // covariance diag / radius / code tuple
    pub mass: f64,                  // fraction of training latents
    pub density: f64,
    pub persistence: Option<f64>,   // topological persistence when available
    pub parents: Vec<RegionId>,     // lineage from previous snapshot
}

pub struct Membership { pub weights: SmallVec<[(RegionId, f32); 4]>, pub novelty: f32 }
```

**Implemented:** the batch contract is `fa_patterns::RegionDiscovery` (`discover` / `assign`; `align` is family-independent: `fa_patterns::lineage::align` on probe assignments). The online contract is `fa_patterns::OnlineRegionDiscovery` (`observe → Assignment{region_id, lineage_id, distance, novelty, memberships, events}`, `stats`, `export_state` / `restore_state`), implemented by the leader-clustering baseline `OnlineRegionSet`. Both see only the encoder's per-expert `mean` vectors, so every `EncoderFamily` (statistical or neural) plugs in unchanged. `Membership` is `Assignment.memberships` (k nearest regions, Gaussian-kernel weights summing to 1) plus `novelty`. `Region.mass / density / persistence / parents` live in `RegionStat` (`mass`, `occupancy`, `spread`, `age`) and in the lineage edges.

### 4.2 Bus subjects (spec 21)
- `latent.membership.<timescale>.<instrument>`: live soft memberships of every expert (`fa.v1.RegionMembership`: per expert `weights[]`, `novelty`, `out_of_region`). *(Proposed as `latent.region.<ts>.<expert>.<instrument>`; moved off `latent.region.>` because that wildcard is consumed as `RegionEvent` by strategy-runtime, control-api and replays.)*
- `latent.lifecycle.<timescale>`: births, carries, splits, merges and deaths with event time, ancestors (`kind`, `overlap`), centroid, mass, occupancy and the regime-break flag (`fa.v1.RegionLifecycleEvent`). *(Proposed as `latent.region.lifecycle`, moved for the same reason.)*
- `latent.region.<timescale>`: the D-039 `RegionEvent` (created / merged / split / retired) stays for existing consumers; a split publishes `KIND_SPLIT` with the child as `region_id` and the parent in `related_region_ids`.
- `latent.hier.<fine>.<coarse>`: cross-timescale hierarchy per expert index (`fa.v1.RegionHierarchy`, §6.5).

### 4.3 Consumers
- spec 33: membership stream → transition counts / likelihoods.
- spec 34: retrieval decoders query Helix for historical memberships of the same region.
- spec 35: GP primitives `current_region`, `neighbors`, `region_mass`, `region_age`.
- spec 51: latent/graph explorer.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `RegionSet` | Helix node | snapshot_id, timescale, expert_id, family, method, fit_time, n_regions, cfg_hash; **implemented** as `RegionSet{region_set_id = expert_space_id}` with the serialised online set (`state`) and kernel (`kernel_state`), linked `ExpertSpace -HasRegionSet->` |
| `Region` | Helix node + vector | id, local_id, centroid (vector), scale, mass, density, persistence, born_at, died_at; **implemented** as `LatentRegion` (`region_id`, `lineage_id`, `radius`, `status`, `mass`, `occupancy`, `spread`, `age`, `created_at`, `died_at`, `retire_reason`) + `LatentVec{kind: centroid, ref_id: region_id}` (replaced, not appended) |
| `MEMBER_OF` | Helix edge (Region → RegionSet) | — ; **implemented** as `InSpace` (Region → ExpertSpace, 1:1 with the set) |
| `DESCENDS_FROM` | Helix edge (Region → Region) | overlap weight, kind (carry, merge, split); **implemented** as `DescendsFrom` child → parent (`kind = carry \| split \| merge \| partial`, `overlap`), plus the trainer's `RegionMatch{score}` (carry) |
| `REFINES` | Helix edge (fine-timescale Region → coarse-timescale Region) | conditional co-occurrence P(coarse ∣ fine), support; **implemented** as `Hier` (`p_coarse_given_fine`, `p_fine_given_coarse`, `support`, `pmi_bits`, `mi_bits`) |
| `NEIGHBOR` | Helix edge (Region ↔ Region, same set) | latent distance, boundary density; *not stored*: neighbours are answered from the kernel (spec 33 `neighbors`) and the centroids |
| `Occupancy` | VictoriaMetrics series | `region_weight{ts,expert,region,instrument}` sampled at the timescale's step |
| Region episode | Helix node (optional) | region, instrument, t_enter, t_exit, max weight. Used by retrieval |

## 6. Behaviour & algorithms

### 6.1 What counts as a region
A region is a connected subset of an expert manifold that:
1. holds non-trivial probability mass (≥ `min_mass`);
2. is separated from neighbours by a density drop or a code boundary; and
3. is **predictively coherent**: next-step residual statistics inside the region differ from the global ones (proposal: KS or energy distance on residuals, p < `coherence_alpha`).

Point 3 ties regions to what matters for trading. It is a **(proposal)** and may be relaxed for families where it is too costly.

### 6.2 Family-specific discovery (initial mapping, proposal)
| Encoder family (spec 31) | Discovery method | Notes |
|---|---|---|
| Hybrid FSQ/VQ + continuous residual | Codebook usage: each used code (or code tuple) is a candidate region; merge codes with similar residual statistics; split codes with multimodal continuous residuals | Cheapest; ids follow code indices |
| Hierarchical VAE + flow-matching prior | Density clustering (HDBSCAN-like) on the top-level latent; density taken from the flow prior | The prior gives exact log-density |
| JEPA + VAE head | Density clustering on the VAE head latent; optional Mapper over the JEPA embedding | |
| Latent diffusion | Mode finding: sample from the prior, then cluster sample endpoints; or score-field basins (follow the score to fixed points) | Expensive; run on fit only |

All methods must be implementable in Rust. HDBSCAN and Mapper need a Rust implementation or crate (see open question 4).

**Implemented (2026-11-11, `fa_patterns::methods`, pure Rust, on any encoder output — they see only the per-expert latent means):**
- `DensityClusters` (`density`): HDBSCAN — core distance to the `min_samples` = 5th neighbour, Prim MST on mutual-reachability distance, single-linkage dendrogram, condensed tree with `min_cluster_size` = max(10, 1 % of the points), excess-of-mass selection (root only when nothing else qualifies), noise points unlabelled; ≤ 4000 points per fit (even stride over the canonical order).
- `MapperCover` (`mapper`): filter = first principal component (power iteration, sign-normalised), 10 intervals with 30 % overlap, single-linkage clusters per interval at 2 × the median 5-NN distance, nerve over shared points; regions = connected components of the nerve holding ≥ 1 % of the points.
- `CodebookRegions` (`codebook`): VQ codebook of 32 codes by deterministic k-means++ / Lloyd; codes with ≥ 0.5 % usage are candidates, codes whose centroid distance ≤ the sum of their RMS spreads merge (union-find), a code whose 2-means halves are ≥ 4 pooled sds apart splits.
- All are deterministic and order-free (canonical point order, rounded centroids emitted sorted) and pass `conformance::assert_conformance` (6 clusters, recovery 100 %, self-alignment ≥ 0.9) plus a 16-d unequal-size mixture; `discovery_for_method(name, radius, max_regions)` maps `FamilyDescriptor::region_discovery` (`density | mapper | codebook | leader`) to an implementation and `methods::fit_assign` fits + assigns any of them like `LeaderClustering::fit_assign`. (`fa_encoders::neural::regions::DensityModes`, the mean-shift density of D-070, remains the families' current default; the codebook and Mapper families can switch from the leader fallback to these through `discovery_for_method`.)

### 6.3 Online membership
- The live latent `z_t` from the active snapshot is assigned via `RegionDiscovery::assign`, giving soft weights plus a **novelty** score (distance to nearest centroid normalised by scale, or prior log-density below threshold).
- High novelty (≥ `novelty_threshold`) marks the latent as **out-of-region**. The MCMC graph sees a special `Unknown` state per set, and strategies can read novelty. Sustained high novelty feeds the snapshot-refresh trigger (spec 36).
- Membership is computed at each timescale step, per instrument (or per asset set where the encoder embeds sets).

**Implemented:** novelty = distance to the nearest centroid / radius; `out_of_region` at `novelty_threshold` (3.0); the heartbeat counts `out_of_region`. Sustained novelty surfaces as occupancy drift (§6.6).

### 6.4 Region identity across refits and snapshots
Regions must keep identity across retraining, otherwise the MCMC graph, strategy genomes and retrieval history break. **This is core blocker #1** (open question 1). Proposed default (proposal):
1. Keep a fixed **probe set** of historical windows (stratified over time, assets and modalities).
2. Encode the probe set with both old and new snapshots, then assign to old and new regions.
3. Build the overlap matrix `O[i,j] = |probes in old i ∩ new j| / |probes in old i ∪ new j|`.
4. Solve the assignment with Hungarian matching on `O` above `carry_threshold`. Matched regions **inherit the old `RegionId`**.
5. One old region overlapping several new ones above `split_threshold` → split (new ids, `DESCENDS_FROM` edges). The reverse → merge.
6. Unmatched new regions get fresh ids; unmatched old regions are marked dead (`died_at`).
7. Downstream components receive `RegionLifecycleEvent`s and remap: MCMC transfers kernel posteriors (spec 33), strategies holding region literals remap through `DESCENDS_FROM` (spec 35).

**Implemented (D-007 fallback, both offline and online):**
- *Snapshot alignment* — `fa_patterns::lineage::align`: Jaccard overlap matrix of the probe assignments, **Hungarian** maximum-weight assignment (`fa_patterns::hungarian`, exact, tested against brute force) on the entries ≥ `carry_threshold`; every other pair ≥ `split_threshold` is a lineage edge classified `carry | split | merge | partial` (partial = one partner between the two thresholds, new lineage). Unmatched new regions are born, parents without any partner die. The trainer's `align_expert` uses it unchanged; `region_map.json` gains per-region `ancestors`, per-expert `dead_parents`, `split_parents`, `regime_break` (all `serde(default)`, older maps load).
- *Lineage ids*: a carried region inherits `lineage_id`; `region_id` stays unique per snapshot. On load / hot swap model-runtime announces the whole alignment on `latent.lifecycle.<ts>` (dead parents in the parent's space) and persists it as `DescendsFrom` edges and death marks.
- *Online lifecycle* in `OnlineRegionSet`: birth (outside every radius), merge (centroids within `merge_radius`), **split** (streaming 2-means per region; when the sub-centroids are `split_distance · radius` apart and both hold `min_split_mass` points, the lighter mode becomes a new region and lineage with `DescendsFrom{kind: split, overlap = mass share}`), **death by support** (decayed occupancy, half-life `support_halflife`, share < `min_support` after `min_age`; reason `support`).
- *Regime break* (§8): carried probe mass < 0.5 sets `regime_break`; model-runtime then restarts that expert's kernel from the prior instead of transferring and reports `regime_break.<ts>`.

### 6.5 Cross-timescale hierarchy
- For each pair of adjacent timescales (s, s+1), estimate `P(region at s+1 | region at s)` over aligned time.
- Edges with mutual information above `hier_mi_min` become `REFINES` edges.
- This gives a multi-scale graph: a coarse (e.g. daily) regime contains finer (hourly) regions. Strategies and the MCMC can condition on it.

**Implemented:** `fa_patterns::HierarchyTracker` per (fine timescale, next coarser served timescale, expert index): every closed fine bar of an instrument is paired with that instrument's current coarse region (latest state, previews included); joint counts are halved past `max_total` (forgetting); the plug-in MI is Miller–Madow corrected; when MI ≥ `hier_mi_min`, pairs with support ≥ 20 and `P(coarse | fine)` ≥ 0.05 are edges. A tracker resets when either snapshot changes; retired regions are forgotten. Published every `--graph-every-secs` (60 s) on `latent.hier.<fine>.<coarse>`, persisted as `Hier` edges, heartbeat `hier.<fine>.<coarse>.e<k>.mi_bits|edges`. Experts are paired by index (true for the statistical family; a family whose expert indices do not correspond across timescales gets a meaningless but harmless low MI).

### 6.6 Refit cadence
- Full refit on every new encoder snapshot (spec 36).
- Online adapters (spec 36) shift latents slightly. Membership stays computed against the snapshot's region set. A **drift monitor** tracks per-region occupancy vs the training distribution and triggers a partial refit when divergence exceeds `drift_refit_kl`.

**Implemented:** `fa_patterns::DriftMonitor`: `KL(live occupancy ‖ reference)` with additive smoothing; the reference is the snapshot's fitted region masses, or (baseline without a map) the live occupancy captured after `drift_warmup` = 2000 observations. Checked with every kernel summary (`KernelUpdate.drift_kl / refit_due`, heartbeat `mcmc.<ts>.e<k>.drift_kl`). Three consecutive `refit_due` summaries request a partial refit: a `train.adapter` job (`{snapshot_id, timescale, window_hours, reason: region_drift}`) on `jobs.train.adapter`, at most once per `--drift-refit-cooldown-secs` (6 h) per timescale, so a single shock does not refit (§8).

**Persistence and restart (§8 "never only in memory"):** every `--hub-state-secs` (120 s) model-runtime writes each expert's `RegionSet` (online set + kernel), region statistics and moved centroids; on start it restores the sets of the serving snapshots (`--no-region-restore` to skip), so region ids, lineages and posteriors survive a restart.

## 7. Tunables
| Name | Default (placeholder) | Notes |
|---|---|---|
| `min_mass` | 0.002 | Minimum fraction of latents per region |
| `max_regions_per_set` | 512 | Hard cap for graph size |
| `coherence_alpha` | 0.01 | Predictive-coherence test level |
| `novelty_threshold` | 3.0 (scale units) | Out-of-region cutoff |
| `carry_threshold` | 0.5 | Jaccard overlap for identity carry-over |
| `split_threshold` | 0.2 | Overlap to count as split/merge partner |
| `probe_set_size` | 50k windows / timescale | Fixed per probe-set version |
| `hier_mi_min` | 0.05 bits | Cross-timescale edge cutoff |
| `drift_refit_kl` | 0.1 | Occupancy KL that triggers partial refit |
| `support_halflife` / `min_age` | 20 000 / 2 000 observations | Death by support (online) |
| `split_distance` / `min_split_mass` | 1.5 radii / 30 points | Online split |
| `membership_k` | 4 | Regions in a soft membership |

Implemented defaults: `min_mass` → `min_support` 0.002, `max_regions_per_set` → `--max-regions` 64 (online cap), `novelty_threshold` 3.0, `carry_threshold` 0.5 (trainer `carry_threshold`), `split_threshold` 0.2, `hier_mi_min` 0.05 bits, `drift_refit_kl` 0.1. `coherence_alpha` and `probe_set_size` (trainer: the fit's window ends) are not used by the baseline.

## 8. Failure modes
| Failure | Response |
|---|---|
| Discovery collapses into 1 region or hits `max_regions_per_set` | Mark the set invalid. Keep the previous region set active. Alert in UI |
| Identity alignment matches < 50% of mass | Treat as a **regime break of the representation**. Downstream components must re-learn (MCMC resets priors to weak, strategies flagged for re-evaluation) |
| Novelty spike across many instruments | Expected in market shocks. Don't refit automatically mid-event; raise model-health signal (spec 40 health halts may consume it) |
| Helix write failure | Buffer memberships in NATS (JetStream retention). Regions are never computed only in memory without persistence |

## 9. Acceptance criteria
1. Every encoder family registered in spec 31 provides a `RegionDiscovery` implementation passing a shared conformance test (fit / assign / align on synthetic Gaussian-mixture latents recovers ≥ 90% of true clusters).
2. Region sets, regions and lineage edges are queryable from Helix by (snapshot, timescale, expert).
3. Re-fitting on the same snapshot and data with the same seed yields identical region ids.
4. On a new snapshot, alignment emits lifecycle events, and ≥ `carry_threshold` overlap regions keep their ids.
5. Live membership latency per instrument per step ≤ 5 ms at the minute timescale on the reference workstation (placeholder).
6. The latent explorer (spec 51) can render a region set with occupancy and hierarchy edges.

**Status of the criteria:** 1 — `fa_patterns::conformance::assert_conformance` (6 clusters × 8 dims; recovery ≥ 90 %, determinism, self-alignment ≥ 0.9) passes for `LeaderClustering`, `DensityClusters` (HDBSCAN), `MapperCover` and `CodebookRegions` (and fa-encoders' `DensityModes`); family crates call it from their tests. 2 — `snapshotGraph`, `regionSet`, `snapshotKernels`, `regionLineage`, `snapshotHierarchy` by business id. 3 — fits are deterministic in point order; online ids are deterministic and survive restarts via `RegionSet`. 4 — `lineage::align` + `latent.lifecycle`, tested. 5 — **benchmarked** (`cargo test -p model-runtime --release -- --ignored --nocapture bench_step_latency_per_instrument`: 1m baseline pipeline, 20 instruments, ~5000 timed closed-bar steps of encode + route + region assignment + soft memberships + six-kernel arena + transition summaries + the staggered kernel summaries): mean 0.37–1.0 ms, p50 0.27–0.79 ms, **p99 1.3–3.6 ms ≤ 5 ms**, max 7–16 ms (a once-a-minute `KernelUpdate`; summaries are staggered to one expert space per step) on the 12-core workstation, the upper figures under a load average of ~25 from parallel builds. 6 — UI work (spec 51).

## 10. Open questions
1. **[Core blocker #1] Region identity across retraining.** Blocks specs 33, 34 (retrieval history), 35 (region literals in genomes). Options:
   - (a) Probe-set alignment + Hungarian matching (default proposal, §6.4).
   - (b) **Anchored training**: the new snapshot gets a loss term that keeps the old region centroids mapping to the same codes/locations (codebook warm-start + anchor loss). Identity is then mostly preserved by construction.
   - (c) **Region ids defined in outcome space**: identify regions by their predictive signature (residual distribution, forward-return profile) instead of geometry, and match on signature.
   - (d) Accept discontinuity: every snapshot has a fresh graph and downstream components retrain each generation.
   - Likely answer: (b) + (a) as fallback.
   - **Resolved for the baselines (D-007):** (a) is implemented end to end (Hungarian alignment, lineage ids, split/merge/death, regime break, kernel transfer). (b) needs a trainable family and lives with the neural encoders (spec 31/36).
2. Is predictive coherence (§6.1, point 3) mandatory, or a soft score? Affects cost and region count.
3. Soft or hard membership for MCMC transitions? Soft keeps information but complicates count-based kernels (spec 33). *Both exist:* memberships are published soft; the serving kernel observes hard transitions; `MarkovGraph::observe_soft` implements the soft update for a kernel that wants it.
4. Rust availability of HDBSCAN / Mapper / persistent homology. Write our own in `fa-patterns`, or restrict v1 to codebook and k-means/GMM-style methods? *Resolved: our own HDBSCAN, Mapper and codebook discovery in `fa_patterns::methods` (§6.2); persistent homology stays deferred (§11).*
5. Should regions be per instrument or per asset set? Depends on the asset-selection outcome in spec 31.
6. Lifetime and storage cost of region episodes in Helix at the seconds timescale.

## 11. Deferred
- Topological (persistent homology) summaries as strategy features.
- Visual rendering of regions as decoder targets (could become a visual decoder family).
- Cross-expert region products (joint regions across experts). Not needed while experts are decorrelated by design.

## 12. Implementation map
| Piece | Code |
|---|---|
| Discovery contracts, conformance test | `crates/fa-patterns/src/{lib,online,conformance}.rs` |
| HDBSCAN / Mapper / codebook discovery | `crates/fa-patterns/src/methods.rs` |
| Alignment, Hungarian, region map | `crates/fa-patterns/src/{lineage,hungarian,region_map}.rs` |
| Hierarchy, drift | `crates/fa-patterns/src/{hierarchy,drift}.rs` |
| Live stage (memberships, lifecycle, persistence, restore, drift trigger) | `bins/model-runtime/src/{pipeline,regions,hub_sink}.rs` |
| Helix rows | `crates/fa-hub/src/regions.rs`, `helix/schema.hx` (`RegionSet`, `DescendsFrom`, `Hier`, `LatentRegion` statistics) |
| Read API | `bins/control-api/src/mcmc_routes.rs`: `GET /api/latent/regions/{region_id}/lineage`, `/api/mcmc/{snapshot}/hierarchy` |
