# 30 — Latent Regions

| | |
|---|---|
| **Status** | draft |
| **Maturity** | Low-medium |
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

### 4.2 Bus subjects (spec 21)
- `latent.region.<timescale>.<expert>.<instrument>`: live membership updates (`fa.v1.RegionMembership`).
- `latent.region.lifecycle`: region set refits, births, merges, splits and deaths (`fa.v1.RegionLifecycleEvent`).

### 4.3 Consumers
- spec 33: membership stream → transition counts / likelihoods.
- spec 34: retrieval decoders query Helix for historical memberships of the same region.
- spec 35: GP primitives `current_region`, `neighbors`, `region_mass`, `region_age`.
- spec 51: latent/graph explorer.

## 5. Data model
| Entity | Store | Fields |
|---|---|---|
| `RegionSet` | Helix node | snapshot_id, timescale, expert_id, family, method, fit_time, n_regions, cfg_hash |
| `Region` | Helix node + vector | id, local_id, centroid (vector), scale, mass, density, persistence, born_at, died_at |
| `MEMBER_OF` | Helix edge (Region → RegionSet) | — |
| `DESCENDS_FROM` | Helix edge (Region → Region) | overlap weight, kind (carry, merge, split) |
| `REFINES` | Helix edge (fine-timescale Region → coarse-timescale Region) | conditional co-occurrence P(coarse ∣ fine), support |
| `NEIGHBOR` | Helix edge (Region ↔ Region, same set) | latent distance, boundary density |
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

### 6.3 Online membership
- The live latent `z_t` from the active snapshot is assigned via `RegionDiscovery::assign`, giving soft weights plus a **novelty** score (distance to nearest centroid normalised by scale, or prior log-density below threshold).
- High novelty (≥ `novelty_threshold`) marks the latent as **out-of-region**. The MCMC graph sees a special `Unknown` state per set, and strategies can read novelty. Sustained high novelty feeds the snapshot-refresh trigger (spec 36).
- Membership is computed at each timescale step, per instrument (or per asset set where the encoder embeds sets).

### 6.4 Region identity across refits and snapshots
Regions must keep identity across retraining, otherwise the MCMC graph, strategy genomes and retrieval history break. **This is core blocker #1** (open question 1). Proposed default (proposal):
1. Keep a fixed **probe set** of historical windows (stratified over time, assets and modalities).
2. Encode the probe set with both old and new snapshots, then assign to old and new regions.
3. Build the overlap matrix `O[i,j] = |probes in old i ∩ new j| / |probes in old i ∪ new j|`.
4. Solve the assignment with Hungarian matching on `O` above `carry_threshold`. Matched regions **inherit the old `RegionId`**.
5. One old region overlapping several new ones above `split_threshold` → split (new ids, `DESCENDS_FROM` edges). The reverse → merge.
6. Unmatched new regions get fresh ids; unmatched old regions are marked dead (`died_at`).
7. Downstream components receive `RegionLifecycleEvent`s and remap: MCMC transfers kernel posteriors (spec 33), strategies holding region literals remap through `DESCENDS_FROM` (spec 35).

### 6.5 Cross-timescale hierarchy
- For each pair of adjacent timescales (s, s+1), estimate `P(region at s+1 | region at s)` over aligned time.
- Edges with mutual information above `hier_mi_min` become `REFINES` edges.
- This gives a multi-scale graph: a coarse (e.g. daily) regime contains finer (hourly) regions. Strategies and the MCMC can condition on it.

### 6.6 Refit cadence
- Full refit on every new encoder snapshot (spec 36).
- Online adapters (spec 36) shift latents slightly. Membership stays computed against the snapshot's region set. A **drift monitor** tracks per-region occupancy vs the training distribution and triggers a partial refit when divergence exceeds `drift_refit_kl`.

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

## 10. Open questions
1. **[Core blocker #1] Region identity across retraining.** Blocks specs 33, 34 (retrieval history), 35 (region literals in genomes). Options:
   - (a) Probe-set alignment + Hungarian matching (default proposal, §6.4).
   - (b) **Anchored training**: the new snapshot gets a loss term that keeps the old region centroids mapping to the same codes/locations (codebook warm-start + anchor loss). Identity is then mostly preserved by construction.
   - (c) **Region ids defined in outcome space**: identify regions by their predictive signature (residual distribution, forward-return profile) instead of geometry, and match on signature.
   - (d) Accept discontinuity: every snapshot has a fresh graph and downstream components retrain each generation.
   - Likely answer: (b) + (a) as fallback.
2. Is predictive coherence (§6.1, point 3) mandatory, or a soft score? Affects cost and region count.
3. Soft or hard membership for MCMC transitions? Soft keeps information but complicates count-based kernels (spec 33).
4. Rust availability of HDBSCAN / Mapper / persistent homology. Write our own in `fa-patterns`, or restrict v1 to codebook and k-means/GMM-style methods?
5. Should regions be per instrument or per asset set? Depends on the asset-selection outcome in spec 31.
6. Lifetime and storage cost of region episodes in Helix at the seconds timescale.

## 11. Deferred
- Topological (persistent homology) summaries as strategy features.
- Visual rendering of regions as decoder targets (could become a visual decoder family).
- Cross-expert region products (joint regions across experts). Not needed while experts are decorrelated by design.
