# fa-patterns

Latent region discovery and region graph maintenance.

**Status:** implemented (D-007, D-058, D-069, D-075, D-080) — Region/lineage types, the batch `RegionDiscovery` trait (`discover`, `discover_with_residuals`, `assign`) and the online `OnlineRegionDiscovery` trait (`observe`, `stats`, `export_state` / `restore_state`); all discovery methods pass the shared conformance test (`conformance::assert_conformance`).

## Modules
| Module | What |
|---|---|
| `lib` | `Region`, `DiscoveryMethod`, the batch `RegionDiscovery` trait (`discover`, `discover_with_residuals` — NaN = unknown residual, default ignores them — `assign`), `nearest`, carry-only `align_by_probes` |
| `online` | `OnlineRegionSet`: streaming leader clustering with birth / merge / **split** (streaming 2-means per region) / **death by decayed support** lifecycle events, deterministic ids, soft memberships (k nearest, Gaussian kernel), novelty = distance / radius, serialisable state (persisted by model-runtime in Helix `RegionSet`); batch `LeaderClustering`; `online_form` (only `leader` streams, so every declared method is served as a leader set seeded with that method's regions) |
| `methods` | pure-Rust batch discovery for any encoder output: `DensityClusters` (HDBSCAN), `MapperCover` (PCA filter, overlapping cover, nerve components), `CodebookRegions` (k-means++ codebook; Gabriel-neighbour codes merge while a KS test of their pooled next-step residuals gives p ≥ 0.05, `ks_pvalue`; geometric merge without residuals); `discovery_for_method(density\|mapper\|codebook\|leader, radius, max_regions)`, `fit_assign`. Neural families route through `fa_encoders::neural::regions::discovery_for` (spec 30 §6.2) |
| `lineage`, `hungarian` | `align`: Jaccard probe overlap, Hungarian carry ≥ `carry_threshold`, other pairs ≥ `split_threshold` as `DescendsFrom{carry\|split\|merge\|partial}`, births, deaths, regime break (carried mass < 0.5) |
| `hierarchy` | `HierarchyTracker`: cross-timescale `REFINES` edges by bias-corrected mutual information |
| `drift` | `DriftMonitor`: occupancy KL vs the fitted masses, `refit_due` |
| `coherence` | predictive coherence (two-sample KS of next-step residuals inside vs outside a region, `p < coherence_alpha` 0.01): `CoherenceTracker` (online, rolling windows, cached scores), `coherence_of` (batch, labels + residuals) |
| `validity` | collapse guard: `check_set(n, cap)` → `Collapse::Single` (≤ 1 region) / `AtCap` (≥ `max_regions_per_set`), `check_online` (after warm-up), `RegionMap::collapsed(cap)` |
| `conformance` | shared fit / assign / align test on synthetic Gaussian mixtures (spec 30 §9.1) |

## Region maps (`region_map.rs`, spec 30 §6.4, D-058)
- `FittedRegion` (region_id, lineage_id, centroid, radius, mass, parent_region_id, score) and `RegionMap` (per expert: regions, matched count, `carried_mass`) are the `regions.json` / `region_map.json` of a trained snapshot (written by `trainer`).
- `align_expert(AlignInput)` (`align_expert_with` for a non-default `split_threshold`, 0.2) runs `lineage::align` on the parent's and the new snapshot's assignment of the same probe windows: a Hungarian match ≥ `carry_threshold` **inherits the parent's `lineage_id`**; per-region `ancestors`, per-expert `dead_parents`, `split_parents` and `regime_break` are recorded in the map (`align_by_probes` remains as the carry-only view); region ids stay unique per snapshot (`<snapshot>-e<k>-r<nnn>`, `region_map::region_id`).
- `LeaderClustering::fit_assign(points)` discovers a region set and assigns every point (inside a radius, else the nearest centroid), deterministic in point order; shared by the trainer's region fit and its feedback search.
- `OnlineRegionSet::seeded(cfg, snapshot, expert, prefix, &[FittedRegion])` starts a live set from a map, so `model-runtime` continues the snapshot's lineages after a load or hot swap; later births continue the ordinal series.

## Region model (lasting content of the retired spec 30)
Spec 30 (latent regions) was retired on 2026-11-11 once every acceptance criterion held and its open questions were decided; its last version is `specs/30-latent-regions.md` in git history (section numbers cited in code comments as "spec 30 §x" refer to it). What remains authoritative:

**What a region is.** A region of one **expert space** of one encoder **snapshot** at one **timescale** (a `(snapshot, timescale, expert)` triple) — a region of latent space, not a time window or an asset cluster. Discovery is family-specific (`FamilyDescriptor::region_discovery`: `leader` statistical, `density` HVAE-flow / latent diffusion, `mapper` JEPA-VAE, `codebook` FSQ hybrid); only leader clustering streams, so a batch method serves as a leader set seeded with its regions (`online_form`). A region holds mass ≥ `min_mass`, is separated by a density drop or code boundary, and carries a **predictive-coherence score** (below).

**Identity across refits (D-007, D-069, D-070).** Probe-set alignment: both snapshots assign the same probe windows; Jaccard overlap; Hungarian carry ≥ `carry_threshold` (0.5) inherits the `lineage_id`; other pairs ≥ `split_threshold` (0.2) are `DescendsFrom{carry|split|merge|partial}`; unmatched new regions are born, parents without partner die; carried probe mass < 0.5 is a **regime break** (the kernel restarts from the prior). Neural generations warm-started from a neural parent add the anchor loss (λ 0.1), so alignment mostly confirms identity that training preserved. Online: birth outside every radius, merge within `merge_radius`, split by streaming 2-means (`split_distance` 1.5 radii, `min_split_mass` 30), death by decayed support (`support_halflife` 20 000, `min_age` 2 000, `min_support` 0.002).

**Decisions (spec 30 open questions).**
1. Identity: probe alignment + anchored training (above).
2. *Predictive coherence is a soft score, not a mandatory criterion* (2026-11-11, provisional). Discovery never drops or merges a region for being incoherent outside the FSQ codebook merge (KS on the residuals of neighbouring codes, `merge_alpha` 0.05); every region carries `p` of the KS test of its next-step residuals vs the residuals elsewhere, and consumers weight by it. Mandatory coherence would tie the region count to the residual sample size (thin regions never reject) and break lineages whenever a regime's residual law drifts toward the global one. Consumer today: model-runtime's expert residual uses a region's conditional mean only while the region is coherent (`coherence_alpha` 0.01), which feeds the cross-asset priors; gauges `coherence.<ts>.e<k>.tested|coherent`.
3. Memberships are published soft; the serving kernel observes hard transitions (`MarkovGraph::observe_soft` exists).
4. HDBSCAN, Mapper and the codebook are implemented here in pure Rust; persistent homology is deferred.
5. *Regions are per expert space, memberships per instrument* (2026-11-11, provisional). A region set is shared by every instrument the encoder embeds; asset sets enter only through what the encoder embeds (spec 31 asset selection / cross-asset attention), never as separate region sets. Per-instrument region sets would multiply graph size by the universe and starve thin instruments; per-asset-set sets wait for an encoder that embeds sets.
6. *Region episodes* (2026-11-11, provisional): stored as Helix `RegionEpisode` nodes for timescales 1m and slower only (1 s would write one node per instrument and expert every few seconds); one node per finished stay `(region, instrument, t_enter, t_exit, bars, max_weight, mean_weight)`; dropped by model-runtime's retention with the observation latents' horizon (`--hub-retention`, `1m=14d, 1h=180d`, slower kept). Estimated volume: one node per region change, ≈ 2·10⁴/day at 1m for 20 instruments × 3 experts with a mean dwell of 5 bars (~6 MB/day).

**Collapse guard (spec 30 §8).** A set that collapsed into ≤ 1 region or reached `max_regions_per_set` is invalid (`validity`). model-runtime refuses a hot swap onto a snapshot whose map has a collapsed expert (the previous snapshot and its sets stay active; counter `regions_invalid_refused.<ts>`), starts a collapsed expert unseeded when there is nothing to keep (process start, shadow), checks every live set after warm-up (2 000 observations) and raises `regions_invalid.<ts>.e<k>` (alert `RegionSetInvalid`).

**Data model (Helix, written by model-runtime).** `RegionSet{region_set_id = expert_space_id, method, state, kernel_state}` (`ExpertSpace -HasRegionSet->`); `LatentRegion{region_id, lineage_id, method, radius, status, mass, occupancy, spread, age, created_at, died_at, retire_reason}` + centroid `LatentVec{kind: centroid}`; `InSpace`; `DescendsFrom{kind, overlap}`; `Hier{p_coarse_given_fine, p_fine_given_coarse, support, pmi_bits, mi_bits}` (`REFINES`, MI ≥ `hier_mi_min` 0.05 bits, support ≥ 20, `P(coarse|fine)` ≥ 0.05); `RegionStatusVersion` (as-of reads, D-078); `RegionEpisode` (above). VictoriaMetrics: `fa_region_weight{tf, inst, expert, region}` = soft-membership weight at each closed bar (1m and slower, only the ≤ 4 regions of each membership). Neighbours are answered from the kernel and the centroids (no `NEIGHBOR` edge).

**Tunables (implemented defaults).** `min_support` 0.002, online cap `--max-regions` 64 (= `max_regions_per_set`), `novelty_threshold` 3.0 radii (out-of-region), `membership_k` 4, `carry_threshold` 0.5, `split_threshold` 0.2, `coherence_alpha` 0.01 (`CoherenceCfg`: `min_n` 20, windows 256 / 2 048), `hier_mi_min` 0.05 bits, `drift_refit_kl` 0.1 (three consecutive → `train.adapter`, cooldown 6 h).

**Failure modes.** Collapse → guard above. Alignment keeps < 50 % of the mass → regime break. Novelty spike across many instruments → no automatic mid-event refit; the occupancy drift trigger needs three consecutive summaries. Helix write failure → model-runtime's observation log ([README](../../bins/model-runtime/README.md#helixdb-srchub_sinkrs-d-048)); region sets are flushed every `--hub-state-secs` and restored on start, so regions are never only in memory.

**Acceptance (held at retirement).** Conformance of every discovery (recovery ≥ 90 %, determinism, self-alignment ≥ 0.9); sets / regions / lineage queryable by business id; identical ids on a deterministic refit; lifecycle events and carried ids on a new snapshot; membership ≤ 5 ms per instrument per step at 1m (`bench_step_latency_per_instrument`, p99 1.3–3.6 ms); the console's latent explorer renders occupancy and hierarchy.

## Owns
Region discovery per encoder family, region identity across snapshots, cross-timescale edges, occupancy drift.

## Governing specs
- spec 30 (latent regions) — retired 2026-11-11; its lasting content is the *Region model* section above
