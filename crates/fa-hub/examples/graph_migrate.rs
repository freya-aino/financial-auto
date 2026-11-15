//! Copies the model-state graph between backends (D-108 cutover): snapshots with their spaces,
//! regions, transitions (current values; the event-time history stays), lineage, hierarchy,
//! per-space state and the reference mirror.
//!
//!   cargo run -p fa-hub --example graph_migrate -- --from helix --to postgres [--dry-run]
//!
//! Sources: `FA_HELIX_URL` (default http://127.0.0.1:6969), `DATABASE_URL` (the platform
//! Postgres with migrations applied). Idempotent: everything writes through upserts, so a
//! re-run converges. Not copied (still Helix-resident, see D-108): genomes/fitness/evolution,
//! news/media embeddings (Qdrant), the TrainedOn/FeedbackFrom links of pre-migration genomes.
use fa_hub::graph::GraphStore;
use fa_hub::{HelixGraph, HubClient, PgGraph, RefEntity, RefInstrument, RefListing, RefVenue, TransitionUpsert};

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let from = arg("--from").unwrap_or_else(|| "helix".into());
    let to = arg("--to").unwrap_or_else(|| "postgres".into());
    let dry = std::env::args().any(|a| a == "--dry-run");
    let helix = HelixGraph::new(HubClient::from_env().with_graph(None));
    let pg = PgGraph::connect_lazy(&std::env::var("DATABASE_URL").expect("DATABASE_URL"));
    match (from.as_str(), to.as_str()) {
        ("helix", "postgres") => migrate(&helix, &pg, dry).await,
        ("postgres", "helix") => migrate(&pg, &helix, dry).await,
        other => panic!("unsupported direction {other:?}"),
    }
}

async fn migrate<S: GraphStore, D: GraphStore>(src: &S, dst: &D, dry: bool) {
    let snapshots = src.list_snapshots().await.expect("list snapshots");
    println!("snapshots: {}", snapshots.len());
    let mut counts = [0usize; 6]; // spaces, regions, transitions, descents, hier, adapters
    for snap in &snapshots {
        let Some(g) = src.snapshot_graph(&snap.snapshot_id).await.expect("snapshot graph") else { continue };
        if dry {
            counts[0] += g.spaces.len();
            counts[1] += g.regions.len();
            counts[2] += g.transitions.len();
            continue;
        }
        let new = dst
            .ensure_snapshot(&fa_hub::NewSnapshot {
                snapshot_id: snap.snapshot_id.clone(),
                timescale: snap.timescale.clone(),
                encoder_family: snap.encoder_family.clone(),
                generation: snap.generation,
                garage_key: snap.garage_key.clone(),
                status: snap.status.clone(),
            })
            .await
            .expect("snapshot");
        let copied = dst.set_snapshot_status(new.id, &snap.status).await.expect("status");
        if let Some(m) = &snap.manifest {
            let _ = dst.set_snapshot_manifest(copied.id, m, snap.carried_mass.unwrap_or(0.0)).await;
        }
        if let Some(prev) = &snap.previous_active {
            let _ = dst.set_snapshot_previous_active(copied.id, prev).await;
        }
        for sp in &g.spaces {
            let new_space = dst
                .ensure_expert_space(
                    new.id,
                    &fa_hub::NewExpertSpace {
                        expert_space_id: sp.expert_space_id.clone(),
                        snapshot_id: sp.snapshot_id.clone(),
                        index: sp.index,
                        dim: sp.dim,
                    },
                )
                .await
                .expect("space");
            counts[0] += 1;
            for r in g.regions.iter().filter(|r| r.expert_space_id == sp.expert_space_id) {
                dst.upsert_region(
                    new_space.id,
                    &fa_hub::NewRegion {
                        region_id: r.region_id.clone(),
                        lineage_id: r.lineage_id.clone(),
                        expert_space_id: r.expert_space_id.clone(),
                        method: r.method.clone(),
                        radius: r.radius,
                        density: r.density,
                    },
                )
                .await
                .expect("region");
                if let (Some(st), Some(rel)) = (r.status.as_deref(), r.related_region_id.as_deref())
                    && st != "live"
                {
                    let _ = dst.mark_region_status(&r.region_id, st, rel).await;
                }
                counts[1] += 1;
                // direct lineage links of this region
                if let Some(lin) = src.region_lineage(&r.region_id, 1).await.expect("lineage") {
                    let descents: Vec<fa_hub::NewDescent> = lin
                        .ancestors
                        .iter()
                        .filter_map(|l| {
                            let parent = l.region.properties.get("region_id")?.as_str()?.to_owned();
                            Some(fa_hub::NewDescent {
                                child_region_id: r.region_id.clone(),
                                parent_region_id: parent,
                                kind: l.kind.clone(),
                                overlap: l.overlap,
                            })
                        })
                        .collect();
                    dst.upsert_descents(&descents).await.expect("descents");
                    counts[3] += descents.len();
                }
            }
            if let Some(rs) = src.region_set(&sp.expert_space_id).await.expect("region set") {
                dst.upsert_region_set(&fa_hub::NewRegionSet {
                    expert_space_id: sp.expert_space_id.clone(),
                    snapshot_id: rs.snapshot_id.clone(),
                    timescale: rs.timescale.clone(),
                    expert_index: rs.expert_index,
                    family: String::new(),
                    method: String::new(),
                    n_regions: rs.n_regions,
                    cfg_hash: rs.cfg_hash.clone(),
                    state: rs.state.clone(),
                    kernel_state: rs.kernel_state.clone(),
                })
                .await
                .expect("region set");
            }
        }
        let ups: Vec<TransitionUpsert> = g
            .transitions
            .iter()
            .filter_map(|t| {
                Some(TransitionUpsert {
                    from_region_id: t.from_region_id.clone()?,
                    to_region_id: t.to_region_id.clone()?,
                    transition: fa_hub::NewTransition {
                        sampler_id: t.sampler_id.clone(),
                        count: t.count,
                        mean: t.mean,
                        lower: t.lower,
                        upper: t.upper,
                        kernel_params: t.kernel_params.clone(),
                    },
                })
            })
            .collect();
        for chunk in ups.chunks(100) {
            dst.upsert_transitions(chunk).await.expect("transitions");
        }
        counts[2] += ups.len();
        for e in src.snapshot_hierarchy(&snap.snapshot_id).await.expect("hierarchy") {
            let get = |k: &str| e.properties.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let id = |k: &str| e.properties.get(k).and_then(|v| v.as_str()).unwrap_or("").to_owned();
            dst.upsert_hier_edges(&[fa_hub::NewHierEdge {
                fine_region_id: id("fine_region_id"),
                coarse_region_id: id("coarse_region_id"),
                p_coarse_given_fine: get("p_coarse_given_fine"),
                p_fine_given_coarse: get("p_fine_given_coarse"),
                support: get("support"),
                pmi_bits: get("pmi_bits"),
                mi_bits: get("mi_bits"),
            }])
            .await
            .expect("hier");
            counts[4] += 1;
        }
        for a in src.snapshot_adapters(&snap.snapshot_id, 10_000).await.expect("adapters") {
            let parse = |t: &Option<String>| {
                t.as_deref()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map_or(0, |d| d.timestamp_millis())
            };
            let _ = dst
                .add_adapter_state(
                    new.id,
                    &fa_hub::NewAdapterState {
                        snapshot_id: a.snapshot_id.clone(),
                        timescale: a.timescale.clone(),
                        version: a.version,
                        garage_key: a.garage_key.clone(),
                        drift: a.drift,
                        max_drift: a.max_drift,
                        frozen: a.frozen,
                        n_windows: a.n_windows,
                        window_start_ms: parse(&a.window_start),
                        window_end_ms: parse(&a.window_end),
                    },
                )
                .await; // unique violation on re-run: already copied
            counts[5] += 1;
        }
    }
    // reference mirror: whole instruments via their reference reads
    let ids = src.ref_instrument_ids().await.expect("ref ids");
    println!("ref instruments: {}", ids.len());
    if !dry {
        for (id, _status) in &ids {
            let Some(r) = src.instrument_reference(id).await.expect("ref read") else { continue };
            let gs = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_owned();
            for e in &r.issuer {
                dst.upsert_ref_entities(&[RefEntity {
                    entity_id: gs(e, "entity_id"),
                    lei: gs(e, "lei"),
                    name: gs(e, "name"),
                    country: gs(e, "country"),
                    sector: gs(e, "sector"),
                }])
                .await
                .expect("entity");
            }
            for v in &r.venues {
                dst.upsert_ref_venues(&[RefVenue {
                    venue_id: gs(v, "venue_id"),
                    code: gs(v, "code"),
                    mic: gs(v, "mic"),
                    name: gs(v, "name"),
                    kind: gs(v, "kind"),
                    timezone: gs(v, "timezone"),
                }])
                .await
                .expect("venue");
            }
            let i = &r.instrument;
            let listings = r
                .listings
                .iter()
                .map(|l| RefListing {
                    venue_id: l.properties.get("venue_id").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                    venue_symbol: l.properties.get("venue_symbol").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                    source_id: l.properties.get("source_id").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                })
                .collect();
            dst.upsert_ref_instruments(&[RefInstrument {
                instrument_id: gs(i, "instrument_id"),
                asset_class: gs(i, "asset_class"),
                kind: gs(i, "kind"),
                name: gs(i, "name"),
                status: gs(i, "status"),
                base_ccy: gs(i, "base_ccy"),
                quote_ccy: gs(i, "quote_ccy"),
                issuer_id: r.issuer.first().map(|e| gs(e, "entity_id")).filter(|s| !s.is_empty()),
                listings,
                exposures: Vec::new(),
            }])
            .await
            .expect("instrument");
        }
    }
    println!(
        "copied: spaces {} regions {} transitions {} descents {} hier {} adapters {} (dry-run: {dry})",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    );
}
