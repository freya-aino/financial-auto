# helix

HelixDB project: `schema.hx` (nodes, edges, vector types) and `queries.hx` (query catalog). Governed by [spec 20 §5.2](../specs/20-storage.md) and D-012.

**Status:** schema and queries drafted. **Unvalidated:** the `helix` CLI isn't installed, so `helix check` hasn't run. Expect syntax fixes on first check.

## Contents
| Area | Nodes / vectors | Edges |
|---|---|---|
| Reference (mirror of Postgres `ref`) | Instrument, Entity, Venue | IssuedBy, ListedOn, SameExposure |
| Snapshots | Snapshot, ExpertSpace | HasSpace |
| Regions / Markov graph | LatentRegion, Window, `V::LatentVec` | InSpace, Transition, Hier, RegionMatch, Contains |
| Evolution | Genome, FitnessRecord | ParentOf, Scored, TrainedOn, FeedbackFrom |
| Text / media | `V::NewsDoc`, `V::MediaSegment` | Mentions |

## Known design gaps
- **Vector types are static** in HelixQL, but expert spaces appear at runtime with varying dims. v1 puts all latents in one `LatentVec` type filtered by `expert_space_id`, which means mixed-dimension vectors in one index. Whether Helix tolerates that is unverified. The alternative is one V-type per (timescale, dim) bucket. Part of the spec 20 benchmark spike.
- No upsert semantics: `upsertInstrument` currently always adds.

## Run
```sh
cd helix
helix check
helix push dev   # serves POST /<query_name> on 127.0.0.1:6969
```
