//! Disk-backed observation log of the Helix sink (spec 30 §8 "Helix write failure → buffer",
//! spec 33 §8 "Helix lag / outage → observation log, replay on reconnect").
//!
//! While Helix is unreachable (or answers 5xx / 429 / retryable), and afterwards until the
//! backlog is written, [`crate::hub_sink`] **defers** every write here instead of dropping it:
//!
//! - **Ordered log** (`<dir>/<id>.seg`, 64 MiB segments, records `u32 LE length + payload`):
//!   events whose every instance matters and whose order matters — snapshots, region births and
//!   status marks, lifecycle (lineage edges, deaths, centroids), closed-bar latents, region
//!   episodes. Replayed oldest first; the replay cursor (`<dir>/cursor`, `"<segment> <offset>"`)
//!   is written after each replayed batch, so a crash replays at most one batch twice (all of
//!   them are upserts except the observation latents, which may then appear twice).
//! - **Keyed state** (`<dir>/keyed/<prio>-<kind>-<hash>.rec`, one record each, atomic
//!   rename): idempotent upserts superseded by the next write of the same key — the `RegionSet`
//!   persist rows per (snapshot, timescale), `KernelModel` per expert space, `Hier` edges per
//!   timescale pair and expert, `CrossPrior` rows per timescale, and the coalesced `Transition`
//!   rows (latest per edge). Only the latest value is kept, so a long outage costs disk in
//!   proportion to the number of keys, not to its duration. Replayed after the ordered log.
//!
//! The ordered log is bounded by `max_bytes` (`--hub-log-max-mb`, default 2048): beyond it an
//! append is refused and counted (`hub.log_full`, alert `HubBacklog`), never silently dropped.
//! Undecodable records are skipped and counted (`hub.log_corrupt`). A torn tail (crash during
//! an append) is truncated at open.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use fa_hub::{NewRegionEpisode, NewTransition, TransitionUpsert};
use fa_proto::{
    Message,
    v1::{
        CrossAssetPriors, ExpertLatent, ExpertSpaceRef, KernelUpdate, RegionEvent, RegionHierarchy,
        RegionLifecycleEvent,
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    hub_sink::{HubEvent, SnapshotDesc},
    pipeline::ExpertPersist,
};

const SEG_BYTES: u64 = 64 << 20;
const CURSOR: &str = "cursor";
const KEYED: &str = "keyed";

/// Position in the ordered log: (segment id, byte offset).
pub type Pos = (u64, u64);

pub struct HubLog {
    dir: PathBuf,
    max_bytes: u64,
    /// segment id → size in bytes.
    segments: BTreeMap<u64, u64>,
    /// Open append handle of the newest segment.
    write: Option<(u64, File)>,
    cursor: Pos,
}

fn seg_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("{id:012}.seg"))
}

/// Length of the valid prefix of a segment (complete records only).
fn valid_prefix(path: &Path) -> io::Result<u64> {
    let mut r = BufReader::new(File::open(path)?);
    let mut off = 0u64;
    let mut len = [0u8; 4];
    loop {
        if r.read_exact(&mut len).is_err() {
            return Ok(off);
        }
        let n = u64::from(u32::from_le_bytes(len));
        let mut buf = vec![0u8; n as usize];
        if r.read_exact(&mut buf).is_err() {
            return Ok(off);
        }
        off += 4 + n;
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_data()?;
    }
    fs::rename(tmp, path)
}

impl HubLog {
    /// Opens (or creates) the log in `dir`; validates the newest segment and drops consumed ones.
    pub fn open(dir: impl Into<PathBuf>, max_bytes: u64) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(dir.join(KEYED))?;
        let mut segments = BTreeMap::new();
        for e in fs::read_dir(&dir)? {
            let p = e?.path();
            if p.extension().is_some_and(|x| x == "seg")
                && let Some(id) = p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<u64>().ok())
            {
                segments.insert(id, fs::metadata(&p)?.len());
            }
        }
        if let Some((&last, &size)) = segments.iter().next_back() {
            let valid = valid_prefix(&seg_path(&dir, last))?;
            if valid < size {
                tracing::warn!(segment = last, size, valid, "hub log: truncating a torn tail");
                OpenOptions::new().write(true).open(seg_path(&dir, last))?.set_len(valid)?;
                segments.insert(last, valid);
            }
        }
        let first = segments.keys().next().copied().unwrap_or(0);
        let cursor = fs::read_to_string(dir.join(CURSOR))
            .ok()
            .and_then(|s| {
                let mut it = s.split_whitespace().map(|x| x.parse::<u64>().ok());
                Some((it.next()??, it.next()??))
            })
            .filter(|(seg, _)| segments.contains_key(seg) || segments.keys().all(|k| k > seg))
            .unwrap_or((first, 0));
        let mut log = Self { dir, max_bytes, segments, write: None, cursor };
        log.gc()?;
        Ok(log)
    }

    /// Replay cursor (the next record to replay starts here).
    pub fn cursor(&self) -> Pos {
        self.cursor
    }

    /// Bytes held by the ordered log.
    pub fn bytes(&self) -> u64 {
        self.segments.values().sum()
    }

    /// Ordered records not replayed yet.
    pub fn pending(&self) -> bool {
        self.segments.iter().any(|(id, size)| (*id, *size) > self.cursor && *size > 0 && *id >= self.cursor.0)
    }

    /// Appends one record; `Ok(false)` when the log is at `max_bytes`.
    pub fn append(&mut self, payload: &[u8]) -> io::Result<bool> {
        let rec = 4 + payload.len() as u64;
        if self.bytes() + rec > self.max_bytes {
            return Ok(false);
        }
        let roll = match &self.write {
            None => true,
            Some((id, _)) => self.segments.get(id).copied().unwrap_or(0) >= SEG_BYTES,
        };
        if roll {
            let id = self.segments.keys().next_back().map_or(self.cursor.0, |l| l + 1);
            let f = OpenOptions::new().create(true).append(true).open(seg_path(&self.dir, id))?;
            self.segments.entry(id).or_insert(0);
            self.write = Some((id, f));
        }
        let Some((id, f)) = self.write.as_mut() else { return Ok(false) };
        let mut buf = Vec::with_capacity(rec as usize);
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(payload);
        f.write_all(&buf)?;
        *self.segments.entry(*id).or_insert(0) += rec;
        Ok(true)
    }

    /// Up to `max` records from the cursor, each with the position just after it.
    pub fn read(&mut self, max: usize) -> io::Result<Vec<(Vec<u8>, Pos)>> {
        if let Some((_, f)) = self.write.as_mut() {
            f.flush()?;
        }
        let mut out = Vec::new();
        let (mut seg, mut off) = self.cursor;
        while out.len() < max {
            let Some(&size) = self.segments.get(&seg) else {
                match self.segments.range(seg..).next() {
                    Some((&next, _)) => {
                        (seg, off) = (next, 0);
                        continue;
                    }
                    None => break,
                }
            };
            if off >= size {
                match self.segments.range(seg + 1..).next() {
                    Some((&next, _)) => {
                        (seg, off) = (next, 0);
                        continue;
                    }
                    None => break,
                }
            }
            let mut f = File::open(seg_path(&self.dir, seg))?;
            f.seek(SeekFrom::Start(off))?;
            let mut r = BufReader::new(f);
            let mut len = [0u8; 4];
            while out.len() < max && off < size {
                r.read_exact(&mut len)?;
                let n = u64::from(u32::from_le_bytes(len));
                let mut buf = vec![0u8; n as usize];
                r.read_exact(&mut buf)?;
                off += 4 + n;
                out.push((buf, (seg, off)));
            }
        }
        Ok(out)
    }

    /// Moves the replay cursor to `pos` (persisted) and deletes fully replayed segments.
    pub fn commit(&mut self, pos: Pos) -> io::Result<()> {
        self.cursor = pos;
        self.gc()?;
        atomic_write(&self.dir.join(CURSOR), format!("{} {}", self.cursor.0, self.cursor.1).as_bytes())
    }

    fn gc(&mut self) -> io::Result<()> {
        let (cseg, coff) = self.cursor;
        let done: Vec<u64> = self
            .segments
            .iter()
            .filter(|(id, size)| {
                **id < cseg || (**id == cseg && coff >= **size && self.segments.range(cseg + 1..).next().is_some())
            })
            .map(|(id, _)| *id)
            .collect();
        for id in done {
            self.remove_segment(id)?;
        }
        // everything replayed: start over with an empty log
        if !self.pending()
            && let Some((&last, _)) = self.segments.iter().next_back()
        {
            let ids: Vec<u64> = self.segments.keys().copied().collect();
            for id in ids {
                self.remove_segment(id)?;
            }
            self.cursor = (last + 1, 0);
        }
        Ok(())
    }

    fn remove_segment(&mut self, id: u64) -> io::Result<()> {
        if self.write.as_ref().is_some_and(|(w, _)| *w == id) {
            self.write = None;
        }
        self.segments.remove(&id);
        match fs::remove_file(seg_path(&self.dir, id)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }

    /// Stores the latest value of a keyed state record (replaces the previous one).
    pub fn keyed_put(&self, name: &str, payload: &[u8]) -> io::Result<()> {
        atomic_write(&self.dir.join(KEYED).join(name), payload)
    }

    /// Keyed records, in replay order (by their priority prefix, then name).
    pub fn keyed_names(&self) -> io::Result<Vec<String>> {
        let mut v: Vec<String> = fs::read_dir(self.dir.join(KEYED))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with(".rec"))
            .collect();
        v.sort();
        Ok(v)
    }

    pub fn keyed_get(&self, name: &str) -> io::Result<Option<Vec<u8>>> {
        match fs::read(self.dir.join(KEYED).join(name)) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    pub fn keyed_remove(&self, name: &str) -> io::Result<()> {
        match fs::remove_file(self.dir.join(KEYED).join(name)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        }
    }

    /// Bytes held by the keyed records.
    pub fn keyed_bytes(&self) -> u64 {
        fs::read_dir(self.dir.join(KEYED))
            .map(|d| d.filter_map(|e| e.ok()?.metadata().ok()).map(|m| m.len()).sum())
            .unwrap_or(0)
    }
}

// ------------------------------------------------------------------------------------------
// Codec: tag byte, then length-prefixed fields (prost bytes for protos, JSON for the rest).

/// Name of the keyed file of the transition rows (one record, latest per edge).
pub const TRANSITIONS_KEY: &str = "5-transitions.rec";

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

fn space_key(s: Option<&ExpertSpaceRef>) -> String {
    s.map_or_else(String::new, |s| format!("{}/{}/{}", s.snapshot_id, s.timescale, s.expert_index))
}

/// Keyed-state file of an event superseded by the next one of the same key; `None` for events
/// that belong in the ordered log.
pub fn keyed_name(ev: &HubEvent) -> Option<String> {
    let (prio, kind, key) = match ev {
        HubEvent::Kernel(k) => (1, "kernel", space_key(k.space.as_ref())),
        HubEvent::Hierarchy(h) => {
            (2, "hier", format!("{}>{}", space_key(h.fine.as_ref()), space_key(h.coarse.as_ref())))
        }
        HubEvent::CrossPriors(c) => (3, "cross", c.timescale.clone()),
        HubEvent::Persist(rows) => {
            (4, "persist", rows.iter().map(|r| space_key(Some(&r.space))).collect::<Vec<_>>().join(","))
        }
        HubEvent::TransitionRows(_) => return Some(TRANSITIONS_KEY.to_owned()),
        _ => return None,
    };
    Some(format!("{prio}-{kind}-{:016x}.rec", fnv(&key)))
}

#[derive(Serialize, Deserialize)]
struct TransitionRec {
    from: String,
    to: String,
    sampler: String,
    count: i64,
    mean: f64,
    lower: f64,
    upper: f64,
    kernel_params: String,
}

#[derive(Serialize, Deserialize)]
struct EpisodeRec {
    region_id: String,
    expert_space_id: String,
    timescale: String,
    instrument_id: String,
    t_enter_ms: i64,
    t_exit_ms: i64,
    bars: u32,
    max_weight: f64,
    mean_weight: f64,
}

#[derive(Serialize, Deserialize)]
struct PersistRec {
    family: String,
    region_state: Option<serde_json::Value>,
    kernel_state: Option<String>,
    stats: Vec<fa_patterns::RegionStat>,
}

struct W(Vec<u8>);

impl W {
    fn new(tag: u8) -> Self {
        Self(vec![tag])
    }
    fn field(&mut self, b: &[u8]) -> &mut Self {
        self.0.extend_from_slice(&(b.len() as u32).to_le_bytes());
        self.0.extend_from_slice(b);
        self
    }
    fn proto(&mut self, m: &impl Message) -> &mut Self {
        self.field(&m.encode_to_vec())
    }
    fn json(&mut self, v: &impl Serialize) -> &mut Self {
        self.field(&serde_json::to_vec(v).unwrap_or_default())
    }
}

struct R<'a> {
    b: &'a [u8],
}

impl<'a> R<'a> {
    fn field(&mut self) -> Result<&'a [u8], String> {
        if self.b.len() < 4 {
            return Err("truncated field length".into());
        }
        let n = u32::from_le_bytes([self.b[0], self.b[1], self.b[2], self.b[3]]) as usize;
        if self.b.len() < 4 + n {
            return Err("truncated field".into());
        }
        let (f, rest) = self.b[4..].split_at(n);
        self.b = rest;
        Ok(f)
    }
    fn proto<M: Message + Default>(&mut self) -> Result<M, String> {
        M::decode(self.field()?).map_err(|e| e.to_string())
    }
    fn json<T: for<'de> Deserialize<'de>>(&mut self) -> Result<T, String> {
        serde_json::from_slice(self.field()?).map_err(|e| e.to_string())
    }
    fn more(&self) -> bool {
        !self.b.is_empty()
    }
}

/// Serialises an event for the log.
pub fn encode(ev: &HubEvent) -> Vec<u8> {
    let w = match ev {
        HubEvent::Snapshot(d) => {
            let mut w = W::new(0);
            w.json(d);
            w
        }
        HubEvent::Region { event, radius } => {
            let mut w = W::new(1);
            w.proto(event).field(&radius.to_le_bytes());
            w
        }
        HubEvent::Transition { update, observed_to } => {
            let mut w = W::new(2);
            w.proto(update).json(observed_to);
            w
        }
        HubEvent::Latent { space, instrument_id, ts_ms, experts } => {
            let mut w = W::new(3);
            w.proto(space).field(instrument_id.as_bytes()).field(&ts_ms.to_le_bytes());
            for e in experts {
                w.proto(e);
            }
            w
        }
        HubEvent::Lifecycle(e) => {
            let mut w = W::new(4);
            w.proto(e);
            w
        }
        HubEvent::Kernel(k) => {
            let mut w = W::new(5);
            w.proto(k);
            w
        }
        HubEvent::Persist(rows) => {
            let mut w = W::new(6);
            for r in rows {
                w.proto(&r.space).json(&PersistRec {
                    family: r.family.clone(),
                    region_state: r.region_state.clone(),
                    kernel_state: r.kernel_state.clone(),
                    stats: r.stats.clone(),
                });
            }
            w
        }
        HubEvent::Hierarchy(h) => {
            let mut w = W::new(7);
            w.proto(h);
            w
        }
        HubEvent::CrossPriors(c) => {
            let mut w = W::new(8);
            w.proto(c);
            w
        }
        HubEvent::TransitionRows(rows) => {
            let recs: Vec<TransitionRec> = rows
                .iter()
                .map(|r| TransitionRec {
                    from: r.from_region_id.clone(),
                    to: r.to_region_id.clone(),
                    sampler: r.transition.sampler_id.clone(),
                    count: r.transition.count,
                    mean: r.transition.mean,
                    lower: r.transition.lower,
                    upper: r.transition.upper,
                    kernel_params: r.transition.kernel_params.clone(),
                })
                .collect();
            let mut w = W::new(9);
            w.json(&recs);
            w
        }
        HubEvent::Episodes(rows) => {
            let recs: Vec<EpisodeRec> = rows
                .iter()
                .map(|e| EpisodeRec {
                    region_id: e.region_id.clone(),
                    expert_space_id: e.expert_space_id.clone(),
                    timescale: e.timescale.clone(),
                    instrument_id: e.instrument_id.clone(),
                    t_enter_ms: e.t_enter_ms,
                    t_exit_ms: e.t_exit_ms,
                    bars: e.bars,
                    max_weight: e.max_weight,
                    mean_weight: e.mean_weight,
                })
                .collect();
            let mut w = W::new(10);
            w.json(&recs);
            w
        }
    };
    w.0
}

fn i64_of(b: &[u8]) -> Result<i64, String> {
    Ok(i64::from_le_bytes(b.try_into().map_err(|_| "bad i64")?))
}

/// Inverse of [`encode`].
pub fn decode(bytes: &[u8]) -> Result<HubEvent, String> {
    let (&tag, rest) = bytes.split_first().ok_or("empty record")?;
    let mut r = R { b: rest };
    Ok(match tag {
        0 => HubEvent::Snapshot(r.json::<SnapshotDesc>()?),
        1 => {
            let event: RegionEvent = r.proto()?;
            let radius = f64::from_le_bytes(r.field()?.try_into().map_err(|_| "bad radius")?);
            HubEvent::Region { event, radius }
        }
        2 => {
            let update = r.proto()?;
            HubEvent::Transition { update, observed_to: r.json()? }
        }
        3 => {
            let space: ExpertSpaceRef = r.proto()?;
            let instrument_id = String::from_utf8(r.field()?.to_vec()).map_err(|e| e.to_string())?;
            let ts_ms = i64_of(r.field()?)?;
            let mut experts = Vec::new();
            while r.more() {
                experts.push(r.proto::<ExpertLatent>()?);
            }
            HubEvent::Latent { space, instrument_id, ts_ms, experts }
        }
        4 => HubEvent::Lifecycle(r.proto::<RegionLifecycleEvent>()?),
        5 => HubEvent::Kernel(r.proto::<KernelUpdate>()?),
        6 => {
            let mut rows = Vec::new();
            while r.more() {
                let space: ExpertSpaceRef = r.proto()?;
                let p: PersistRec = r.json()?;
                rows.push(ExpertPersist {
                    space,
                    family: p.family,
                    region_state: p.region_state,
                    kernel_state: p.kernel_state,
                    stats: p.stats,
                });
            }
            HubEvent::Persist(rows)
        }
        7 => HubEvent::Hierarchy(r.proto::<RegionHierarchy>()?),
        8 => HubEvent::CrossPriors(r.proto::<CrossAssetPriors>()?),
        9 => HubEvent::TransitionRows(
            r.json::<Vec<TransitionRec>>()?
                .into_iter()
                .map(|t| TransitionUpsert {
                    from_region_id: t.from,
                    to_region_id: t.to,
                    transition: NewTransition {
                        sampler_id: t.sampler,
                        count: t.count,
                        mean: t.mean,
                        lower: t.lower,
                        upper: t.upper,
                        kernel_params: t.kernel_params,
                    },
                })
                .collect(),
        ),
        10 => HubEvent::Episodes(
            r.json::<Vec<EpisodeRec>>()?
                .into_iter()
                .map(|e| NewRegionEpisode {
                    region_id: e.region_id,
                    expert_space_id: e.expert_space_id,
                    timescale: e.timescale,
                    instrument_id: e.instrument_id,
                    t_enter_ms: e.t_enter_ms,
                    t_exit_ms: e.t_exit_ms,
                    bars: e.bars,
                    max_weight: e.max_weight,
                    mean_weight: e.mean_weight,
                })
                .collect(),
        ),
        t => return Err(format!("unknown tag {t}")),
    })
}

/// Default directory: `FA_HUB_LOG_DIR`, else `var/model-runtime/hub-log` under the working dir.
pub fn default_dir() -> PathBuf {
    std::env::var_os("FA_HUB_LOG_DIR").map_or_else(|| PathBuf::from("var/model-runtime/hub-log"), PathBuf::from)
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::region_event;

    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fa-hub-log-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn region(i: usize) -> HubEvent {
        HubEvent::Region {
            event: RegionEvent {
                region_id: format!("r{i}"),
                kind: region_event::Kind::Created as i32,
                ..Default::default()
            },
            radius: 1.5,
        }
    }

    #[test]
    fn ordered_records_replay_in_order_across_segments_and_restarts() {
        let dir = tmp("ordered");
        let mut log = HubLog::open(&dir, 1 << 30).unwrap();
        assert!(!log.pending());
        for i in 0..10 {
            assert!(log.append(&encode(&region(i))).unwrap());
        }
        assert!(log.pending());
        let batch = log.read(4).unwrap();
        assert_eq!(batch.len(), 4);
        log.commit(batch[3].1).unwrap();
        drop(log);
        // a restart resumes after the committed batch
        let mut log = HubLog::open(&dir, 1 << 30).unwrap();
        let rest = log.read(100).unwrap();
        let ids: Vec<String> = rest
            .iter()
            .map(|(b, _)| match decode(b).unwrap() {
                HubEvent::Region { event, radius } => {
                    assert_eq!(radius, 1.5);
                    event.region_id
                }
                _ => panic!(),
            })
            .collect();
        assert_eq!(ids, (4..10).map(|i| format!("r{i}")).collect::<Vec<_>>());
        log.commit(rest.last().unwrap().1).unwrap();
        assert!(!log.pending());
        assert_eq!(log.bytes(), 0, "a fully replayed log is deleted");
        // and starts again cleanly
        assert!(log.append(&encode(&region(11))).unwrap());
        assert_eq!(log.read(10).unwrap().len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_full_log_refuses_and_a_torn_tail_is_truncated() {
        let dir = tmp("full");
        let rec = encode(&region(1));
        let mut log = HubLog::open(&dir, (rec.len() as u64 + 4) * 2).unwrap();
        assert!(log.append(&rec).unwrap());
        assert!(log.append(&rec).unwrap());
        assert!(!log.append(&rec).unwrap(), "over max_bytes");
        drop(log);
        // tear the last record
        let seg = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.path().extension().is_some_and(|x| x == "seg"))
            .unwrap()
            .path();
        let len = fs::metadata(&seg).unwrap().len();
        OpenOptions::new().write(true).open(&seg).unwrap().set_len(len - 3).unwrap();
        let mut log = HubLog::open(&dir, 1 << 20).unwrap();
        assert_eq!(log.read(10).unwrap().len(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn keyed_state_keeps_the_latest_per_key_and_every_event_round_trips() {
        let dir = tmp("keyed");
        let log = HubLog::open(&dir, 1 << 20).unwrap();
        let k = |e: u32, mass: f64| {
            HubEvent::Kernel(KernelUpdate {
                space: Some(ExpertSpaceRef { snapshot_id: "s".into(), timescale: "1m".into(), expert_index: e }),
                observation_mass: mass,
                ..Default::default()
            })
        };
        for (e, m) in [(0, 1.0), (1, 2.0), (0, 3.0)] {
            let ev = k(e, m);
            log.keyed_put(&keyed_name(&ev).unwrap(), &encode(&ev)).unwrap();
        }
        let names = log.keyed_names().unwrap();
        assert_eq!(names.len(), 2);
        let masses: Vec<f64> = names
            .iter()
            .map(|n| match decode(&log.keyed_get(n).unwrap().unwrap()).unwrap() {
                HubEvent::Kernel(k) => k.observation_mass,
                _ => panic!(),
            })
            .collect();
        assert!(masses.contains(&3.0) && masses.contains(&2.0) && !masses.contains(&1.0));
        assert!(keyed_name(&region(0)).is_none());
        // the rest of the codec
        let rows = vec![TransitionUpsert {
            from_region_id: "a".into(),
            to_region_id: "b".into(),
            transition: NewTransition {
                sampler_id: "conjugate".into(),
                count: 3,
                mean: 0.5,
                lower: 0.1,
                upper: 0.9,
                kernel_params: "{}".into(),
            },
        }];
        match decode(&encode(&HubEvent::TransitionRows(rows.clone()))).unwrap() {
            HubEvent::TransitionRows(r) => assert_eq!(r, rows),
            _ => panic!(),
        }
        let latent = HubEvent::Latent {
            space: ExpertSpaceRef { snapshot_id: "s".into(), timescale: "1m".into(), expert_index: 0 },
            instrument_id: "i".into(),
            ts_ms: 42,
            experts: vec![ExpertLatent { expert_index: 1, mean: vec![1.0, 2.0], ..Default::default() }; 2],
        };
        match decode(&encode(&latent)).unwrap() {
            HubEvent::Latent { experts, ts_ms, instrument_id, .. } => {
                assert_eq!((experts.len(), ts_ms, instrument_id.as_str()), (2, 42, "i"));
            }
            _ => panic!(),
        }
        let ep = NewRegionEpisode {
            region_id: "r".into(),
            expert_space_id: "s-e0".into(),
            timescale: "1m".into(),
            instrument_id: "i".into(),
            t_enter_ms: 1,
            t_exit_ms: 2,
            bars: 1,
            max_weight: 0.9,
            mean_weight: 0.8,
        };
        match decode(&encode(&HubEvent::Episodes(vec![ep.clone()]))).unwrap() {
            HubEvent::Episodes(v) => assert_eq!(v, vec![ep]),
            _ => panic!(),
        }
        assert!(decode(&[99]).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
