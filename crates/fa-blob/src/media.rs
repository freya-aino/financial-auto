//! Media downsampling (spec 20 §6.5, `garage.media_downsample_after_d`): video is downsampled
//! after N days, audio and transcripts are kept forever. Media lives at
//! `fa-media/<source>/<media_id>/{video,audio,frames,transcript}/…` (§5.5); the downsampled form
//! of a video is its sampled `frames/`, so once a media item has frames, its full `video/`
//! objects older than the horizon are deleted. Items without frames keep their video (nothing to
//! downsample to), as does every object of unknown age.
use std::collections::BTreeSet;

use chrono::{DateTime, Duration, Utc};

use crate::{BlobError, BlobStore, ObjectKey, ObjectMeta, buckets};

/// Default `garage.media_downsample_after_d`.
pub const DEFAULT_AFTER_DAYS: i64 = 30;

/// `(source, media_id, part)` of a media key, `None` for keys outside the layout.
fn parts(key: &str) -> Option<(&str, &str, &str)> {
    let mut it = key.splitn(4, '/');
    let (source, media, part) = (it.next()?, it.next()?, it.next()?);
    it.next().filter(|name| !name.is_empty())?;
    Some((source, media, part))
}

fn modified(o: &ObjectMeta) -> Option<DateTime<Utc>> {
    o.last_modified.as_deref().and_then(|t| DateTime::parse_from_rfc3339(t).ok()).map(|t| t.with_timezone(&Utc))
}

/// The `video/` keys to delete from a listing of `fa-media`. Pure: no I/O.
pub fn plan(objects: &[ObjectMeta], now: DateTime<Utc>, after: Duration) -> Vec<String> {
    let with_frames: BTreeSet<(&str, &str)> = objects
        .iter()
        .filter_map(|o| parts(&o.key))
        .filter(|(_, _, part)| *part == "frames")
        .map(|(s, m, _)| (s, m))
        .collect();
    objects
        .iter()
        .filter(|o| {
            parts(&o.key).is_some_and(|(s, m, part)| part == "video" && with_frames.contains(&(s, m)))
                && modified(o).is_some_and(|t| now - t >= after)
        })
        .map(|o| o.key.clone())
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct DownsampleReport {
    pub objects: usize,
    pub deleted: usize,
    pub deleted_bytes: u64,
    pub dry_run: bool,
}

/// Lists `fa-media`, plans and (unless `dry_run`) deletes the downsampled videos.
pub async fn downsample<S: BlobStore>(
    store: &S,
    now: DateTime<Utc>,
    after: Duration,
    dry_run: bool,
) -> Result<DownsampleReport, BlobError> {
    let listing = store.list(buckets::MEDIA, "").await?;
    let delete = plan(&listing, now, after);
    let size: std::collections::HashMap<&str, u64> = listing.iter().map(|o| (o.key.as_str(), o.size)).collect();
    let mut report = DownsampleReport { objects: listing.len(), dry_run, ..Default::default() };
    for key in &delete {
        if !dry_run {
            store.delete(&ObjectKey { bucket: buckets::MEDIA, key: key.clone() }).await?;
        }
        report.deleted += 1;
        report.deleted_bytes += size.get(key.as_str()).copied().unwrap_or(0);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(key: &str, days_old: Option<i64>, now: DateTime<Utc>) -> ObjectMeta {
        ObjectMeta {
            key: key.into(),
            size: 10,
            last_modified: days_old.map(|d| (now - Duration::days(d)).to_rfc3339()),
            etag: None,
        }
    }

    #[test]
    fn deletes_only_old_video_that_has_frames() {
        let now = Utc::now();
        let listing = vec![
            obj("yt/m1/video/full.mp4", Some(40), now),
            obj("yt/m1/frames/0001.jpg", Some(40), now),
            obj("yt/m1/audio/a.opus", Some(40), now),
            obj("yt/m1/transcript/t.json", Some(40), now),
            obj("yt/m2/video/full.mp4", Some(40), now), // no frames: kept
            obj("yt/m3/video/full.mp4", Some(5), now),  // too young
            obj("yt/m3/frames/0001.jpg", Some(5), now),
            obj("yt/m4/video/full.mp4", None, now), // unknown age
            obj("yt/m4/frames/0001.jpg", None, now),
            obj("stray.mp4", Some(400), now),
        ];
        assert_eq!(plan(&listing, now, Duration::days(DEFAULT_AFTER_DAYS)), ["yt/m1/video/full.mp4"]);
    }

    #[tokio::test]
    async fn downsample_deletes_from_the_store() {
        let store = crate::MemoryStore::default();
        for part in ["video/v.mp4", "frames/f.jpg", "audio/a.opus"] {
            let key = ObjectKey { bucket: buckets::MEDIA, key: format!("src/m/{part}") };
            store.put(&key, vec![0; 4], "application/octet-stream").await.unwrap();
        }
        // MemoryStore reports no modification time, so nothing is old enough: kept
        let r = downsample(&store, Utc::now(), Duration::days(30), false).await.unwrap();
        assert_eq!((r.objects, r.deleted), (3, 0));
        let dry = downsample(&store, Utc::now(), Duration::zero(), true).await.unwrap();
        assert_eq!(dry.deleted, 0, "unknown age is never deleted, even with a zero horizon");
    }
}
