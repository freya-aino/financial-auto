//! Frame timeline of the `visual.frame` decoders (`nn_frame`, spec 34 §6.1 "broadcast frames"):
//! `media.>` envelopes (`MediaChunk` of kind image / video, recorded in the `MEDIA` stream) over
//! the window, their blobs fetched from Garage (`blob_ref`, bucket `fa-media` unless the ref names
//! another) and hashed into frame features (`fa_decoders::neural::data::frame_features`, the
//! frozen byte-n-gram front-end; no pretrained vision tower, D-010), keyed by receive time.
//!
//! Without a media producer the stream is empty (spec 10: no capture connector), so a frame
//! genome fails with "no frames" — the family is trainable as soon as frames arrive.

use fa_blob::{BlobStore, ObjectKey, S3BlobStore};
use fa_bus::replay::RecordedMessage;
use fa_decoders::neural::data::{TextTimeline, frame_features};
use fa_proto::{
    Message,
    v1::{Envelope, MediaKind, envelope::Payload},
};
use serde_json::{Value, json};

/// Cap on media chunks read per job.
pub const MAX_FRAMES: usize = 20_000;

/// Garage key of a `blob_ref` (`<bucket>/<key>` for a known bucket, else a key in `fa-media`).
pub fn blob_key(blob_ref: &str) -> Option<ObjectKey> {
    let r = blob_ref.trim().trim_start_matches("s3://");
    if r.is_empty() {
        return None;
    }
    for bucket in [fa_blob::buckets::MEDIA, fa_blob::buckets::RAW, fa_blob::buckets::ARTIFACTS] {
        if let Some(key) = r.strip_prefix(bucket).and_then(|k| k.strip_prefix('/')) {
            return Some(ObjectKey { bucket, key: key.to_owned() });
        }
    }
    Some(ObjectKey { bucket: fa_blob::buckets::MEDIA, key: r.to_owned() })
}

/// `(receive ms, blob ref)` of the image / video chunks among recorded `media.>` envelopes, and
/// the count of undecodable or non-visual messages.
pub fn frame_refs(msgs: &[RecordedMessage]) -> (Vec<(i64, String)>, usize) {
    let mut out = Vec::new();
    let mut skipped = 0;
    for m in msgs {
        match Envelope::decode(m.payload.as_slice()).ok().and_then(|e| e.payload) {
            Some(Payload::Media(c))
                if matches!(MediaKind::try_from(c.kind), Ok(MediaKind::Image | MediaKind::Video))
                    && !c.blob_ref.is_empty() =>
            {
                out.push((m.published_ms, c.blob_ref));
            }
            _ => skipped += 1,
        }
    }
    (out, skipped)
}

/// Builds the frame timeline (blobs that cannot be read are counted, not fatal).
pub async fn timeline(blob: Option<&S3BlobStore>, refs: &[(i64, String)]) -> (TextTimeline, Value) {
    let mut t = TextTimeline::frames();
    let (mut fetched, mut missing) = (0usize, 0usize);
    if let Some(blob) = blob {
        for (ts, r) in refs.iter().take(MAX_FRAMES) {
            let Some(key) = blob_key(r) else {
                missing += 1;
                continue;
            };
            match blob.get(&key).await {
                Ok(bytes) => {
                    t.push_features(*ts, frame_features(&bytes), &[]);
                    fetched += 1;
                }
                Err(_) => missing += 1,
            }
        }
    }
    t.finish();
    let block = json!({"source": "MEDIA", "chunks": refs.len(), "frames": t.len(), "fetched": fetched,
                       "missing": missing, "garage": blob.is_some(), "front_end": "hashed byte 4-grams"});
    (t, block)
}

#[cfg(test)]
mod tests {
    use fa_proto::v1::MediaChunk;

    use super::*;

    #[test]
    fn refs_keep_visual_chunks_and_keys_resolve() {
        let env = |kind: MediaKind, r: &str| Envelope {
            payload: Some(Payload::Media(MediaChunk { kind: kind as i32, blob_ref: r.into(), ..Default::default() })),
            ..Default::default()
        };
        let msgs: Vec<RecordedMessage> = [
            env(MediaKind::Image, "fa-media/yt/abc/frames/0001.jpg"),
            env(MediaKind::Audio, "fa-media/yt/abc/audio/0001.opus"),
            env(MediaKind::Video, "yt/abc/video/seg1.mp4"),
        ]
        .iter()
        .enumerate()
        .map(|(i, e)| RecordedMessage {
            subject: "media.chunk.yt".into(),
            stream_seq: i as u64,
            published_ms: 1000 * i as i64,
            payload: e.encode_to_vec(),
        })
        .collect();
        let (refs, skipped) = frame_refs(&msgs);
        assert_eq!((refs.len(), skipped), (2, 1));
        let k = blob_key(&refs[0].1).unwrap();
        assert_eq!((k.bucket, k.key.as_str()), ("fa-media", "yt/abc/frames/0001.jpg"));
        let k = blob_key(&refs[1].1).unwrap();
        assert_eq!((k.bucket, k.key.as_str()), ("fa-media", "yt/abc/video/seg1.mp4"));
        assert!(blob_key(" ").is_none());
    }
}
