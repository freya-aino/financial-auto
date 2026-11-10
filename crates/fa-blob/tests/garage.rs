//! Round-trip against the compose Garage (`just up`, `docker/garage-init.sh`). Opt in with
//! `FA_GARAGE_TEST=1`; credentials from `config/secrets.toml` (or `FA_GARAGE_*`).
use std::time::Duration;

use fa_blob::{BlobStore, JSON, ObjectKey, S3BlobStore, buckets, eval_report};
use fa_core::{Secrets, config::PlatformConfig};

fn store() -> Option<S3BlobStore> {
    if std::env::var("FA_GARAGE_TEST").ok().as_deref() != Some("1") {
        eprintln!("FA_GARAGE_TEST!=1: skipping live Garage test");
        return None;
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let cfg = PlatformConfig::load(format!("{root}/config/platform.toml")).expect("platform.toml");
    let secrets_path = std::env::var_os("FA_SECRETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| format!("{root}/config/secrets.toml").into());
    let secrets = Secrets::load_from(secrets_path).expect("secrets");
    Some(S3BlobStore::from_config(&cfg, &secrets).expect("[garage] credentials in config/secrets.toml"))
}

#[tokio::test]
async fn garage_roundtrip() {
    let Some(s) = store() else { return };
    let run = ulid::Ulid::generate().to_string();
    let key = eval_report(&format!("test-{run}"), "job-1");
    let body = format!(r#"{{"protocol":"test","run":"{run}"}}"#).into_bytes();
    let prefix = format!("evals/test-{run}/");

    assert!(!s.exists(&key).await.unwrap(), "fresh key must not exist");
    s.put(&key, body.clone(), JSON).await.unwrap();
    assert!(s.exists(&key).await.unwrap());
    assert_eq!(s.get(&key).await.unwrap(), body);

    let second = ObjectKey { bucket: buckets::ARTIFACTS, key: format!("{prefix}job 2.json") };
    s.put(&second, b"{}".to_vec(), JSON).await.unwrap();
    let mut listed: Vec<String> =
        s.list(buckets::ARTIFACTS, &prefix).await.unwrap().into_iter().map(|o| o.key).collect();
    listed.sort();
    assert_eq!(listed, vec![second.key.clone(), key.key.clone()]);
    let meta = s.list(buckets::ARTIFACTS, &key.key).await.unwrap();
    assert_eq!(meta.len(), 1);
    assert_eq!(meta[0].size, body.len() as u64);

    // Presigned GET works without credentials.
    let url = s.presigned_get(&key, Duration::from_secs(300)).unwrap();
    let fetched = reqwest::get(url.clone()).await.unwrap();
    assert_eq!(fetched.status(), 200, "presigned url {url}");
    assert_eq!(fetched.bytes().await.unwrap().to_vec(), body);

    s.delete(&key).await.unwrap();
    s.delete(&second).await.unwrap();
    s.delete(&key).await.unwrap(); // idempotent
    assert!(!s.exists(&key).await.unwrap());
    assert!(matches!(s.get(&key).await, Err(fa_blob::BlobError::NotFound(..))));
    assert!(s.list(buckets::ARTIFACTS, &prefix).await.unwrap().is_empty());
}
