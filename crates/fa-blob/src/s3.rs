//! S3 client for Garage over plain `reqwest` + SigV4 (path-style addressing).
use std::time::Duration;

use bytes::Bytes;
use chrono::Utc;
use fa_core::{Secrets, config::PlatformConfig};
use reqwest::{Method, StatusCode, Url};
use serde::Deserialize;

use crate::{
    BlobError, BlobStore, ObjectKey, ObjectMeta,
    sigv4::{self, Credentials, EMPTY_SHA256, Scope},
};

/// Garage/S3 object store. Cheap to clone (shares the HTTP client).
#[derive(Clone)]
pub struct S3BlobStore {
    client: reqwest::Client,
    endpoint: Url,
    host: String,
    region: String,
    creds: Credentials,
}

impl std::fmt::Debug for S3BlobStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3BlobStore")
            .field("endpoint", &self.endpoint.as_str())
            .field("region", &self.region)
            .field("access_key_id", &self.creds.access_key_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ListBucketResult {
    #[serde(default)]
    contents: Vec<Contents>,
    #[serde(default)]
    is_truncated: bool,
    next_continuation_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Contents {
    key: String,
    #[serde(default)]
    size: u64,
    last_modified: Option<String>,
    #[serde(rename = "ETag")]
    e_tag: Option<String>,
}

/// Parses a ListObjectsV2 response body.
pub(crate) fn parse_list(xml: &str) -> Result<(Vec<ObjectMeta>, Option<String>), BlobError> {
    let r: ListBucketResult = quick_xml::de::from_str(xml).map_err(|e| BlobError::Backend(format!("list xml: {e}")))?;
    let objs = r
        .contents
        .into_iter()
        .map(|c| ObjectMeta {
            key: c.key,
            size: c.size,
            last_modified: c.last_modified,
            etag: c.e_tag.map(|t| t.trim_matches('"').to_owned()),
        })
        .collect();
    Ok((objs, if r.is_truncated { r.next_continuation_token } else { None }))
}

impl S3BlobStore {
    /// `endpoint` like `http://127.0.0.1:3900`; `region` as the Garage config's `s3_api.s3_region`.
    pub fn new(endpoint: &str, region: &str, access_key_id: &str, secret_access_key: &str) -> Result<Self, BlobError> {
        if access_key_id.is_empty() || secret_access_key.is_empty() {
            return Err(BlobError::NotConfigured);
        }
        let endpoint = Url::parse(endpoint).map_err(|e| BlobError::Backend(format!("endpoint `{endpoint}`: {e}")))?;
        let host = match (endpoint.host_str(), endpoint.port()) {
            (Some(h), Some(p)) => format!("{h}:{p}"),
            (Some(h), None) => h.to_owned(),
            (None, _) => return Err(BlobError::Backend(format!("endpoint `{endpoint}` has no host"))),
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| BlobError::Backend(e.to_string()))?;
        Ok(Self {
            client,
            endpoint,
            host,
            region: region.to_owned(),
            creds: Credentials { access_key_id: access_key_id.into(), secret_access_key: secret_access_key.into() },
        })
    }

    /// From `[garage]` in `config/platform.toml` and `[garage]` in `config/secrets.toml`.
    /// `Err(NotConfigured)` when the credentials are empty.
    pub fn from_config(cfg: &PlatformConfig, secrets: &Secrets) -> Result<Self, BlobError> {
        Self::new(
            &cfg.garage.endpoint,
            &cfg.garage.region,
            &secrets.garage.access_key_id,
            &secrets.garage.secret_access_key,
        )
    }

    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    fn scope(&self) -> Scope<'_> {
        Scope { region: &self.region, service: "s3" }
    }

    /// Path-style object path `/<bucket>/<key>` (un-encoded; the signer and URL builder encode).
    fn path(bucket: &str, key: &str) -> String {
        if key.is_empty() { format!("/{bucket}") } else { format!("/{bucket}/{key}") }
    }

    fn url(&self, path: &str, query: &[(String, String)]) -> Result<Url, BlobError> {
        let mut s = format!("{}://{}{}", self.endpoint.scheme(), self.host, sigv4::uri_encode(path, true));
        if !query.is_empty() {
            s.push('?');
            s.push_str(&sigv4::canonical_query(query));
        }
        Url::parse(&s).map_err(|e| BlobError::Backend(format!("url `{s}`: {e}")))
    }

    async fn send(
        &self,
        method: Method,
        path: &str,
        query: &[(String, String)],
        extra_headers: &[(String, String)],
        body: Option<(Bytes, &str)>,
    ) -> Result<reqwest::Response, BlobError> {
        let payload_hash = match &body {
            Some((b, _)) => sigv4::sha256_hex(b),
            None => EMPTY_SHA256.to_owned(),
        };
        let mut headers = vec![("host".to_owned(), self.host.clone())];
        if let Some((b, ct)) = &body {
            headers.push(("content-type".into(), (*ct).to_owned()));
            headers.push(("content-length".into(), b.len().to_string()));
        }
        headers.extend_from_slice(extra_headers);
        let signed = sigv4::sign_headers(
            &self.creds,
            &self.scope(),
            Utc::now(),
            method.as_str(),
            path,
            query,
            &headers,
            &payload_hash,
        );
        let mut req = self.client.request(method, self.url(path, query)?);
        for (k, v) in &signed.headers {
            if k == "host" || k == "content-length" {
                continue; // set by the HTTP stack from the URL / body
            }
            req = req.header(k.as_str(), v.as_str());
        }
        if let Some((b, _)) = body {
            req = req.body(b);
        }
        req.send().await.map_err(|e| BlobError::Backend(format!("{e}")))
    }

    async fn check(resp: reqwest::Response, key: &ObjectKey) -> Result<reqwest::Response, BlobError> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        if status == StatusCode::NOT_FOUND {
            return Err(BlobError::NotFound(key.bucket, key.key.clone()));
        }
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(300).collect();
        Err(BlobError::Backend(format!("{status} on {}/{}: {snippet}", key.bucket, key.key)))
    }

    /// Presigned GET URL valid for `expires` (max 7 days per S3; Garage accepts the same).
    pub fn presigned_get(&self, key: &ObjectKey, expires: Duration) -> Result<Url, BlobError> {
        let path = Self::path(key.bucket, &key.key);
        let q = sigv4::presign_query(
            &self.creds,
            &self.scope(),
            Utc::now(),
            expires.as_secs().max(1),
            "GET",
            &path,
            &self.host,
            &[],
        );
        let s = format!("{}://{}{}?{q}", self.endpoint.scheme(), self.host, sigv4::uri_encode(&path, true));
        Url::parse(&s).map_err(|e| BlobError::Backend(format!("url `{s}`: {e}")))
    }

    /// One page of ListObjectsV2.
    async fn list_page(
        &self,
        bucket: &'static str,
        prefix: &str,
        token: Option<&str>,
    ) -> Result<(Vec<ObjectMeta>, Option<String>), BlobError> {
        let mut q = vec![("list-type".to_owned(), "2".to_owned())];
        if !prefix.is_empty() {
            q.push(("prefix".into(), prefix.to_owned()));
        }
        if let Some(t) = token {
            q.push(("continuation-token".into(), t.to_owned()));
        }
        let key = ObjectKey { bucket, key: prefix.to_owned() };
        let resp = self.send(Method::GET, &Self::path(bucket, ""), &q, &[], None).await?;
        let resp = Self::check(resp, &key).await?;
        let text = resp.text().await.map_err(|e| BlobError::Backend(e.to_string()))?;
        parse_list(&text)
    }
}

impl BlobStore for S3BlobStore {
    async fn put(&self, key: &ObjectKey, bytes: Vec<u8>, content_type: &str) -> Result<(), BlobError> {
        let resp = self
            .send(Method::PUT, &Self::path(key.bucket, &key.key), &[], &[], Some((Bytes::from(bytes), content_type)))
            .await?;
        Self::check(resp, key).await.map(drop)
    }

    async fn get(&self, key: &ObjectKey) -> Result<Vec<u8>, BlobError> {
        let resp = self.send(Method::GET, &Self::path(key.bucket, &key.key), &[], &[], None).await?;
        let resp = Self::check(resp, key).await?;
        Ok(resp.bytes().await.map_err(|e| BlobError::Backend(e.to_string()))?.to_vec())
    }

    async fn exists(&self, key: &ObjectKey) -> Result<bool, BlobError> {
        let resp = self.send(Method::HEAD, &Self::path(key.bucket, &key.key), &[], &[], None).await?;
        match Self::check(resp, key).await {
            Ok(_) => Ok(true),
            Err(BlobError::NotFound(..)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn delete(&self, key: &ObjectKey) -> Result<(), BlobError> {
        let resp = self.send(Method::DELETE, &Self::path(key.bucket, &key.key), &[], &[], None).await?;
        match Self::check(resp, key).await {
            Ok(_) | Err(BlobError::NotFound(..)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn list(&self, bucket: &'static str, prefix: &str) -> Result<Vec<ObjectMeta>, BlobError> {
        let mut all = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let (page, next) = self.list_page(bucket, prefix, token.as_deref()).await?;
            all.extend(page);
            match next {
                Some(t) if !t.is_empty() => token = Some(t),
                _ => break,
            }
        }
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_list_objects_v2() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>fa-artifacts</Name><Prefix>evals/</Prefix><KeyCount>2</KeyCount><MaxKeys>1000</MaxKeys>
  <IsTruncated>true</IsTruncated><NextContinuationToken>tok1</NextContinuationToken>
  <Contents><Key>evals/g1/j1.json</Key><LastModified>2026-11-10T04:00:00.000Z</LastModified>
    <ETag>"d41d8cd98f00b204e9800998ecf8427e"</ETag><Size>12</Size><StorageClass>STANDARD</StorageClass></Contents>
  <Contents><Key>evals/g1/j2.json</Key><Size>0</Size></Contents>
</ListBucketResult>"#;
        let (objs, next) = parse_list(xml).unwrap();
        assert_eq!(objs.len(), 2);
        assert_eq!(objs[0].key, "evals/g1/j1.json");
        assert_eq!(objs[0].size, 12);
        assert_eq!(objs[0].etag.as_deref(), Some("d41d8cd98f00b204e9800998ecf8427e"));
        assert_eq!(objs[0].last_modified.as_deref(), Some("2026-11-10T04:00:00.000Z"));
        assert_eq!(objs[1].size, 0);
        assert_eq!(next.as_deref(), Some("tok1"));

        let empty = r#"<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>b</Name><IsTruncated>false</IsTruncated></ListBucketResult>"#;
        let (objs, next) = parse_list(empty).unwrap();
        assert!(objs.is_empty());
        assert!(next.is_none());
    }

    #[test]
    fn urls_and_config() {
        let s = S3BlobStore::new("http://127.0.0.1:3900", "garage", "AK", "SK").unwrap();
        assert_eq!(s.host, "127.0.0.1:3900");
        let k = ObjectKey { bucket: crate::buckets::ARTIFACTS, key: "evals/g 1/j.json".into() };
        let u = s.url(&S3BlobStore::path(k.bucket, &k.key), &[("list-type".into(), "2".into())]).unwrap();
        assert_eq!(u.as_str(), "http://127.0.0.1:3900/fa-artifacts/evals/g%201/j.json?list-type=2");
        let p = s.presigned_get(&k, Duration::from_secs(600)).unwrap();
        assert!(
            p.as_str()
                .starts_with("http://127.0.0.1:3900/fa-artifacts/evals/g%201/j.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&")
        );
        assert!(p.as_str().contains("X-Amz-Expires=600"));
        assert!(p.as_str().contains("&X-Amz-Signature="));
        assert!(!format!("{s:?}").contains("SK"), "secret must not be printed");

        assert!(matches!(S3BlobStore::new("http://x", "r", "", ""), Err(BlobError::NotConfigured)));
        let no_port = S3BlobStore::new("https://s3.example.org", "garage", "AK", "SK").unwrap();
        assert_eq!(no_port.host, "s3.example.org");
    }

    #[test]
    fn from_repo_config() {
        let cfg = PlatformConfig::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/platform.toml")).unwrap();
        let empty = Secrets::default();
        assert!(matches!(S3BlobStore::from_config(&cfg, &empty), Err(BlobError::NotConfigured)));
        let mut s = Secrets::default();
        s.garage.access_key_id = "AK".into();
        s.garage.secret_access_key = "SK".into();
        let store = S3BlobStore::from_config(&cfg, &s).unwrap();
        assert_eq!(store.region, "garage");
    }
}
