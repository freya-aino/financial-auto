//! AWS Signature Version 4 (header and query/presigned variants), implemented from the
//! documented algorithm so that no S3 SDK is needed for Garage.
//!
//! <https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html>
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

pub const ALGORITHM: &str = "AWS4-HMAC-SHA256";
pub const UNSIGNED_PAYLOAD: &str = "UNSIGNED-PAYLOAD";
/// `sha256("")`, the payload hash of body-less requests.
pub const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Clone)]
pub struct Credentials {
    pub access_key_id: String,
    pub secret_access_key: String,
}

#[derive(Debug, Clone)]
pub struct Scope<'a> {
    pub region: &'a str,
    pub service: &'a str,
}

/// `YYYYMMDD'T'HHMMSS'Z'` and its date part.
pub fn amz_date(t: DateTime<Utc>) -> (String, String) {
    (t.format("%Y%m%dT%H%M%SZ").to_string(), t.format("%Y%m%d").to_string())
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// AWS URI encoding: unreserved characters stay, everything else is `%XX` (upper-case hex).
/// `/` is kept when `keep_slash` (path segments) and encoded otherwise (query parts).
pub fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            b'/' if keep_slash => out.push('/'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Canonical query string: parameters URI-encoded and sorted by name, then value.
pub fn canonical_query(params: &[(String, String)]) -> String {
    let mut enc: Vec<(String, String)> =
        params.iter().map(|(k, v)| (uri_encode(k, false), uri_encode(v, false))).collect();
    enc.sort();
    enc.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
}

/// Lower-cases names, trims values, sorts by name. Returns `(canonical_headers, signed_headers)`.
pub fn canonical_headers(headers: &[(String, String)]) -> (String, String) {
    let mut h: Vec<(String, String)> = headers
        .iter()
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.split_whitespace().collect::<Vec<_>>().join(" ")))
        .collect();
    h.sort();
    let canonical = h.iter().map(|(k, v)| format!("{k}:{v}\n")).collect::<String>();
    let signed = h.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>().join(";");
    (canonical, signed)
}

/// The canonical request (step 1) and the signed-headers list.
pub fn canonical_request(
    method: &str,
    path: &str,
    query: &[(String, String)],
    headers: &[(String, String)],
    payload_hash: &str,
) -> (String, String) {
    let (ch, signed) = canonical_headers(headers);
    let path = if path.is_empty() { "/" } else { path };
    let cr =
        format!("{method}\n{}\n{}\n{ch}\n{signed}\n{payload_hash}", uri_encode(path, true), canonical_query(query));
    (cr, signed)
}

pub fn credential_scope(date: &str, scope: &Scope<'_>) -> String {
    format!("{date}/{}/{}/aws4_request", scope.region, scope.service)
}

/// The string to sign (step 2).
pub fn string_to_sign(amz_date: &str, credential_scope: &str, canonical_request: &str) -> String {
    format!("{ALGORITHM}\n{amz_date}\n{credential_scope}\n{}", sha256_hex(canonical_request.as_bytes()))
}

/// The derived signing key (step 3).
pub fn signing_key(secret: &str, date: &str, scope: &Scope<'_>) -> Vec<u8> {
    let k_date = hmac(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = hmac(&k_date, scope.region.as_bytes());
    let k_service = hmac(&k_region, scope.service.as_bytes());
    hmac(&k_service, b"aws4_request")
}

/// The signature (step 4), hex.
pub fn signature(signing_key: &[u8], string_to_sign: &str) -> String {
    hex::encode(hmac(signing_key, string_to_sign.as_bytes()))
}

/// Output of header signing: the headers to send (`Authorization` included).
#[derive(Debug, Clone)]
pub struct SignedRequest {
    pub headers: Vec<(String, String)>,
    pub signature: String,
}

/// Signs a request with the `Authorization` header. `headers` must already contain `host`;
/// `x-amz-date` and `x-amz-content-sha256` are added.
#[allow(clippy::too_many_arguments)]
pub fn sign_headers(
    creds: &Credentials,
    scope: &Scope<'_>,
    now: DateTime<Utc>,
    method: &str,
    path: &str,
    query: &[(String, String)],
    headers: &[(String, String)],
    payload_hash: &str,
) -> SignedRequest {
    let (amz, date) = amz_date(now);
    let mut headers = headers.to_vec();
    headers.push(("x-amz-date".into(), amz.clone()));
    headers.push(("x-amz-content-sha256".into(), payload_hash.to_owned()));
    let (cr, signed) = canonical_request(method, path, query, &headers, payload_hash);
    let cs = credential_scope(&date, scope);
    let sts = string_to_sign(&amz, &cs, &cr);
    let sig = signature(&signing_key(&creds.secret_access_key, &date, scope), &sts);
    headers.push((
        "authorization".into(),
        format!("{ALGORITHM} Credential={}/{cs}, SignedHeaders={signed}, Signature={sig}", creds.access_key_id),
    ));
    SignedRequest { headers, signature: sig }
}

/// Presigned (query-string) authentication: returns the full canonical query to append to the
/// URL, including `X-Amz-Signature`. Only `host` is signed; the payload is unsigned.
#[allow(clippy::too_many_arguments)]
pub fn presign_query(
    creds: &Credentials,
    scope: &Scope<'_>,
    now: DateTime<Utc>,
    expires_secs: u64,
    method: &str,
    path: &str,
    host: &str,
    extra_query: &[(String, String)],
) -> String {
    let (amz, date) = amz_date(now);
    let cs = credential_scope(&date, scope);
    let mut query = extra_query.to_vec();
    query.push(("X-Amz-Algorithm".into(), ALGORITHM.into()));
    query.push(("X-Amz-Credential".into(), format!("{}/{cs}", creds.access_key_id)));
    query.push(("X-Amz-Date".into(), amz.clone()));
    query.push(("X-Amz-Expires".into(), expires_secs.to_string()));
    query.push(("X-Amz-SignedHeaders".into(), "host".into()));
    let headers = [("host".to_owned(), host.to_owned())];
    let (cr, _) = canonical_request(method, path, &query, &headers, UNSIGNED_PAYLOAD);
    let sts = string_to_sign(&amz, &cs, &cr);
    let sig = signature(&signing_key(&creds.secret_access_key, &date, scope), &sts);
    format!("{}&X-Amz-Signature={sig}", canonical_query(&query))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// AWS "Examples of how to derive a signing key" (IAM, 2012-02-15).
    #[test]
    fn signing_key_example() {
        let k = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20120215",
            &Scope { region: "us-east-1", service: "iam" },
        );
        assert_eq!(hex::encode(k), "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d");
    }

    fn suite_creds() -> Credentials {
        Credentials {
            access_key_id: "AKIDEXAMPLE".into(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
        }
    }

    /// aws-sig-v4-test-suite `get-vanilla`.
    #[test]
    fn get_vanilla() {
        let scope = Scope { region: "us-east-1", service: "service" };
        let headers = [
            ("Host".to_owned(), "example.amazonaws.com".to_owned()),
            ("X-Amz-Date".to_owned(), "20150830T123600Z".to_owned()),
        ];
        let (cr, signed) = canonical_request("GET", "/", &[], &headers, EMPTY_SHA256);
        assert_eq!(
            cr,
            "GET\n/\n\nhost:example.amazonaws.com\nx-amz-date:20150830T123600Z\n\nhost;x-amz-date\n\
             e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(signed, "host;x-amz-date");
        let cs = credential_scope("20150830", &scope);
        let sts = string_to_sign("20150830T123600Z", &cs, &cr);
        assert_eq!(
            sts,
            "AWS4-HMAC-SHA256\n20150830T123600Z\n20150830/us-east-1/service/aws4_request\n\
             bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63"
        );
        let sig = signature(&signing_key(&suite_creds().secret_access_key, "20150830", &scope), &sts);
        assert_eq!(sig, "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31");
    }

    /// aws-sig-v4-test-suite `get-vanilla-query-order-key-case` (`?Param2=value2&Param1=value1`).
    #[test]
    fn get_vanilla_query_order_key_case() {
        let scope = Scope { region: "us-east-1", service: "service" };
        let headers = [
            ("Host".to_owned(), "example.amazonaws.com".to_owned()),
            ("X-Amz-Date".to_owned(), "20150830T123600Z".to_owned()),
        ];
        let query = [("Param2".to_owned(), "value2".to_owned()), ("Param1".to_owned(), "value1".to_owned())];
        let (cr, _) = canonical_request("GET", "/", &query, &headers, EMPTY_SHA256);
        assert!(cr.starts_with("GET\n/\nParam1=value1&Param2=value2\n"), "{cr}");
        let sts = string_to_sign("20150830T123600Z", &credential_scope("20150830", &scope), &cr);
        let sig = signature(&signing_key(&suite_creds().secret_access_key, "20150830", &scope), &sts);
        assert_eq!(sig, "b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500");
    }

    /// aws-sig-v4-test-suite `get-utf8` (path `/ሴ`), checks URI encoding of the path.
    #[test]
    fn get_utf8_path() {
        let scope = Scope { region: "us-east-1", service: "service" };
        let headers = [
            ("Host".to_owned(), "example.amazonaws.com".to_owned()),
            ("X-Amz-Date".to_owned(), "20150830T123600Z".to_owned()),
        ];
        let (cr, _) = canonical_request("GET", "/ሴ", &[], &headers, EMPTY_SHA256);
        assert!(cr.starts_with("GET\n/%E1%88%B4\n\n"), "{cr}");
        let sts = string_to_sign("20150830T123600Z", &credential_scope("20150830", &scope), &cr);
        let sig = signature(&signing_key(&suite_creds().secret_access_key, "20150830", &scope), &sts);
        assert_eq!(sig, "8318018e0b0f223aa2bbf98705b62bb787dc9c0e678f255a891fd03141be5d85");
    }

    /// S3 docs "Authenticating Requests (AWS Signature Version 4) — GET object" example.
    #[test]
    fn s3_get_object_header_example() {
        let creds = Credentials {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
        };
        let scope = Scope { region: "us-east-1", service: "s3" };
        let now = Utc.with_ymd_and_hms(2013, 5, 24, 0, 0, 0).unwrap();
        let headers = [
            ("host".to_owned(), "examplebucket.s3.amazonaws.com".to_owned()),
            ("range".to_owned(), "bytes=0-9".to_owned()),
        ];
        let s = sign_headers(&creds, &scope, now, "GET", "/test.txt", &[], &headers, EMPTY_SHA256);
        assert_eq!(s.signature, "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41");
        let auth = s.headers.iter().find(|(k, _)| k == "authorization").map(|(_, v)| v.as_str()).unwrap();
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, \
             Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
    }

    /// S3 docs "Authenticating Requests: Using Query Parameters" presigned GET example.
    #[test]
    fn s3_presigned_get_example() {
        let creds = Credentials {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
        };
        let scope = Scope { region: "us-east-1", service: "s3" };
        let now = Utc.with_ymd_and_hms(2013, 5, 24, 0, 0, 0).unwrap();
        let q = presign_query(&creds, &scope, now, 86400, "GET", "/test.txt", "examplebucket.s3.amazonaws.com", &[]);
        assert_eq!(
            q,
            "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request\
             &X-Amz-Date=20130524T000000Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host\
             &X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"
        );
    }

    #[test]
    fn encoding_rules() {
        assert_eq!(uri_encode("a b/c~d_e.f-g", true), "a%20b/c~d_e.f-g");
        assert_eq!(uri_encode("a/b", false), "a%2Fb");
        assert_eq!(uri_encode("ü+&=", false), "%C3%BC%2B%26%3D");
        assert_eq!(
            canonical_query(&[("b".into(), "2".into()), ("a".into(), String::new()), ("b".into(), "1".into())]),
            "a=&b=1&b=2"
        );
        let (ch, signed) = canonical_headers(&[("X-B".into(), "  two   words ".into()), ("a".into(), "1".into())]);
        assert_eq!(ch, "a:1\nx-b:two words\n");
        assert_eq!(signed, "a;x-b");
        assert_eq!(sha256_hex(b""), EMPTY_SHA256);
    }
}
