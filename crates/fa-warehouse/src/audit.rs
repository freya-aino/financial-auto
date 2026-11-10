//! Append-only audit log (`audit.log`, spec 42 §6): every manual action, halt/resume, limit
//! change and reconciliation adjustment. Rows are never updated or deleted.
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{PgPool, Row};

use crate::WarehouseError;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditRow {
    pub id: i64,
    pub at: DateTime<Utc>,
    pub actor: String,
    pub action: String,
    pub subject: Option<String>,
    pub payload: Option<Value>,
}

/// Appends one entry. `actor` is `system:<service>` or `ui`. Returns the row id.
pub async fn append(
    pool: &PgPool,
    actor: &str,
    action: &str,
    subject: Option<&str>,
    payload: Value,
) -> Result<i64, WarehouseError> {
    let (id,): (i64,) =
        sqlx::query_as("INSERT INTO audit.log (actor, action, subject, payload) VALUES ($1, $2, $3, $4) RETURNING id")
            .bind(actor)
            .bind(action)
            .bind(subject)
            .bind(payload)
            .fetch_one(pool)
            .await?;
    Ok(id)
}

/// Entries at or after `since` (all when `None`), oldest first, at most `limit`.
pub async fn since(pool: &PgPool, since: Option<DateTime<Utc>>, limit: i64) -> Result<Vec<AuditRow>, WarehouseError> {
    let rows = sqlx::query(
        "SELECT id, at, actor, action, subject, payload FROM audit.log \
         WHERE ($1::timestamptz IS NULL OR at >= $1) ORDER BY at, id LIMIT $2",
    )
    .bind(since)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .iter()
        .map(|r| AuditRow {
            id: r.get("id"),
            at: r.get("at"),
            actor: r.get("actor"),
            action: r.get("action"),
            subject: r.get("subject"),
            payload: r.get("payload"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn append_then_read_since() {
        let Ok(url) = std::env::var("DATABASE_URL") else { return };
        let pool = sqlx::postgres::PgPoolOptions::new().max_connections(2).connect(&url).await.unwrap();
        let subject = format!("test-{}", uuid::Uuid::now_v7());
        let t0 = Utc::now();
        let id = append(&pool, "system:test", "halt", Some(&subject), serde_json::json!({"k": 1})).await.unwrap();
        let rows = since(&pool, Some(t0 - chrono::Duration::seconds(5)), 1000).await.unwrap();
        let mine = rows.iter().find(|r| r.id == id).expect("row visible");
        assert_eq!((mine.actor.as_str(), mine.action.as_str()), ("system:test", "halt"));
        assert_eq!(mine.subject.as_deref(), Some(subject.as_str()));
        assert_eq!(mine.payload, Some(serde_json::json!({"k": 1})));
        assert!(since(&pool, Some(Utc::now() + chrono::Duration::hours(1)), 10).await.unwrap().is_empty());
        sqlx::query("DELETE FROM audit.log WHERE id = $1").bind(id).execute(&pool).await.unwrap();
    }
}
