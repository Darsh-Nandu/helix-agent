//! HelixDB client.
//!
//! HelixDB (v3 / enterprise-dev) exposes a single `POST /v1/query` endpoint
//! that accepts a dynamic query AST as JSON. Rather than depend on the full
//! `helix-db` crate (it *is* the database, and pulls a huge native build), we
//! talk to the running instance over plain HTTP and hand-build the same AST.
//!
//! The exact JSON shapes here were generated with the official TypeScript DSL
//! (`@helix-db/helix-db`, see `scripts/gen_queries.mjs`), which the package
//! documents as producing byte-compatible output with the Rust SDK's serde
//! AST. So these are the real shapes the server expects, not guesses.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

/// A memory row returned from vector search. `id`/`ts` are surfaced for callers
/// even though the current REPL only prints text + distance.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MemoryHit {
    pub id: i64,
    pub distance: f32,
    pub role: String,
    pub text: String,
    pub ts: String,
}

/// A memory row from a graph traversal (`valueMap` exposes the id as `$id`).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MemoryRow {
    #[serde(rename = "$id")]
    pub id: i64,
    pub role: String,
    pub text: String,
    pub ts: String,
}

/// One fact in the knowledge graph: `<subject> <predicate> <object>`. The
/// subject is whichever entity the fact was read from.
#[derive(Debug, Deserialize)]
pub struct Fact {
    pub predicate: String,
    pub object: String,
}

pub struct HelixClient {
    http: reqwest::Client,
    endpoint: String,
}

impl HelixClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: format!("{}/v1/query", base_url.trim_end_matches('/')),
        }
    }

    /// POST a request body and return the parsed JSON response, surfacing any
    /// HTTP/remote error with the server's message.
    async fn send(&self, body: Value) -> Result<Value> {
        let resp = self
            .http
            .post(&self.endpoint)
            .json(&body)
            .send()
            .await
            .context("POST /v1/query failed; is the HelixDB dev instance running?")?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("HelixDB returned {status}: {text}"));
        }
        serde_json::from_str(&text).context("could not parse HelixDB response as JSON")
    }

    /// Create the HNSW vector index on `Memory.embedding`. Idempotent
    /// (`if_not_exists`), so it's safe to call on every startup. Must run
    /// before any vector search, and the index fixes the embedding width.
    pub async fn ensure_vector_index(&self) -> Result<()> {
        let body = json!({
            "request_type": "write",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "idx",
                    "steps": [{ "CreateIndex": {
                        "spec": { "NodeVector": { "label": "Memory", "property": "embedding" } },
                        "if_not_exists": true
                    }}],
                    "condition": null
                }}],
                "returns": ["idx"]
            }
        });
        self.send(body).await.map(|_| ())
    }

    /// Insert a memory node and return its assigned id.
    pub async fn add_memory(
        &self,
        role: &str,
        text: &str,
        embedding: &[f32],
        ts: &str,
    ) -> Result<i64> {
        let body = json!({
            "request_type": "write",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "m",
                    "steps": [
                        { "AddN": { "label": "Memory", "properties": [
                            ["role",      { "Expr": { "Param": "role" } }],
                            ["text",      { "Expr": { "Param": "text" } }],
                            ["embedding", { "Expr": { "Param": "embedding" } }],
                            ["ts",        { "Expr": { "Param": "ts" } }]
                        ]}},
                        { "Project": [{ "source": "$id", "alias": "id" }] }
                    ],
                    "condition": null
                }}],
                "returns": ["m"]
            },
            "parameters": { "role": role, "text": text, "embedding": embedding, "ts": ts },
            "parameter_types": {
                "role": "String", "text": "String",
                "embedding": { "Array": "F32" }, "ts": "String"
            }
        });
        let resp = self.send(body).await?;
        resp["m"]["properties"][0]["id"]
            .as_i64()
            .ok_or_else(|| anyhow!("add_memory: missing id in response: {resp}"))
    }

    /// k-NN search over stored embeddings; results are nearest-first.
    pub async fn search_memories(&self, query_vector: &[f32], k: i64) -> Result<Vec<MemoryHit>> {
        let body = json!({
            "request_type": "read",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "hits",
                    "steps": [
                        { "VectorSearchNodes": {
                            "label": "Memory", "property": "embedding",
                            "query_vector": { "Expr": { "Param": "query_vector" } },
                            "k": { "Expr": { "Param": "k" } }
                        }},
                        { "Project": [
                            { "source": "$id",       "alias": "id" },
                            { "source": "$distance", "alias": "distance" },
                            { "source": "role",      "alias": "role" },
                            { "source": "text",      "alias": "text" },
                            { "source": "ts",        "alias": "ts" }
                        ]}
                    ],
                    "condition": null
                }}],
                "returns": ["hits"]
            },
            "parameters": { "query_vector": query_vector, "k": k },
            "parameter_types": { "query_vector": { "Array": "F32" }, "k": "I64" }
        });
        let resp = self.send(body).await?;
        let rows = resp["hits"]["properties"].clone();
        Ok(serde_json::from_value(rows).unwrap_or_default())
    }

    /// Create a directed edge `from -[label]-> to`. The label is a literal in
    /// the AST (not a param), so it's interpolated here. Used for `NEXT`
    /// (conversation chain) and `HAS` (entity → statement).
    pub async fn add_edge(&self, from_id: i64, to_id: i64, label: &str) -> Result<()> {
        let body = json!({
            "request_type": "write",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "e",
                    "steps": [
                        { "N": { "Param": "from_id" } },
                        { "AddE": { "label": label, "to": { "Param": "to_id" }, "properties": [] } }
                    ],
                    "condition": null
                }}],
                "returns": ["e"]
            },
            "parameters": { "from_id": from_id, "to_id": to_id },
            "parameter_types": { "from_id": "I64", "to_id": "I64" }
        });
        self.send(body).await.map(|_| ())
    }

    /// Convenience: chain one memory to the next with a `NEXT` edge.
    pub async fn link_memories(&self, from_id: i64, to_id: i64) -> Result<()> {
        self.add_edge(from_id, to_id, "NEXT").await
    }

    // ---- Knowledge graph -------------------------------------------------

    /// Ensure a unique-equality index on `Entity.name` so entities dedupe.
    pub async fn ensure_entity_index(&self) -> Result<()> {
        let body = json!({
            "request_type": "write",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "ix",
                    "steps": [{ "CreateIndex": {
                        "spec": { "NodeEquality": { "label": "Entity", "property": "name", "unique": true } },
                        "if_not_exists": true
                    }}],
                    "condition": null
                }}],
                "returns": ["ix"]
            }
        });
        self.send(body).await.map(|_| ())
    }

    /// Find an existing `Entity` by name, returning its id if present.
    async fn find_entity(&self, name: &str) -> Result<Option<i64>> {
        let body = json!({
            "request_type": "read",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "e",
                    "steps": [
                        { "NWhere": { "Eq": ["$label", { "String": "Entity" }] } },
                        { "Where": { "EqExpr": ["name", { "Param": "name" }] } },
                        { "Project": [{ "source": "$id", "alias": "id" }] }
                    ],
                    "condition": null
                }}],
                "returns": ["e"]
            },
            "parameters": { "name": name },
            "parameter_types": { "name": "String" }
        });
        let resp = self.send(body).await?;
        Ok(resp["e"]["properties"][0]["id"].as_i64())
    }

    async fn add_entity(&self, name: &str) -> Result<i64> {
        let body = json!({
            "request_type": "write",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "e",
                    "steps": [
                        { "AddN": { "label": "Entity", "properties": [["name", { "Expr": { "Param": "name" } }]] } },
                        { "Project": [{ "source": "$id", "alias": "id" }] }
                    ],
                    "condition": null
                }}],
                "returns": ["e"]
            },
            "parameters": { "name": name },
            "parameter_types": { "name": "String" }
        });
        let resp = self.send(body).await?;
        resp["e"]["properties"][0]["id"]
            .as_i64()
            .ok_or_else(|| anyhow!("add_entity: missing id: {resp}"))
    }

    /// Find-or-create an entity, returning its id (dedupes on name).
    pub async fn upsert_entity(&self, name: &str) -> Result<i64> {
        match self.find_entity(name).await? {
            Some(id) => Ok(id),
            None => self.add_entity(name).await,
        }
    }

    /// Add a reified `Statement` node `{ predicate, object }`, returning its id.
    pub async fn add_statement(&self, predicate: &str, object: &str) -> Result<i64> {
        let body = json!({
            "request_type": "write",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "s",
                    "steps": [
                        { "AddN": { "label": "Statement", "properties": [
                            ["predicate", { "Expr": { "Param": "predicate" } }],
                            ["object", { "Expr": { "Param": "object" } }]
                        ]}},
                        { "Project": [{ "source": "$id", "alias": "id" }] }
                    ],
                    "condition": null
                }}],
                "returns": ["s"]
            },
            "parameters": { "predicate": predicate, "object": object },
            "parameter_types": { "predicate": "String", "object": "String" }
        });
        let resp = self.send(body).await?;
        resp["s"]["properties"][0]["id"]
            .as_i64()
            .ok_or_else(|| anyhow!("add_statement: missing id: {resp}"))
    }

    /// Record a `(subject, predicate, object)` triple as graph structure:
    /// subject Entity --HAS--> Statement, and Statement --ABOUT--> object Entity
    /// so the two entities are genuinely connected in the graph.
    pub async fn add_fact(&self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        let subj_id = self.upsert_entity(subject).await?;
        let obj_id = self.upsert_entity(object).await?;
        let stmt_id = self.add_statement(predicate, object).await?;
        self.add_edge(subj_id, stmt_id, "HAS").await?;
        self.add_edge(stmt_id, obj_id, "ABOUT").await?;
        Ok(())
    }

    /// Read all facts about an entity: `Entity(name) --HAS--> Statement`.
    pub async fn facts_of(&self, name: &str) -> Result<Vec<Fact>> {
        let body = json!({
            "request_type": "read",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "f",
                    "steps": [
                        { "NWhere": { "Eq": ["$label", { "String": "Entity" }] } },
                        { "Where": { "EqExpr": ["name", { "Param": "name" }] } },
                        { "Out": "HAS" },
                        { "ValueMap": ["predicate", "object"] }
                    ],
                    "condition": null
                }}],
                "returns": ["f"]
            },
            "parameters": { "name": name },
            "parameter_types": { "name": "String" }
        });
        let resp = self.send(body).await?;
        let rows = resp["f"]["properties"].clone();
        Ok(serde_json::from_value(rows).unwrap_or_default())
    }

    /// Follow the `NEXT` edge(s) out of a memory: `g().n(id).out("NEXT")`.
    pub async fn next_of(&self, id: i64) -> Result<Vec<MemoryRow>> {
        let body = json!({
            "request_type": "read",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "nxt",
                    "steps": [
                        { "N": { "Param": "id" } },
                        { "Out": "NEXT" },
                        { "ValueMap": ["$id", "role", "text", "ts"] }
                    ],
                    "condition": null
                }}],
                "returns": ["nxt"]
            },
            "parameters": { "id": id },
            "parameter_types": { "id": "I64" }
        });
        let resp = self.send(body).await?;
        let rows = resp["nxt"]["properties"].clone();
        Ok(serde_json::from_value(rows).unwrap_or_default())
    }

    /// Count of stored memories.
    pub async fn count_memories(&self) -> Result<i64> {
        let body = json!({
            "request_type": "read",
            "query_name": null,
            "query": {
                "queries": [{ "Query": {
                    "name": "c",
                    "steps": [
                        { "NWhere": { "Eq": ["$label", { "String": "Memory" }] } },
                        "Count"
                    ],
                    "condition": null
                }}],
                "returns": ["c"]
            }
        });
        let resp = self.send(body).await?;
        Ok(resp["c"]["count"].as_i64().unwrap_or(0))
    }
}
