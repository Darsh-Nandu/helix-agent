//! The agent loop: recall → respond → remember.
//!
//! For each user turn we embed the message, vector-search HelixDB for the most
//! relevant past memories, ask the LLM to respond using them as context, then
//! store both the user message and the assistant reply back as new memories
//! (each with its own embedding) so the agent's memory grows over time.

use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::HelixClient;
use crate::embed;
use crate::llm::Llm;

/// How many memories to recall per turn.
const RECALL_K: i64 = 5;

pub struct Agent {
    db: HelixClient,
    llm: Llm,
    /// Id of the most recently stored memory, so each new turn can be chained
    /// onto it with a `NEXT` edge.
    last_id: Option<i64>,
}

impl Agent {
    pub fn new(db: HelixClient, llm: Llm) -> Self {
        Self {
            db,
            llm,
            last_id: None,
        }
    }

    /// Handle one user turn and return the assistant's reply.
    pub async fn turn(&mut self, user_msg: &str) -> Result<String> {
        let q = embed::embed(user_msg);
        let recalled = self.db.search_memories(&q, RECALL_K).await?;

        let reply = self.llm.respond(user_msg, &recalled).await?;

        // Persist both sides of the exchange as memory nodes...
        let user_id = self.db.add_memory("user", user_msg, &q, &now_ts()).await?;
        let reply_vec = embed::embed(&reply);
        let bot_id = self
            .db
            .add_memory("assistant", &reply, &reply_vec, &now_ts())
            .await?;

        // ...then wire up the conversation chain as graph edges:
        // prev -> user -> assistant.
        if let Some(prev) = self.last_id {
            self.db.link_memories(prev, user_id).await?;
        }
        self.db.link_memories(user_id, bot_id).await?;
        self.last_id = Some(bot_id);

        Ok(reply)
    }

    /// Walk the `NEXT` chain from `start_id`, returning (role, text) in order.
    /// Demonstrates HelixDB graph traversal over the same nodes used for vectors.
    pub async fn thread(&self, start_id: i64) -> Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        let mut id = start_id;
        // Guard against cycles / runaways.
        for _ in 0..1000 {
            let nexts = self.db.next_of(id).await?;
            let Some(row) = nexts.into_iter().next() else {
                break;
            };
            out.push((row.role, row.text));
            id = row.id;
        }
        Ok(out)
    }

    pub fn db(&self) -> &HelixClient {
        &self.db
    }
}

/// A lexicographically-sortable timestamp (epoch millis, zero-padded) so
/// HelixDB's `orderBy(ts, Desc)` returns memories newest-first.
fn now_ts() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("{millis:015}")
}
