//! Groq chat client (OpenAI-compatible Chat Completions API, raw HTTP).
//!
//!   POST https://api.groq.com/openai/v1/chat/completions
//!   headers: Authorization: Bearer <GROQ_API_KEY>, content-type: application/json
//!   body:    { model, max_tokens, messages: [{role, content}] }
//!   resp:    { choices: [{ message: { content } }] }
//!
//! If `GROQ_API_KEY` is unset the agent still runs in an offline "echo" mode so
//! the HelixDB memory loop can be demonstrated without a key.

use anyhow::{Context, Result};
use serde_json::json;

use crate::db::MemoryHit;

const API_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
/// A solid, fast default on Groq; override with GROQ_MODEL.
const DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";

pub struct Llm {
    http: reqwest::Client,
    api_key: Option<String>,
    model: String,
}

impl Llm {
    pub fn from_env() -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: std::env::var("GROQ_API_KEY").ok().filter(|k| !k.is_empty()),
            model: std::env::var("GROQ_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
        }
    }

    pub fn online(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Generate a reply to `user_msg`, given memories recalled from HelixDB.
    pub async fn respond(&self, user_msg: &str, recalled: &[MemoryHit]) -> Result<String> {
        let Some(key) = &self.api_key else {
            return Ok(offline_reply(user_msg, recalled));
        };

        // Recalled memories become grounding context in the system prompt.
        let mut system = String::from(
            "You are a helpful assistant with long-term memory backed by HelixDB. \
             Below are notes recalled from past conversations, most relevant first. \
             Use them when relevant; do not mention them unless useful.\n\n",
        );
        if recalled.is_empty() {
            system.push_str("(no relevant past memories)\n");
        } else {
            for m in recalled {
                system.push_str(&format!("- ({}) {}\n", m.role, m.text));
            }
        }

        let body = json!({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user_msg },
            ],
        });

        let resp = self
            .http
            .post(API_URL)
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Groq request failed")?;

        let status = resp.status();
        let v: serde_json::Value = resp.json().await.context("invalid Groq JSON")?;
        if !status.is_success() {
            let msg = v["error"]["message"].as_str().unwrap_or("unknown error");
            anyhow::bail!("Groq API {status}: {msg}");
        }
        let text = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();
        Ok(if text.is_empty() {
            "[empty response]".to_string()
        } else {
            text
        })
    }
}

/// Deterministic stand-in when no API key is set: it still exercises HelixDB by
/// reporting what was recalled, so the memory loop is visible offline.
fn offline_reply(user_msg: &str, recalled: &[MemoryHit]) -> String {
    let mut out = format!(
        "[offline mode — set GROQ_API_KEY for real replies]\nYou said: {user_msg}\n"
    );
    if recalled.is_empty() {
        out.push_str("I have no related memories yet.");
    } else {
        out.push_str(&format!("HelixDB recalled {} related memo", recalled.len()));
        out.push_str(if recalled.len() == 1 { "ry:\n" } else { "ries:\n" });
        for m in recalled {
            out.push_str(&format!("  • ({}, dist {:.3}) {}\n", m.role, m.distance, m.text));
        }
    }
    out
}
