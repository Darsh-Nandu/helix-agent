//! helix-agent — a tiny AI agent with long-term memory backed by HelixDB.
//!
//! Usage:
//!   helix-agent            # interactive chat REPL (default)
//!   helix-agent seed       # insert a few example memories, then exit
//!   helix-agent ask "..."  # one-shot question
//!
//! Config (env):
//!   HELIX_URL     HelixDB base URL          (default http://localhost:6969)
//!   GROQ_API_KEY  Groq key for real replies (optional; offline echo without it)
//!   GROQ_MODEL    Groq model                (default llama-3.3-70b-versatile)

mod agent;
mod db;
mod embed;
mod llm;

use std::io::{self, Write};

use agent::Agent;
use anyhow::Result;
use db::HelixClient;
use llm::Llm;

#[tokio::main]
async fn main() -> Result<()> {
    // Load GROQ_API_KEY etc. from a local .env if present (ignored if absent).
    let _ = dotenvy::dotenv();

    let base_url = std::env::var("HELIX_URL").unwrap_or_else(|_| "http://localhost:6969".into());
    let db = HelixClient::new(&base_url);
    let llm = Llm::from_env();

    // Indexes must exist before search / entity dedup; both idempotent.
    db.ensure_vector_index().await?;
    db.ensure_entity_index().await?;

    let online = llm.online();
    let model = llm.model().to_string();
    let mut agent = Agent::new(db, llm);

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("seed") => seed(&mut agent).await?,
        Some("ask") => {
            let q = args.collect::<Vec<_>>().join(" ");
            if q.trim().is_empty() {
                eprintln!("usage: helix-agent ask \"your question\"");
            } else {
                let reply = agent.turn(&q).await?;
                println!("{reply}");
            }
        }
        Some("thread") => {
            let start: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let chain = agent.thread(start).await?;
            println!("conversation chain following NEXT edges from memory #{start}:");
            if chain.is_empty() {
                println!("  (no outgoing edges — seed or chat first)");
            }
            for (i, (role, text)) in chain.iter().enumerate() {
                let one_line = text.replace('\n', " ");
                println!("  {}. ({role}) {one_line}", i + 1);
            }
        }
        Some("facts") => {
            let name = args.collect::<Vec<_>>().join(" ");
            let name = if name.trim().is_empty() { "user".to_string() } else { name };
            let facts = agent.facts(&name).await?;
            println!("knowledge-graph facts about '{name}':");
            if facts.is_empty() {
                println!("  (none yet — chat or seed first, with GROQ_API_KEY set)");
            }
            for f in &facts {
                println!("  {name} --{}--> {}", f.predicate, f.object);
            }
        }
        Some(other) => {
            eprintln!(
                "unknown command '{other}'. try: seed | ask \"...\" | thread [id] | facts [name] | (no args for chat)"
            );
        }
        None => repl(&mut agent, online, &model, &base_url).await?,
    }
    Ok(())
}

/// Interactive chat loop.
async fn repl(agent: &mut Agent, online: bool, model: &str, base_url: &str) -> Result<()> {
    let count = agent.db().count_memories().await.unwrap_or(0);
    println!("helix-agent — memory backed by HelixDB ({base_url})");
    println!(
        "LLM: {}",
        if online {
            format!("Groq ({model})")
        } else {
            "offline echo mode (set GROQ_API_KEY for real replies)".into()
        }
    );
    println!("{count} memories stored. Type a message, or 'quit' to exit.\n");

    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break; // EOF
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        if matches!(msg, "quit" | "exit" | ":q") {
            break;
        }
        match agent.turn(msg).await {
            Ok(reply) => println!("bot> {reply}\n"),
            Err(e) => eprintln!("error: {e:#}\n"),
        }
    }
    println!("bye!");
    Ok(())
}

/// Insert a handful of example memories so vector recall has something to find.
async fn seed(agent: &mut Agent) -> Result<()> {
    let samples = [
        "My favorite programming language is Rust.",
        "I'm building an AI agent to test HelixDB, a graph-vector database.",
        "I live in Pune and love filter coffee.",
        "HelixDB stores both graph edges and vector embeddings natively.",
        "My cat's name is Tuna and she sleeps on my keyboard.",
    ];
    for s in samples {
        let reply = agent.turn(s).await?;
        println!("seeded: {s}\n   -> {reply}\n");
    }
    let count = agent.db().count_memories().await?;
    println!("done. {count} memories now stored in HelixDB.");
    Ok(())
}
