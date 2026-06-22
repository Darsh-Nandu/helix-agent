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
    let base_url = std::env::var("HELIX_URL").unwrap_or_else(|_| "http://localhost:6969".into());
    let db = HelixClient::new(&base_url);
    let llm = Llm::from_env();

    // The vector index must exist before any search; idempotent.
    db.ensure_vector_index().await?;

    let online = llm.online();
    let model = llm.model().to_string();
    let agent = Agent::new(db, llm);

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("seed") => seed(&agent).await?,
        Some("ask") => {
            let q = args.collect::<Vec<_>>().join(" ");
            if q.trim().is_empty() {
                eprintln!("usage: helix-agent ask \"your question\"");
            } else {
                let reply = agent.turn(&q).await?;
                println!("{reply}");
            }
        }
        Some(other) => {
            eprintln!("unknown command '{other}'. try: seed | ask \"...\" | (no args for chat)");
        }
        None => repl(&agent, online, &model, &base_url).await?,
    }
    Ok(())
}

/// Interactive chat loop.
async fn repl(agent: &Agent, online: bool, model: &str, base_url: &str) -> Result<()> {
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
async fn seed(agent: &Agent) -> Result<()> {
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
