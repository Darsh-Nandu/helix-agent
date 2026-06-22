# helix-agent

A tiny AI agent with **long-term memory**, written in Rust and backed by
[HelixDB](https://helix-db.com) — a native graph-vector database built for AI.

Each turn the agent:

1. **embeds** your message,
2. **vector-searches** HelixDB for the most relevant past memories,
3. **responds** (via Groq) using those memories as context,
4. **stores** both your message and its reply back into HelixDB as new vectors.

So the agent remembers across runs, and recall is semantic — powered entirely by
HelixDB's built-in vector index. The whole point of this repo is to give HelixDB
a real workout: storing nodes, creating an HNSW vector index, and doing k-NN
search, all over its dynamic `POST /v1/query` API.

## How it talks to HelixDB

HelixDB v3 (enterprise-dev) exposes one endpoint — `POST /v1/query` — that takes
a query AST as JSON. Rather than pull in the full `helix-db` crate (it *is* the
database), this client hand-builds that AST and posts it with `reqwest`. The
exact JSON shapes were generated with HelixDB's official TypeScript DSL
(`@helix-db/helix-db`), which produces byte-compatible output with the Rust SDK —
see [`scripts/gen_queries.mjs`](scripts/gen_queries.mjs). The operations used:

| Operation | HelixDB step |
|-----------|--------------|
| Create vector index | `CreateIndex { NodeVector }` (idempotent, on startup) |
| Store a memory | `AddN("Memory", { role, text, embedding, ts })` |
| Recall | `VectorSearchNodes("Memory", "embedding", q, k)` → `$distance` |
| Chain turns (graph) | `n(from).addE("NEXT", to)` linking each turn to the next |
| Walk a thread (graph) | `n(id).out("NEXT")` to follow the conversation chain |
| Recent / count | `nWithLabel("Memory").orderBy(ts).limit(k)` / `.count()` |

So the same `Memory` nodes carry **both** a vector embedding (for semantic
recall) and `NEXT` edges (for ordered conversation threads) — graph and vector
in one store, which is HelixDB's whole pitch.

Embeddings are a dependency-free hashing embedder (`src/embed.rs`) — good enough
to demonstrate semantic recall without an embedding API. Swap it for a real one
to improve quality; the HelixDB plumbing is unchanged.

## Prerequisites

- [HelixDB CLI](https://docs.helix-db.com) + Docker (`curl -sSL https://install.helix-db.com | bash`)
- Rust (this repo pins the **GNU** toolchain — see "Windows note" below)
- A [Groq](https://groq.com) API key (optional — runs in an offline echo mode without one)

## Run

```bash
# 1. Start the local HelixDB instance (in-memory; restart wipes data)
helix start dev

# 2. Point the agent at your Groq key (optional)
export GROQ_API_KEY=gsk_...        # PowerShell: $env:GROQ_API_KEY="gsk_..."

# 3. Build & run
cargo run                 # interactive chat
cargo run -- seed         # insert a few example memories
cargo run -- ask "what language do I like?"
cargo run -- thread 0     # walk the NEXT-edge conversation chain from memory #0
```

Config via env: `HELIX_URL` (default `http://localhost:6969`),
`GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`).

## Windows note (no MSVC)

This machine has no MSVC linker, so the project pins the mingw-w64 **GNU**
toolchain (`rust-toolchain.toml`) and links with Rust's bundled self-contained
CRT (`.cargo/config.toml`). `reqwest` uses native-TLS (SChannel) to avoid a
native `aws-lc-rs` build. Build from PowerShell with `C:\gcc\bin` on `PATH`.
