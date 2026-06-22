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
| Dedupe entities | unique-equality index on `Entity.name` |
| Store a fact (KG) | `Entity --HAS--> Statement{predicate, object} --ABOUT--> Entity` |
| Read facts (KG) | `nWithLabel("Entity").where(name==?).out("HAS").valueMap(...)` |
| Recent / count | `nWithLabel("Memory").orderBy(ts).limit(k)` / `.count()` |

So one HelixDB instance holds **three views of memory at once**:

- **vectors** on `Memory` nodes → semantic recall,
- **`NEXT` edges** between memories → ordered conversation threads,
- a **knowledge graph** → the LLM distils each message into
  `(subject, predicate, object)` triples, stored as `Entity` and `Statement`
  nodes connected by edges (`user --OWNS--> a cat named tuna`).

That's the whole point: the kind of memory that used to mean juggling a vector
DB *and* a separate graph DB *and* sync glue is one store here.

## How I built this

The interesting part was getting the query AST right without guessing. HelixDB
v3 doesn't take a query string — `POST /v1/query` wants a JSON AST (the
`{ "AddN": {...} }`, `{ "VectorSearchNodes": {...} }` shapes you see in
`src/db.rs`). Hand-writing those by trial-and-error would be fragile.

Instead I used HelixDB's **official TypeScript DSL as a compiler**. The
`@helix-db/helix-db` npm package builds the exact same AST as the Rust SDK and
exposes a `.toDynamicJson(params, args)` method that prints the full request
body. So [`scripts/gen_queries.mjs`](scripts/gen_queries.mjs) writes each query
in the readable DSL —

```js
writeBatch().varAs("m",
  g().addN("Memory", { role: p.role, text: p.text, embedding: p.embedding })
).returning(["m"]).toDynamicJson(...)
```

— runs it once under Node, and prints the ground-truth JSON. I validated each
shape against the live instance with `helix query dev --file`, then transcribed
the confirmed AST into the Rust client as `serde_json::json!` templates. The
Rust side stays dependency-light (no `helix-db` crate, just `reqwest`), but the
query shapes are exactly what the database expects — generated, not guessed.

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
cargo run -- facts user   # print knowledge-graph facts about an entity (needs GROQ key)
```

Config via env: `HELIX_URL` (default `http://localhost:6969`),
`GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`).

## Windows note (no MSVC)

This machine has no MSVC linker, so the project pins the mingw-w64 **GNU**
toolchain (`rust-toolchain.toml`) and links with Rust's bundled self-contained
CRT (`.cargo/config.toml`). `reqwest` uses native-TLS (SChannel) to avoid a
native `aws-lc-rs` build. Build from PowerShell with `C:\gcc\bin` on `PATH`.
