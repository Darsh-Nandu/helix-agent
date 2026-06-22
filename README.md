# helix-agent

A small AI agent with real long-term memory, written in Rust and backed by
[HelixDB](https://helix-db.com), a native graph-vector database for AI.

The point of this repo is to give HelixDB a genuine workout. Building an agent
that remembers you usually means stitching together a vector database for
recall, a separate graph database for relationships, and glue code to keep them
in sync. Here that is one store. The same HelixDB instance holds **three views
of memory at once**:

1. **Vectors** on `Memory` nodes for semantic recall.
2. **`NEXT` edges** between memories for ordered conversation threads.
3. **A knowledge graph** of `Entity` and `Statement` nodes for facts and
   relationships.

## What it looks like

```text
$ cargo run -- ask "what's my pet's name?"
Your cat's name is Tuna. She's quite the character, especially with her
fondness for sleeping on your keyboard.
```

No chat history is passed in. The agent recalls that from an earlier session by
vector search in HelixDB.

```text
$ cargo run -- facts user
knowledge-graph facts about 'user':
  user --LIKES--> rust
  user --LIVES_IN--> pune
  user --LOVES--> filter coffee
  user --OWNS--> a cat named tuna

$ cargo run -- thread 0
conversation chain following NEXT edges from memory #0:
  1. (assistant) Rust is a great language, known for its focus on safety...
  2. (user) I'm building an AI agent to test HelixDB, a graph-vector database.
  3. (assistant) That's a fascinating project...
```

## How a turn works

Every message flows through the same loop: **recall, respond, remember.**

1. **Embed** the message into a vector.
2. **Vector-search** HelixDB for the most relevant past memories.
3. **Respond** (via Groq) using those memories as grounding context.
4. **Remember**: store the message and reply as new `Memory` vectors, chain
   them with `NEXT` edges, and distil the message into knowledge-graph triples.

## How it maps to HelixDB

HelixDB v3 exposes one endpoint, `POST /v1/query`, which takes a query AST as
JSON. The operations this agent uses:

| Operation              | HelixDB step                                                    |
| ---------------------- | --------------------------------------------------------------- |
| Create vector index    | `CreateIndex { NodeVector }` (idempotent, on startup)           |
| Store a memory         | `AddN("Memory", { role, text, embedding, ts })`                 |
| Semantic recall        | `VectorSearchNodes("Memory", "embedding", q, k)` with `$distance`  |
| Chain turns (graph)    | `n(from).addE("NEXT", to)` linking each turn to the next        |
| Walk a thread (graph)  | `n(id).out("NEXT")` to follow the conversation chain            |
| Dedupe entities        | unique-equality index on `Entity.name`                          |
| Store a fact (KG)      | `Entity --HAS--> Statement{predicate, object} --ABOUT--> Entity`   |
| Read facts (KG)        | `nWithLabel("Entity").where(name == ?).out("HAS").valueMap(...)`   |
| Recent / count         | `nWithLabel("Memory").orderBy(ts).limit(k)` and `.count()`      |

Embeddings come from a dependency-free hashing embedder in
[`src/embed.rs`](src/embed.rs). It is good enough to demonstrate semantic
recall without an embedding API. Swap it for a real one and the HelixDB
plumbing stays the same.

## How the query shapes were built

HelixDB v3 wants a JSON query AST, not a query string. Rather than guess those
shapes, this project uses HelixDB's official TypeScript DSL as a compiler. The
`@helix-db/helix-db` package builds the same AST as the Rust SDK and exposes a
`.toDynamicJson(params, args)` method that prints the full request body. So
[`scripts/gen_queries.mjs`](scripts/gen_queries.mjs) writes each query in the
readable DSL:

```js
writeBatch().varAs("m",
  g().addN("Memory", { role: p.role, text: p.text, embedding: p.embedding })
).returning(["m"]).toDynamicJson(...)
```

runs it once under Node, and prints the ground-truth JSON. Each shape was
validated against the live instance with `helix query dev --file`, then
transcribed into the Rust client as `serde_json::json!` templates. The Rust side
stays dependency-light (just `reqwest`), but the query shapes are exactly what
the database expects: generated, not guessed.

> One quirk worth knowing: HelixDB does not let you `project` or `valueMap` an
> edge (those need node state). So relationships are reified as `Statement`
> nodes, which keeps the knowledge graph fully queryable.

## Quickstart

Prerequisites:

- The [HelixDB CLI](https://docs.helix-db.com) plus Docker
  (`curl -sSL https://install.helix-db.com | bash`)
- Rust (this repo pins the GNU toolchain, see the Windows note below)
- A [Groq](https://groq.com) API key (optional; runs in an offline echo mode
  without one)

```bash
# 1. Start the local HelixDB instance (in-memory; restart wipes data)
helix start dev

# 2. Put your Groq key in a .env file (gitignored)
echo "GROQ_API_KEY=gsk_..." > .env

# 3. Build and run
cargo run                 # interactive chat
cargo run -- seed         # insert a few example memories
cargo run -- ask "what language do I like?"
cargo run -- thread 0     # walk the NEXT-edge conversation chain
cargo run -- facts user   # print knowledge-graph facts about an entity
```

Configuration via environment (also read from `.env`):

| Variable       | Default                     | Purpose                        |
| -------------- | --------------------------- | ------------------------------ |
| `HELIX_URL`    | `http://localhost:6969`     | HelixDB base URL               |
| `GROQ_API_KEY` | (unset)                     | Enables real replies + KG extraction |
| `GROQ_MODEL`   | `llama-3.3-70b-versatile`   | Groq model                     |

## Project layout

```
src/
  main.rs    CLI: chat REPL plus seed / ask / thread / facts commands
  agent.rs   the recall -> respond -> remember loop
  db.rs      HelixDB client over POST /v1/query (vectors, edges, KG)
  embed.rs   dependency-free hashing embedder
  llm.rs     Groq chat + knowledge-graph triple extraction
scripts/
  gen_queries.mjs   generates ground-truth query JSON from the TS DSL
```

## Windows note (no MSVC)

This was built on a machine with no MSVC linker, so the project pins the
mingw-w64 GNU toolchain (`rust-toolchain.toml`) and links with Rust's bundled
self-contained CRT (`.cargo/config.toml`). `reqwest` uses native-TLS (SChannel)
to avoid a native `aws-lc-rs` build. Build from PowerShell with `C:\gcc\bin` on
`PATH`. On a normal MSVC setup, none of this applies and `cargo run` just works.
