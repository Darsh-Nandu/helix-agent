// One-shot generator: builds each query the Rust agent needs and prints the
// exact POST /v1/query request JSON (which the package documents as matching
// the Rust serde AST). We bake these shapes into the Rust client and only
// swap in the `parameters` values at runtime.
import {
  defineParams,
  param,
  g,
  readBatch,
  writeBatch,
  PropertyProjection,
  Order,
} from "@helix-db/helix-db";

function show(name, json) {
  console.log("\n===== " + name + " =====");
  console.log(json);
}

// 1) add_memory — store a chat turn as a Memory node (embedding is a property)
const addParams = defineParams({
  role: param.string(),
  text: param.string(),
  embedding: param.array(param.f32()),
  ts: param.string(),
});
const addMemory = (p = addParams) =>
  writeBatch()
    .varAs(
      "m",
      g()
        .addN("Memory", {
          role: p.role,
          text: p.text,
          embedding: p.embedding,
          ts: p.ts,
        })
        .project([PropertyProjection.renamed("$id", "id")]),
    )
    .returning(["m"]);
show(
  "add_memory",
  addMemory().toDynamicJson(addParams, {
    role: "user",
    text: "hello",
    embedding: [0.1, 0.2, 0.3],
    ts: "2026-06-22T00:00:00Z",
  }),
);

// 2) search_memories — k-NN over embeddings
const searchParams = defineParams({
  query_vector: param.array(param.f32()),
  k: param.i64(),
});
const searchMemories = (p = searchParams) =>
  readBatch()
    .varAs(
      "hits",
      g()
        .vectorSearchNodesWith("Memory", "embedding", p.query_vector, p.k)
        .project([
          PropertyProjection.renamed("$id", "id"),
          PropertyProjection.renamed("$distance", "distance"),
          PropertyProjection.new("role"),
          PropertyProjection.new("text"),
          PropertyProjection.new("ts"),
        ]),
    )
    .returning(["hits"]);
show(
  "search_memories",
  searchMemories().toDynamicJson(searchParams, {
    query_vector: [0.1, 0.2, 0.3],
    k: 5n,
  }),
);

// 3) recent_memories — newest-first by timestamp
const recentParams = defineParams({ k: param.i64() });
const recentMemories = (p = recentParams) =>
  readBatch()
    .varAs(
      "rows",
      g()
        .nWithLabel("Memory")
        .orderBy("ts", Order.Desc)
        .limit(p.k)
        .valueMap(["$id", "role", "text", "ts"]),
    )
    .returning(["rows"]);
show(
  "recent_memories",
  recentMemories().toDynamicJson(recentParams, { k: 10n }),
);

// 4) count_memories
const countMemories = () =>
  readBatch()
    .varAs("c", g().nWithLabel("Memory").count())
    .returning(["c"]);
show("count_memories", countMemories().toDynamicJson());
