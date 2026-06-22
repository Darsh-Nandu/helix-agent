//! A tiny, dependency-free text embedder.
//!
//! The point of this project is to exercise HelixDB's vector search, not to
//! ship a state-of-the-art encoder. So instead of calling out to an embedding
//! API, we use the classic "hashing trick": every token is hashed into a fixed
//! number of buckets with a signed contribution, then the vector is L2
//! normalized. Two texts that share words land close together in cosine space,
//! which is exactly what we need to demonstrate semantic-ish recall.

/// Embedding dimensionality. Every vector stored in HelixDB uses this width;
/// the vector index is created for it on first insert, so keep it stable.
pub const DIM: usize = 256;

/// FNV-1a 64-bit hash — small, fast, good enough for bucketing tokens.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Lowercase, then split on anything that isn't a letter or digit.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Embed `text` into a unit-length `DIM`-dimensional vector.
pub fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; DIM];
    for tok in tokenize(text) {
        let h = fnv1a(tok.as_bytes());
        let idx = (h % DIM as u64) as usize;
        // Use a different bit of the hash to pick the sign, so collisions in
        // the bucket index don't all push in the same direction.
        let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
