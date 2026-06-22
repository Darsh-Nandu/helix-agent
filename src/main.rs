// Linker smoke test. Replaced by the real agent next.
#[tokio::main]
async fn main() {
    let _client = reqwest::Client::new();
    println!("build + link OK");
}
