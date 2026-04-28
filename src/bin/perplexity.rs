//! Perplexity benchmark for the packed Phi-3 mixed-quant model.
//!
//! Computes `PPL = exp(mean -log P(token_{i+1} | token_{≤i}))` over a plain-
//! text corpus, using fixed-size context chunks with a configurable stride.
//!
//! Usage:
//! ```text
//! perplexity <packed-shard1.safetensors> [shard2.safetensors] <corpus.txt>
//! ```
//!
//! Tunable via env vars:
//!   * `PHI3_PPL_CTX`         — context length per chunk (default 512)
//!   * `PHI3_PPL_STRIDE`      — token stride between chunks (default = ctx, i.e. non-overlapping)
//!   * `PHI3_PPL_MAX_CHUNKS`  — cap the number of chunks (handy for smoke tests)
//!
//! Why this matters: it gives us a *quantitative* knob to compare quantization
//! recipes (Q8K vs Q4K, perm on/off, future Q8_0 KV cache, …) without
//! eyeballing chat output. For the suggested baseline run we recommend
//! `wikitext-2-raw` test split with `PHI3_PPL_CTX=2048`.

use candle::{DType, Device, Tensor, D};
use std::path::PathBuf;
use std::time::Instant;

use phi3_mixed_quant::cache::Cache;
use phi3_mixed_quant::loader::{
    build_model, build_model_multi_shard, load_config, load_tokenizer, DEFAULT_HF_REPO,
};
use phi3_mixed_quant::model::Phi3;
use phi3_mixed_quant::types::Phi3Config;

const USAGE: &str = "Usage: perplexity <packed.safetensors> [shard2.safetensors] <corpus.txt>";

struct Args {
    shard1: PathBuf,
    shard2: Option<PathBuf>,
    corpus: PathBuf,
}

fn parse_args() -> candle::Result<Args> {
    let positional: Vec<String> = std::env::args().skip(1).collect();
    match positional.len() {
        2 => Ok(Args {
            shard1: positional[0].clone().into(),
            shard2: None,
            corpus: positional[1].clone().into(),
        }),
        3 => Ok(Args {
            shard1: positional[0].clone().into(),
            shard2: Some(positional[1].clone().into()),
            corpus: positional[2].clone().into(),
        }),
        _ => Err(candle::Error::Msg(USAGE.to_string())),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_opt_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

fn main() -> candle::Result<()> {
    let args = parse_args()?;
    let device = Device::Cpu;
    let config = load_config(&args.shard1, DEFAULT_HF_REPO)?;
    let model = load_model(&args, &device, &config)?;
    let tokenizer = load_tokenizer()?;

    let corpus = std::fs::read_to_string(&args.corpus)
        .map_err(|e| candle::Error::Msg(format!("read {}: {e}", args.corpus.display())))?;
    println!(
        "Corpus: {} ({} bytes)",
        args.corpus.display(),
        corpus.len()
    );

    let encoded = tokenizer
        .encode(corpus, false)
        .map_err(|e| candle::Error::Msg(format!("tokenize: {e}")))?;
    let tokens: Vec<u32> = encoded.get_ids().to_vec();
    println!("Tokens: {}", tokens.len());

    let ctx = env_usize("PHI3_PPL_CTX", 512).min(config.max_position_embeddings);
    let stride = env_usize("PHI3_PPL_STRIDE", ctx);
    let max_chunks = env_opt_usize("PHI3_PPL_MAX_CHUNKS");
    if ctx < 2 {
        candle::bail!("PHI3_PPL_CTX must be >= 2");
    }
    if stride == 0 {
        candle::bail!("PHI3_PPL_STRIDE must be > 0");
    }
    if tokens.len() < 2 {
        candle::bail!("Corpus is too short ({} tokens) for perplexity", tokens.len());
    }

    println!(
        "Eval : ctx={} stride={} max_chunks={:?}",
        ctx, stride, max_chunks
    );

    let head_dim = config.hidden_size / config.num_attention_heads;
    let mut cache = Cache::new(
        true,
        DType::F32,
        config.max_position_embeddings,
        head_dim,
        config.num_hidden_layers,
        &device,
    )?;

    let mut total_nll = 0f64;
    let mut total_count: usize = 0;
    let mut chunks_done: usize = 0;
    let mut last_print = Instant::now();
    let started = Instant::now();

    let mut start = 0usize;
    while start + 1 < tokens.len() {
        if let Some(cap) = max_chunks {
            if chunks_done >= cap {
                println!("Reached PHI3_PPL_MAX_CHUNKS={}, stopping.", cap);
                break;
            }
        }

        let end = (start + ctx).min(tokens.len());
        let chunk = &tokens[start..end];
        if chunk.len() < 2 {
            break;
        }

        // Fresh cache per chunk so positions start at 0 and KV from the
        // previous chunk doesn't leak. We could reuse the cache for a
        // sliding-window evaluation, but per-chunk reset is the textbook
        // formulation and what HF `evaluate` does.
        cache.reset_for_new_turn();

        let (chunk_nll, chunk_count) = score_chunk(&model, &mut cache, &device, chunk)?;
        total_nll += chunk_nll;
        total_count += chunk_count;
        chunks_done += 1;

        // Progress every 2 s.
        if last_print.elapsed().as_secs_f32() >= 2.0 {
            let avg_nll = total_nll / total_count as f64;
            let ppl = avg_nll.exp();
            println!(
                "  chunk {} | tokens scored {} | running PPL = {:.4} | NLL = {:.4}",
                chunks_done, total_count, ppl, avg_nll
            );
            last_print = Instant::now();
        }

        if end == tokens.len() {
            break;
        }
        start = start.saturating_add(stride);
    }

    if total_count == 0 {
        candle::bail!("No tokens were scored");
    }

    let avg_nll = total_nll / total_count as f64;
    let ppl = avg_nll.exp();
    let elapsed = started.elapsed().as_secs_f32();

    println!();
    println!("Result:");
    println!("  chunks scored : {}", chunks_done);
    println!("  tokens scored : {}", total_count);
    println!("  mean NLL      : {:.6}", avg_nll);
    println!("  perplexity    : {:.4}", ppl);
    println!("  wall time     : {:.2} s", elapsed);
    if elapsed > 0.0 {
        println!(
            "  throughput    : {:.1} tok/s",
            total_count as f32 / elapsed
        );
    }

    Ok(())
}

fn load_model(args: &Args, device: &Device, config: &Phi3Config) -> candle::Result<Phi3> {
    if let Some(ref s2) = args.shard2 {
        build_model_multi_shard(&args.shard1, s2, device, config)
    } else {
        build_model(&args.shard1, device, config)
    }
}

/// Forward one chunk and return `(sum_nll, count)` over the `chunk.len()-1`
/// next-token predictions it contains.
fn score_chunk(
    model: &Phi3,
    cache: &mut Cache,
    device: &Device,
    chunk: &[u32],
) -> candle::Result<(f64, usize)> {
    let n = chunk.len();
    debug_assert!(n >= 2);

    // (1, n) input.
    let input = Tensor::new(chunk, device)?.unsqueeze(0)?;
    let logits = model.forward_full(&input, 0, cache)?; // (1, n, vocab)
    let logits = logits.squeeze(0)?; // (n, vocab)

    // log_softmax over vocab; pick the rows that actually predict something
    // (positions 0..n-1 → predict tokens[1..n]).
    let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let preds = log_probs.narrow(0, 0, n - 1)?; // (n-1, vocab)
    let preds_vec: Vec<f32> = preds.flatten_all()?.to_vec1()?;
    let vocab = log_probs.dim(D::Minus1)?;

    let mut sum_nll = 0f64;
    for i in 0..(n - 1) {
        let target = chunk[i + 1] as usize;
        if target >= vocab {
            candle::bail!("Token id {} >= vocab {}", target, vocab);
        }
        let lp = preds_vec[i * vocab + target] as f64;
        sum_nll += -lp;
    }

    Ok((sum_nll, n - 1))
}
