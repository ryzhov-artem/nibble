//! Token-by-token autoregressive generation loop.
//!
//! Pulled out of `main.rs` so the REPL only has to deal with conversation
//! state and prompt assembly. The loop is shared between the first-turn
//! prompt (full system+user template) and subsequent turns (just the new
//! `<|user|>...` block).

use candle::{bail, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::io::{self, Write};
use tokenizers::Tokenizer;

use crate::cache::Cache;
use crate::model::Phi3;

/// Runtime knobs for one generation pass.
pub struct GenConfig {
    pub max_new_tokens: usize,
    /// Repetition penalty applied to the recent window of generated tokens.
    pub repeat_penalty: f32,
    /// Width of that window (in tokens).
    pub repeat_last_n: usize,
}

/// Outcome of a single `generate` call.
pub struct GenStats {
    /// Number of tokens produced (excluding the bootstrap sample).
    pub token_count: usize,
}

/// Run the autoregressive loop, streaming decoded text to stdout as tokens arrive.
///
/// `position` and `generated` are mutated in place so the caller (the REPL)
/// can keep going across multiple user turns with a warm KV cache.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &Phi3,
    cache: &mut Cache,
    tokenizer: &Tokenizer,
    device: &Device,
    logits_processor: &mut LogitsProcessor,
    eos_ids: &[u32],
    input_tokens: &[u32],
    position: &mut usize,
    generated: &mut Vec<u32>,
    response: &mut String,
    printed_chars: &mut usize,
    cfg: &GenConfig,
) -> candle::Result<GenStats> {
    let mut token_count = 0usize;

    // ── prompt prefill ────────────────────────────────────────────────────
    let input = Tensor::new(input_tokens, device)?.unsqueeze(0)?;
    let initial_logits = model.forward(&input, *position, cache)?;
    *position += input_tokens.len();

    let logits = squeeze_logits(initial_logits)?;
    let mut last_token = logits_processor.sample(&logits)?;
    generated.push(last_token);
    *position += 1;

    // ── decode loop ───────────────────────────────────────────────────────
    for _step in 0..cfg.max_new_tokens {
        let input = Tensor::new(&[last_token], device)?.unsqueeze(0)?;
        let logits = squeeze_logits(model.forward(&input, *position, cache)?)?;

        let logits = if cfg.repeat_penalty == 1.0 {
            logits
        } else {
            let start_at = generated.len().saturating_sub(cfg.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                cfg.repeat_penalty,
                &generated[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        token_count += 1;

        if eos_ids.contains(&next_token) {
            break;
        }

        last_token = next_token;
        generated.push(next_token);
        *position += 1;

        // Stream decoded text once we have enough tokens to amortise BPE merges.
        if generated.len() >= 4 {
            let batch_text = tokenizer
                .decode(generated, true)
                .unwrap_or_else(|_| String::from("[DECODE_ERROR]"));
            let total_chars = batch_text.chars().count();
            if total_chars > *printed_chars {
                let new_part: String = batch_text.chars().skip(*printed_chars).collect();
                print!("{new_part}");
                io::stdout().flush().ok();
                *printed_chars = total_chars;
            }
            *response = batch_text;
        }
    }

    Ok(GenStats { token_count })
}

fn squeeze_logits(logits: Tensor) -> candle::Result<Tensor> {
    match logits.dims() {
        [_v] => Ok(logits),
        [1, _v] => logits.i((0, ..)),
        other => bail!("Unsupported logits shape: {:?}", other),
    }
}
