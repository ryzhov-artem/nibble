mod cache;
mod conversation;
mod loader;
mod model;
mod quant_linear;
mod types;

use candle::{bail, DType, Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use cache::Cache;
use conversation::Conversation;
use loader::{build_model, build_model_multi_shard, load_tokenizer};
use types::Phi3Config;

fn main() -> candle::Result<()> {
    let mut args = std::env::args().skip(1);
    let first_arg = args.next().ok_or_else(|| {
        candle::Error::Msg(
            "Usage: phi3_quant <packed-model.safetensors> [shard2.safetensors]\n\
             Single file: phi3_quant model.safetensors\n\
             Two shards:  phi3_quant shard1.safetensors shard2.safetensors"
                .into(),
        )
    })?;

    let shard1_path: PathBuf = first_arg.into();
    let shard2_path: Option<PathBuf> = args.next().map(PathBuf::from);

    let device = Device::Cpu;
    let config = Phi3Config::phi3_mini();

    println!("Loading Phi-3 Q8K model...");

    let model = if let Some(ref shard2) = shard2_path {
        build_model_multi_shard(&shard1_path, shard2, &device, &config)?
    } else {
        build_model(&shard1_path, &device, &config)?
    };

    let mut cache = Cache::new(
        true,
        DType::F32,
        config.max_position_embeddings,
        config.hidden_size / config.num_attention_heads,
        config.num_hidden_layers,
        &device,
    )?;

    let tokenizer = load_tokenizer()?;

    let system_prompt = "You are a helpful AI assistant.".to_string();
    let max_history_tokens = 3072;
    let mut conversation = Conversation::new(system_prompt, max_history_tokens);


    println!("\nPhi-3 Mini Q8K Conversational AI");
    println!("Commands: 'exit' to quit | 'reset' to clear history\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut global_position = 0usize;
    let mut is_first_turn = true;

    loop {
        print!("\nYou: ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        reader.read_line(&mut user_input)?;
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            println!("\nGoodbye!");
            break;
        }

        if user_input.eq_ignore_ascii_case("reset") {
            conversation = Conversation::new(conversation.system_prompt.clone(), max_history_tokens);
            cache.reset_for_new_turn();
            global_position = 0;
            is_first_turn = true;
            println!("Conversation history cleared!");
            continue;
        }

        if user_input.is_empty() {
            continue;
        }

        // ── Per-turn sampling: lower temperature for code questions ────────────
        let is_code_question = user_input.contains("code")
            || user_input.contains("example")
            || user_input.contains("```")
            || user_input.contains("function")
            || user_input.contains("loop")
            || user_input.contains("similar");

                let (sampling, repeat_pen, rep_last_n) = if is_code_question {
            (Sampling::TopKThenTopP { k: 25, p: 0.95, temperature: 0.35 }, 1.3f32, 128usize)
        } else {
            (Sampling::TopKThenTopP { k: 50, p: 0.9, temperature: 0.6 }, 1.35f32, 256usize)
        };
        let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);
        // ───────────────────────────────────────────────────────────────────────

        conversation.add_user_message(user_input.to_string());
        conversation.apply_sliding_window(&tokenizer)?;

        let gen_start = std::time::Instant::now();
        let mut token_count = 0usize;

        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let mut assistant_response = String::new();
        let mut printed_chars = 0usize;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let max_new_tokens = 1024;
        let eos_token_id = 32000u32;

        // ── helper: one autoregressive generation loop ─────────────────────────
                let mut run_generation = |input_tokens: &[u32],
                                  cache: &mut Cache,
                                  position: &mut usize,
                                  generated: &mut Vec<u32>,
                                  response: &mut String,
                                  printed_chars: &mut usize,
                                  tp_count: &mut usize|
         -> candle::Result<()> {
            let input = Tensor::new(input_tokens, &device)?.unsqueeze(0)?;
            let initial_logits = model.forward(&input, *position, cache)?;
            *position += input_tokens.len();

            let logits = match initial_logits.dims() {
                [_v] => initial_logits,
                [1, _v] => initial_logits.i((0, ..))?,
                _ => bail!("Unsupported logits shape: {:?}", initial_logits.dims()),
            };

            let mut last_token = logits_processor.sample(&logits)?;
            generated.push(last_token);
            *position += 1;

            for _step in 0..max_new_tokens {
                let input = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
                let logits = model.forward(&input, *position, cache)?;

                let logits = match logits.dims() {
                    [_v] => logits,
                    [1, _v] => logits.i((0, ..))?,
                    _ => bail!("Unsupported logits shape"),
                };

                let logits = if repeat_pen == 1.0 {
                    logits
                } else {
                    let start_at = generated.len().saturating_sub(rep_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        repeat_pen,
                        &generated[start_at..],
                    )?
                };

                let next_token = logits_processor.sample(&logits)?;
                *tp_count += 1;

                if next_token == eos_token_id {
                    break;
                }

                last_token = next_token;
                generated.push(next_token);
                *position += 1;

                if generated.len() >= 4 {
                    let batch_text = tokenizer
                        .decode(generated, true)
                        .unwrap_or_else(|_| String::from("[DECODE_ERROR]"));
                    let total_chars = batch_text.chars().count();
                    if total_chars > *printed_chars {
                        let new_part: String = batch_text.chars().skip(*printed_chars).collect();
                        print!("{}", new_part);
                        io::stdout().flush().unwrap();
                        *printed_chars = total_chars;
                    }
                    *response = batch_text;
                }
            }
            Ok(())
        };
        // ───────────────────────────────────────────────────────────────────────

        if is_first_turn {
            let full_prompt = conversation.format_prompt(&tokenizer)?;
            let prompt_tokens = tokenizer
                .encode(full_prompt, false)
                .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
                .get_ids()
                .to_vec();

            run_generation(
                &prompt_tokens,
                &mut cache,
                &mut global_position,
                &mut generated_tokens,
                &mut assistant_response,
                &mut printed_chars,
                &mut token_count,
            )?;

            is_first_turn = false;
        } else {
            let user_msg = format!("<|user|>\n{}<|end|>\n<|assistant|>\n", user_input);
            let user_tokens = tokenizer
                .encode(user_msg, false)
                .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
                .get_ids()
                .to_vec();

            run_generation(
                &user_tokens,
                &mut cache,
                &mut global_position,
                &mut generated_tokens,
                &mut assistant_response,
                &mut printed_chars,
                &mut token_count,
            )?;
        }

        let final_text = tokenizer
            .decode(&generated_tokens, true)
            .unwrap_or_else(|_| String::from("[DECODE_ERROR]"));
        let remaining: String = final_text.chars().skip(printed_chars).collect();
        if !remaining.is_empty() {
            print!("{}", remaining);
            io::stdout().flush().unwrap();
        }
        assistant_response = final_text;

        println!();
        conversation.add_assistant_message(assistant_response.trim().to_string());

        let elapsed = gen_start.elapsed().as_secs_f32();
        let tps = if elapsed > 0.0 && token_count > 0 { token_count as f32 / elapsed } else { 0.0 };

        println!(
            "[Pos: {} | Cache: {} tok ({:.1} MB) | Speed: {:.1} t/s | Hist: {} msgs]",
            global_position,
            cache.estimate_tokens(),
            cache.memory_mb(),
            tps,
            conversation.messages.len()
        );
    }

    Ok(())
}