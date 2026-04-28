//! Interactive chat REPL: load model + tokenizer, read lines from stdin,
//! delegate token streaming to [`crate::generation`].

use candle::{DType, Device};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::{self, BufRead, Write};

use crate::cache::Cache;
use crate::cli::CliArgs;
use crate::conversation::Conversation;
use crate::generation::{generate, GenConfig};
use crate::loader::{
    build_model, build_model_multi_shard, eos_token_ids, load_config, load_tokenizer,
    DEFAULT_HF_REPO,
};

const SYSTEM_PROMPT: &str = "You are a helpful AI assistant.";
const MAX_HISTORY_TOKENS: usize = 3072;
const MAX_NEW_TOKENS: usize = 1024;

/// Boot the model, set up the conversation, and run the chat loop until EOF / "exit".
pub fn run(args: CliArgs) -> candle::Result<()> {
    let device = Device::Cpu;
    let config = load_config(&args.shard1, DEFAULT_HF_REPO)?;

    println!(
        "Loading Phi-3 Q8K model (layers={}, hidden={})...",
        config.num_hidden_layers, config.hidden_size
    );

    let model = if let Some(ref shard2) = args.shard2 {
        build_model_multi_shard(&args.shard1, shard2, &device, &config)?
    } else {
        build_model(&args.shard1, &device, &config)?
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
    let eos_ids = eos_token_ids(&tokenizer);
    println!("EOS token ids: {:?}", eos_ids);

    let mut conversation = Conversation::new(SYSTEM_PROMPT.to_string(), MAX_HISTORY_TOKENS);

    println!("\nPhi-3 Mini Q8K Conversational AI");
    println!("Commands: 'exit' to quit | 'reset' to clear history\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut global_position = 0usize;
    let mut is_first_turn = true;

    loop {
        print!("\nYou: ");
        io::stdout().flush().ok();

        let mut user_input = String::new();
        if reader.read_line(&mut user_input)? == 0 {
            // EOF
            println!();
            break;
        }
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            println!("\nGoodbye!");
            break;
        }

        if user_input.eq_ignore_ascii_case("reset") {
            conversation =
                Conversation::new(conversation.system_prompt.clone(), MAX_HISTORY_TOKENS);
            cache.reset_for_new_turn();
            global_position = 0;
            is_first_turn = true;
            println!("Conversation history cleared!");
            continue;
        }

        if user_input.is_empty() {
            continue;
        }

        let (sampling, gen_cfg) = sampling_for_query(user_input);
        let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

        conversation.add_user_message(user_input.to_string());
        conversation.apply_sliding_window(&tokenizer)?;

        let gen_start = std::time::Instant::now();

        print!("Assistant: ");
        io::stdout().flush().ok();

        let mut assistant_response = String::new();
        let mut printed_chars = 0usize;
        let mut generated_tokens: Vec<u32> = Vec::new();

        let prompt_tokens = if is_first_turn {
            let full_prompt = conversation.format_prompt(&tokenizer)?;
            tokenizer
                .encode(full_prompt, false)
                .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {e}")))?
                .get_ids()
                .to_vec()
        } else {
            let user_msg = format!("<|user|>\n{user_input}<|end|>\n<|assistant|>\n");
            tokenizer
                .encode(user_msg, false)
                .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {e}")))?
                .get_ids()
                .to_vec()
        };

        let stats = generate(
            &model,
            &mut cache,
            &tokenizer,
            &device,
            &mut logits_processor,
            &eos_ids,
            &prompt_tokens,
            &mut global_position,
            &mut generated_tokens,
            &mut assistant_response,
            &mut printed_chars,
            &gen_cfg,
        )?;

        is_first_turn = false;

        // Flush any tail not yet streamed.
        let final_text = tokenizer
            .decode(&generated_tokens, true)
            .unwrap_or_else(|_| String::from("[DECODE_ERROR]"));
        let remaining: String = final_text.chars().skip(printed_chars).collect();
        if !remaining.is_empty() {
            print!("{remaining}");
            io::stdout().flush().ok();
        }
        assistant_response = final_text;

        println!();
        conversation.add_assistant_message(assistant_response.trim().to_string());

        let elapsed = gen_start.elapsed().as_secs_f32();
        let tps = if elapsed > 0.0 && stats.token_count > 0 {
            stats.token_count as f32 / elapsed
        } else {
            0.0
        };

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

/// Pick sampling parameters and a generation config based on a heuristic
/// classification of the user's query (code-flavoured vs. general chat).
fn sampling_for_query(user_input: &str) -> (Sampling, GenConfig) {
    let is_code_question = user_input.contains("code")
        || user_input.contains("example")
        || user_input.contains("```")
        || user_input.contains("function")
        || user_input.contains("loop")
        || user_input.contains("similar");

    if is_code_question {
        (
            Sampling::TopKThenTopP {
                k: 25,
                p: 0.95,
                temperature: 0.35,
            },
            GenConfig {
                max_new_tokens: MAX_NEW_TOKENS,
                repeat_penalty: 1.3,
                repeat_last_n: 128,
            },
        )
    } else {
        (
            Sampling::TopKThenTopP {
                k: 50,
                p: 0.9,
                temperature: 0.6,
            },
            GenConfig {
                max_new_tokens: MAX_NEW_TOKENS,
                repeat_penalty: 1.35,
                repeat_last_n: 256,
            },
        )
    }
}
