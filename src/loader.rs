use candle::{Device, Tensor};
use safetensors::tensor::SafeTensors;
use std::{fs::File, path::{Path, PathBuf}};
use tokenizers::Tokenizer;

use crate::model::{Block, CausalSelfAttention, Mlp, Phi3};
use crate::quant_linear::{is_packed_format, QuantLinear};
use crate::types::{tensor_to_f32, Phi3Config};

/// Memory-map a safetensors file read-only.
///
/// Saves ~2.6 GB of heap allocation per shard at startup compared to
/// `fs::read` (the previous approach). The `Mmap` is paged in on demand
/// and can be released as soon as the loader has copied the blocks it needs
/// into owned `QuantLinear` storage — see [`build_model`].
///
/// # Safety
/// `unsafe` is inherited from `memmap2::MmapOptions::map`: the file must
/// not be modified by another process while the mapping is alive. Inference
/// is read-only and the packed shard is treated as an immutable artefact,
/// so this is safe in practice.
fn mmap_file(path: &Path) -> candle::Result<memmap2::Mmap> {
    let file = File::open(path)
        .map_err(|e| candle::Error::Msg(format!("open {}: {e}", path.display())))?;
    // SAFETY: see function docs.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| candle::Error::Msg(format!("mmap {}: {e}", path.display())))?;
    Ok(mmap)
}

/// Default HF repo used as a fallback for tokenizer / config.
pub const DEFAULT_HF_REPO: &str = "microsoft/Phi-3-mini-4k-instruct";

/// Resolve a `config.json` for the requested model.
///
/// Lookup order:
///   1. A `config.json` sitting next to the model file (or in the model dir).
///   2. `$HF_HOME` / `~/.cache/huggingface` snapshot of `repo`.
///   3. Fresh download from the HF Hub.
pub fn load_config(model_path: &Path, repo: &str) -> candle::Result<Phi3Config> {
    if let Some(parent) = model_path.parent() {
        let candidate = parent.join("config.json");
        if candidate.exists() {
            println!("Using config.json: {}", candidate.display());
            return Phi3Config::from_json_file(&candidate);
        }
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| candle::Error::Msg(format!("HF API: {}", e)))?;
    let path = api
        .model(repo.to_string())
        .get("config.json")
        .map_err(|e| candle::Error::Msg(format!("download config.json from {}: {}", repo, e)))?;
    println!("Using config.json: {}", path.display());
    Phi3Config::from_json_file(&path)
}

/// Collect the EOS token ids from the tokenizer's special-tokens map.
/// Falls back to the legacy `32000` if nothing matches.
pub fn eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let candidates = ["<|end|>", "<|endoftext|>", "</s>"];
    let mut ids: Vec<u32> = candidates
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    if ids.is_empty() {
        ids.push(32000);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub fn load_tokenizer() -> candle::Result<Tokenizer> {
    let direct_paths = [
        PathBuf::from(shellexpand::tilde(
            "~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e/tokenizer.json"
        ).as_ref()),
        PathBuf::from("./tokenizer.json"),
    ];

    for path in &direct_paths {
        if path.exists() {
            return Tokenizer::from_file(path).map_err(|e| {
                candle::Error::Msg(format!("Failed to load tokenizer from {}: {}", path.display(), e))
            });
        }
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| candle::Error::Msg(format!("Failed to create HF API: {}", e)))?;
    let repo = api.model(DEFAULT_HF_REPO.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle::Error::Msg(format!("Failed to load downloaded tokenizer: {}", e)))
}

fn load_blocks(
    load_tensor: &impl Fn(&str) -> candle::Result<Vec<f32>>,
    load_quant: &impl Fn(&str) -> candle::Result<QuantLinear>,
    config: &Phi3Config,
    device: &Device,
) -> candle::Result<Vec<Block>> {
    let mut blocks = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let rms_1 = Tensor::from_vec(
            load_tensor(&format!("model.layers.{}.input_layernorm.weight", i))?,
            (config.hidden_size,),
            device,
        )?;
        let rms_2 = Tensor::from_vec(
            load_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i))?,
            (config.hidden_size,),
            device,
        )?;
        let qkv_proj = load_quant(&format!("model.layers.{}.self_attn.qkv_proj.weight", i))?;
        let o_proj   = load_quant(&format!("model.layers.{}.self_attn.o_proj.weight", i))?;
        let gate_up  = load_quant(&format!("model.layers.{}.mlp.gate_up_proj.weight", i))?;
        let down     = load_quant(&format!("model.layers.{}.mlp.down_proj.weight", i))?;

        blocks.push(Block {
            rms_1,
            attn: CausalSelfAttention {
                qkv_proj,
                o_proj,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads(),
                head_dim: config.hidden_size / config.num_attention_heads,
            },
            rms_2,
            mlp: Mlp { gate_up_proj: gate_up, down_proj: down, hidden_size: config.intermediate_size },
        });
    }
    Ok(blocks)
}

pub fn build_model(
    safetensors_path: &Path,
    device: &Device,
    config: &Phi3Config,
) -> candle::Result<Phi3> {
    let mmap = mmap_file(safetensors_path)?;
    let st = SafeTensors::deserialize(&mmap)?;

    if !is_packed_format(&st) {
        candle::bail!("Only packed format supported. Run pack_q8k_safetensors first.");
    }

    let embed_view = st.tensor("model.embed_tokens.weight")?;
    let wte = Tensor::from_vec(
        tensor_to_f32(embed_view.data(), embed_view.dtype())?,
        embed_view.shape(),
        device,
    )?;

    let norm_name = if st.names().iter().any(|n| *n == "model.norm.weight") {
        "model.norm.weight"
    } else if st.names().iter().any(|n| *n == "model.final_layernorm.weight") {
        "model.final_layernorm.weight"
    } else if st.names().iter().any(|n| *n == "norm.weight") {
        "norm.weight"
    } else {
        println!("Available norm tensors:");
        for name in st.names() {
            if name.contains("norm") { println!("  - {}", name); }
        }
        candle::bail!(
            "Could not find final layer norm. Expected one of: \
             model.norm.weight, model.final_layernorm.weight, norm.weight"
        )
    };

    let norm_view = st.tensor(norm_name)?;
    let ln_f = Tensor::from_vec(
        tensor_to_f32(norm_view.data(), norm_view.dtype())?,
        norm_view.shape(),
        device,
    )?;

    let load_t = |name: &str| -> candle::Result<Vec<f32>> {
        let v = st.tensor(name)?;
        tensor_to_f32(v.data(), v.dtype())
    };
    let load_q = |name: &str| -> candle::Result<QuantLinear> {
    let q4k_meta = format!("{}.q4k_meta", name);
    if st.names().iter().any(|n| **n == q4k_meta) {
        return QuantLinear::load_q4k_from_packed_safetensors(&st, name);
    }
    QuantLinear::load_from_packed_safetensors(&st, name)
};

    let blocks = load_blocks(&load_t, &load_q, config, device)?;
    let lm_head = QuantLinear::load_from_packed_safetensors(&st, "lm_head.weight")?;

    Ok(Phi3 { wte, blocks, ln_f, lm_head })
}

pub fn build_model_multi_shard(
    shard1: &Path,
    shard2: &Path,
    device: &Device,
    config: &Phi3Config,
) -> candle::Result<Phi3> {
    println!("Loading from 2 shards...");
    println!("  Shard 1: {}", shard1.display());
    println!("  Shard 2: {}", shard2.display());

    let bytes1 = mmap_file(shard1)?;
    let st1 = SafeTensors::deserialize(&bytes1)?;
    let bytes2 = mmap_file(shard2)?;
    let st2 = SafeTensors::deserialize(&bytes2)?;

    if !is_packed_format(&st1) || !is_packed_format(&st2) {
        candle::bail!("Both shards must be in packed format");
    }

    let load_t = |name: &str| -> candle::Result<Vec<f32>> {
        if let Ok(v) = st1.tensor(name) { return tensor_to_f32(v.data(), v.dtype()); }
        if let Ok(v) = st2.tensor(name) { return tensor_to_f32(v.data(), v.dtype()); }
        candle::bail!("Tensor {} not found in either shard", name)
    };
    let load_q = |name: &str| -> candle::Result<QuantLinear> {
    let q4k_meta = format!("{}.q4k_meta", name);
    let q8k_meta = format!("{}.q8k_meta", name);
    if st1.names().iter().any(|n| **n == q4k_meta) {
        return QuantLinear::load_q4k_from_packed_safetensors(&st1, name);
    }
    if st2.names().iter().any(|n| **n == q4k_meta) {
        return QuantLinear::load_q4k_from_packed_safetensors(&st2, name);
    }
    if st1.names().iter().any(|n| **n == q8k_meta) {
        return QuantLinear::load_from_packed_safetensors(&st1, name);
    }
    if st2.names().iter().any(|n| **n == q8k_meta) {
        return QuantLinear::load_from_packed_safetensors(&st2, name);
    }
    candle::bail!("Quantized tensor {} not found in either shard", name)
};

    let embed_view = st1.tensor("model.embed_tokens.weight")
        .or_else(|_| st2.tensor("model.embed_tokens.weight"))?;
    let wte = Tensor::from_vec(
        tensor_to_f32(embed_view.data(), embed_view.dtype())?,
        embed_view.shape(),
        device,
    )?;

    let norm_data = load_t("model.norm.weight")
        .or_else(|_| load_t("model.final_layernorm.weight"))
        .or_else(|_| load_t("norm.weight"))
        .map_err(|_| candle::Error::Msg("Could not find final layer norm in either shard".into()))?;
    let ln_f = Tensor::from_vec(norm_data, (config.hidden_size,), device)?;

    let blocks = load_blocks(&load_t, &load_q, config, device)?;
    let lm_head = load_q("lm_head.weight")?;

    Ok(Phi3 { wte, blocks, ln_f, lm_head })
}