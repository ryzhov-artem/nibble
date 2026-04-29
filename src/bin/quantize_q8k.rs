//! Standalone Q8_K / Q4_K quantizer for a single `model.safetensors` shard.
//!
//! Usage:
//!   cargo run --release -p tensor-tools --bin quantize_q8k \
//!     <input.safetensors> <output_dir>
//!
//! Layer policy (compile-time):
//!   mlp.down_proj  → Q4K  (50% smaller blocks, less sensitivity)
//!   everything else → Q8K
//!
//! Permutation toggle (requires the `experimental-perm` Cargo feature):
//!   cargo build --release --features experimental-perm
//!   CANDLE_Q8K_PERMUTE=1  cargo run ...
//!
//! Output files per layer:
//!   <name>.q8k  + <name>.q8k_meta  (+ optional <name>.perm)
//!   <name>.q4k  + <name>.q4k_meta
use anyhow::{bail, Context, Result};
use candle::quantized::k_quants::{matmul, BlockQ4K, BlockQ8K, GgmlType, QK_K};
use candle::Device;
use half::{bf16, f16};
#[cfg(feature = "experimental-perm")]
use once_cell::sync::Lazy;
use phi3_mixed_quant::types::{Q8KHeader, DTYPE_Q4K, DTYPE_Q8K, HEADER_VERSION, MAGIC_Q4K, MAGIC_Q8K};
#[cfg(feature = "experimental-perm")]
use phi3_mixed_quant::types::MAGIC_PERM;
#[cfg(feature = "experimental-perm")]
use regex::Regex;
use safetensors::tensor::{Dtype, SafeTensors};
#[cfg(feature = "experimental-perm")]
use std::collections::HashMap;
#[cfg(feature = "experimental-perm")]
use std::sync::Mutex;
#[cfg(feature = "experimental-perm")]
use std::sync::OnceLock;
use std::{
    fs,
    io::{BufWriter, Write},
    mem,
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(feature = "experimental-perm")]
static LAYER_PERM_CACHE: OnceLock<Mutex<LayerPermCache>> = OnceLock::new();
#[cfg(feature = "experimental-perm")]
static ATTN_PROJ_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$").unwrap());

// ── layer routing ──────────────────────────────────────────────────────────

fn is_target_weight(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    if name.contains("embed_tokens") {
        return false;
    }
    if name.contains("norm") {
        return false;
    }
    true
}

fn is_q4k_layer(name: &str) -> bool {
    name.contains("mlp.down_proj")
}

// ── Q8K helpers ────────────────────────────────────────────────────────────

fn validate_quantization(original: &[f32], blocks: &[BlockQ8K], k: usize) -> Result<f32> {
    let rows = original.len() / k;
    let _device = Device::Cpu;
    let test_input = vec![1.0f32; k];
    let mut expected_output = vec![0f32; rows];
    for row in 0..rows {
        let row_data = &original[row * k..(row + 1) * k];
        expected_output[row] = row_data.iter().sum();
    }
    let mut actual_output = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &test_input, blocks, &mut actual_output)
        .map_err(|e| anyhow::anyhow!("matmul failed: {}", e))?;
    let mut mse = 0f32;
    for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
        let diff = expected - actual;
        mse += diff * diff;
    }
    mse /= rows as f32;
    Ok(mse)
}

fn validate_quantization_direct(original: &[f32], blocks: &[BlockQ8K], k: usize) -> Result<f32> {
    let rows = original.len() / k;
    let mut test_input = vec![0f32; k];
    for i in 0..k {
        test_input[i] = (i as f32 + 1.0) / k as f32;
    }
    let mut expected_output = vec![0f32; rows];
    for row in 0..rows {
        let row_data = &original[row * k..(row + 1) * k];
        expected_output[row] = row_data.iter().zip(test_input.iter()).map(|(a, b)| a * b).sum();
    }
    let mut actual_output = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &test_input, blocks, &mut actual_output)
        .map_err(|e| anyhow::anyhow!("direct validation matmul failed: {}", e))?;
    let mut mse = 0f32;
    for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
        let diff = expected - actual;
        mse += diff * diff;
    }
    mse /= rows as f32;
    Ok(mse)
}

fn compute_quantization_error_detailed(
    original: &[f32],
    blocks: &[BlockQ8K],
    rows: usize,
    k: usize,
) -> Result<(f64, f64, f64)> {
    let mut dequantized = vec![0f32; rows * k];
    BlockQ8K::to_float(blocks, &mut dequantized);
    let mut l2_error = 0f64;
    let mut max_error = 0f64;
    let mut relative_error_sum = 0f64;
    for (&orig, &deq) in original.iter().zip(dequantized.iter()) {
        let abs_err = (orig - deq).abs() as f64;
        l2_error += abs_err * abs_err;
        max_error = max_error.max(abs_err);
        if orig.abs() > 1e-10 {
            relative_error_sum += abs_err / orig.abs() as f64;
        }
    }
    let rmse = (l2_error / (rows * k) as f64).sqrt();
    let mean_relative_error = relative_error_sum / (rows * k) as f64;
    Ok((rmse, max_error, mean_relative_error))
}

fn quantize_rows_q8k(rows: usize, k: usize, data: &[f32]) -> Result<Vec<BlockQ8K>> {
    if k % QK_K != 0 {
        bail!("inner dim {k} not multiple of {QK_K}");
    }
    let blocks_per_row = k / QK_K;
    let mut blocks = vec![BlockQ8K::zeros(); rows * blocks_per_row];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        let dst = &mut blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
        BlockQ8K::from_float(row, dst);
    }
    Ok(blocks)
}

fn write_q8k(path: &Path, rows: usize, k: usize, blocks: &[BlockQ8K]) -> Result<()> {
    let header = Q8KHeader {
        magic: MAGIC_Q8K,
        version: HEADER_VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_K) as u32,
        dtype: DTYPE_Q8K,
    };
    let mut w = BufWriter::new(fs::File::create(path)?);
    w.write_all(bytemuck::bytes_of(&header))?;
    let raw = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * mem::size_of::<BlockQ8K>())
    };
    w.write_all(raw)?;
    w.flush()?;
    Ok(())
}

// ── Q4K helpers ────────────────────────────────────────────────────────────

fn quantize_rows_q4k(rows: usize, k: usize, data: &[f32]) -> Result<Vec<BlockQ4K>> {
    if k % QK_K != 0 {
        bail!("inner dim {k} not multiple of {QK_K}");
    }
    let blocks_per_row = k / QK_K;
    let mut blocks = vec![BlockQ4K::zeros(); rows * blocks_per_row];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        let dst = &mut blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
        BlockQ4K::from_float(row, dst);
    }
    Ok(blocks)
}

fn validate_q4k(original: &[f32], blocks: &[BlockQ4K], rows: usize, k: usize) -> Result<(f64, f64, f64)> {
    let mut dequantized = vec![0f32; rows * k];
    BlockQ4K::to_float(blocks, &mut dequantized);
    let mut l2_error = 0f64;
    let mut max_error = 0f64;
    let mut relative_error_sum = 0f64;
    for (&orig, &deq) in original.iter().zip(dequantized.iter()) {
        let abs_err = (orig - deq).abs() as f64;
        l2_error += abs_err * abs_err;
        max_error = max_error.max(abs_err);
        if orig.abs() > 1e-10 {
            relative_error_sum += abs_err / orig.abs() as f64;
        }
    }
    let rmse = (l2_error / (rows * k) as f64).sqrt();
    let mean_relative_error = relative_error_sum / (rows * k) as f64;
    Ok((rmse, max_error, mean_relative_error))
}

/// Write a .q4k file: header (Q8KHeader struct with MAGIC_Q4K) followed by raw BlockQ4K bytes.
/// The packer reads back only the raw block bytes (strips the header), matching
/// what quant_linear.rs::load_q4k_from_packed_safetensors expects.
fn write_q4k(path: &Path, rows: usize, k: usize, blocks: &[BlockQ4K]) -> Result<()> {
    let header = Q8KHeader {
        magic: MAGIC_Q4K,
        version: HEADER_VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_K) as u32,
        dtype: DTYPE_Q4K,
    };
    let mut w = BufWriter::new(fs::File::create(path)?);
    w.write_all(bytemuck::bytes_of(&header))?;
    let raw = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * mem::size_of::<BlockQ4K>())
    };
    w.write_all(raw)?;
    w.flush()?;
    Ok(())
}

// ── shared dtype conversion + permutation helpers ─────────────────────────
//
// `tensor_to_f32` is used by the main loop unconditionally; everything else
// in this section is only compiled with `--features experimental-perm`.

fn tensor_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Dtype::F16 => bytes.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32()).collect(),
        Dtype::BF16 => bytes.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32()).collect(),
        other => bail!("unsupported dtype {other:?}"),
    })
}

#[cfg(feature = "experimental-perm")]
fn column_l2_norms(rows: usize, k: usize, data: &[f32]) -> Vec<f32> {
    let mut sums: Vec<f64> = vec![0.0; k];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        for (j, &v) in row.iter().enumerate() {
            let fv = v as f64;
            sums[j] += fv * fv;
        }
    }
    sums.into_iter().map(|s| s.sqrt() as f32).collect()
}

#[cfg(feature = "experimental-perm")]
fn build_column_permutation(norms: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..norms.len()).collect();
    idx.sort_by(|&a, &b| norms[b].partial_cmp(&norms[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

#[cfg(feature = "experimental-perm")]
fn build_block_wise_permutation(rows: usize, k: usize, data: &[f32]) -> Vec<usize> {
    const BLOCK_SIZE: usize = 64;
    let num_blocks = k / BLOCK_SIZE;
    if k % BLOCK_SIZE != 0 {
        let norms = column_l2_norms(rows, k, data);
        return build_column_permutation(&norms);
    }
    let mut global_perm = vec![0usize; k];
    let col_norms = column_l2_norms(rows, k, data);
    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_norms = &col_norms[block_start..block_start + BLOCK_SIZE];
        let mut local_idx: Vec<usize> = (0..BLOCK_SIZE).collect();
        local_idx.sort_by(|&a, &b| block_norms[b].partial_cmp(&block_norms[a]).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..BLOCK_SIZE {
            global_perm[block_start + i] = block_start + local_idx[i];
        }
    }
    println!("    Block-wise permutation: {} blocks of {}", num_blocks, BLOCK_SIZE);
    global_perm
}

#[cfg(feature = "experimental-perm")]
fn apply_column_permutation(rows: usize, k: usize, data: &[f32], perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        let src = &data[r * k..(r + 1) * k];
        let dst = &mut out[r * k..(r + 1) * k];
        for j in 0..k {
            dst[j] = src[perm[j]];
        }
    }
    out
}

#[cfg(feature = "experimental-perm")]
fn write_perm(path_base: &Path, perm: &[usize]) -> Result<()> {
    let mut p = path_base.to_path_buf();
    p.set_extension("perm");
    let mut w = BufWriter::new(fs::File::create(&p)?);
    w.write_all(&MAGIC_PERM.to_le_bytes())?;
    w.write_all(&(perm.len() as u32).to_le_bytes())?;
    for &u in perm {
        w.write_all(&(u as u32).to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(feature = "experimental-perm")]
#[derive(Debug, Clone, Copy)]
enum PermStrategy {
    Block,
    L2,
    Svd,
    Qr,
}

#[cfg(feature = "experimental-perm")]
impl PermStrategy {
    fn from_env() -> Self {
        match std::env::var("CANDLE_Q8K_PERM_STRATEGY")
            .ok()
            .as_deref()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("l2") | Some("column") => Self::L2,
            Some("svd") | Some("variance") => Self::Svd,
            Some("qr") | Some("qr-pivot") => Self::Qr,
            // default = blockwise (matches prior behaviour)
            _ => Self::Block,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Block => "blockwise",
            Self::L2 => "l2-norm",
            Self::Svd => "svd-importance",
            Self::Qr => "qr-pivot",
        }
    }

    fn build(self, rows: usize, k: usize, data: &[f32]) -> Result<Vec<usize>> {
        Ok(match self {
            Self::Block => build_block_wise_permutation(rows, k, data),
            Self::L2 => build_column_permutation(&column_l2_norms(rows, k, data)),
            Self::Svd => svd_importance_permutation(rows, k, data)?,
            Self::Qr => qr_pivot_permutation(rows, k, data)?,
        })
    }
}

#[cfg(feature = "experimental-perm")]
#[derive(Debug, Clone)]
struct LayerPermCache {
    map: HashMap<u32, Vec<usize>>,
}

#[cfg(feature = "experimental-perm")]
impl LayerPermCache {
    fn new() -> Self {
        Self { map: HashMap::new() }
    }

    fn get_or_compute(
        &mut self,
        strategy: PermStrategy,
        layer_id: u32,
        rows: usize,
        k: usize,
        data: &[f32],
    ) -> Result<Vec<usize>> {
        if let Some(p) = self.map.get(&layer_id) {
            return Ok(p.clone());
        }
        let perm = strategy.build(rows, k, data)?;
        self.map.insert(layer_id, perm.clone());
        Ok(perm)
    }
}

#[cfg(feature = "experimental-perm")]
fn parse_attention_proj(name: &str) -> Option<(u32, &'static str)> {
    if let Some(caps) = ATTN_PROJ_RE.captures(name) {
        let layer_id: u32 = caps[1].parse().ok()?;
        let kind = match &caps[2] {
            "q" => "q",
            "k" => "k",
            "v" => "v",
            "o" => "o",
            _ => return None,
        };
        Some((layer_id, kind))
    } else {
        None
    }
}

#[cfg(feature = "experimental-perm")]
fn svd_importance_permutation(rows: usize, k: usize, data: &[f32]) -> Result<Vec<usize>> {
    let mut col_means: Vec<f64> = vec![0.0; k];
    let mut col_vars: Vec<f64> = vec![0.0; k];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        for (j, &v) in row.iter().enumerate() {
            col_means[j] += v as f64;
        }
    }
    for mean in &mut col_means {
        *mean /= rows as f64;
    }
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        for (j, &v) in row.iter().enumerate() {
            let diff = v as f64 - col_means[j];
            col_vars[j] += diff * diff;
        }
    }
    for var in &mut col_vars {
        *var /= rows as f64;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| col_vars[b].partial_cmp(&col_vars[a]).unwrap_or(std::cmp::Ordering::Equal));
    println!("    SVD-importance: sorted by column variance");
    Ok(idx)
}

#[cfg(feature = "experimental-perm")]
fn qr_pivot_permutation(rows: usize, k: usize, data: &[f32]) -> Result<Vec<usize>> {
    let mut perm: Vec<usize> = (0..k).collect();
    let qr_steps = match k {
        k if k <= 64 => k,
        k if k <= 256 => (k * 7) / 8,
        k if k <= 512 => (k * 3) / 4,
        k if k <= 1024 => (k * 2) / 3,
        k if k <= 2048 => k / 2,
        _ => (k / 4).max(256).min(512),
    };
    println!("    QR pivot: processing {}/{} columns ({:.1}%)", qr_steps, k, (qr_steps as f64 / k as f64) * 100.0);
    let mut col_norms: Vec<f64> = vec![0.0; k];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        for (j, &v) in row.iter().enumerate() {
            let fv = v as f64;
            col_norms[j] += fv * fv;
        }
    }
    for norm in &mut col_norms {
        *norm = norm.sqrt();
    }
    for step in 0..qr_steps.min(rows).min(k) {
        let mut max_norm = col_norms[step];
        let mut max_idx = step;
        for j in (step + 1)..k {
            if col_norms[j] > max_norm {
                max_norm = col_norms[j];
                max_idx = j;
            }
        }
        if max_idx != step {
            perm.swap(step, max_idx);
            col_norms.swap(step, max_idx);
        }
        if col_norms[step] > 1e-10 {
            for j in (step + 1)..k {
                col_norms[j] *= 0.99;
            }
        }
    }
    if qr_steps < k {
        let mut remaining_idx: Vec<usize> = (qr_steps..k).collect();
        remaining_idx.sort_by(|&a, &b| col_norms[b].partial_cmp(&col_norms[a]).unwrap_or(std::cmp::Ordering::Equal));
        for (i, &orig_idx) in remaining_idx.iter().enumerate() {
            perm[qr_steps + i] = perm[orig_idx];
        }
    }
    Ok(perm)
}

/// Check if a string looks like a HF model ID (contains exactly one slash, no path separators)
fn is_hf_model_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 {
        return false;
    }
    if s.starts_with('.') || s.starts_with('/') || s.starts_with('~') {
        return false;
    }
    // Reject anything that already exists as a path on disk (so a local
    // "foo/bar.safetensors" is treated as a file, not an HF model id).
    if Path::new(s).exists() {
        return false;
    }
    // org/repo: both parts non-empty, no nested slashes, no extension.
    !parts[0].is_empty() && !parts[1].is_empty() && !parts[1].contains('.')
}

/// Download model safetensors from Hugging Face Hub, returns paths to downloaded files.
fn download_safetensors(model_id: &str, filename: Option<&str>) -> Result<Vec<PathBuf>> {
    use hf_hub::api::sync::Api;

    println!("Downloading from Hugging Face: {model_id}");
    let api = Api::new().context("Failed to create HF Hub API client")?;
    let repo = api.model(model_id.to_string());

    if let Some(name) = filename {
        // User specified a single file
        let path = repo
            .get(name)
            .with_context(|| format!("Failed to download {name} from {model_id}"))?;
        println!("  Downloaded: {}", path.display());
        Ok(vec![path])
    } else {
        // Try single model.safetensors first, then fall back to sharded pattern
        match repo.get("model.safetensors") {
            Ok(path) => {
                println!("  Downloaded: {}", path.display());
                Ok(vec![path])
            }
            Err(_) => {
                // Try sharded format: model-00001-of-NNNNN.safetensors
                println!("  No single model.safetensors, trying sharded format...");
                let mut paths = Vec::new();
                for i in 1..=100 {
                    let shard_name = format!("model-{i:05}-of-{:05}.safetensors", 0);
                    // We don't know total count yet; try to get index file first
                    let _ = shard_name; // placeholder
                    break;
                }
                // Use the model index to find all shards
                let index_path = repo
                    .get("model.safetensors.index.json")
                    .context("No model.safetensors or model.safetensors.index.json found")?;
                let index_bytes = fs::read(&index_path)?;
                let index: serde_json::Value = serde_json::from_slice(&index_bytes)
                    .context("Failed to parse model.safetensors.index.json")?;

                if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                    let mut shard_files: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    shard_files.sort();
                    shard_files.dedup();

                    for shard_name in &shard_files {
                        let path = repo
                            .get(shard_name)
                            .with_context(|| format!("Failed to download shard {shard_name}"))?;
                        println!("  Downloaded: {} ({})", shard_name, path.display());
                        paths.push(path);
                    }
                }

                if paths.is_empty() {
                    bail!("No safetensors files found in {model_id}");
                }
                Ok(paths)
            }
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() || args.len() < 2 {
        bail!(
            "Usage: quantize_q8k <input.safetensors | org/model> <output_dir> [--file <name>]\n\
             \n\
             Examples:\n\
             \x20 quantize_q8k model.safetensors ./quantized\n\
             \x20 quantize_q8k microsoft/Phi-3-mini-4k-instruct ./quantized\n\
             \x20 quantize_q8k microsoft/Phi-3-mini-4k-instruct ./quantized --file model-00001-of-00002.safetensors"
        );
    }

    let input_arg = &args[0];
    let out_dir: PathBuf = args[1].clone().into();

    // Parse optional --file flag
    let explicit_file = if let Some(pos) = args.iter().position(|a| a == "--file") {
        Some(
            args.get(pos + 1)
                .context("--file requires a filename argument")?
                .as_str(),
        )
    } else {
        None
    };

    // Resolve input: either a local file or a HF model ID to download
    let input_files: Vec<PathBuf> = if is_hf_model_id(input_arg) {
        download_safetensors(input_arg, explicit_file)?
    } else {
        let p: PathBuf = input_arg.into();
        if !p.exists() {
            bail!("Input file does not exist: {}", p.display());
        }
        vec![p]
    };

    fs::create_dir_all(&out_dir)?;

    #[cfg(feature = "experimental-perm")]
    let use_permute = std::env::var("CANDLE_Q8K_PERMUTE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    #[cfg(not(feature = "experimental-perm"))]
    {
        if std::env::var("CANDLE_Q8K_PERMUTE").ok().as_deref() == Some("1") {
            eprintln!(
                "warning: CANDLE_Q8K_PERMUTE=1 ignored — \
                 rebuild with `--features experimental-perm` to enable."
            );
        }
    }

    #[cfg(feature = "experimental-perm")]
    let perm_strategy = PermStrategy::from_env();

    println!("Output  : {}", out_dir.display());
    #[cfg(feature = "experimental-perm")]
    println!(
        "Permute : {}{}",
        if use_permute { "on" } else { "off" },
        if use_permute {
            format!(" (strategy: {})", perm_strategy.name())
        } else {
            String::new()
        }
    );
    #[cfg(not(feature = "experimental-perm"))]
    println!("Permute : disabled (build without --features experimental-perm)");

    let t0 = Instant::now();
    let mut q8k_count = 0usize;
    let mut q4k_count = 0usize;
    let mut skipped_count = 0usize;

    for in_file in &input_files {
        println!("Processing: {}", in_file.display());
        let bytes = fs::read(in_file)?;
        let st = SafeTensors::deserialize(&bytes)?;
        println!("  Tensors: {}", st.len());

        for name in st.names() {
        let t = st.tensor(name)?;
        let shape = t.shape();
        if shape.len() != 2 || !is_target_weight(name) {
            skipped_count += 1;
            continue;
        }
        let (rows, k) = (shape[0], shape[1]);
        if k % QK_K != 0 {
            println!("skip (k % {QK_K} != 0): {name} [{rows} x {k}]");
            skipped_count += 1;
            continue;
        }

        let data_f32 = tensor_to_f32(t.data(), t.dtype())?;

        if is_q4k_layer(name) {
            // ── Q4K path ───────────────────────────────────────────────────
            println!("quantizing Q4K {name} ({rows} x {k})");
            let blocks = quantize_rows_q4k(rows, k, &data_f32)?;
            let (rmse, max_err, rel_err) = validate_q4k(&data_f32, &blocks, rows, k)?;
            println!("  RMSE: {:.6e}, Max Err: {:.6e}, Rel Err: {:.6e}", rmse, max_err, rel_err);
            if max_err > 1e-2 || rel_err > 0.01 {
                println!("    WARNING: High Q4K quantization error for {name}");
            }
            let block_bytes = blocks.len() * mem::size_of::<BlockQ4K>();
            println!("  Q4K size: {:.2} MB  ({:.0}% of Q8K)",
                block_bytes as f64 / 1_048_576.0,
                block_bytes as f64 / (blocks.len() * mem::size_of::<BlockQ8K>()) as f64 * 100.0);
            let out_path = out_dir.join(format!("{name}.q4k"));
            write_q4k(&out_path, rows, k, &blocks)?;
            q4k_count += 1;
        } else {
            // ── Q8K path (with optional permutation) ─────────────
            println!("quantizing Q8K {name} ({rows} x {k})");

            #[cfg(feature = "experimental-perm")]
            let (data_for_quant, maybe_perm): (Vec<f32>, Option<Vec<usize>>) = if use_permute {
                if let Some((layer_id, kind)) = parse_attention_proj(name) {
                    if kind == "o" {
                        (data_f32, None)
                    } else {
                        let perm = {
                            let mut cache = LAYER_PERM_CACHE
                                .get_or_init(|| Mutex::new(LayerPermCache::new()))
                                .lock()
                                .unwrap();
                            cache.get_or_compute(perm_strategy, layer_id, rows, k, &data_f32)?
                        };
                        let permuted = apply_column_permutation(rows, k, &data_f32, &perm);
                        (permuted, Some(perm))
                    }
                } else {
                    let perm = perm_strategy.build(rows, k, &data_f32)?;
                    let permuted = apply_column_permutation(rows, k, &data_f32, &perm);
                    (permuted, Some(perm))
                }
            } else {
                (data_f32, None)
            };

            #[cfg(not(feature = "experimental-perm"))]
            let (data_for_quant, maybe_perm): (Vec<f32>, Option<Vec<usize>>) = (data_f32, None);

            let blocks = quantize_rows_q8k(rows, k, &data_for_quant)?;

            let mse_matmul = validate_quantization(&data_for_quant, &blocks, k)?;
            let mse_direct = validate_quantization_direct(&data_for_quant, &blocks, k)?;
            let (rmse, max_err, rel_err) = compute_quantization_error_detailed(&data_for_quant, &blocks, rows, k)?;

            println!("  MSE (matmul): {:.6e}", mse_matmul);
            println!("  MSE (direct): {:.6e}", mse_direct);
            println!("  RMSE: {:.6e}, Max Err: {:.6e}, Rel Err: {:.6e}", rmse, max_err, rel_err);

            if max_err > 1e-2 || rel_err > 0.01 {
                println!("    WARNING: High quantization error detected!");
                println!("       Consider using block-wise permutation or reducing compression");
            }
            let diff = (mse_matmul - mse_direct).abs();
            if diff > 1e-6 {
                println!("    [INFO] Validation methods differ by {:.8e}", diff);
            }
            if mse_matmul > 1e-2 || mse_direct > 1e-2 {
                println!("    [WARN] High MSE detected - quantization may be lossy");
            }

            let out_path = out_dir.join(format!("{name}.q8k"));
            write_q8k(&out_path, rows, k, &blocks)?;
            #[cfg(feature = "experimental-perm")]
            if let Some(perm) = &maybe_perm {
                write_perm(&out_path, perm)?;
            }
            #[cfg(not(feature = "experimental-perm"))]
            let _ = &maybe_perm;
            q8k_count += 1;
        }
    }
    }

    println!(
        "Done in {:.2}s. Q8K: {q8k_count}, Q4K: {q4k_count}, skipped: {skipped_count}",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
