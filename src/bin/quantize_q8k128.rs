//! Experimental Q8K128 quantizer for Phi-3 SafeTensors shards.
//!
//! Build with:
//!   cargo build --release --features experimental-q8k128 --bin quantize_q8k128
//!
//! Routing:
//!   CANDLE_Q8K128_POLICY=layer0-qkv  (default)
//!   CANDLE_Q8K128_POLICY=early-qkv   (layers 0..3 qkv_proj)
//!   CANDLE_Q8K128_POLICY=qkv         (all qkv_proj)
//!   CANDLE_Q8K128_POLICY=attn        (all self_attn qkv/o projections)
//!   CANDLE_Q8K128_POLICY=q8k         (all layers that baseline routes to Q8K)
//!   CANDLE_Q8K128_POLICY=all         (all quantized weights, including down_proj)
//!   CANDLE_Q8K128_POLICY=off         (baseline Q8K/Q4K output)

#![cfg(feature = "experimental-q8k128")]

use anyhow::{bail, Context, Result};
use candle::quantized::k_quants::{matmul, BlockQ4K, BlockQ8K, GgmlType, QK_K};
use half::{bf16, f16};
use phi3_mixed_quant::quant_q8k_128::{self, BlockQ8K128};
use phi3_mixed_quant::types::{
    Q8KHeader, DTYPE_Q4K, DTYPE_Q8K, DTYPE_Q8K_128, HEADER_VERSION, MAGIC_Q4K, MAGIC_Q8K,
    MAGIC_Q8K_128, QK_Q8K_128,
};
use safetensors::tensor::{Dtype, SafeTensors};
use std::{
    fs,
    io::{BufWriter, Write},
    mem,
    path::{Path, PathBuf},
    time::Instant,
};

const _: () = assert!(std::mem::size_of::<Q8KHeader>() == 24);
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);
const _: () = assert!(std::mem::size_of::<BlockQ8K128>() == 148);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantFormat {
    Q8K128,
    Q8K,
    Q4K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Q8K128Policy {
    Off,
    Layer0Qkv,
    EarlyQkv,
    Qkv,
    Attn,
    Q8K,
    All,
}

impl Q8K128Policy {
    fn from_env() -> Result<Self> {
        let raw = std::env::var("CANDLE_Q8K128_POLICY")
            .unwrap_or_else(|_| "layer0-qkv".to_string())
            .to_ascii_lowercase();
        match raw.as_str() {
            "off" | "0" | "false" => Ok(Self::Off),
            "layer0-qkv" | "layer0_qkv" | "l0-qkv" | "l0_qkv" => Ok(Self::Layer0Qkv),
            "early-qkv" | "early_qkv" | "layers0-3-qkv" | "0-3-qkv" => Ok(Self::EarlyQkv),
            "qkv" | "all-qkv" | "all_qkv" => Ok(Self::Qkv),
            "attn" | "attention" | "self-attn" | "self_attn" => Ok(Self::Attn),
            "q8k" | "q8k-layers" | "non-down" => Ok(Self::Q8K),
            "all" => Ok(Self::All),
            other => bail!(
                "unsupported CANDLE_Q8K128_POLICY={other:?}; expected off, layer0-qkv, early-qkv, qkv, attn, q8k, or all"
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Layer0Qkv => "layer0-qkv",
            Self::EarlyQkv => "early-qkv",
            Self::Qkv => "qkv",
            Self::Attn => "attn",
            Self::Q8K => "q8k",
            Self::All => "all",
        }
    }
}

fn is_target_weight(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains("embed_tokens") && !name.contains("norm")
}

fn is_q4k_layer(name: &str) -> bool {
    name.contains("mlp.down_proj")
}

fn layer_id(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

fn is_q8k128_layer(name: &str, policy: Q8K128Policy) -> bool {
    match policy {
        Q8K128Policy::Off => false,
        Q8K128Policy::Layer0Qkv => layer_id(name) == Some(0) && name.contains("self_attn.qkv_proj"),
        Q8K128Policy::EarlyQkv => {
            matches!(layer_id(name), Some(0..=3)) && name.contains("self_attn.qkv_proj")
        }
        Q8K128Policy::Qkv => name.contains("self_attn.qkv_proj"),
        Q8K128Policy::Attn => {
            name.contains("self_attn.qkv_proj") || name.contains("self_attn.o_proj")
        }
        Q8K128Policy::Q8K => !is_q4k_layer(name),
        Q8K128Policy::All => true,
    }
}

fn choose_quant_format(name: &str, policy: Q8K128Policy) -> QuantFormat {
    if is_q8k128_layer(name, policy) {
        QuantFormat::Q8K128
    } else if is_q4k_layer(name) {
        QuantFormat::Q4K
    } else {
        QuantFormat::Q8K
    }
}

fn quantize_rows_q8k(rows: usize, k: usize, data: &[f32]) -> Result<Vec<BlockQ8K>> {
    if k % QK_K != 0 {
        bail!("Q8K inner dim {k} is not multiple of {QK_K}");
    }
    if data.len() != rows * k {
        bail!(
            "Q8K source length mismatch: got {}, expected {}",
            data.len(),
            rows * k
        );
    }
    let blocks_per_row = k / QK_K;
    let mut blocks = vec![BlockQ8K::zeros(); rows * blocks_per_row];
    for row in 0..rows {
        let src = &data[row * k..(row + 1) * k];
        let dst = &mut blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        BlockQ8K::from_float(src, dst);
    }
    Ok(blocks)
}

fn quantize_rows_q4k(rows: usize, k: usize, data: &[f32]) -> Result<Vec<BlockQ4K>> {
    if k % QK_K != 0 {
        bail!("Q4K inner dim {k} is not multiple of {QK_K}");
    }
    if data.len() != rows * k {
        bail!(
            "Q4K source length mismatch: got {}, expected {}",
            data.len(),
            rows * k
        );
    }
    let blocks_per_row = k / QK_K;
    let mut blocks = vec![BlockQ4K::zeros(); rows * blocks_per_row];
    for row in 0..rows {
        let src = &data[row * k..(row + 1) * k];
        let dst = &mut blocks[row * blocks_per_row..(row + 1) * blocks_per_row];
        BlockQ4K::from_float(src, dst);
    }
    Ok(blocks)
}

fn validate_q8k(
    original: &[f32],
    blocks: &[BlockQ8K],
    rows: usize,
    k: usize,
) -> Result<(f64, f64, f64)> {
    let mut reconstructed = vec![0f32; rows * k];
    BlockQ8K::to_float(blocks, &mut reconstructed);
    reconstruction_stats_from_slices(original, &reconstructed)
}

fn validate_q4k(
    original: &[f32],
    blocks: &[BlockQ4K],
    rows: usize,
    k: usize,
) -> Result<(f64, f64, f64)> {
    let mut reconstructed = vec![0f32; rows * k];
    BlockQ4K::to_float(blocks, &mut reconstructed);
    reconstruction_stats_from_slices(original, &reconstructed)
}

fn reconstruction_stats_from_slices(
    original: &[f32],
    reconstructed: &[f32],
) -> Result<(f64, f64, f64)> {
    if original.len() != reconstructed.len() {
        bail!(
            "reconstruction length mismatch: original={}, reconstructed={}",
            original.len(),
            reconstructed.len()
        );
    }
    let mut sq = 0f64;
    let mut max_abs = 0f64;
    let mut rel_sum = 0f64;
    let mut rel_count = 0u64;
    for (&orig, &deq) in original.iter().zip(reconstructed.iter()) {
        let abs_err = (orig - deq).abs() as f64;
        sq += abs_err * abs_err;
        max_abs = max_abs.max(abs_err);
        if orig.abs() > 1e-10 {
            rel_sum += abs_err / orig.abs() as f64;
            rel_count += 1;
        }
    }
    Ok((
        (sq / original.len().max(1) as f64).sqrt(),
        max_abs,
        rel_sum / rel_count.max(1) as f64,
    ))
}

fn validate_q8k128_matmul(
    original: &[f32],
    blocks: &[BlockQ8K128],
    rows: usize,
    k: usize,
) -> Result<f32> {
    let x: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) / k as f32).collect();
    let mut expected = vec![0f32; rows];
    for row in 0..rows {
        expected[row] = original[row * k..(row + 1) * k]
            .iter()
            .zip(x.iter())
            .map(|(&w, &xv)| w * xv)
            .sum();
    }
    let mut actual = vec![0f32; rows];
    quant_q8k_128::matmul_scalar((1, k, rows), &x, blocks, &mut actual)
        .map_err(|e| anyhow::anyhow!("Q8K128 matmul validation failed: {e}"))?;
    let mse = expected
        .iter()
        .zip(actual.iter())
        .map(|(&e, &a)| {
            let diff = e - a;
            diff * diff
        })
        .sum::<f32>()
        / rows.max(1) as f32;
    Ok(mse)
}

fn validate_q8k_matmul(
    original: &[f32],
    blocks: &[BlockQ8K],
    rows: usize,
    k: usize,
) -> Result<f32> {
    let x: Vec<f32> = (0..k).map(|i| (i as f32 + 1.0) / k as f32).collect();
    let mut expected = vec![0f32; rows];
    for row in 0..rows {
        expected[row] = original[row * k..(row + 1) * k]
            .iter()
            .zip(x.iter())
            .map(|(&w, &xv)| w * xv)
            .sum();
    }
    let mut actual = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &x, blocks, &mut actual)
        .map_err(|e| anyhow::anyhow!("Q8K matmul validation failed: {e}"))?;
    let mse = expected
        .iter()
        .zip(actual.iter())
        .map(|(&e, &a)| {
            let diff = e - a;
            diff * diff
        })
        .sum::<f32>()
        / rows.max(1) as f32;
    Ok(mse)
}

fn write_q8k128(path: &Path, rows: usize, k: usize, blocks: &[BlockQ8K128]) -> Result<()> {
    if k % QK_Q8K_128 != 0 {
        bail!("Q8K128 inner dim {k} is not multiple of {QK_Q8K_128}");
    }
    let expected_blocks = rows * (k / QK_Q8K_128);
    if blocks.len() != expected_blocks {
        bail!(
            "Q8K128 block count mismatch for {}: got {}, expected {}",
            path.display(),
            blocks.len(),
            expected_blocks
        );
    }
    let header = Q8KHeader {
        magic: MAGIC_Q8K_128,
        version: HEADER_VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_Q8K_128) as u32,
        dtype: DTYPE_Q8K_128,
    };
    let mut writer = BufWriter::new(fs::File::create(path)?);
    writer.write_all(bytemuck::bytes_of(&header))?;
    let raw = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * mem::size_of::<BlockQ8K128>(),
        )
    };
    writer.write_all(raw)?;
    writer.flush()?;
    Ok(())
}

fn write_q8k(path: &Path, rows: usize, k: usize, blocks: &[BlockQ8K]) -> Result<()> {
    let expected_blocks = rows * (k / QK_K);
    if blocks.len() != expected_blocks {
        bail!(
            "Q8K block count mismatch for {}: got {}, expected {}",
            path.display(),
            blocks.len(),
            expected_blocks
        );
    }
    let header = Q8KHeader {
        magic: MAGIC_Q8K,
        version: HEADER_VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_K) as u32,
        dtype: DTYPE_Q8K,
    };
    write_blocks(path, &header, blocks)
}

fn write_q4k(path: &Path, rows: usize, k: usize, blocks: &[BlockQ4K]) -> Result<()> {
    let expected_blocks = rows * (k / QK_K);
    if blocks.len() != expected_blocks {
        bail!(
            "Q4K block count mismatch for {}: got {}, expected {}",
            path.display(),
            blocks.len(),
            expected_blocks
        );
    }
    let header = Q8KHeader {
        magic: MAGIC_Q4K,
        version: HEADER_VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_K) as u32,
        dtype: DTYPE_Q4K,
    };
    write_blocks(path, &header, blocks)
}

fn write_blocks<T>(path: &Path, header: &Q8KHeader, blocks: &[T]) -> Result<()> {
    let mut writer = BufWriter::new(fs::File::create(path)?);
    writer.write_all(bytemuck::bytes_of(header))?;
    let raw = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks))
    };
    writer.write_all(raw)?;
    writer.flush()?;
    Ok(())
}

fn tensor_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                bail!("F32 byte length {} is not divisible by 4", bytes.len());
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                bail!("F16 byte length {} is not divisible by 2", bytes.len());
            }
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect()
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                bail!("BF16 byte length {} is not divisible by 2", bytes.len());
            }
            bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
                .collect()
        }
        other => bail!("unsupported dtype {other:?}"),
    })
}

fn is_hf_model_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    parts.len() == 2
        && !s.starts_with('.')
        && !s.starts_with('/')
        && !s.starts_with('~')
        && !Path::new(s).exists()
        && !parts[0].is_empty()
        && !parts[1].is_empty()
        && !parts[1].contains('.')
}

fn download_safetensors(model_id: &str, filename: Option<&str>) -> Result<Vec<PathBuf>> {
    use hf_hub::api::sync::Api;

    println!("Downloading from Hugging Face: {model_id}");
    let api = Api::new().context("Failed to create HF Hub API client")?;
    let repo = api.model(model_id.to_string());

    if let Some(name) = filename {
        let path = repo
            .get(name)
            .with_context(|| format!("Failed to download {name} from {model_id}"))?;
        return Ok(vec![path]);
    }

    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }

    let index_path = repo
        .get("model.safetensors.index.json")
        .context("No model.safetensors or model.safetensors.index.json found")?;
    let index_bytes = fs::read(&index_path)?;
    let index: serde_json::Value =
        serde_json::from_slice(&index_bytes).context("parse model.safetensors.index.json")?;
    let mut shard_files: Vec<String> = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("model index has no weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    let mut paths = Vec::with_capacity(shard_files.len());
    for shard in shard_files {
        let path = repo
            .get(&shard)
            .with_context(|| format!("download shard {shard}"))?;
        println!("  Downloaded: {} ({})", shard, path.display());
        paths.push(path);
    }
    Ok(paths)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!(
            "Usage: quantize_q8k128 <input.safetensors | org/model> <output_dir> [--file <name>]\n\
             Example: CANDLE_Q8K128_POLICY=layer0-qkv quantize_q8k128 model-00001-of-00002.safetensors ./quantized-q8k128"
        );
    }

    let input_arg = &args[0];
    let out_dir: PathBuf = args[1].clone().into();
    let explicit_file = if let Some(pos) = args.iter().position(|a| a == "--file") {
        Some(
            args.get(pos + 1)
                .context("--file requires a filename argument")?
                .as_str(),
        )
    } else {
        None
    };

    let input_files = if is_hf_model_id(input_arg) {
        download_safetensors(input_arg, explicit_file)?
    } else {
        let path = PathBuf::from(input_arg);
        if !path.exists() {
            bail!("Input file does not exist: {}", path.display());
        }
        vec![path]
    };

    fs::create_dir_all(&out_dir)?;

    if std::env::var("CANDLE_Q8K_PERMUTE").is_ok() {
        eprintln!("warning: quantize_q8k128 does not write permutation files");
    }

    let policy = Q8K128Policy::from_env()?;
    println!("Output  : {}", out_dir.display());
    println!("Q8K128  : policy {}", policy.name());

    let t0 = Instant::now();
    let mut q8k128_count = 0usize;
    let mut q8k_count = 0usize;
    let mut q4k_count = 0usize;
    let mut skipped_count = 0usize;

    for input in &input_files {
        println!("Processing: {}", input.display());
        let bytes = fs::read(input)?;
        let st = SafeTensors::deserialize(&bytes)?;
        println!("  Tensors: {}", st.len());

        for name in st.names() {
            let tensor = st.tensor(name)?;
            let shape = tensor.shape();
            if shape.len() != 2 || !is_target_weight(name) {
                skipped_count += 1;
                continue;
            }
            let (rows, k) = (shape[0], shape[1]);
            let format = choose_quant_format(name, policy);
            let required_block = match format {
                QuantFormat::Q8K128 => QK_Q8K_128,
                QuantFormat::Q8K | QuantFormat::Q4K => QK_K,
            };
            if k % required_block != 0 {
                println!("skip (k % {required_block} != 0): {name} [{rows} x {k}]");
                skipped_count += 1;
                continue;
            }

            let data_f32 = tensor_to_f32(tensor.data(), tensor.dtype())
                .with_context(|| format!("convert {name} to f32"))?;

            match format {
                QuantFormat::Q8K128 => {
                    println!("quantizing Q8K128 {name} ({rows} x {k})");
                    let blocks = quant_q8k_128::quantize_rows(rows, k, &data_f32)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let (rmse, max_err, rel_err) =
                        quant_q8k_128::reconstruction_stats(&data_f32, &blocks, rows, k)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let mse = validate_q8k128_matmul(&data_f32, &blocks, rows, k)?;
                    println!(
                        "  RMSE: {:.6e}, Max Err: {:.6e}, Rel Err: {:.6e}, MSE(matmul): {:.6e}",
                        rmse, max_err, rel_err, mse
                    );
                    let out_path = out_dir.join(format!("{name}.q8k128"));
                    write_q8k128(&out_path, rows, k, &blocks)?;
                    q8k128_count += 1;
                }
                QuantFormat::Q8K => {
                    println!("quantizing Q8K {name} ({rows} x {k})");
                    let blocks = quantize_rows_q8k(rows, k, &data_f32)?;
                    let (rmse, max_err, rel_err) = validate_q8k(&data_f32, &blocks, rows, k)?;
                    let mse = validate_q8k_matmul(&data_f32, &blocks, rows, k)?;
                    println!(
                        "  RMSE: {:.6e}, Max Err: {:.6e}, Rel Err: {:.6e}, MSE(matmul): {:.6e}",
                        rmse, max_err, rel_err, mse
                    );
                    let out_path = out_dir.join(format!("{name}.q8k"));
                    write_q8k(&out_path, rows, k, &blocks)?;
                    q8k_count += 1;
                }
                QuantFormat::Q4K => {
                    println!("quantizing Q4K {name} ({rows} x {k})");
                    let blocks = quantize_rows_q4k(rows, k, &data_f32)?;
                    let (rmse, max_err, rel_err) = validate_q4k(&data_f32, &blocks, rows, k)?;
                    println!(
                        "  RMSE: {:.6e}, Max Err: {:.6e}, Rel Err: {:.6e}",
                        rmse, max_err, rel_err
                    );
                    let out_path = out_dir.join(format!("{name}.q4k"));
                    write_q4k(&out_path, rows, k, &blocks)?;
                    q4k_count += 1;
                }
            }
        }
    }

    println!(
        "Done in {:.2}s. Q8K128: {q8k128_count}, Q8K: {q8k_count}, Q4K: {q4k_count}, skipped: {skipped_count}",
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{choose_quant_format, Q8K128Policy, QuantFormat};

    #[test]
    fn default_policy_targets_only_layer0_qkv() {
        assert_eq!(
            choose_quant_format(
                "model.layers.0.self_attn.qkv_proj.weight",
                Q8K128Policy::Layer0Qkv
            ),
            QuantFormat::Q8K128
        );
        assert_eq!(
            choose_quant_format(
                "model.layers.1.self_attn.qkv_proj.weight",
                Q8K128Policy::Layer0Qkv
            ),
            QuantFormat::Q8K
        );
        assert_eq!(
            choose_quant_format(
                "model.layers.0.mlp.down_proj.weight",
                Q8K128Policy::Layer0Qkv
            ),
            QuantFormat::Q4K
        );
    }

    #[test]
    fn q8k_policy_keeps_down_proj_at_q4k() {
        assert_eq!(
            choose_quant_format("model.layers.0.self_attn.o_proj.weight", Q8K128Policy::Q8K),
            QuantFormat::Q8K128
        );
        assert_eq!(
            choose_quant_format("model.layers.0.mlp.down_proj.weight", Q8K128Policy::Q8K),
            QuantFormat::Q4K
        );
    }
}
