//! Pack Q8K/Q6K/Q4K quantized weights + original embeddings/norms into a single SafeTensors file
//!
//! Usage:
//!   cargo run --release -p tensor-tools --bin pack_q8k_safetensors \
//!     model.safetensors \
//!     quantized-weights/ \
//!     model-q8k-packed.safetensors
use anyhow::{bail, Context, Result};
use candle::quantized::k_quants::{BlockQ4K, BlockQ6K, BlockQ8K, GgmlType, QK_K};
#[cfg(feature = "experimental-q8k128")]
use phi3_mixed_quant::quant_q8k_128::BlockQ8K128;
#[cfg(feature = "experimental-perm")]
use phi3_mixed_quant::types::MAGIC_PERM;
use phi3_mixed_quant::types::{
    Q8KHeader, DTYPE_Q4K, DTYPE_Q6K, DTYPE_Q8K, HEADER_VERSION, MAGIC_Q4K, MAGIC_Q6K, MAGIC_Q8K,
};
#[cfg(feature = "experimental-q8k128")]
use phi3_mixed_quant::types::{DTYPE_Q8K_128, MAGIC_Q8K_128, QK_Q8K_128};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const _: () = assert!(std::mem::size_of::<Q8KHeader>() == 24);
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(std::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);
#[cfg(feature = "experimental-q8k128")]
const _: () = assert!(std::mem::size_of::<BlockQ8K128>() == 148);
const Q4K_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4K>();
const Q6K_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6K>();
const Q8K_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8K>();
#[cfg(feature = "experimental-q8k128")]
const Q8K128_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8K128>();

fn read_q8k_file(path: &Path) -> Result<(Vec<BlockQ8K>, Q8KHeader)> {
    let data = fs::read(path)?;
    let hdr_size = std::mem::size_of::<Q8KHeader>();
    if data.len() < hdr_size {
        bail!("Q8K file too small: {}", path.display());
    }
    let hdr = *bytemuck::from_bytes::<Q8KHeader>(&data[..hdr_size]);
    validate_header(&hdr, MAGIC_Q8K, DTYPE_Q8K, "Q8K", path, QK_K)?;
    let blocks_per_row = hdr.k as usize / QK_K;
    let total_blocks = hdr.out as usize * blocks_per_row;
    let expected_len = hdr_size + total_blocks * Q8K_BLOCK_SIZE;
    if data.len() != expected_len {
        bail!(
            "Q8K byte length mismatch in {}: got {}, expected {}",
            path.display(),
            data.len(),
            expected_len
        );
    }
    let mut blocks = vec![BlockQ8K::zeros(); total_blocks];
    let raw = &data[hdr_size..];
    unsafe {
        std::ptr::copy_nonoverlapping(
            raw.as_ptr(),
            blocks.as_mut_ptr() as *mut u8,
            total_blocks * std::mem::size_of::<BlockQ8K>(),
        );
    }
    Ok((blocks, hdr))
}

fn read_q6k_file(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    read_raw_quant_file(path, MAGIC_Q6K, DTYPE_Q6K, "Q6K", Q6K_BLOCK_SIZE)
}

fn read_q4k_file(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    read_raw_quant_file(path, MAGIC_Q4K, DTYPE_Q4K, "Q4K", Q4K_BLOCK_SIZE)
}

#[cfg(feature = "experimental-q8k128")]
fn read_q8k128_file(path: &Path) -> Result<(Vec<u8>, u32, u32)> {
    read_raw_quant_file_with_qk(
        path,
        MAGIC_Q8K_128,
        DTYPE_Q8K_128,
        "Q8K128",
        Q8K128_BLOCK_SIZE,
        QK_Q8K_128,
    )
}

fn read_raw_quant_file(
    path: &Path,
    expected_magic: u32,
    expected_dtype: u32,
    format: &str,
    block_size: usize,
) -> Result<(Vec<u8>, u32, u32)> {
    read_raw_quant_file_with_qk(
        path,
        expected_magic,
        expected_dtype,
        format,
        block_size,
        QK_K,
    )
}

fn read_raw_quant_file_with_qk(
    path: &Path,
    expected_magic: u32,
    expected_dtype: u32,
    format: &str,
    block_size: usize,
    block_elems: usize,
) -> Result<(Vec<u8>, u32, u32)> {
    let data = fs::read(path)?;
    let hdr_size = std::mem::size_of::<Q8KHeader>();
    if data.len() < hdr_size {
        bail!("{format} file too small: {}", path.display());
    }
    let hdr = *bytemuck::from_bytes::<Q8KHeader>(&data[..hdr_size]);
    validate_header(
        &hdr,
        expected_magic,
        expected_dtype,
        format,
        path,
        block_elems,
    )?;
    let expected_len = hdr_size + hdr.out as usize * hdr.blocks_per_row as usize * block_size;
    if data.len() != expected_len {
        bail!(
            "{format} byte length mismatch in {}: got {}, expected {}",
            path.display(),
            data.len(),
            expected_len
        );
    }
    // Strip header — return only raw block bytes
    Ok((data[hdr_size..].to_vec(), hdr.out, hdr.k))
}

fn validate_header(
    hdr: &Q8KHeader,
    expected_magic: u32,
    expected_dtype: u32,
    format: &str,
    path: &Path,
    block_elems: usize,
) -> Result<()> {
    if hdr.magic != expected_magic {
        bail!(
            "{format} bad magic in {}: got 0x{:08X}, expected 0x{:08X}",
            path.display(),
            hdr.magic,
            expected_magic
        );
    }
    if hdr.version != HEADER_VERSION {
        bail!(
            "{format} unsupported header version in {}: got {}, expected {}",
            path.display(),
            hdr.version,
            HEADER_VERSION
        );
    }
    if hdr.dtype != expected_dtype {
        bail!(
            "{format} dtype mismatch in {}: got 0x{:X}, expected 0x{:X}",
            path.display(),
            hdr.dtype,
            expected_dtype
        );
    }
    if hdr.out == 0 || hdr.k == 0 {
        bail!(
            "{format} invalid dimensions in {}: out={}, k={}",
            path.display(),
            hdr.out,
            hdr.k
        );
    }
    if !(hdr.k as usize).is_multiple_of(block_elems) {
        bail!(
            "{format} k mismatch in {}: {} is not divisible by {}",
            path.display(),
            hdr.k,
            block_elems
        );
    }
    let expected_blocks_per_row = hdr.k as usize / block_elems;
    if hdr.blocks_per_row as usize != expected_blocks_per_row {
        bail!(
            "{format} block-count mismatch in {}: header blocks_per_row={}, expected {}",
            path.display(),
            hdr.blocks_per_row,
            expected_blocks_per_row
        );
    }
    Ok(())
}

fn is_quantized_weight(name: &str) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    if name.contains("embed") || name.contains("norm") {
        return false;
    }
    true
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let orig_model: PathBuf = args
        .next()
        .context(
            "Usage: pack_q8k_safetensors <original.safetensors> <q8k-dir> <output.safetensors>",
        )?
        .into();
    let q8k_dir: PathBuf = args
        .next()
        .context(
            "Usage: pack_q8k_safetensors <original.safetensors> <q8k-dir> <output.safetensors>",
        )?
        .into();
    let output: PathBuf = args
        .next()
        .context(
            "Usage: pack_q8k_safetensors <original.safetensors> <q8k-dir> <output.safetensors>",
        )?
        .into();

    println!("Packing Q8K128/Q8K/Q6K/Q4K model into single SafeTensors file");
    println!("==============================================================");
    println!("Input  : {}", orig_model.display());
    println!("Q8K Dir: {}", q8k_dir.display());
    println!("Output : {}", output.display());
    println!();

    let t0 = Instant::now();
    let bytes = fs::read(&orig_model)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut tensors_to_save: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)> = HashMap::new();
    let mut quantized_count = 0usize;
    let mut preserved_count = 0usize;
    #[cfg(feature = "experimental-perm")]
    let mut perm_count = 0usize;
    #[cfg(not(feature = "experimental-perm"))]
    let mut perm_skipped = 0usize;
    #[cfg(not(feature = "experimental-q8k128"))]
    let mut q8k128_skipped = 0usize;

    let tensor_names = st.names();
    let has_final_norm = [
        "model.norm.weight",
        "model.final_layernorm.weight",
        "norm.weight",
    ]
    .iter()
    .any(|name| tensor_names.contains(name));
    if !has_final_norm {
        println!("WARNING: No final layer norm found in original model!");
        println!("         Expected one of: model.norm.weight, model.final_layernorm.weight, norm.weight");
        println!("         Available norms:");
        for name in &tensor_names {
            if name.contains("norm") {
                println!("           - {}", name);
            }
        }
        println!("         The model may not work correctly without a final norm.");
        println!();
    }

    for name in st.names() {
        let tensor = st.tensor(name)?;
        if is_quantized_weight(name) {
            let q8k128_path = q8k_dir.join(format!("{}.q8k128", name));
            let q4k_path = q8k_dir.join(format!("{}.q4k", name));
            let q6k_path = q8k_dir.join(format!("{}.q6k", name));
            let q8k_path = q8k_dir.join(format!("{}.q8k", name));
            #[cfg_attr(not(feature = "experimental-q8k128"), allow(unused_mut))]
            let mut existing = vec![
                ("Q4K", q4k_path.exists()),
                ("Q6K", q6k_path.exists()),
                ("Q8K", q8k_path.exists()),
            ];
            #[cfg(feature = "experimental-q8k128")]
            existing.insert(0, ("Q8K128", q8k128_path.exists()));
            #[cfg(not(feature = "experimental-q8k128"))]
            if q8k128_path.exists() {
                q8k128_skipped += 1;
            }
            let existing_count = existing.iter().filter(|(_, exists)| *exists).count();
            if existing_count > 1 {
                let formats = existing
                    .iter()
                    .filter_map(|(format, exists)| exists.then_some(*format))
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!("Multiple quantized formats found for {name}: {formats}. Remove stale files before packing.");
            }

            if cfg!(feature = "experimental-q8k128") && q8k128_path.exists() {
                #[cfg(feature = "experimental-q8k128")]
                {
                    let (block_bytes, out, k) = read_q8k128_file(&q8k128_path)?;
                    let blob_len = block_bytes.len();
                    tensors_to_save.insert(
                        format!("{}.q8k128", name),
                        (block_bytes, vec![blob_len], Dtype::U8),
                    );
                    let meta_bytes: Vec<u8> = [out as i32, k as i32]
                        .iter()
                        .flat_map(|&x| x.to_le_bytes())
                        .collect();
                    tensors_to_save.insert(
                        format!("{}.q8k128_meta", name),
                        (meta_bytes, vec![2], Dtype::I32),
                    );
                    quantized_count += 1;
                    println!("Q8K128 packed: {} [{} x {}]", name, out, k);
                }
            } else if q4k_path.exists() {
                let (block_bytes, out, k) = read_q4k_file(&q4k_path)?;
                let blob_len = block_bytes.len();
                tensors_to_save.insert(
                    format!("{}.q4k", name),
                    (block_bytes, vec![blob_len], Dtype::U8),
                );
                let meta_bytes: Vec<u8> = [out as i32, k as i32]
                    .iter()
                    .flat_map(|&x| x.to_le_bytes())
                    .collect();
                tensors_to_save.insert(
                    format!("{}.q4k_meta", name),
                    (meta_bytes, vec![2], Dtype::I32),
                );
                quantized_count += 1;
                println!("Q4K packed: {} [{} x {}]", name, out, k);
            } else if q6k_path.exists() {
                let (block_bytes, out, k) = read_q6k_file(&q6k_path)?;
                let blob_len = block_bytes.len();
                tensors_to_save.insert(
                    format!("{}.q6k", name),
                    (block_bytes, vec![blob_len], Dtype::U8),
                );
                let meta_bytes: Vec<u8> = [out as i32, k as i32]
                    .iter()
                    .flat_map(|&x| x.to_le_bytes())
                    .collect();
                tensors_to_save.insert(
                    format!("{}.q6k_meta", name),
                    (meta_bytes, vec![2], Dtype::I32),
                );
                quantized_count += 1;
                println!("Q6K packed: {} [{} x {}]", name, out, k);
            } else if q8k_path.exists() {
                let (blocks, hdr) = read_q8k_file(&q8k_path)?;
                let blocks_bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        std::mem::size_of_val(blocks.as_slice()),
                    )
                }
                .to_vec();
                let full_size = std::mem::size_of::<Q8KHeader>() + blocks_bytes.len();
                let mut full_data = Vec::with_capacity(full_size);
                full_data.extend_from_slice(bytemuck::bytes_of(&hdr));
                full_data.extend_from_slice(&blocks_bytes);
                tensors_to_save.insert(
                    format!("{}.q8k", name),
                    (full_data, vec![full_size], Dtype::U8),
                );
                let metadata_bytes: Vec<u8> = [
                    hdr.out as i32,
                    hdr.k as i32,
                    std::mem::size_of::<Q8KHeader>() as i32,
                ]
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
                tensors_to_save.insert(
                    format!("{}.q8k_meta", name),
                    (metadata_bytes, vec![3], Dtype::I32),
                );
                let perm_path = q8k_path.with_extension("perm");
                #[cfg(feature = "experimental-perm")]
                if perm_path.exists() {
                    let perm_bytes = fs::read(&perm_path)?;
                    if perm_bytes.len() < 8 {
                        bail!("Invalid perm file: {}", perm_path.display());
                    }
                    let magic = u32::from_le_bytes(perm_bytes[0..4].try_into().unwrap());
                    if magic != MAGIC_PERM {
                        bail!("Invalid perm magic: {}", perm_path.display());
                    }
                    let k = u32::from_le_bytes(perm_bytes[4..8].try_into().unwrap()) as usize;
                    let expected_len = 8 + 4 * k;
                    if perm_bytes.len() != expected_len {
                        bail!(
                            "Permutation size mismatch for {}: got {}, expected {}",
                            name,
                            perm_bytes.len(),
                            expected_len
                        );
                    }
                    tensors_to_save.insert(
                        format!("{}.perm", name),
                        (perm_bytes, vec![expected_len], Dtype::U8),
                    );
                    perm_count += 1;
                    println!("  Packed permutation for {}", name);
                }
                #[cfg(not(feature = "experimental-perm"))]
                if perm_path.exists() {
                    perm_skipped += 1;
                }
                quantized_count += 1;
                println!("Q8K packed: {} [{} x {}]", name, hdr.out, hdr.k);
            } else {
                let data = tensor.data().to_vec();
                let shape = tensor.shape().to_vec();
                let dtype = tensor.dtype();
                println!(
                    "WARNING: Missing Q8K/Q6K/Q4K, kept original: {} {:?}",
                    name, shape
                );
                tensors_to_save.insert(name.to_string(), (data, shape, dtype));
                preserved_count += 1;
            }
        } else {
            let data = tensor.data().to_vec();
            let shape = tensor.shape().to_vec();
            let dtype = tensor.dtype();
            println!("Preserved: {} {:?}", name, shape);
            tensors_to_save.insert(name.to_string(), (data, shape, dtype));
            preserved_count += 1;
        }
    }

    let final_norm_packed = tensors_to_save.keys().any(|k| {
        k == "model.norm.weight" || k == "model.final_layernorm.weight" || k == "norm.weight"
    });
    if !final_norm_packed {
        println!();
        println!("ERROR: Final layer norm not included in packed model!");
        println!("       The model will likely fail during inference.");
        println!();
    }

    println!();
    println!("Writing packed model...");
    let tensor_views: Vec<(String, TensorView)> = tensors_to_save
        .iter()
        .map(|(name, (data, shape, dtype))| {
            let view = TensorView::new(*dtype, shape.clone(), data.as_slice())
                .expect("Failed to create tensor view");
            (name.clone(), view)
        })
        .collect();
    let tensor_refs: Vec<(&str, TensorView)> = tensor_views
        .iter()
        .map(|(name, view)| (name.as_str(), view.clone()))
        .collect();
    safetensors::serialize_to_file(tensor_refs, None, &output)?;

    println!();
    println!("Done in {:.2}s", t0.elapsed().as_secs_f32());
    println!("Statistics:");
    println!("   - Quantized weights : {}", quantized_count);
    #[cfg(feature = "experimental-perm")]
    println!("   - Permutation files : {}", perm_count);
    println!("   - Preserved tensors : {}", preserved_count);
    println!("   - Total tensors     : {}", tensors_to_save.len());
    let output_size = fs::metadata(&output)?.len();
    println!(
        "   - Output size       : {:.2} MB",
        output_size as f64 / 1_048_576.0
    );
    #[cfg(feature = "experimental-perm")]
    if perm_count > 0 {
        println!();
        println!("Permutation files included — loader will detect and apply them automatically.");
    }
    #[cfg(not(feature = "experimental-perm"))]
    if perm_skipped > 0 {
        println!();
        println!(
            "note: {} `.perm` file(s) on disk were skipped — \
             rebuild with `--features experimental-perm` to include them.",
            perm_skipped
        );
    }
    #[cfg(not(feature = "experimental-q8k128"))]
    if q8k128_skipped > 0 {
        println!();
        println!(
            "note: {} `.q8k128` file(s) on disk were skipped — \
             rebuild with `--features experimental-q8k128` to include them.",
            q8k128_skipped
        );
    }
    Ok(())
}
