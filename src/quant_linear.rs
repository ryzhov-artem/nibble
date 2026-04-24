use candle::quantized::k_quants::{matmul, BlockQ4K, BlockQ8K, GgmlType, QK_K};
use candle::{DType, Tensor};
use safetensors::tensor::{Dtype, SafeTensors};
use std::{fs, path::Path};

use crate::types::{Q8KHeader, MAGIC_Q8K, MAGIC_PERM};

#[allow(dead_code)]
pub fn read_q8k(path: &Path) -> candle::Result<(Vec<BlockQ8K>, Q8KHeader)> {
    let data = fs::read(path)?;
    if data.len() < std::mem::size_of::<Q8KHeader>() {
        candle::bail!("file too small: {}", path.display());
    }
    let hdr = *bytemuck::from_bytes::<Q8KHeader>(&data[..std::mem::size_of::<Q8KHeader>()]);
    if hdr.magic != MAGIC_Q8K {
        candle::bail!("bad magic in {}", path.display());
    }

    let calculated_blocks_per_row = hdr.k as usize / QK_K;
    let total_blocks = (hdr.out as usize) * calculated_blocks_per_row;
    let header_size = std::mem::size_of::<Q8KHeader>();
    let expected_size = header_size + total_blocks * std::mem::size_of::<BlockQ8K>();

    if data.len() != expected_size {
        candle::bail!("Size mismatch: file={}, expected={}", data.len(), expected_size);
    }

    let mut blocks = vec![BlockQ8K::zeros(); total_blocks];
    let raw = &data[header_size..];
    unsafe {
        std::ptr::copy_nonoverlapping(
            raw.as_ptr(),
            blocks.as_mut_ptr() as *mut u8,
            total_blocks * std::mem::size_of::<BlockQ8K>(),
        );
    }
    Ok((blocks, hdr))
}

pub fn is_packed_format(st: &SafeTensors) -> bool {
    st.names().iter().any(|name| name.ends_with(".q8k") || name.ends_with(".q4k"))
}

/// Holds quantized weight blocks for a single linear layer.
/// Variants correspond to the quantization format used during packing.
pub enum QuantBlocks {
    Q8K(Vec<BlockQ8K>),
    Q4K(Vec<BlockQ4K>),
}

pub struct QuantLinear {
    pub(crate) blocks: QuantBlocks,
    pub(crate) out: usize,
    pub(crate) k: usize,
    pub(crate) name: String,
    pub(crate) perm: Option<Vec<usize>>,
}

impl QuantLinear {
    /// Load a Q8K-quantized layer from a packed safetensors file.
    /// Expects `name.q8k`, `name.q8k_meta` (and optionally `name.perm`) keys.
    pub fn load_from_packed_safetensors(st: &SafeTensors, name: &str) -> candle::Result<Self> {
        let q8k_name = format!("{}.q8k", name);
        let meta_name = format!("{}.q8k_meta", name);

        let meta_tensor = st.tensor(&meta_name)
            .map_err(|_| candle::Error::Msg(format!("Missing metadata: {}", meta_name)))?;

        if meta_tensor.dtype() != Dtype::I32 {
            candle::bail!("Expected I32 metadata, got {:?}", meta_tensor.dtype());
        }

        let meta_data = meta_tensor.data();
        let out = i32::from_le_bytes(meta_data[0..4].try_into().unwrap()) as usize;
        let k = i32::from_le_bytes(meta_data[4..8].try_into().unwrap()) as usize;
        let header_size = i32::from_le_bytes(meta_data[8..12].try_into().unwrap()) as usize;

        let tensor = st.tensor(&q8k_name)
            .map_err(|_| candle::Error::Msg(format!("Missing quantized tensor: {}", q8k_name)))?;

        if tensor.dtype() != Dtype::U8 {
            candle::bail!("Expected U8 dtype for Q8K tensor, got {:?}", tensor.dtype());
        }

        let data = tensor.data();
        if data.len() < header_size {
            candle::bail!("Data too small for header");
        }
        let hdr = *bytemuck::from_bytes::<Q8KHeader>(&data[..header_size]);
        if hdr.magic != MAGIC_Q8K {
            candle::bail!("Invalid Q8K magic in packed tensor");
        }

        let blocks_per_row = k / QK_K;
        let total_blocks = out * blocks_per_row;
        let mut blocks = vec![BlockQ8K::zeros(); total_blocks];
        let blocks_data = &data[header_size..];
        unsafe {
            std::ptr::copy_nonoverlapping(
                blocks_data.as_ptr(),
                blocks.as_mut_ptr() as *mut u8,
                total_blocks * std::mem::size_of::<BlockQ8K>(),
            );
        }

        let perm = Self::load_perm(st, name)?;

        Ok(Self {
            blocks: QuantBlocks::Q8K(blocks),
            out,
            k,
            name: name.to_string(),
            perm,
        })
    }

    /// Load a Q4K-quantized layer from a packed safetensors file.
    /// Expects `name.q4k` (raw BlockQ4K bytes) and `name.q4k_meta` ([out, k] as I32) keys.
    pub fn load_q4k_from_packed_safetensors(st: &SafeTensors, name: &str) -> candle::Result<Self> {
        let q4k_name = format!("{}.q4k", name);
        let meta_name = format!("{}.q4k_meta", name);

        let meta_tensor = st.tensor(&meta_name)
            .map_err(|_| candle::Error::Msg(format!("Missing Q4K metadata: {}", meta_name)))?;

        if meta_tensor.dtype() != Dtype::I32 {
            candle::bail!("Expected I32 Q4K metadata, got {:?}", meta_tensor.dtype());
        }

        let meta_data = meta_tensor.data();
        if meta_data.len() < 8 {
            candle::bail!("Q4K metadata too short for {}", name);
        }
        let out = i32::from_le_bytes(meta_data[0..4].try_into().unwrap()) as usize;
        let k = i32::from_le_bytes(meta_data[4..8].try_into().unwrap()) as usize;

        let tensor = st.tensor(&q4k_name)
            .map_err(|_| candle::Error::Msg(format!("Missing Q4K tensor: {}", q4k_name)))?;

        if tensor.dtype() != Dtype::U8 {
            candle::bail!("Expected U8 dtype for Q4K tensor, got {:?}", tensor.dtype());
        }

        let data = tensor.data();
        let blocks_per_row = k / QK_K;
        let total_blocks = out * blocks_per_row;
        let expected_bytes = total_blocks * std::mem::size_of::<BlockQ4K>();

        if data.len() != expected_bytes {
            candle::bail!(
                "Q4K size mismatch for {}: file={}, expected={}",
                name,
                data.len(),
                expected_bytes
            );
        }

        let mut blocks = vec![BlockQ4K::zeros(); total_blocks];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                blocks.as_mut_ptr() as *mut u8,
                expected_bytes,
            );
        }

        let perm = Self::load_perm(st, name)?;

        Ok(Self {
            blocks: QuantBlocks::Q4K(blocks),
            out,
            k,
            name: name.to_string(),
            perm,
        })
    }

    /// Shared permutation loader used by both Q8K and Q4K constructors.
    fn load_perm(st: &SafeTensors, name: &str) -> candle::Result<Option<Vec<usize>>> {
        let perm_name = format!("{}.perm", name);
        if let Ok(perm_tensor) = st.tensor(&perm_name) {
            let perm_data = perm_tensor.data();
            if perm_data.len() < 8 {
                candle::bail!("Invalid permutation data for {}", name);
            }
            let magic = u32::from_le_bytes(perm_data[0..4].try_into().unwrap());
            if magic != MAGIC_PERM {
                candle::bail!("Invalid permutation magic for {}", name);
            }
            let k_perm = u32::from_le_bytes(perm_data[4..8].try_into().unwrap()) as usize;
            let expected_len = 8 + 4 * k_perm;
            if perm_data.len() != expected_len {
                candle::bail!("Permutation size mismatch for {}", name);
            }
            let mut perm_indices = Vec::with_capacity(k_perm);
            for i in 0..k_perm {
                let off = 8 + 4 * i;
                let idx = u32::from_le_bytes(perm_data[off..off + 4].try_into().unwrap()) as usize;
                perm_indices.push(idx);
            }
            Ok(Some(perm_indices))
        } else {
            Ok(None)
        }
    }

    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        match x.dims() {
            [b, k] => self.forward_2d(x, *b, *k),
            [b, s, k] => {
                let (b, s, k) = (*b, *s, *k);
                let flat = x.reshape((b * s, k))?;
                let y = self.forward_2d(&flat, b * s, k)?;
                y.reshape((b, s, self.out))
            }
            _ => candle::bail!("{}: unsupported input dims {:?}", self.name, x.dims()),
        }
    }

    fn forward_2d(&self, x2d: &Tensor, b: usize, k_in: usize) -> candle::Result<Tensor> {
        if k_in != self.k {
            candle::bail!("{}: k mismatch got {} expected {}", self.name, k_in, self.k);
        }
        let x_f32 = x2d.to_dtype(DType::F32)?.contiguous()?;
        let mut x_vec = x_f32.flatten_all()?.to_vec1::<f32>()?;
        if let Some(ref perm) = self.perm {
            x_vec = self.apply_permutation_to_input(&x_vec, b, perm);
        }
        let mut out_buf = vec![0f32; b * self.out];
        match &self.blocks {
            QuantBlocks::Q8K(blocks) => {
                matmul::<BlockQ8K>((b, self.k, self.out), &x_vec, blocks, &mut out_buf)?
            }
            QuantBlocks::Q4K(blocks) => {
                matmul::<BlockQ4K>((b, self.k, self.out), &x_vec, blocks, &mut out_buf)?
            }
        }
        Tensor::from_vec(out_buf, (b, self.out), x2d.device())
    }

    fn apply_permutation_to_input(&self, x: &[f32], batch_size: usize, perm: &[usize]) -> Vec<f32> {
        let k = self.k;
        let mut permuted = vec![0f32; x.len()];
        unsafe {
            for b in 0..batch_size {
                let src_ptr = x.as_ptr().add(b * k);
                let dst_ptr = permuted.as_mut_ptr().add(b * k);
                for (dst_idx, &src_idx) in perm.iter().enumerate() {
                    *dst_ptr.add(dst_idx) = *src_ptr.add(src_idx);
                }
            }
        }
        permuted
    }
}