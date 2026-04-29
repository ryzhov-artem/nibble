use candle::quantized::k_quants::{matmul, BlockQ4K, BlockQ8K, GgmlType, QK_K};
use candle::{CpuStorage, DType, Storage, Tensor};
use safetensors::tensor::{Dtype, SafeTensors};
use std::{fs, path::Path};

use crate::scratch;
use crate::types::{Q8KHeader, MAGIC_Q8K};
#[cfg(feature = "experimental-perm")]
use crate::types::MAGIC_PERM;

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
    ///
    /// Behind the non-default `experimental-perm` Cargo feature: returns the
    /// parsed permutation indices when a `<name>.perm` tensor is present.
    /// In the default build: returns `Ok(None)`, but emits a one-line warning
    /// per layer if the packed file *does* contain a `.perm` tensor so the
    /// user knows it's being ignored.
    #[cfg(feature = "experimental-perm")]
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

    #[cfg(not(feature = "experimental-perm"))]
    fn load_perm(st: &SafeTensors, name: &str) -> candle::Result<Option<Vec<usize>>> {
        let perm_name = format!("{}.perm", name);
        if st.tensor(&perm_name).is_ok() {
            eprintln!(
                "warning: ignoring permutation tensor `{}` — \
                 rebuild with `--features experimental-perm` to enable",
                perm_name
            );
        }
        Ok(None)
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

        // Make sure the activation tensor is F32 and contiguous on the CPU.
        // Both calls are no-ops when the tensor already satisfies them, so this
        // is essentially free for the hot path (decoder layers always feed us
        // F32-contiguous activations from the KV cache).
        let x_f32 = x2d.to_dtype(DType::F32)?.contiguous()?;
        let total = b * self.k;

        // Output buffer must remain a fresh `Vec` because `Tensor::from_vec`
        // takes ownership. It's only `b * out` floats (≈12 KB for batch=1,
        // out=3072), so the win from pooling it would be marginal.
        let mut out_buf = vec![0f32; b * self.out];

        // B3: read the activation slice straight out of candle's CPU storage.
        // No `to_vec1::<f32>()` copy. The `RwLockReadGuard` is held only for
        // the lifetime of this inner block; rayon-parallel `matmul` runs
        // inside it (it just reads, doesn't mutate the storage).
        {
            let (storage, layout) = x_f32.storage_and_layout();
            let raw_full: &[f32] = match &*storage {
                Storage::Cpu(CpuStorage::F32(v)) => v.as_slice(),
                Storage::Cpu(_) => candle::bail!(
                    "{}: forward_2d expected F32 CPU storage",
                    self.name
                ),
                _ => candle::bail!(
                    "{}: forward_2d only supports the CPU device",
                    self.name
                ),
            };
            let off = layout.start_offset();
            let raw = &raw_full[off..off + total];

            // B1: route the permuted activation copy through a thread-local
            // scratch pool. Without pooling we would `vec![0; b*k]` once per
            // matmul call (~128 calls per token); with it, the same allocation
            // is recycled for the entire decode loop.
            let perm_scratch: Option<scratch::Buf> = self.perm.as_ref().map(|perm| {
                let mut buf = scratch::take_f32(total);
                apply_permutation_into(raw, b, self.k, perm, buf.as_mut_slice());
                buf
            });
            let x_for_matmul: &[f32] = match &perm_scratch {
                Some(buf) => buf.as_slice(),
                None => raw,
            };

            // B2 note: `candle::quantized::k_quants::matmul` already
            // parallelises across output columns with rayon
            // (k_quants.rs ~line 2305). No extra threading needed here.
            match &self.blocks {
                QuantBlocks::Q8K(blocks) => matmul::<BlockQ8K>(
                    (b, self.k, self.out),
                    x_for_matmul,
                    blocks,
                    &mut out_buf,
                )?,
                QuantBlocks::Q4K(blocks) => matmul::<BlockQ4K>(
                    (b, self.k, self.out),
                    x_for_matmul,
                    blocks,
                    &mut out_buf,
                )?,
            }
            // `perm_scratch` drops here → buffer returns to the pool.
        }

        Tensor::from_vec(out_buf, (b, self.out), x2d.device())
    }
}

/// Apply a column permutation in place into `dst`. Caller owns both slices;
/// they must be the same length and `perm.len() == k`.
#[inline]
fn apply_permutation_into(src: &[f32], batch: usize, k: usize, perm: &[usize], dst: &mut [f32]) {
    debug_assert_eq!(perm.len(), k);
    debug_assert_eq!(src.len(), batch * k);
    debug_assert_eq!(dst.len(), batch * k);
    for b in 0..batch {
        let s = &src[b * k..(b + 1) * k];
        let d = &mut dst[b * k..(b + 1) * k];
        for (di, &si) in perm.iter().enumerate() {
            d[di] = s[si];
        }
    }
}