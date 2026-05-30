//! Experimental Q8K block with a 128-value super-block.
//!
//! This keeps Q8K's single-scale arithmetic but halves the block width from
//! GGML's 256 values to 128. It is scalar-only for now; use it for PPL and
//! reconstruction-error experiments before adding SIMD.

use bytemuck::{Pod, Zeroable};

use crate::types::QK_Q8K_128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8K128 {
    pub d: f32,
    pub qs: [i8; QK_Q8K_128],
    pub bsums: [i16; QK_Q8K_128 / 16],
}

impl BlockQ8K128 {
    pub const QK: usize = QK_Q8K_128;
    pub const BSUM_GROUP: usize = 16;
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn zeros() -> Self {
        Self {
            d: 0.0,
            qs: [0; QK_Q8K_128],
            bsums: [0; QK_Q8K_128 / Self::BSUM_GROUP],
        }
    }

    pub fn from_float_row(src: &[f32]) -> Self {
        debug_assert_eq!(src.len(), Self::QK);
        let amax = src.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv_d = 1.0 / d;

        let mut qs = [0i8; QK_Q8K_128];
        for (q, &w) in qs.iter_mut().zip(src.iter()) {
            *q = (w * inv_d).round().clamp(-127.0, 127.0) as i8;
        }

        let mut bsums = [0i16; QK_Q8K_128 / Self::BSUM_GROUP];
        for (idx, chunk) in qs.chunks_exact(Self::BSUM_GROUP).enumerate() {
            let sum: i32 = chunk.iter().map(|&q| q as i32).sum();
            bsums[idx] = sum as i16;
        }

        Self { d, qs, bsums }
    }

    pub fn to_float_row(&self, dst: &mut [f32]) {
        debug_assert_eq!(dst.len(), Self::QK);
        for (out, &q) in dst.iter_mut().zip(self.qs.iter()) {
            *out = self.d * q as f32;
        }
    }
}

const _: () = assert!(std::mem::size_of::<BlockQ8K128>() == 148);
const _: () = assert!(std::mem::align_of::<BlockQ8K128>() == 4);

pub fn quantize_rows(rows: usize, k: usize, src: &[f32]) -> candle::Result<Vec<BlockQ8K128>> {
    if !k.is_multiple_of(BlockQ8K128::QK) {
        candle::bail!(
            "Q8K128 inner dim {k} is not divisible by {}",
            BlockQ8K128::QK
        );
    }
    let expected_len = rows * k;
    if src.len() != expected_len {
        candle::bail!(
            "Q8K128 source length mismatch: got {}, expected {}",
            src.len(),
            expected_len
        );
    }

    let blocks_per_row = k / BlockQ8K128::QK;
    let mut out = vec![BlockQ8K128::zeros(); rows * blocks_per_row];
    for row in 0..rows {
        for block in 0..blocks_per_row {
            let src_offset = row * k + block * BlockQ8K128::QK;
            let dst_offset = row * blocks_per_row + block;
            out[dst_offset] =
                BlockQ8K128::from_float_row(&src[src_offset..src_offset + BlockQ8K128::QK]);
        }
    }
    Ok(out)
}

pub fn dequantize_rows(
    blocks: &[BlockQ8K128],
    rows: usize,
    k: usize,
    dst: &mut [f32],
) -> candle::Result<()> {
    if !k.is_multiple_of(BlockQ8K128::QK) {
        candle::bail!(
            "Q8K128 inner dim {k} is not divisible by {}",
            BlockQ8K128::QK
        );
    }
    let expected_len = rows * k;
    if dst.len() != expected_len {
        candle::bail!(
            "Q8K128 destination length mismatch: got {}, expected {}",
            dst.len(),
            expected_len
        );
    }
    let blocks_per_row = k / BlockQ8K128::QK;
    let expected_blocks = rows * blocks_per_row;
    if blocks.len() != expected_blocks {
        candle::bail!(
            "Q8K128 block count mismatch: got {}, expected {}",
            blocks.len(),
            expected_blocks
        );
    }

    for row in 0..rows {
        for block in 0..blocks_per_row {
            let dst_offset = row * k + block * BlockQ8K128::QK;
            let block_offset = row * blocks_per_row + block;
            blocks[block_offset].to_float_row(&mut dst[dst_offset..dst_offset + BlockQ8K128::QK]);
        }
    }
    Ok(())
}

pub fn matmul_scalar(
    dims: (usize, usize, usize),
    x: &[f32],
    blocks: &[BlockQ8K128],
    y: &mut [f32],
) -> candle::Result<()> {
    let (batch, k, out) = dims;
    if !k.is_multiple_of(BlockQ8K128::QK) {
        candle::bail!(
            "Q8K128 matmul k={} is not divisible by {}",
            k,
            BlockQ8K128::QK
        );
    }
    if x.len() != batch * k {
        candle::bail!(
            "Q8K128 matmul input length mismatch: got {}, expected {}",
            x.len(),
            batch * k
        );
    }
    if y.len() != batch * out {
        candle::bail!(
            "Q8K128 matmul output length mismatch: got {}, expected {}",
            y.len(),
            batch * out
        );
    }
    let blocks_per_row = k / BlockQ8K128::QK;
    let expected_blocks = out * blocks_per_row;
    if blocks.len() != expected_blocks {
        candle::bail!(
            "Q8K128 matmul block count mismatch: got {}, expected {}",
            blocks.len(),
            expected_blocks
        );
    }

    for batch_idx in 0..batch {
        let x_row = &x[batch_idx * k..(batch_idx + 1) * k];
        for out_idx in 0..out {
            let mut acc = 0f64;
            let block_row = &blocks[out_idx * blocks_per_row..(out_idx + 1) * blocks_per_row];
            for (block_idx, block) in block_row.iter().enumerate() {
                let x_offset = block_idx * BlockQ8K128::QK;
                let x_block = &x_row[x_offset..x_offset + BlockQ8K128::QK];
                let mut dot = 0f32;
                for (&q, &xv) in block.qs.iter().zip(x_block.iter()) {
                    dot += q as f32 * xv;
                }
                acc += (block.d * dot) as f64;
            }
            y[batch_idx * out + out_idx] = acc as f32;
        }
    }
    Ok(())
}

pub fn reconstruction_stats(
    original: &[f32],
    blocks: &[BlockQ8K128],
    rows: usize,
    k: usize,
) -> candle::Result<(f64, f64, f64)> {
    let mut reconstructed = vec![0f32; rows * k];
    dequantize_rows(blocks, rows, k, &mut reconstructed)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_stable() {
        assert_eq!(std::mem::size_of::<BlockQ8K128>(), 148);
        assert_eq!(std::mem::align_of::<BlockQ8K128>(), 4);
    }

    #[test]
    fn round_trip_is_within_one_step() {
        let src: Vec<f32> = (0..BlockQ8K128::QK)
            .map(|i| (i as f32 - 64.0) / 37.0)
            .collect();
        let block = BlockQ8K128::from_float_row(&src);
        let mut deq = vec![0f32; BlockQ8K128::QK];
        block.to_float_row(&mut deq);
        let max_err = src
            .iter()
            .zip(deq.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err <= block.d * 0.51,
            "max_err={max_err}, d={}",
            block.d
        );
    }

    #[test]
    fn matmul_matches_dequantized_weights() {
        let rows = 7;
        let k = 256;
        let weights: Vec<f32> = (0..rows * k)
            .map(|i| ((i * 17 % 251) as f32 - 125.0) / 500.0)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) / 100.0)
            .collect();
        let blocks = quantize_rows(rows, k, &weights).unwrap();
        let mut deq = vec![0f32; rows * k];
        dequantize_rows(&blocks, rows, k, &mut deq).unwrap();

        let mut expected = vec![0f32; rows];
        for row in 0..rows {
            expected[row] = deq[row * k..(row + 1) * k]
                .iter()
                .zip(x.iter())
                .map(|(&w, &xv)| w * xv)
                .sum();
        }

        let mut actual = vec![0f32; rows];
        matmul_scalar((1, k, rows), &x, &blocks, &mut actual).unwrap();
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "row={idx} actual={a} expected={e}");
        }
    }
}
