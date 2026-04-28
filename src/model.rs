use candle::{DType, IndexOp, Tensor, D};
use candle_nn::ops::silu;

use crate::cache::Cache;
use crate::quant_linear::QuantLinear;

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> candle::Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / x.dim(D::Minus1)? as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps)?.sqrt()?)?;
    let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)?;
    Ok(x)
}

pub struct CausalSelfAttention {
    pub qkv_proj: QuantLinear,
    pub o_proj: QuantLinear,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> candle::Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _head_dim) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn repeat_kv(&self, x: Tensor) -> candle::Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> candle::Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        let qkv = self.qkv_proj.forward(x)?;
        let q_size = self.num_attention_heads * self.head_dim;
        let kv_size = self.num_key_value_heads * self.head_dim;

        let q = qkv.narrow(D::Minus1, 0, q_size)?;
        let k = qkv.narrow(D::Minus1, q_size, kv_size)?;
        let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?;

        let q = q.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let mut v = v.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let att_scaled = q.matmul(&k.t()?)?;
        let att_max = att_scaled.max_keepdim(D::Minus1)?;
        let att_shifted = att_scaled.broadcast_sub(&att_max)?;
        let att = (att_shifted / (self.head_dim as f64).sqrt())?;

        let kv_seq_len = k.dims()[2];
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask_query_kv(seq_len, kv_seq_len)?;
            let mask_broadcast = mask.broadcast_as(att.shape().dims())?;
            masked_fill(&att, &mask_broadcast, f32::NEG_INFINITY)?
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }
}

pub struct Mlp {
    pub gate_up_proj: QuantLinear,
    pub down_proj: QuantLinear,
    pub hidden_size: usize,
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(D::Minus1, 0, self.hidden_size)?;
        let up = gate_up.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
        let x = (silu(&gate)? * up)?;
        self.down_proj.forward(&x)
    }
}

pub struct Block {
    pub rms_1: Tensor,
    pub attn: CausalSelfAttention,
    pub rms_2: Tensor,
    pub mlp: Mlp,
}

impl Block {
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> candle::Result<Tensor> {
        let residual = x;
        let x = rms_norm(x, &self.rms_1, 1e-5)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = rms_norm(&x, &self.rms_2, 1e-5)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        Ok(x)
    }
}

pub struct Phi3 {
    pub wte: Tensor,
    pub blocks: Vec<Block>,
    pub ln_f: Tensor,
    pub lm_head: QuantLinear,
}

impl Phi3 {
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> candle::Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let hidden = self.wte.dim(1)?;

        // Gather embeddings in a single op instead of an N-element python-style loop.
        let flat_ids = x.flatten_all()?;
        let mut x = self
            .wte
            .index_select(&flat_ids, 0)?
            .reshape((b_sz, seq_len, hidden))?;

        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }

        let x = rms_norm(&x, &self.ln_f, 1e-5)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    /// Full-sequence forward pass that returns logits for **every** position
    /// `(b_sz, seq_len, vocab)` instead of just the last one.
    ///
    /// Used by the perplexity benchmark to score every target token in a
    /// chunk with a single transformer pass. The autoregressive REPL path
    /// keeps using [`Self::forward`] which is cheaper because `lm_head` only
    /// runs on one row.
    pub fn forward_full(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> candle::Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;
        let hidden = self.wte.dim(1)?;

        let flat_ids = x.flatten_all()?;
        let mut x = self
            .wte
            .index_select(&flat_ids, 0)?
            .reshape((b_sz, seq_len, hidden))?;

        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }

        let x = rms_norm(&x, &self.ln_f, 1e-5)?;
        // QuantLinear::forward already accepts a 3-D input and reshapes
        // internally, so we do NOT slice the last position here.
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }
}