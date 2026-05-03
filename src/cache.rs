use candle::{DType, Device, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Cache {
    pub masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    pub cos: Tensor,
    pub sin: Tensor,
    pub device: Device,
}

impl Cache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        max_seq_len: usize,
        head_dim: usize,
        num_layers: usize,
        device: &Device,
    ) -> candle::Result<Self> {
        let theta = 10000.0f32;
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / head_dim as f32))
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (inv_freq_len,), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        let freqs = t.matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    pub fn mask_query_kv(&mut self, q_len: usize, kv_len: usize) -> candle::Result<Tensor> {
        let key = (q_len, kv_len);
        if let Some(mask) = self.masks.get(&key) {
            return Ok(mask.clone());
        }
        let offset = kv_len.saturating_sub(q_len);
        let mask: Vec<u8> = (0..q_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        let mask_tensor = Tensor::from_slice(&mask, (q_len, kv_len), &self.device)?;
        self.masks.insert(key, mask_tensor.clone());
        Ok(mask_tensor)
    }

    pub fn reset_for_new_turn(&mut self) {
        self.kvs = vec![None; self.kvs.len()];
        self.masks.clear();
    }

    pub fn estimate_tokens(&self) -> usize {
        self.kvs
            .iter()
            .filter_map(|kv| kv.as_ref())
            .map(|(k, _)| k.dims()[2])
            .max()
            .unwrap_or(0)
    }

    pub fn memory_mb(&self) -> f32 {
        let mut total_bytes = 0;
        for kv in &self.kvs {
            if let Some((k, v)) = kv {
                total_bytes += k.elem_count() * 4;
                total_bytes += v.elem_count() * 4;
            }
        }
        total_bytes as f32 / (1024.0 * 1024.0)
    }
}
