use candle::bail;
use half::{bf16, f16};
use safetensors::tensor::Dtype;
use serde::Deserialize;
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Q8KHeader {
    pub magic: u32,
    pub version: u32,
    pub out: u32,
    pub k: u32,
    pub blocks_per_row: u32,
    pub dtype: u32,
}

pub const MAGIC_Q8K: u32 = 0x4B51_3838;
pub const MAGIC_Q4K: u32 = 0x4B51_3834;
pub const MAGIC_Q6K: u32 = 0x4B51_3836;
#[cfg(feature = "experimental-q8k128")]
pub const MAGIC_Q8K_128: u32 = 0x4B51_3831;
pub const MAGIC_PERM: u32 = 0x4D52_4550;

/// Header version stamped into freshly-quantized .q8k / .q4k files.
pub const HEADER_VERSION: u32 = 1;

/// dtype tags carried in `Q8KHeader::dtype` (mirrors GGML).
pub const DTYPE_Q8K: u32 = 0x18;
pub const DTYPE_Q4K: u32 = 0x14;
pub const DTYPE_Q6K: u32 = 0x16;
#[cfg(feature = "experimental-q8k128")]
pub const DTYPE_Q8K_128: u32 = 0x28;

#[cfg(feature = "experimental-q8k128")]
pub const QK_Q8K_128: usize = 128;

#[derive(Debug, Clone)]
pub struct Phi3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
}

impl Phi3Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn phi3_mini() -> Self {
        Self {
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: 4096,
        }
    }

    /// Parse a Hugging Face `config.json` for any Phi-3 variant.
    pub fn from_json_file(path: &Path) -> candle::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| candle::Error::Msg(format!("read {}: {}", path.display(), e)))?;
        let parsed: HfConfigJson = serde_json::from_str(&raw)
            .map_err(|e| candle::Error::Msg(format!("parse {}: {}", path.display(), e)))?;
        Ok(Self {
            hidden_size: parsed.hidden_size,
            intermediate_size: parsed.intermediate_size,
            num_hidden_layers: parsed.num_hidden_layers,
            num_attention_heads: parsed.num_attention_heads,
            num_key_value_heads: parsed.num_key_value_heads,
            max_position_embeddings: parsed.max_position_embeddings,
        })
    }
}

#[derive(Debug, Deserialize)]
struct HfConfigJson {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    max_position_embeddings: usize,
}

pub fn tensor_to_f32(bytes: &[u8], dtype: Dtype) -> candle::Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        _ => bail!("unsupported dtype"),
    })
}
