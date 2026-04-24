# phi3-mixed-quant

Mixed-precision Q8K/Q4K quantized inference for Phi-3 Mini 3.8B on CPU.
Built on top of [candle](https://github.com/huggingface/candle) and its GGML-compatible quantization system.

## What It Does

Three-stage pipeline that takes a full-precision Phi-3 model, quantizes it with a per-layer mixed strategy, packs the result into a single SafeTensors file, and runs interactive chat inference -- all on CPU, no GPU required.

**Stage 1 -- Quantize.** Reads BF16/F16/F32 SafeTensors (local file or HuggingFace Hub model ID). Routes each weight tensor to Q8K or Q4K format based on layer sensitivity. Attention projections and MLP gate/up layers get Q8K (8-bit). MLP down projections get Q4K (4-bit). Embeddings and layer norms stay unquantized.

**Stage 2 -- Pack.** Combines per-layer quantized blocks, metadata, optional permutation indices, and unquantized tensors into a single packed SafeTensors file ready for inference.

**Stage 3 -- Inference.** Loads the packed model, runs a streaming conversational chatbot with KV cache, RoPE, sliding-window history, and adaptive sampling parameters for code vs. natural language queries.

## Quantization Strategy

| Layer | Format | Bits/Weight | Rationale |
|---|---|---|---|
| Attention Q/K/V/O | Q8K | 8 | High information density |
| MLP gate_proj, up_proj | Q8K | 8 | Activation-critical |
| MLP down_proj | Q4K | 4 | Less sensitive, 50% block size reduction |
| Embeddings, norms | F32 | 32 | Preserved unquantized |

Result: BF16 model (~7.6 GB) compresses to roughly 2 GB packed SafeTensors while retaining high output quality on attention-critical paths.

## Optional Column Permutation

Setting `CANDLE_Q8K_PERMUTE=1` before quantization enables column-wise reordering by L2-norm magnitude prior to block quantization. Permutation vectors are stored alongside quantized blocks and applied at inference time inside `QuantLinear::forward()`. This can reduce quantization error for layers with high variance across columns.

## Usage

### Quantize

```bash
# From local SafeTensors
quantize_q8k model.safetensors ./quantized

# From HuggingFace Hub (downloads automatically)
quantize_q8k microsoft/Phi-3-mini-4k-instruct ./quantized

# Specific shard
quantize_q8k microsoft/Phi-3-mini-4k-instruct ./quantized --file model-00001-of-00002.safetensors
```

### Pack

```bash
pack_q8k_safetensors model.safetensors ./quantized model-q8k-packed.safetensors
```

### Run

```bash
# Single shard
phi3-mixed-quant model-q8k-packed.safetensors

# Two shards
phi3-mixed-quant shard1.safetensors shard2.safetensors
```

Interactive session: type a query, get streaming output. Commands: `exit`, `reset`.

## Build

```bash
cargo build --release
```

Binaries appear in `target/release/`: `phi3-mixed-quant`, `quantize_q8k`, `pack_q8k_safetensors`.

## Project Structure

```
src/
  main.rs              -- Inference binary: chat loop, sampling, streaming
  cache.rs             -- KV cache, RoPE cos/sin tables, memory estimation
  conversation.rs      -- Message history, sliding window, Phi-3 prompt format
  loader.rs            -- SafeTensors loader, HF Hub tokenizer, multi-shard
  model.rs             -- Phi-3 transformer: attention, MLP, RMSNorm
  quant_linear.rs      -- Q8K/Q4K dispatch, permutation, dequant matmul
  types.rs             -- Phi3Config, Q8KHeader, type conversions
  bin/
    quantize_q8k.rs    -- Stage 1: quantizer with HF Hub download
    pack_q8k_safetensors.rs -- Stage 2: packer into single SafeTensors
```

## Model Architecture

Phi-3 Mini 4K Instruct -- 3.8B parameters, 32 transformer blocks, 32 attention heads, 3072 hidden dimension, 4096 max context, 32000 vocabulary.

Forward pass: token embeddings -> 32x (RMSNorm -> multi-head attention with RoPE + KV cache -> residual -> RMSNorm -> SiLU-gated MLP -> residual) -> final RMSNorm -> lm_head -> logits.

## Dependencies

Core: [candle-core](https://github.com/huggingface/candle), candle-nn, candle-transformers (rev 3b39794c1). SafeTensors 0.7 for model I/O. Tokenizers 0.21 with onig backend. hf-hub 0.4.1 for model/tokenizer downloads. half 2.4 for F16/BF16 handling. bytemuck for zero-copy block casting.

## Runtime Characteristics

- CPU-only, no CUDA/Metal dependency
- Streaming token generation with real-time speed reporting (tokens/sec)
- KV cache with live memory tracking
- Adaptive sampling: temperature 0.6 / top-k 50 / top-p 0.9 for chat; temperature 0.35 / top-k 25 / top-p 0.95 for code
- Sliding window conversation history (3072 tokens) with automatic trimming

## Acknowledgments

This project relies on [candle](https://github.com/huggingface/candle) by Hugging Face for tensor operations and the GGML-compatible quantization primitives (Q8K, Q4K block formats, `from_float` conversions). The quantization block structures and dequantization kernels originate from candle's `quantized` module.

## License

MIT OR Apache-2.0
