# Perplexity benchmark results

Versioned outputs of the [`perplexity`](../src/bin/perplexity.rs) binary
(roadmap item **C1** in
[`pthi3_standalone_analysis.md`](https://github.com/artem1984A/nibble/blob/master/README.md)).

## How to reproduce

```bash
# 1. Build the release binaries
cargo build --release --bin quantize_q8k --bin pack_q8k_safetensors --bin perplexity

# 2. Quantize + pack a Phi-3-mini-4k checkpoint (defaults: permutation OFF, blockwise)
SNAP=~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/<rev>
./target/release/quantize_q8k     "$SNAP/model-00001-of-00002.safetensors" ./quantized
./target/release/quantize_q8k     "$SNAP/model-00002-of-00002.safetensors" ./quantized
./target/release/pack_q8k_safetensors "$SNAP/model-00001-of-00002.safetensors" ./quantized ./packed-shard1.safetensors
./target/release/pack_q8k_safetensors "$SNAP/model-00002-of-00002.safetensors" ./quantized ./packed-shard2.safetensors

# 3. Fetch wikitext-2 test split as a plain UTF-8 file (e.g. via duckdb on the parquet)
#    -> /tmp/wt2.txt   (~1.3 MB, ~335 k tokens)

# 4. Run the benchmark
PHI3_PPL_CTX=2048 PHI3_PPL_MAX_CHUNKS=30 \
  ./target/release/perplexity \
    ./packed-shard1.safetensors ./packed-shard2.safetensors /tmp/wt2.txt \
    | tee ppl_results/wikitext2_q8k-q4k_<variant>_ctx2048_chunks30_$(date +%Y%m%d-%H%M%S).log
```

To enable the experimental column-permutation path:

```bash
CANDLE_Q8K_PERMUTE=1 CANDLE_Q8K_PERM_STRATEGY=blockwise ./target/release/quantize_q8k ...
# (then re-pack + re-run perplexity into a separate log)
```

## Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `PHI3_PPL_CTX` | 512 (capped to `max_position_embeddings`) | Tokens per chunk |
| `PHI3_PPL_STRIDE` | `PHI3_PPL_CTX` | Sliding-window stride between chunks |
| `PHI3_PPL_MAX_CHUNKS` | unbounded | Upper bound on chunks evaluated |
| `CANDLE_Q8K_PERMUTE` | `0` | Apply column permutation during quantize (then auto-applied at inference) |
| `CANDLE_Q8K_PERM_STRATEGY` | `blockwise` | `blockwise` \| `l2`/`column` \| `svd`/`variance` \| `qr`/`qr-pivot` |

## Results

| Date | Variant | Permutation | Strategy | Ctx | Chunks | Tokens | PPL | tok/s | Log |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-28 | Mixed Q8K/Q4K | off | n/a | 2048 | 30 | 61 410 | **6.4602** | 17.8 | [`wikitext2_q8k-q4k_perm_ctx2048_chunks30_20260428-182240.log`](./wikitext2_q8k-q4k_perm_ctx2048_chunks30_20260428-182240.log) |
| 2026-04-28 | Mixed Q8K/Q4K | on (blockwise) | blockwise | 2048 | 30 | 61 410 | _pending_ | _pending_ | _pending — see `pipeline_perm-on_*.log`_ |

> The first row's filename retains the historical "perm" tag from when the
> baseline was first captured; the actual run was permutation-OFF (no
> `.perm` files were produced by the quantizer pre-run). Future logs use the
> explicit `perm-on` / `perm-off` naming.

## Reference numbers

| System | Format | WikiText-2 PPL |
|---|---|---|
| Phi-3-mini-4k, BF16 (HF reference, similar settings) | BF16 | 6.0 – 6.3 |
| Phi-3-mini-4k, llama.cpp `Q4_K_M` | Q4_K_M | 6.3 – 6.7 |
| **This project, mixed Q8K (attn + gate/up) + Q4K (down_proj), perm OFF** | mixed | **6.46** |

So the mixed scheme sits within ~2–8 % of BF16 quality at ~60 % of the file
size, in line with established 4-bit baselines from the llama.cpp ecosystem.
