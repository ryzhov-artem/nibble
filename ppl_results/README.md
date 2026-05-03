# Perplexity Benchmark Results

Versioned outputs of the [`perplexity`](../src/bin/perplexity.rs) binary
(roadmap item **C1** in
[`pthi3_standalone_analysis.md`](https://github.com/artem1984A/nibble/blob/master/README.md)).

These logs are used as the quality gate for quantization experiments. The
working rule is strict: reconstruction-error histograms can justify a PPL run,
but a format is not worth optimizing unless it improves PPL on the same corpus.

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

For the Q6K comparison, use a clean quantized directory and set one policy:

```bash
# Replace layers that normally use Q8K with Q6K.
CANDLE_Q6K_POLICY=q8k ./target/release/quantize_q8k "$SNAP/model-00001-of-00002.safetensors" ./quantized-q6k
CANDLE_Q6K_POLICY=q8k ./target/release/quantize_q8k "$SNAP/model-00002-of-00002.safetensors" ./quantized-q6k
./target/release/pack_q8k_safetensors "$SNAP/model-00001-of-00002.safetensors" ./quantized-q6k ./packed-shard1-q6k.safetensors
./target/release/pack_q8k_safetensors "$SNAP/model-00002-of-00002.safetensors" ./quantized-q6k ./packed-shard2-q6k.safetensors
```

The packer rejects multiple quantized files for one tensor, so stale mixed-policy directories fail instead of producing ambiguous PPL results.

For the Q8K128 comparison, use the experimental feature and a clean directory:

```bash
# Build the experimental binary and loader support.
cargo build --release --features experimental-q8k128 \
  --bin quantize_q8k128 --bin pack_q8k_safetensors --bin perplexity

# Replace only the layer-0 qkv projection with Q8K128.
CANDLE_Q8K128_POLICY=layer0-qkv ./target/release/quantize_q8k128 "$SNAP/model-00001-of-00002.safetensors" ./quantized-q8k128-layer0-qkv
CANDLE_Q8K128_POLICY=layer0-qkv ./target/release/quantize_q8k128 "$SNAP/model-00002-of-00002.safetensors" ./quantized-q8k128-layer0-qkv

# Replace all qkv projections with Q8K128.
CANDLE_Q8K128_POLICY=qkv ./target/release/quantize_q8k128 "$SNAP/model-00001-of-00002.safetensors" ./quantized-q8k128-qkv
CANDLE_Q8K128_POLICY=qkv ./target/release/quantize_q8k128 "$SNAP/model-00002-of-00002.safetensors" ./quantized-q8k128-qkv
```

When packing one shard at a time, shard 1 can print:

```text
ERROR: Final layer norm not included in packed model!
```

That warning is harmless for the two-shard Phi-3 checkpoint when shard 2
preserves `model.norm.weight`. The final inference loader uses both shards and
will fail if the final norm is truly absent.

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
| `CANDLE_Q6K_POLICY` | `off` | Quantizer routing: `off` keeps Q8K/Q4K baseline, `down` replaces Q4K down projections with Q6K, `q8k` replaces Q8K-routed layers with Q6K, `all` quantizes every target weight to Q6K |
| `CANDLE_Q8K128_POLICY` | `layer0-qkv` in `quantize_q8k128` | Experimental Q8K128 routing: `off`, `layer0-qkv`, `early-qkv`, `qkv`, `attn`, `q8k`, or `all` |
| `CANDLE_Q8K_PERMUTE` | `0` | Apply column permutation during quantize (then auto-applied at inference) |
| `CANDLE_Q8K_PERM_STRATEGY` | `blockwise` | `blockwise` \| `l2`/`column` \| `svd`/`variance` \| `qr`/`qr-pivot` |

## Results

### 2026-05-02: Q6K and Q8K128 Experiments

The Q8K128 runs use a recreated plain-text WikiText-2 test file:

```text
Corpus bytes : 1,256,449
Token count  : 337,885
SHA256       : d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0
Eval         : ctx=2048, stride=2048, max_chunks=30
Scored tokens: 61,410
```

These numbers should only be compared with other runs on the same corpus.

| Date | Variant | Corpus | Q8K128 Scope | Ctx | Chunks | Tokens | Mean NLL | PPL | Delta vs same-corpus base | tok/s | Log |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-05-02 | Mixed Q8K/Q4K | current wt2 | none | 2048 | 30 | 61 410 | 1.983096 | **7.2652** | - | 16.9 | [`wikitext2_q8k-q4k_base_current-wt2_ctx2048_chunks30_20260502-152707.log`](./wikitext2_q8k-q4k_base_current-wt2_ctx2048_chunks30_20260502-152707.log) |
| 2026-05-02 | Q8K128 + Q8K/Q4K | current wt2 | layer 0 qkv only | 2048 | 30 | 61 410 | 1.984034 | **7.2720** | +0.0068 | 14.2 | [`wikitext2_q8k128-layer0-qkv_ctx2048_chunks30_20260502-114745.log`](./wikitext2_q8k128-layer0-qkv_ctx2048_chunks30_20260502-114745.log) |
| 2026-05-03 | Q8K128 + Q8K/Q4K | current wt2 | all qkv projections | 2048 | 30 | 61 410 | 1.983343 | **7.2670** | +0.0018 | 2.0 | [`wikitext2_q8k128-qkv_ctx2048_chunks30_20260502-164550.log`](./wikitext2_q8k128-qkv_ctx2048_chunks30_20260502-164550.log) |

**Q8K128 verdict.** Q8K128 improved local qkv reconstruction RMSE in the
histogram tool (weighted qkv RMSE improved by about 8.9%, with all 32 qkv
layers improving), but this did not translate into a PPL win. The all-qkv run
was nearly neutral on PPL (+0.0018) and much slower because the custom Q8K128
matmul is scalar. Do not implement SIMD for Q8K128 Variant A unless a later
quality experiment shows a real PPL gain.

The Q6K comparison below used the earlier WikiText-2 corpus. It is comparable
to the 2026-05-01 Q8K/Q4K baseline, not to the current-corpus Q8K128 runs.

| Date | Variant | Corpus | Ctx | Chunks | Tokens | Mean NLL | PPL | Delta vs same-corpus base | tok/s | Log |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-05-01 | Mixed Q8K/Q4K | earlier wt2 | 2048 | 30 | 61 410 | 1.865666 | **6.4602** | - | 16.8 | [`wikitext2_q8k-q4k_base_ctx2048_chunks30_20260501-182733.log`](./wikitext2_q8k-q4k_base_ctx2048_chunks30_20260501-182733.log) |
| 2026-05-01 | Q6K/Q4K | earlier wt2 | 2048 | 30 | 61 410 | 1.867840 | **6.4743** | +0.0141 | 16.1 | [`wikitext2_q6k-q4k_ctx2048_chunks30_20260501-193048.log`](./wikitext2_q6k-q4k_ctx2048_chunks30_20260501-193048.log) |

**Q6K verdict.** Replacing Q8K-routed layers with Q6K reduced model size but
regressed PPL slightly and was marginally slower. It does not beat the
Q8K/Q4K baseline.

### 2026-04-28 to 2026-04-29: Column-Permutation Experiment

| Date | Variant | Permutation | Strategy | Ctx | Chunks | Tokens | PPL | Δ vs base | tok/s | Log |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-04-28 | Mixed Q8K/Q4K | off | n/a | 2048 | 30 | 61 410 | **6.4602** | — | 17.8 | [`wikitext2_q8k-q4k_perm_ctx2048_chunks30_20260428-182240.log`](./wikitext2_q8k-q4k_perm_ctx2048_chunks30_20260428-182240.log) |
| 2026-04-28 | Mixed Q8K/Q4K | on | blockwise | 2048 | 30 | 61 410 | **6.4606** | +0.0004 | 17.8 | [`wikitext2_q8k-q4k_perm-on_blockwise_ctx2048_chunks30_20260428-201231.log`](./wikitext2_q8k-q4k_perm-on_blockwise_ctx2048_chunks30_20260428-201231.log) |
| 2026-04-29 | Mixed Q8K/Q4K | on | l2 | 2048 | 30 | 61 410 | **6.4625** | +0.0023 | 17.5 | [`wikitext2_q8k-q4k_perm-on_l2_ctx2048_chunks30_20260429-120122.log`](./wikitext2_q8k-q4k_perm-on_l2_ctx2048_chunks30_20260429-120122.log) |
| 2026-04-29 | Mixed Q8K/Q4K | on | svd | 2048 | 30 | 61 410 | **6.4666** | +0.0064 | 17.2 | [`wikitext2_q8k-q4k_perm-on_svd_ctx2048_chunks30_20260429-120122.log`](./wikitext2_q8k-q4k_perm-on_svd_ctx2048_chunks30_20260429-120122.log) |

**Verdict.** Across all three implemented strategies
(`blockwise` / `l2` / `svd`) the column-permutation path is at best
neutral (Δ ≤ +0.001 PPL, blockwise) and at worst a small regression
(Δ ≤ +0.007 PPL, svd). None of them improve on the perm-off baseline.
The inference-time cost is also non-zero on the sub-blockwise strategies
(L2 − 0.3 tok/s, SVD − 0.6 tok/s), driven by the per-token gather in
`apply_permutation_into`. The `qr` strategy remains unevaluated but is not
expected to change the conclusion.

**Action.** As of commit `<perm-feature-gate>`, the permutation code path
is behind a non-default Cargo feature (`experimental-perm`). The default
build no longer compiles it in; existing packed files containing `.perm`
tensors are loaded but the indices are ignored (a one-line warning is
emitted). To re-enable it for further experimentation:

```bash
cargo build --release --features experimental-perm
```

> The 2026-04-28 perm-off filename retains the historical `perm` tag from
> when the baseline was first captured; the actual run was permutation-OFF
> (no `.perm` files were produced by the quantizer pre-run). Later logs use
> the explicit `perm-on` / `perm-off` naming.

## Corpus Reproducibility Note

Several early runs used `/tmp/wt2.txt` as a scratch path. The path was later
recreated from a different plain-text WikiText-2 source, which changed both
byte count and token count. For this reason:

- compare Q8K128 only against `wikitext2_q8k-q4k_base_current-wt2_*`;
- compare Q6K only against `wikitext2_q8k-q4k_base_ctx2048_chunks30_20260501-182733.log`;
- include corpus byte count, token count, and preferably SHA256 in future logs.

## Reference numbers

| System | Format | WikiText-2 PPL |
|---|---|---|
| Phi-3-mini-4k, BF16 (HF reference, similar settings) | BF16 | 6.0 – 6.3 |
| Phi-3-mini-4k, llama.cpp `Q4_K_M` | Q4_K_M | 6.3 – 6.7 |
| **This project, mixed Q8K (attn + gate/up) + Q4K (down_proj), perm OFF** | mixed | **6.46** |

So the mixed scheme sits within ~2–8 % of BF16 quality at ~60 % of the file
size, in line with established 4-bit baselines from the llama.cpp ecosystem.
