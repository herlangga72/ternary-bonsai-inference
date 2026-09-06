# qwen35 forward-pass study notes (M6 reference)

Goal: reproduce llama.cpp's `qwen35` decode numerics in Rust. These notes
capture what is known from the GGUF tensor layout and llama.cpp sources, and
what still needs to be reverse-engineered from `src/llama-graph.cpp`
(`LLM_ARCH_QWEN35`) and `src/llama-context.cpp` before M6 can be implemented.

## Model facts (verified from GGUF)

- 64 layers, n_embd 5120, 24 q-heads, 4 kv-heads, key/value length 256.
- Hybrid: layers with `idx % 4 == 3` are full attention (KV cache allocated on
  layers 3,7,...,63 = 16 layers); the other 48 layers use the SSM branch.
- SSM per-layer hyperparams: conv_kernel 4, state_size 128, group_count 16,
  time_step_rank 48, inner_size 6144.
- RoPE: `rope.dimension_sections = [11,11,10,0]`, `dimension_count = 64`,
  freq_base 1e7.
- All linear weight matrices are PQ2_0 (ternary, group 128); 1-d norms/scales
  and small vectors (ssm_a, ssm_dt.bias, ...) are f32.

## Per-layer tensor names

### SSM layer example (blk.0) - actual index

```
blk.0.attn_gate.weight          pq2 (5120, 6144)
blk.0.attn_norm.weight          f32 (5120)
blk.0.attn_qkv.weight           pq2 (5120, 10240)
blk.0.ffn_down.weight           pq2 (17408, 5120)
blk.0.ffn_gate.weight           pq2 (5120, 17408)
blk.0.ffn_up.weight             pq2 (5120, 17408)
blk.0.post_attention_norm.weight f32 (5120)
blk.0.ssm_a                     f32 (48)
blk.0.ssm_alpha.weight          pq2 (5120, 48)
blk.0.ssm_beta.weight           pq2 (5120, 48)
blk.0.ssm_conv1d.weight         f32 (4, 10240)
blk.0.ssm_dt.bias               f32 (48)
blk.0.ssm_norm.weight           f32 (128)
blk.0.ssm_out.weight            pq2 (6144, 5120)
```

### Attention layer example (blk.3) - actual index

```
blk.3.attn_k.weight             pq2 (5120, 1024)     # 4 kv-heads * 256
blk.3.attn_k_norm.weight        f32 (256)
blk.3.attn_norm.weight          f32 (5120)
blk.3.attn_output.weight        pq2 (6144, 5120)
blk.3.attn_q.weight             pq2 (5120, 12288)    # 24 heads * 512 ?
blk.3.attn_q_norm.weight        f32 (256)
blk.3.attn_v.weight             pq2 (5120, 1024)     # 4 kv-heads * 256
blk.3.ffn_*                     pq2 (as above)
blk.3.post_attention_norm.weight f32 (5120)
```

Open questions to resolve from `llama-graph.cpp` during M6:

- blk.0 is an SSM layer but still has `attn_qkv` (5120 -> 10240) and
  `attn_gate` (5120 -> 6144). Is a q-projection used with SSM-derived
  key/value (linear/gated attention style), or is qkv only relevant on
  attention layers and simply present-but-unused on SSM layers?
- Attention layers: q projects to 12288 (= 24 * 512) while kv is 4 * 256, and
  q_norm/k_norm are length 256. Determine head dims, how 512 splits into
  rotary bands per `rope.dimension_sections=[11,11,10,0]` (dimension_count 64),
  and where q_norm/k_norm apply.
- `attn_output` and `attn_gate` are 5120 -> 6144 -> 5120 paths; confirm exact
  gating/combination and residual order for both layer types.
- SSM state recurrence details: conv1d over 10240 channels with kernel 4, dt
  from rank-48 projection + bias, A matrix expansion 48 -> 128 (group_count 16),
  `ssm_norm`(128) location, alpha/beta (->48) gating, `ssm_out` 6144 -> 5120.

## What must be ported (M6 checklist)

1. Exact graph order for both layer types from `llama-graph.cpp`
   (`LLM_ARCH_QWEN35` case): norms, projections, gates, residual add points.
2. Partial RoPE application over the q/k head dims given
   `rope.dimension_sections = [11,11,10,0]` (three rotary bands?).
3. Full attention: q/k/v projection, KV cache (f16), scaled dot-product,
   output projection.
4. SSM scan: conv1d over channels, dt projection + softplus/bias, A matrix
   (ssm_a, 48 -> 128 with grouping), state recurrence, alpha/beta gating,
   out projection.
5. LM head = output.weight (PQ2_0, 248320 x 5120) on the final normed hidden.
6. KV + recurrent-state caching and positions for decode steps.

## Verification artifacts (golden/)

`golden/qa.logits.bin` and `golden/code.logits.bin` are llama.cpp logits for
two fixed prompts (format documented in the golden dir). M6 validates the Rust
forward pass by comparing argmax token ids and logits (tolerance) against these.
