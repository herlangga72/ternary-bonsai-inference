# qwen35 forward pass: operation spec (from llama-model qwen35.cpp)

Reference: `src/models/qwen35.cpp` (llama_model_qwen35::graph) in the PrismML
fork. Decode pass only (no MTP/NextN blocks; the model GGUF has none).

## Shared skeleton per layer (both types)

```
cur  = rms_norm(inpL, attn_norm)                 # + residual kept as inpSA
cur  = attention(cur)                            # recurrent or full-attention
cur += inpSA                                     # attn residual
ffn_residual = cur
cur  = rms_norm(cur, attn_post_norm)
cur  = ffn(cur)                                  # silu(gate)*up -> down, no norm
cur += ffn_residual
inpL = cur
... after all layers:
h   = rms_norm(inpL, output_norm)
logits = output.weight @ h                       # per token row dot
```

FFN (LLM_FFN_PAR): `down( silu(gate @ x) ⊙ (up @ x) )` with PQ2_0 weights
`ffn_gate (5120,17408)`, `ffn_up (5120,17408)`, `ffn_down (17408,5120)`.

## Full-attention layers (is_recr = false, il % 4 == 3)

Hyper: n_head 24, n_head_kv 4, n_embd_head = 256 (q_norm/k_norm length 256).

```
Qfull = wq  @ x            # (5120, 12288) ; layout per head: [q(256) g(256)]
Q     = Qfull[..., :256] per head -> view (256, 24, tokens)
gate  = Qfull[...,256:] per head -> (6144, tokens)
Q     = rms_norm_per_head(Q, attn_q_norm)        # norm along 256 per head
K     = wk  @ x            # (5120,1024) -> (256, 4, tokens)
K     = rms_norm_per_head(K, attn_k_norm)
V     = wv  @ x            # (5120,1024) -> (256, 4, tokens)
Q,K   = rope_multi(Q,K,pos, sections=[11,11,10,0], n_rot)   # partial multi-band
attn  = softmax(Q·Kᵀ/sqrt(256) + mask) @ V      # GQA: 24 q-heads over 4 kv
cur   = (wo @ attn) * sigmoid(gate)             # wo (6144,5120)
```

## Recurrent (linear attention / gated delta net) layers

Hyper from GGUF metadata mapping:
- `num_k_heads = ssm.group_count` (16), `head_k_dim = ssm.state_size` (128)
- `num_v_heads = ssm.time_step_rank` (48), `head_v_dim = d_inner/num_v_heads` (6144/48=128)
- key_dim 2048, value_dim 6144, conv channels = key_dim*2 + value_dim = 10240
- qkv rows: [q(2048) | k(2048) | v(6144)]

```
qkv  = wqkv @ x                      # (5120,10240), one token per column
z    = wqkv_gate @ x                 # (5120,6144) -> z (6144,)
beta = sigmoid( ssm_beta @ x )       # (5120,48) -> 48 values, one per v-head
alpha= softplus( (ssm_alpha @ x) + ssm_dt.bias )   # (5120,48)+bias(48) -> 48
gate = alpha * ssm_a                 # ssm_a(48) per v-head (log-decay)

# temporal conv over the qkv channels (kernel 4, cached conv state)
conv = silu( conv1d(qkv, ssm_conv1d (4,10240)) )
q    = l2_norm(conv[0:2048].view(128,16))       # per (head_k_dim) rows
k    = l2_norm(conv[2048:4096].view(128,16))
v    = conv[4096:10240].view(128,48)

# fused GDN recurrence (per sequence state S: (128,128,48), cached):
#   given q,k,v,gate,beta and prev state -> new state + per-token output
out  = recurrent_attn(S, q, k, v, gate, beta)

# gated output norm:
out  = rms_norm_per_vhead(out, ssm_norm(128)) * silu(z.view(128,48))
cur  = ssm_out @ out.reshape(6144)     # (6144,5120)
```

Remaining unknowns to pin down from `build_recurrent_attn` (in
`llama-memory-recurrent`/gated-delta-net fused op) and the exact `rope_multi`
band math before implementation:
- fused GDN state update equation (delta rule using q/k/v, gate=exp(-dt·A),
  beta blend), state layout (128,128,48) and output formula.
- ggml `l2_norm` and RMS norm axis conventions for the 4-d views.
- `rope_multi` section semantics with n_rot (dimension_count 64) over
  head dim 256 and three bands [11,11,10,0].
- ggml_silu/softplus/sigmoid are elementwise; conv1d is causal with cached
  per-seq state of (kernel-1) previous inputs per channel.

## MRoPE port notes (from ggml rope kernel)

qwen35 uses `ggml_rope_multi` with `rope.dimension_count` = n_rot/n_dims = 64,
sections `[11,11,10,0]` (sum 32 = n_dims/2 pairs), head dim 256, freq_base
1e7, ext_factor 0.

- Cache: for pair j in 0..31 pick theta band t (j<11), h (11<=j<22),
  w (22<=j<32); e unused (sections[3]=0). Four theta bases start at the four
  position ids p_t/p_h/p_w/p_e and are each multiplied by
  `theta_scale = freq_base^(-2/n_dims)` every pair (global index j).
- ext_factor 0 and mscale 1 reduce YaRN to `cos = cos(theta)`,
  `sin = sin(theta)`.
- Application: NEOX ordering, rotate first n_dims=64 dims pairing
  `(i, i + n_dims/2)` with cache pair j = i/2; other head dims unchanged.
- Text-only decode: the four position ids should equal the token position;
  confirm against golden logits during layer validation.
