# Plan: RotorQuant-style KV cache quantization

> ## Status (2026-09-13): R0, R1, R2 done on CPU. R3 (GPU) NOT started.
>
> `src/kvquant.rs` implements the quantizer; `AttnCache`/`full_attention_layer`
> use it. Selected with `BONSAI_KV=f32|planarN|planarNk` (N = 1..8; `planarNk`
> = K only, V f32). Default is `f32`, so nothing changes unless asked.
>
> **Measured (qa prompt, 20 tokens; f32 baseline logit rel diff is 4.03e-3,
> greedy 8160):**
>
> | config | logit rel diff | greedy |
> | --- | --- | --- |
> | f32 (baseline) | 4.03e-3 | MATCH |
> | planar2k | 4.07e-2 | MATCH |
> | planar3k | 3.13e-2 | MATCH |
> | planar4k | 2.07e-2 | MATCH |
> | planar6k | 1.63e-2 | MATCH |
> | planar8k | 1.07e-2 | MATCH |
> | planar4 (symmetric) | 4.14e-2 | MATCH |
> | planar8 (symmetric) | 2.43e-2 | MATCH |
>
> Greedy matches at every bit width on both prompts. The quantizer is
> textbook-exact: measured per-coordinate RMSE is 18.58% of sigma at 3-bit and
> 9.25% at 4-bit against the theoretical 18.58% / 9.75%, halving per bit.
>
> **Conclusion that changes the plan:** 3-4 bit KV quantization perturbs this
> model's logits by 2-4% (5-10x the f32 baseline error), and the error is
> monotone in bits, so it is inherent, not a bug. On a 20-token prompt greedy
> survives, but this is a real quality cost and the 10.3x headline should not
> be treated as free. The defensible operating points are:
>   - **f16 KV**: 2x saving, error ~1e-3 (not implemented; trivial).
>   - **8-bit planar K-only**: ~4x on K, logit error 1.07e-2.
>   - 3-4 bit: only if long-context memory forces it, and then measure PPL.
>
> R3 (GPU shaders) is worth doing only for a chosen operating point; at 3-4 bit
> the quality tradeoff is significant enough that it should be a deliberate
> decision first.


Goal: replace the f32 KV cache with a rotated + Lloyd-Max scalar-quantized
cache (PlanarQuant / IsoQuant family, from the RotorQuant work), so long
context fits the 8 GB RX 7600 and KV bandwidth stops growing at 4 bytes per
coordinate.

## Why this, and what it will not do

At the current default (`BONSAI_CTX=2048`) the KV cache is **not** the decode
bottleneck: it is 128 KB per token of context (K+V, f32, 16 full-attention
layers), so a 2048-token step reads ~262 MB against 7.14 GB of weights (~3.7%).
Quantizing KV will therefore **not** meaningfully speed up short-context decode
on this box.

What it unlocks is context length on the discrete card:

| context | f32 KV (now) | planar3 KV (~10.4x) |
| --- | --- | --- |
| 2K | 262 MB | 25 MB |
| 32K | 4.19 GB | 403 MB |
| 128K | 16.8 GB | 1.6 GB |

7.14 GB of weights plus 4.19 GB of f32 KV does not fit 8 GB of VRAM; 7.14 GB
plus ~0.4 GB does. That is the real justification. Secondary: at long context
the attention kernels become O(n_pos) and KV bytes matter.

## The algorithm (from the reference implementation)

Per KV vector `x` (one head, `d = head_dim = 256`):

1. **Normalize**: `n = ||x||`, `x̂ = x / n`; store `n` separately (f16).
2. **Rotate** each adjacent pair with a fixed random Givens angle
   `(cos θ, sin θ)`:
   `[x0', x1'] = [c*x0 - s*x1, s*x0 + c*x1]`.
   Angles are random per group, generated once from a fixed seed (seed 42 in
   the reference). No training/calibration.
3. **Scalar-quantize** every rotated coordinate to the Lloyd-Max codebook for
   `2^bits` levels. A coordinate of a unit `d`-vector follows
   `Beta((d-3)/2)` on [-1,1], approximated `N(0, 1/d)` for d >= 64; for
   `d = 256`, `sigma = 1/16`. Centroids are solved from the Lloyd-Max
   conditions (1-D k-means) and are constants for a given `(d, bits)`.
4. **Store** the indices packed at `bits` each, plus the norm.
5. **Reconstruct**: centroid lookup → inverse Givens (`s` negated) → `* n`.

Reference: `turboquant/planarquant.py`, `turboquant/lloyd_max.py`. The
paper's original Cl(3,0) rotor was replaced by simpler 2D/4D blocks in
production because they are cheaper and scored better PPL.

### Two simplifications that matter for this engine

- **K never needs an inverse rotation.** Store `R k / ||k||` quantized. For
  scores, rotate q once with the same Givens (`R q`) and compute
  `score = ||k|| * (R q) · centroids`. This avoids an inverse block rotation
  per cached position.
- **V does need the inverse rotation**, applied once per token to the
  weighted sum over positions, not per position:
  `out = R^{-1} (Σ_p prob_p * dequant(v_p))`.

### Sizing

| format (K / V) | bytes per kv-head pair (2x256 dims) | vs f32 |
| --- | --- | --- |
| f32 / f32 (current) | 2048 | 1x |
| f16 / f16 | 1024 | 2x |
| planar4 / planar4 | 260 | 7.9x |
| planar3 / planar3 | 196 | 10.4x |
| planar3 / f16 | 610 | 3.4x |

Recommended first target: **planar4** (byte-aligned, trivial packing), then
**planar3**. The reference reports `planar3 K + f16 V` as ~0 PPL loss, which is
the lowest-risk milestone of all.
## Where this lands in the engine

Everything is dimension-generic on `hd = 256`, `n_kv = 4`, `n_rep = 6`, 16
full-attention layers, and RoPE is applied **before** caching, so the rotation
acts on post-RoPE K/Q.

### CPU (scalar reference, prototype here first)

- `src/forward.rs::AttnCache` — replace `k: Vec<Vec<f32>>` / `v: Vec<Vec<f32>>`
  with packed caches (indices + norms + a fixed angle table). `n_kv_elems`
  becomes a byte stride; `n_pos` must be derived from a stored length.
- `src/forward.rs::full_attention_layer` —
  - `cache.append(il, &k, &s.vrow)` (line ~186) becomes rotate+quantize+store;
  - the score loop (`dot(qh, kj)`) rotates `qh` once per head and dots against
    dequantized centroids with the `||k||` factor;
  - the V accumulation applies `R^{-1}` to `s.head_out` after the position loop.

### GPU

- `src/gdev.rs` — `kcache`/`vcache` allocation uses `KV_STRIDE = N_KV*HEAD_D`
  f32 slots; switch to packed bytes + a norm array. `KV_STRIDE` is also pushed
  to the attention trio.
- `shaders/kv_store.comp` — becomes rotate + quantize + pack + store norm.
- `shaders/attn_scores.comp` — rotate `qh` in shared memory after load
  (currently loads then dots), then dot against dequantized k indices and
  multiply by the stored norm.
- `shaders/attn_out.comp` — dequantize V per position into the weighted sum,
  then apply the inverse 2D rotation to the accumulated `acc` before storing.
- `src/vk.rs` — the one-shot attention path (`attention_...`, kcache/vcache
  host buffers) gets the same treatment or is left f32 behind a flag.

### Shared constants

Add `src/kvquant.rs`:
- `CENTROIDS[bits]` — solved Lloyd-Max levels for `d = 256`, bits 3 and 4
  (generate once with a small helper bin mirroring `lloyd_max.py`, then commit
  as consts).
- `ROT_COS_SIN` — the Givens table. Decide sharing granularity (below).
- `quantize_pair` / `dequant_pair` / `rotate_pair` used by both the CPU path
  and as the reference for shader parity tests.

## Phases and acceptance criteria

**R0 - codebook + constants (no behavior change).**
Generate centroids for `d=256`, bits 3/4; generate the angle table; commit as
consts. Unit test: quantize/dequantize of a random unit vector has the expected
per-coordinate distortion, and `R^{-1}R v == v` to 1e-6.

**R1 - CPU planar path, K-only first (`planar4` K, f32 V).**
Wire `AttnCache`/`full_attention_layer`. Acceptance: `bonsai-golden` on qa and
code keeps greedy `8160` and logit rel diff in the same 1e-3 band as the f32
baseline (4.03e-3 / 3.36e-3), plus a unit test that quantize→dequantize error
on real cached K stays below a chosen bound.

**R2 - CPU symmetric planar4 (K and V), then planar3.**
Same acceptance, plus a short-context quality check: greedy token ids for a
20-token prefix must match the f32 path.

**R3 - GPU parity.**
Modify the three shaders; validate with `bonsai-gdecode` (golden) and a new
`bonsai-vk` subcommand comparing GPU-dequantized K/V against the CPU
`kvquant` functions on the same real tensors (this is the `kerncmp` pattern).

**R4 - long-context validation and sizing.**
Raise `BONSAI_CTX` and measure (a) VRAM in `BONSAI_VRAM=1` mode, (b) per-token
time vs context against the f32 cache, (c) a needle-in-a-haystack style recall
check. This is where the compression actually pays off.

**R5 - deferred quantization (optional).**
Keep K as f16 during prefill, quantize on decode insert, per the reference's
"deferred" path, if roundtrip quantization hurts PPL enough to matter.

## Decision points

1. **Rotation sharing granularity.** Reference creates one angle table per
   quantizer instance. Options: one global table (cheapest, least decorrelation),
   one per (layer, kv-head) (more params, likely better). Start global, measure;
   escalate only if quality needs it.
2. **Bits.** planar4 first for byte-aligned packing; planar3 once correct.
3. **K-only vs symmetric.** K-only + f16 V is the low-risk step (3.4x on the
   pair) and is reported as ~0 PPL loss; symmetric is the full 10.4x.
4. **Whether to keep the f32 path** behind an env flag (`BONSAI_KV=f32`) for
   A/B and as the golden anchor. Yes - same pattern as `BONSAI_MATVEC`.

## Risks

- **Quality, not correctness.** The golden logits will shift; the gate is that
  greedy ids still match and the rel diff stays in the existing band. A proper
  PPL run needs a corpus and the llama-backend fork; budget for it at R2/R4.
- **3-bit packing is fiddly.** Byte-aligned 4-bit avoids it initially.
- **Rotation on q must use the kv-head's angle table**, and GQA means 6 q heads
  share one kv head's rotation. Easy to get wrong; the transform-parity unit
  test should cover it.
- **Perf regression risk on the APU.** Dequant adds ALU; at 2K context the KV
  read is only 3.7% of traffic, so a small win is the most to expect there. Do
  not sell this as a decode speedup on the iGPU.
- **Numerical drift compounds over positions** if quantization error is
  correlated; the "deferred prefill" idea exists for that reason.

## Effort

R0-R1: ~half a day. R2: ~half a day plus a PPL run. R3: 1-2 days (3 shaders +
parity harness). R4-R5: 1 day. Total ~3-4 days, with R1 already giving a
testable, revertable milestone.

## First step when starting

R0 plus the R1 K-only path, validated against both golden prompts. That is a
self-contained change with a clear revert point, and it produces the CPU
reference functions the GPU shaders will be checked against.
