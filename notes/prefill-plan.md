# Prefill plan: batched (parallel) prompt processing

Status: 2026-09-09. Scope: make *prompt processing* (prefill) faster, i.e. cut
time-to-first-token. This is the G3 work named in `notes/perf-plan.md`, made
concrete against the current GDev single-stream engine (`src/gdev.rs`).

## Why prefill is slow today (root cause)

Prompt tokens are fed through the **decode loop one at a time**, each token a
full single-token forward pass. In `src/bin/bonsai-grun.rs` and `src/main.rs`:

```rust
for (pos, &tok) in toks.iter().enumerate() {
    let emb = dec.w.row_f32("token_embd.weight", tok as u64)?; // one embed row
    hidden = dev.forward_token(pos, &emb)?;                     // full 64-layer pass
}
```

`forward_token` (`gdev.rs:507`) records and submits one command buffer per token
that streams **all ~7.1 GB of weights**. So an N-token prompt reads the weight
stream **N times**:

- measured today on the gfx902 iGPU (shared DRAM): ~1.4-1.6 s/token =>
  a 150-token reasoning prompt costs ~210-240 s (~3.5-4 min) before the first
  generated token;
- even on the RX 7600 (288 GB/s dedicated VRAM), the token loop would be ~N x
  ~0.03-0.05 s, so a 150-token prompt ~5-7 s. Acceptable but wasteful: the
  weights are re-read per position.

**The fix is the same for both**: during prefill all N positions are known up
front, and their weights are the *same* matrices, so read each weight block
**once** and produce N outputs (a GEMM over the N-wide activation tile) instead
of N separate matvec passes. Weight memory traffic drops from N passes to ~1
pass; prefill time becomes ~a single full-model pass (plus the N-wide
attention and the small GDN recurrence), independent of N up to the window cap.

Projected (unmeasured) bounds:

| engine | today (token loop) | batched (weight stream once) |
| --- | --- | --- |
| gfx902 iGPU | ~1.4-1.6 s x N | ~1.5-2 s per window, independent of N |
| RX 7600 | ~0.03-0.05 s x N | ~0.05-0.3 s (short prompts = fixed cost) |

## Architecture facts the plan must respect

Read `notes/qwen35-arch.md`, `notes/qwen35-forward-spec.md`, `src/forward.rs`,
`src/gdn.rs`, `src/gdev.rs` before coding. Two layer kinds (from `forward.rs`
and `gdev.rs::rec_attn_layer` / `rec_gdn_layer`):

1. **Full-attention layers** (`il % interval == 3`, 16 of 64): attn_norm,
   wq/wk/wv projections, split [q|gate] per head, per-head RMSNorm + IMROPE
   rope, KV-store at `pos`, GQA causal attention (scores -> softmax -> weighted
   v), gate+out projection (wo), residual. Prefill is **parallel across the N
   positions within the layer**: only the causal attention and the KV writes
   couple positions.
2. **GDN recurrent layers** (48 of 64): attn_norm, projections (attn_qkv,
   attn_gate, ssm_beta, ssm_alpha), causal conv1d over channels (kernel 4),
   l2-norm q/k, the **sequential** gated-delta-net state recurrence
   (`gdn_step`), gated output norm + silu(z), ssm_out projection, residual.
   Projections and conv are parallel across N; **only the state recurrence is
   sequential across positions within the layer**.

Every layer also has an FFN (gate/up -> silu*h -> down) and residuals, and the
whole model has output_norm + a CPU-side LM head / embed gather.

Key kernel/plumbing today (`src/vk.rs`):
- All PQ2_0 matvecs are the **two-pass v3** kernel: `pq2_partial` (per-128-block
  partial dots into `partials`) then `pq2_rowsum` (`gdev.rs` `partials` buffer
  is `N_FF * (N_EMBD/128)`). It multiplies **one** x vector by the weight
  matrix (`rec_matvec`, `vk.rs:1938`). This is the function to generalize to
  N columns.
- Weights live in VRAM once (G1 done). Small ops (rms, rope, elem, gdn, conv,
  attn) are recorded dispatches with explicit barriers; one in-order command
  buffer per token.

Correctness contract for generation after prefill (do not break):
- KV caches are laid out `[pos][kv_head][head_dim]`, appended per position, so
  generation continues by appending at `pos = N`.
- GDN conv cache holds the last (kernel-1)=3 pre-conv inputs per channel; GDN
  state holds the final recurrence matrices (48 x 128 x 128 f32). After batched
  prefill these must equal what the token loop leaves behind, so
  `forward_token` after prefill is bit-consistent.

## Plan (do in order)

### P1 - Batched activation tile + embedding gather (correctness, no new kernels)
- Introduce an N-wide activation layout `X[t]` (decide `[t][n]` column-major vs
  `[n][t]`; pick whatever the existing kernels address most cheaply - weights
  are row-streamed, so an `[n][t]` tile keeps one weight row contiguous with N
  outputs - verify against the two-pass kernel's addressing).
- Batch embed gather: read N token rows of `token_embd.weight` into one device
  buffer up front (still host-side, per-row read as today but once for the
  batch; no 0.3 GB weight upload - embeddings stay CPU/row-fetched per
  forward.rs convention).
- Gate on an env/CLI like `BONSAI_BATCH` (default 0 = current token loop) so
  every phase ships behind a flag and decode is untouched.
- Verify: gather N rows == N individual `row_f32` reads, bit-exact.

### P2 - Generalize the two-pass PQ2 matvec to an N-column GEMM (biggest lift)
- Change `pq2_partial` / `pq2_rowsum` (`shaders/pq2_partial.comp`,
  `pq2_rowsum.comp`, `vk.rs::rec_matvec`) so one workgroup that streams one
  weight row computes N partial dot products (one per activation column), and
  `pq2_rowsum` writes N outputs per row. Partial buffer grows to
  `rows * (n_blocks) * N`; dimension the group/tile so a weight block is read
  once and reused across the N columns (LDS-tile the activation columns). This
  is where weight traffic drops from N passes to one.
- Keep the single-vector path as the decode fallback; the N=1 result of the
  new kernel must match the old matvec bit/rel-exactly.
- Verify: for small N, CPU batched matmul vs new GPU GEMM <= ~1e-3 rel on real
  tensors (extend the existing row-check style used by G0/G1).

### P3 - Batched full-attention prefill (parallel across N)
- Run wq/wk/wv (and the layer's FFN) through the P2 GEMM for all N positions.
- rope each position with its own `pos` (rope dispatch already takes `pos` in
  the push-constant - make it per-position).
- Write all N k/v blocks into the KV caches in one store pass, at positions
  `0..N`, keeping the `[pos][kv][hd]` layout (so decode appends at N).
- Causal attention over the batch: per query position t attend to keys `<= t`.
  Build on the existing `attn_scores` / `softmax_inplace` / `attn_out` kernels
  but with an N-wide query tile and an N x N causal score matrix (masked to the
  lower triangle), GQA grouping unchanged. Softmax is per-row.
- Output the per-position layer out; residuals and the next layer operate on
  the whole X tile.
- Verify: after a batched prefill, the KV cache == KV cache from the token loop
  for the same prompt; hidden at each position matches within ~1e-3 rel.

### P4 - Batched GDN (recurrent) layers (the sequential part)
- Projections (attn_qkv / attn_gate / ssm_beta / ssm_alpha) all via P2 GEMM over
  N positions.
- **Causal conv1d**: replace the running-3-sample window with a per-channel
  causal convolution over the N-window, seeded by the pre-prompt conv state
  (normally zeros at the start of a sequence). Output all N positions in
  parallel, then silu.
- **State recurrence**: keep it simple first - run the existing `gdn_step` N
  times per recurrent layer (one or a few small sequential dispatches over the
  already-computed projections). State is small (48*128*128 f32 ~ 3 MB/layer),
  so this is cheap next to the (once-read) weight stream. Defer a chunked
  parallel scan (associative scan over the linear delta recurrence) to P6 only
  if large-N prefill latency demands it.
- Gated output norm + ssm_out GEMM batched.
- Verify: final conv cache + GDN state == token-loop state for the same prompt;
  continuation after batched prefill is greedy-identical for K tokens.

### P5 - Integrate `prefill_batch(N)`, replace the prompt loop
- New entry (mirror `forward_token` but over N): gather N embeddings, run the
  batched 64-layer forward, output_norm on the batch, compute logits for
  position N-1 only (LM head stays CPU) to sample the first generated token,
  then hand off to the existing single-stream `forward_token` decode loop at
  `pos = N`.
- Fallback: for N <= 1 (or tiny prompts) keep the token loop.
- Window large prompts: if N exceeds a cap (scores/partial/activation buffers,
  BONSAI_CTX, VRAM), run the batch as sequential windows that append KV/conv/
  state window by window; weights stay resident across windows so traffic is
  still ~1 pass per window.
- Verify: `bonsai-grun` with a real prompt, greedy continuation after batched
  prefill == current engine for the first K tokens; golden qa + code still
  greedy 8160.

### P6 - Tune + measure on target hardware
- Per-phase micro-benchmarks (GMAC/s and GB/s) like `notes/gpu-numbers.md`.
- Measure time-to-first-token before/after on: golden qa (~20 tok), the code
  prompt, and a ~150-200-token reasoning prompt.
- Tuning levers: wave32/LDS/vectorization of the N-column GEMM, KV f16 for long
  prompts, batch window size, GDN parallel scan if the sequential loop shows
  up in profiling.
- Record a new table in this doc (and `notes/gpu-numbers.md`).

## Verification harness (run every phase)

```sh
cargo build --release
./scripts/verify_gpu.sh                       # existing GPU engine checks stay green
# new: batched-prefill vs token-loop greedy-equivalence + golden id 8160
./target/release/bonsai-grun Ternary-Bonsai-27B-PQ2_0.gguf "What is the capital of France?" 24
./target/release/bonsai-gdecode Ternary-Bonsai-27B-PQ2_0.gguf golden/prompts/qa.txt golden/qa.logits.bin
```

Conventions to preserve:
- CPU scalar kernels and the single-vector decode path stay untouched as the
  numeric anchor (handoff and README).
- Every new kernel gets a CPU-vs-GPU check before it is wired in.
- Numeric drift from batched summation order is expected (~1e-6..1e-3 rel);
  golden checks (greedy id 8160, ~1e-3 rel) tolerate it.

## Risks / watch-outs

- **The P2 GEMM rewrite is the main risk**; it touches the kernel the whole
  engine depends on. Do it behind the batch flag, keep the N=1 path exact, and
  prove N=1 equivalence before N>1.
- **GDN sequential scan** at very large N: mitigated by windowing and, if
  needed, a chunked parallel scan in P6. Conv must be seeded so window edges
  match the running-window semantics exactly.
- **VRAM is tight on the RX 7600** (7.1 GB weights + batch tiles + KV + scores):
  cap the window, keep activations/partials host-visible on APUs, KV f16 later.
- **Numeric drift** across the batched attention/softmax and GEMM: keep rel
  diffs within the golden tolerance; validate hidden per position in P3/P4.
- **Small prompts** may not repay batching (fixed cost); the fallback keeps
  them on the token loop.

## Implementation status (2026-09-09) - committed on master

| phase | commit | what landed | verified on gfx902 |
| --- | --- | --- | --- |
| P1 | `3ff0bec` | `src/prefill.rs` N-wide Tile + `batch_embed`, `bonsai-batch` CPU reference; gated | embed gather + ref hidden bit-exact |
| P2 | `3a69d8b` | `pq2_partial_n`/`pq2_rowsum_n` N-column two-pass GEMM (`rec_matvec_batch`), `bonsai-batchgemm` | N=1 bit-identical to old matvec; N=8 <= 3.3e-7 rel vs CPU |
| P3 | `5ee8e58` | `GDev::prefill_batch` batched full-attention layers + FFN over N; `bonsai-bprefill` | KV/hidden <= ~5e-4 rel, 0 > 1e-3; greedy equal |
| P4 | `2249a1d` | batched GDN projections/conv over N, sequential per-position recurrence (all 64 layers batched) | state/conv <= ~5e-4 rel; greedy equal |
| P5 | `1a34188` | `prefill_batch_windowed` + `batch_cfg`, replaces per-token prompt loop in `bonsai-grun`; `bonsai-gprefill` end-to-end | greedy continuation equal token loop K=8, qa 248068 / code 7734 |

`cargo build --release` and `cargo test --release` are green at every step; the
single-stream decode path used for generation is untouched throughout.

## P6 measured result (2026-09-09, gfx902 iGPU) - batched is NOT faster here

Time-to-first-token (prefill time), real model, RADV gfx902 APU:

| prompt (tokens) | token loop | batched | ratio |
| --- | --- | --- | --- |
| qa (11) | 13.8 s | 14.4 s | 1.04x |
| code (17) | 21.1 s | 24.0 s | 1.14x |
| long (146) | ~183 s* | 299.3 s | ~1.6x |

*token-loop time extrapolated at the measured ~1.25 s/token for this box.

Because batching is not faster on this shared-bus APU, `bonsai-grun`'s default
is now **device-aware** (`batch_cfg(discrete)`): on an APU the default is the
token loop; batching is the default only on a discrete (dedicated-VRAM) GPU and
is always available explicitly via `BONSAI_BATCH=<window>` (e.g. `=64`). On this
gfx902: default took the token loop (21 tok / 26.0 s); `BONSAI_BATCH=64` took
the batched path (21 tok / 34.8 s).

**Conclusion: on the gfx902 this APU, batched prefill is correctness-equivalent
but does not reduce time-to-first-token** - it is roughly equal at small N and
~1.6x slower at N=146. Reason (consistent with `notes/gpu-numbers.md`): the
iGPU decodes at only ~5 GB/s of an 18 GB/s shared bus, i.e. it is small-op /
launch bound, NOT weight-bandwidth bound. Batching only removes the per-token
weight re-read; the per-position causal-attention and GDN recurrent work still
run sequentially per position and dominate on this hardware, and the windowed
batch adds tile-copy + per-layer-submit overhead on top.

The weight-read-once optimization is designed to pay off where decode is
weight-bandwidth bound - the discrete RX 7600 (288 GB/s dedicated VRAM, decode
~0.03-0.05 s/token is ~7.1 GB/288 GB/s bandwidth-limited). That GPU is not
installed here, so the projected speedup is **unvalidated on this box**.
Remaining work to claim the speedup: install the RX 7600 and measure; if the
sequential per-position GDN/attention cost still dominates there, batch the
causal attention over N (real prefill GEMM) and add a chunked parallel scan for
the GDN recurrence.

## Exit criteria (revised)

- [x] Batched prefill replaces the token loop in `bonsai-grun`; golden qa/code
  greedy-first-token equal; generation after prefill greedy-identical to the
  token loop (K=8) on gfx902.
- [ ] A ~150-token prompt reaches its first generated token in roughly a single
  full-model pass **on the RX 7600** (not measurable on this shared-bus iGPU).
Numbers recorded in this doc and `notes/gpu-numbers.md`.
