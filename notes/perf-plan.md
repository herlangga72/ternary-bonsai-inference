# Perf plan: making Ternary-Bonsai-27B inference fast on this box

Status: 2026-09-06. The M1..M7 Rust migration is functionally complete
(correct, golden-verified, standalone). The remaining problem is speed:
~23-34 s/token on the 4-core Ryzen 3200G. This document is the plan to close
most of that gap, in dependency order, with measured evidence and a target for
each step.

## Current state (measured)

| quantity | value | source |
| --- | --- | --- |
| model | qwen35 64L, n_embd 5120, n_ff 17408, 24q/4kv heads, vocab 248 320 | `bonsai-weights config` |
| weight elements | 26.89e9 (~27B), 7.14 GB of PQ2_0 payload | computed from dims |
| decode | ~23-34 s/token (prefill same per token) | README, M6-5 smoke |
| matvec throughput | 0.34 GMAC/s (1 thr), 0.88 GMAC/s (4 thr) | `bonsai-matbench` on blk.0.ffn_up |
| matvec rows per token | ~4.0M (every row is a seek+read syscall today) | computed |
| page-cache RAM scan | ~18 GB/s | membw probe (gcc) |
| disk | SATA SSDs (no HDD); 30 GB zram swap | `lsblk` |

Model weight budget per decode token = stream ~7.1 GB through the CPU.
Memory-bandwidth floor on this machine: 7.1 GB / 18 GB/s ≈ **0.4 s/token**.
A llama.cpp-quality kernel would land near 0.5-0.8 s/token here (decode is
bandwidth-bound, not compute-bound, once the kernel is sane). Today we are
~30-70x above that floor, so there is a large, well-understood gap to close.

## Bottleneck analysis (why 30 s/token)

1. **Scalar inner loop.** `kernels::pq2_matvec_range` decodes every weight
   with per-element integer shifts/masks plus *two* float multiplies
   (`(code-1)*scale`, then `* x[j]`). matbench proves this kernel alone gives
   ~0.88 GMAC/s across 4 threads; the whole model at that rate is ~30 s/token.
   This is the dominant cost (~90%+ of decode time).
2. **Syscall per row.** `gguf::read_bytes_at` does `seek` + `read_exact` per
   row. Decode touches ~4M rows/token -> ~4M syscalls/token plus a fresh
   kernel-side copy into a scratch `raw` buffer per matvec. Roughly a tenth of
   decode time and an obstacle to threading (shared `File` position).
3. **Single-threaded pipeline.** Only the standalone matbench threads; the
   engine's matvecs run on one core. 4 cores sit idle.
4. **Prefill is decode repeated.** `main.rs` forwards prompt tokens one at a
   time, each a full 7.1 GB model pass. A 20-token prompt is 20 model passes.
5. Minor: per-layer String tensor-name lookups, `.clone()` of norm vectors and
   small tensors, ~450 `Vec` allocations/token, whole-model re-read for every
   matvec slice (weight data is not cached per layer across the layer loop,
   but each tensor is used once per token anyway, so no reuse is lost).

## Target

- Decode: 30 s/token -> **~1 s/token** (bandwidth-bound regime). Budget: M8.
- Prefill: per-token model passes -> **one batched pass per prompt** for
  multi-token prompts. Budget: M9 (stretch, after M8 lands).
- Keep the golden checks green at every step (qa/code prompts: greedy id
  8160, logit rel diff in the 1e-3 range).

## M8 - fast decode path (the main event)

Order matters: mmap first (unblocks threading), then kernel, then SIMD.
Each step compiles, passes golden, and is benchmarked before the next.

### M8.1 mmap the tensor payload (removes syscalls + enables threads)

- Replace per-row `File::seek/read` with a read-only mmap of the whole file
  (`memmap2` or `libc::mmap` directly; the repo is dependency-light today, so
  prefer a tiny `libc` shim or `memmap2` - decision at implementation).
- `GGUF` keeps parsing the header/tensor index; add a `payload: &[u8]` view
  over the data section and a `tensor_slice(info) -> &[u8]` accessor.
- Refactor `pq2_matvec_range` to take a `&[u8]` row window (exactly what
  matbench already does) instead of a `&mut GGUF` handle. Numerically
  identical - same bytes, same decode.
- `madvise(MADV_SEQUENTIAL)` on the data section; keep the 7.1 GB working set
  in page cache (9.7 GB available - fits, but nothing else should compete).

Expected: removes ~4M syscalls + row-copy churn per token. ~10-20% alone,
but it is the prerequisite for M8.3.

### M8.2 tighten the scalar row kernel (small, safe, reversible)

- Hoist the group scale: accumulate `x[j] * (code-1)` per element, apply the
  block scale once per 128-block instead of per element (halves the float
  multiplies). Rounding changes at ~1e-7 relative; golden greedy ids are
  unaffected (existing logit agreement is only ~4e-3 anyway).
- Unroll 4 elements per byte (the 2-bit fields of one `qs` byte), keep the
  decode order stable so row-level unit tests stay meaningful.

### M8.3 thread the matvecs

- Row-parallelize each PQ2_0 matvec over the 4 cores: split the row range
  into contiguous chunks; each worker dots its own rows against the shared `x`
  from the mmap slice. Pure read-only parallelism, no cache mutation, join per
  matvec (layer dependencies stay serialized in `forward_hidden`).
- Use `std::thread::scope` (no new dependency) or a tiny persistent pool
  (avoid 4M-thread spawn churn per token; a per-call scoped spawn of 4 threads
  is acceptable at ~450 matvecs/token, but a pool is nicer).

Expected M8.1-3 together: ~3-5x over today -> decode ~5-8 s/token.

### M8.4 AVX2 SIMD dequant-dot

Ternary values are `(code-1)*scale` with code in 0..3, i.e. {-1, 0, +1, +2}.
A block of 128 weights is 34 bytes (2-byte scale + 32 bytes of codes), so the
natural kernel processes 128 floats of `x` (512 B, one cache line) per block:

- Scalar-Rust reference math to preserve: `y_block = scale * (2*S3 + S2 - S0)`
  where `Sc = sum of x[j]` whose 2-bit code equals c. (code 1 contributes 0.)
- AVX2 + FMA implementation sketch: stream the 32 code bytes, expand to
  per-element masks/planes (`_mm256_*` bit tricks or a 16-entry nibble LUT
  through `_mm256_shuffle_epi8`), accumulate the three bucket sums with fused
  multiply-add over 8 x-lanes at a time. Final combine per block.
- Use `std::arch` with `#[target_feature(enable = "avx2,fma")]` + a
  compile-time fallback to the M8.2 scalar path for non-AVX2 machines.
  Runtime detection at load (`is_x86_feature_detected!`).
- The matvec then approaches the ~18 GB/s stream limit; with 4 threads the
  decode becomes bandwidth-bound.

Expected: decode ~1-2 s/token (15-30x vs today), i.e. the llama.cpp-class
number for this CPU/RAM.

### M8 verification

- `cargo build --release`; `bonsai-golden` on qa + code prompts must keep
  greedy id 8160 and logit rel diff in the same 1e-3 ballpark.
- Extend `bonsai-matbench` (or add a matvec microbench variant) so every step
  reports GMAC/s single- and multi-threaded against the same tensor, and a
  `--check` mode compares SIMD results to the scalar decode on random rows
  (exact or <1e-6 rel).
- `bonsai-run -n 2 -v` timing after each step; keep a table in this file.

## M9 - batched prefill (stretch)

Problem today: a 20-token prompt = 20 full model passes (~11 min at 30 s/tok;
even at 1 s/tok it is 20 s of avoidable work). Fix: process the whole prompt
in one weight pass per layer:

- Embedding: gather the N prompt rows at once.
- FFN + attention projections: matrix * `[N x n_embd]` activation block
  instead of `[1 x n_embd]`; weights stream once for all N tokens.
- Full-attention layers: compute q/k/v for all N, causal scores over the N
  positions, output all N at once (no per-token KV append needed for prefill).
- Recurrent (gdn) layers: projections and the conv1d are batch-friendly; the
  `gdn_step` recurrence stays sequential per position over precomputed q/k/v
  (the state update is order-dependent by definition). State math is ~150M
  MAC/token/layer-set, small next to the matvecs, so serializing it is fine.
- Workspaces sized `[N x n_embd]`; N = prompt length.

Expected: a 150-token prompt from ~75 min (today) to tens of seconds.
The block-matmul kernel from M8.4 generalizes (same row kernel, batch of x).

## Later / optional (not in the fast-decode critical path)

- Self-contained profiling flag (`-v` per-stage timings, or `#[feature]`
  counters) to catch the next bottleneck after M8 (likely attention KV +
  softmax at long context, then the GDN state update).
- Speculative decoding with the dspark 3.6B sidecar (`*dspark*.gguf` present
  in the repo): real 2-3x decode win, but it lives in the fork's C++
  acceptance loop today and porting it is a large separate effort. Keep as a
  documented follow-up (README already flags this).
- Layer fusion (norm + matvec + residual in one pass) and scratch reuse are
  small constant-factor wins once M8 lands; do them only if profiling shows
  they matter.

## GPU? (question, 2026-09-06)

Hardware facts on this box:

- One GPU: the Vega 8 iGPU (Picasso/Raven2, gfx902, 8 CUs / 512 SP, ~1.25 GHz,
  ~1.3 TFLOPS FP32 peak). No discrete card in the PCIe slots.
- It is usable today: `/dev/kfd` present, OpenCL platform works
  (`clinfo`: AMD-APP, gfx902), 512 MB carve-out, ~8 GB host-visible global.
  ROCm 6.0 no longer officially supports gfx900-class, so treat HIP as
  experimental.
- The iGPU shares the same DDR4 as the CPU. Measured RAM scan is ~18 GB/s for
  the CPU; the iGPU reads the same memory controller, so it does not add
  bandwidth.

Why the iGPU does not change the decode plan:

- Decode streams ~7.1 GB of weights per token. The floor is DRAM bandwidth
  (~0.4 s/token), and that floor is shared with the CPU. Adding the iGPU
  cannot lower it; it would only move the same FMA work to a device that is
  also waiting on the same memory.
- The packed-PQ2_0 dot is not iGPU-compute-limited either: streaming 7.1 GB
  at 18 GB/s needs only ~68 GMAC/s (3.76 MAC/byte), well under what the Vega
  can do once dequantized, so the iGPU would idle on bandwidth like the CPU.
- 15 GB total RAM with the 7.1 GB model page-cached already leaves ~9 GB
  free. GPU-side copies of weights would fight the CPU page cache for the
  same 15 GB and push mapped pages out (SSD re-read stalls).

Where a GPU would actually pay:

1. **Discrete card with its own VRAM/bandwidth** (e.g. any 8 GB+ card:
   224-288 GB/s vs 18 GB/s) - that is the only hardware on this box that
   lowers the decode wall (floor ~30-60 ms/token compute-side, real-world
   maybe 3-8 tok/s for this ternary model) and makes batched prefill fast.
   Cost: weight upload over PCIe at load, a HIP/OpenCL/Vulkan PQ2_0 kernel,
   and reworking the pure-Rust no-llama.cpp architecture. Separate project.
2. **Batched prefill (M9)** is compute-bound once weights stream once per
   prompt (N tokens x 27e9 MAC); extra FLOPS help there. The iGPU could cut
   prefill time for long prompts even though it cannot help decode.
3. **dspark speculative decode** - once M8/M9 land, the verifier is
   bandwidth-bound; a GPU only helps the draft-model compute, not the wall.

Recommendation: do M8 on CPU first - it reaches the shared-DRAM decode wall
(~0.5-1 s/token) with a fraction of the effort of any GPU path and is a
prerequisite data point (kernel design, golden checks) for a later GPU port.
Revisit "add GPU" only if (a) a discrete card becomes available, or (b) M9
prefill on long prompts is the dominant user pain and the iGPU's extra FLOPs
are worth an OpenCL experiment.

## Risks / watch-outs

- **Autovectorization will not happen by itself.** Rust rarely vectorizes
  these byte-scattered loops; the AVX2 path must be explicit intrinsics.
  Validate against the scalar path per tensor with a check mode.
- **Numeric drift**: M8.2/M8.4 reorder summation and scale application.
  Expected ~1e-6 relative on logits; golden comparisons are greedy-id + 1e-3
  level, so this is safe, but keep `decode_pq2_0_row` and the exactness tests
  untouched as the numeric anchor.
- **Threading**: all parallel work is read-only over the mmap + shared `x`;
  KV/GDN caches mutate only on the single decode thread. Never share the
  current `File`-based reader across threads (seek position races) - that is
  exactly why mmap must land first.
- **Memory**: 7.1 GB mapped + page cache, 9.7 GB available, zram swap 30 GB.
  Fine today, but anything else that grabs RAM during a long reasoning run
  will push mapped pages out and stall decode on SSD re-reads.
- The **LM head** is 1.27e9 MAC (248k rows) per token - ~5% of the model but
  248k of the ~4M rows. After mmap it is just bandwidth; no special casing
  needed initially.

## Suggested sequencing (commit per step, mirroring M1..M7 style)

1. `M8.1 mmap` - GGUF payload view, slice-based row kernel, same numbers.
2. `M8.2 scalar` - hoisted scale + 4-per-byte unroll, bench in matbench.
3. `M8.3 threads` - scoped row-parallel matvecs, bench whole decode.
4. `M8.4 AVX2` - intrinsics dot with scalar fallback + check mode, bench.
5. `M9 batched prefill` (optional) - block matvec over prompt tokens.
