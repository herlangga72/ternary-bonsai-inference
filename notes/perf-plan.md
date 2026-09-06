# Perf plan: Ternary-Bonsai-27B inference fast on CPU, then on an RX 7600

Status: 2026-09-06. The M1..M7 Rust migration is functionally complete
(correct, golden-verified, standalone). Speed today is ~23-34 s/token on the
4-core Ryzen 3200G. An RX 7600 arrives in a few days; this document tracks
both the interim CPU work and the GPU build-out that is the real target.

## Current state (measured)

| quantity | value | source |
| --- | --- | --- |
| model | qwen35 64L, n_embd 5120, n_ff 17408, 24q/4kv heads, vocab 248 320 | `bonsai-weights config` |
| weight elements | 26.89e9 (~27B), 7.14 GB of PQ2_0 payload | computed from dims |
| decode | ~23-34 s/token (prefill same per token) | README, M6-5 smoke |
| matvec throughput | 0.34 GMAC/s (1 thr), 0.88 GMAC/s (4 thr) | `bonsai-matbench` on blk.0.ffn_up |
| matvec rows per token | ~4.0M (every row is a seek+read syscall today) | computed |
| page-cache RAM scan | ~18 GB/s | membw probe (gcc) |
| iGPU (today) | Vega 8 / gfx902, OpenCL works, shares the 18 GB/s DRAM | clinfo |
| disk | SATA SSDs (no HDD); 30 GB zram swap | lsblk |

## Bottleneck analysis (why 30 s/token)

1. **Scalar inner loop.** `kernels::pq2_matvec_range` decodes every weight
   with per-element integer shifts/masks plus two float multiplies. matbench
   proves this kernel alone gives ~0.88 GMAC/s across 4 threads; the whole
   model at that rate is ~30 s/token. This is >90% of decode time.
2. **Syscall per row.** `gguf::read_bytes_at` does `seek` + `read_exact` per
   row. Decode touches ~4M rows/token -> ~4M syscalls/token, plus row-copy
   scratch per matvec. Also blocks threading (shared File position).
3. **Single-threaded pipeline.** Only the standalone matbench threads.
4. **Prefill = decode repeated.** `main.rs` forwards prompt tokens one at a
   time, each a full 7.1 GB model pass.

## Decision (2026-09-06): GPU-first, RX 7600

An RX 7600 (Navi 33, gfx1102) is being added to this box. Its relevant specs:

- 8 GB GDDR6 on a 128-bit bus @ 18 Gbps -> **288 GB/s** (vs ~18 GB/s today)
- 2048 shaders, up to ~2.66 GHz, ~21.7 TFLOPS FP32 peak
- PCIe 4.0 x8 (~14 GB/s one-way; fine for a one-time weight upload)
- Supported by ROCm 6.0 (`gfx1102`), which is already installed at /opt/rocm

New decode floor on the 7600: 7.1 GB / 288 GB/s = **~25 ms/token** for the
weight stream. With a sane PQ2_0 kernel that is ~20-40 tok/s decode (600-1000x
today), and prefill for a normal prompt drops to ~1 s. That is the level this
repo is now being built toward.

### Compute API choice: OpenCL first

Recommended: **OpenCL**, not HIP, for the first GPU backend.

- It is the only compute API verified working on this box right now
  (AMD-APP platform enumerates gfx902), so kernels can be written, run, and
  validated against the CPU decoder *before* the 7600 arrives.
- ROCm 6.0 ships the OpenCL runtime for gfx1102, so the same kernels run on
  the 7600 when it lands.
- Kernels are plain C with runtime `clBuildProgram` compilation: fast
  iteration, and trivially mirrors the scalar CPU kernels one-to-one for
  validation.
- HIP remains the fallback if OpenCL perf on RDNA3 disappoints; the kernel
  math ports nearly line-for-line.

Keep the CPU path alive as the numeric reference and fallback (it is what the
golden checks run against). Structurally this is a **compute-backend split**,
not a rewrite: weights + forward semantics stay in Rust; only matvec-scale
kernels move behind a trait.

### Two-track work plan

The CPU track keeps value flowing before the card arrives and produces the
mmap/upload plumbing + numeric reference the GPU track needs.

#### Track CPU (do now, slim)

| step | work | expected |
| --- | --- | --- |
| M8.1 | mmap the payload; slice-based row kernel (kills 4M syscalls/token, enables threads and GPU upload) | ~10-20% + prerequisite |
| M8.3 | thread matvecs over 4 cores (read-only row chunks, scoped threads) | decode ~5-8 s/token |
| (defer) | M8.2 scalar tightening, M8.4 AVX2 - only if CPU stays useful after the GPU lands; the 7600 makes CPU SIMD a hobby path | - |

#### Track GPU (target: 7600, build on gfx902 now)

| step | work | validates on | expected on 7600 |
| --- | --- | --- | --- |
| G0 | OpenCL kernels: PQ2_0 matvec (2-bit unpack, block scales), RMSNorm, silu/softplus/sigmoid, softmax, IMROPE, GDN step. Micro-bench each vs the CPU kernel on random rows | gfx902 (OpenCL works today) | kernel math identical to CPU |
| G1 | device weight store: upload packed 2-bit tensors from the mmap (no repack, no f32 blowup); KV (f32) + recurrent state buffers on device | gfx902 | 7.1 GB fits 8 GB; KV 64 KB/token |
| G2 | single-stream GPU decode: layer loop launches kernels per tensor, KV/state stay on device, logits sampled on host | gfx902, then 7600 | decode ~20-40 tok/s |
| G3 | batched prefill on device (matmul over N prompt tokens), then tune: wave32/LDS/vectorization, KV f16 at long ctx | 7600 | prefill ~1 s for a 150-token prompt |

G0-G2 are designed to be validated on the current iGPU for *correctness*
(golden greedy id 8160 must match), knowing gfx902 throughput is irrelevant;
the 7600 numbers are the goal.

## GPU kernel design notes

- Keep weights packed as PQ2_0 (34 B per 128-weight block = 2 bits + one fp16
  scale). Dequantize in-kernel: one workgroup tile loads x (5120 floats =
  20 KB) into LDS once, then streams weight blocks; each 128-block maps to
  wave/lane unpacking (32 code bytes -> 128 two-bit codes). Scale applied per
  block at the end: `block_dot = scale * (2*S3 + S2 - S0)` over x buckets.
- A decode token is 7.1 GB of device-memory reads; the kernel must stream
  weight rows sequentially and saturate the 288 GB/s bus, so row-block
  work assignment and vectorized 16-byte code loads matter more than FMA
  count. Target ~60-75% of bus bandwidth (200+ GB/s).
- Embedding and LM head are 0.34 GB tensors each (1.27e9 elems): plain
  gather/matvec on device. Embedding row lookup per token is a single row.
- Recurrent (gdn) state is 48 x 128 x 128 f32 = 786 KB/layer x 48 layers =
  151 MB; keep it on device and run the GDN update as a small kernel (order-
  dependent per token, cheap FLOPS).
- KV for full-attention layers is f32 64 KB/token (4 kv heads x 256 x 4 B x
  16 layers). 32K ctx = ~2 GB; switch KV to f16 if long-context runs matter.
- Weight upload once at load: 7.1 GB over PCIe 4.0 x8 (~10-14 GB/s) ~ 0.5-1
  min. Do not copy weights per token.

## Verification (both tracks)

- Every step: `cargo build --release`; `bonsai-golden` on qa + code prompts
  must keep greedy id 8160 and logit rel diff in the same 1e-3 ballpark.
- OpenCL kernels get a CPU-vs-GPU row check mode (exact or <1e-6 rel) before
  they are wired into the graph.
- Keep `decode_pq2_0_row` and the CPU scalar kernels untouched as the numeric
  anchor.
- Micro-benchmarks per kernel (GMAC/s and GB/s) so each G step reports a
  number; record the table here.

## Risks / watch-outs

- **Toolchain risk on gfx902**: ROCm 6.0 does not officially support
  gfx900-class; OpenCL (AMD-APP) does enumerate it today. If gfx902 kernels
  misbehave, validate G0/G1 kernels against CPU math on the same data without
  needing the GPU to be fast - only correct.
- **OpenCL perf on RDNA3**: officially supported but historically a bit under
  HIP. Budget a HIP port of the matvec kernel as plan B (same math).
- **Numeric drift**: GPU summation order differs from CPU; expected ~1e-6 rel
  on logits. Golden checks are greedy-id + 1e-3 level, so safe.
- **8 GB VRAM is tight**: 7.1 GB weights + ~0.5-1 GB runtime/KV/workspace.
  Watch context growth (KV f16 later) and driver reserved memory.
- **Autovectorization will not happen by itself** (CPU). Rust rarely
  vectorizes byte-scattered loops; the CPU AVX2 path (if ever needed) must be
  explicit intrinsics.
- **Threading (CPU)**: parallel work is read-only over the mmap + shared x;
  KV/GDN caches mutate only on the single decode thread. mmap must land before
  any threading (never share the File-based reader).

## Sequencing

1. M8.1 mmap - GGUF payload view, slice-based row kernel, same numbers.
2. M8.3 threads - scoped row-parallel matvecs (CPU decode ~5-8 s/token while
   waiting for the card).
3. G0 OpenCL kernels + CPU-vs-GPU row checks (runs on the iGPU today).
4. G1 device weight store + G2 single-stream GPU decode; golden id 8160 on the
   iGPU.
5. When the RX 7600 arrives: benchmark, tune to bus bandwidth, then G3
   batched prefill and (later) dspark speculative decode.
