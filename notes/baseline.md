# Baseline and optimization log (2026-09-12/13)

Machine: Ryzen 3 3200G (4 cores), Vega 8 iGPU (gfx902, RADV Mesa 25.3.5),
15 GB DDR4. Branch `master`, HEAD `48c8f25` at baseline. The box is shared
(browser + editor), so timings fluctuate; **min of N** is used as the estimator.

## Baseline as found

| item | value |
| --- | --- |
| `cargo build --release` | clean |
| `cargo test --release` | 24 test binaries, 0 failures |
| CPU decode (`bonsai-decode`, 8-token min) | 1.61 s/token |
| CPU golden qa | greedy 8160 MATCH, logit rel 4.03e-3 |
| GPU decode (`bonsai-gdecode`, 20 token) | ~29-30 s total, ~1.4 s/token |
| GPU golden qa | greedy 8160 MATCH, logit rel 4.41e-3 |
| GPU record / submit+read | ~2.5 ms / ~1.29 s per token |
| `bonsai-matbench` (scalar kernel) | 0.97 GMAC/s 4-thread |

## What was changed

### CPU AVX2 PQ2_0 row dot (`src/kernels.rs`)

Before: per 128-weight block, the 256-bit accumulator was stored and its 8 lanes
summed, then scaled; each code byte was a separate 8-bit load.

After:
- four independent 256-bit accumulators (breaks the FMA dependency chain),
- eight code bytes per iteration via one unaligned `u64` load,
- one accumulator kept across blocks, scale folded with a single FMA per block,
- one 8-lane horizontal sum per row.

Measured, alternating A/B, min of 6 pairs: **1.42 -> 1.08 s/token** (new faster
in all 6 pairs; ~1.3x). Isolated kernel: **9.2 GMAC/s** single-thread, 28 GMAC/s
across 4 threads, 27x the scalar reference.

### GPU fused single-pass matvec (`shaders/pq2_matvec_fused.comp`, `src/vk.rs`)

Before: `pq2_partial` wrote one float per (row, 128-block) pair to a `partials`
buffer, then `pq2_rowsum` read them back. For this model that is 840 MB written
+ 840 MB read per token, ~24% extra traffic on top of the 7.14 GB weight stream.

After: one workgroup per row, blocks strided across the workgroup, reduced in
shared memory; `partials` is gone and each matvec is one dispatch.

Measured on gfx902: 30 -> 31 s per 20 tokens, i.e. **~3% slower here** (this
part is latency/occupancy-bound, not bandwidth-bound). Kept as the default
because the removed traffic is what limits the bandwidth-bound RX 7600 target;
`BONSAI_MATVEC=2pass` restores the old path. Golden identical in both modes
(greedy 8160, rel 4.409e-3).

### Build tuning: no effect

`lto = "fat"` + `codegen-units = 1` and `-C target-cpu=native` were both A/B
tested and gave no measurable decode change (min 8.6 vs 8.7 s and 8.8 vs 9.0 s
over 8 tokens). The hot loop is hand-written SIMD, so there is nothing for the
compiler to improve. Reverted; builds stay fast and portable.

### Tooling

`bonsai-matbench` now measures the **shipped** kernel (plus a scalar reference)
with best-of-N instead of a private scalar reimplementation, which understated
the real throughput by ~27x.

Shader build wiring: `build.rs` now guarantees `OUT_DIR` holds a SPIR-V module
for every `.comp` (compile with `glslangValidator`, or copy the committed
`spv/` fallback only when the compiler is absent), and `vk.rs` includes from
`OUT_DIR`. Before, `vk.rs` read the committed `spv/`, so editing a shader
silently had no effect, and a broken shader was not caught. A bad shader now
fails the build with the glslang error.

## Reproduce

```sh
cargo test --release
./target/release/bonsai-decode  Ternary-Bonsai-27B-PQ2_0.gguf 8
./target/release/bonsai-matbench Ternary-Bonsai-27B-PQ2_0.gguf 7
./target/release/bonsai-gdecode Ternary-Bonsai-27B-PQ2_0.gguf \
  golden/prompts/qa.txt golden/qa.logits.bin
BONSAI_MATVEC=2pass ./target/release/bonsai-gdecode Ternary-Bonsai-27B-PQ2_0.gguf \
  golden/prompts/qa.txt golden/qa.logits.bin
```

## Still open

- Fuse the paired/independent matvecs in a layer (wq/wk/wv; ffn gate/up) into
  one dispatch: ~1700 dispatches/token, each with a full 3-buffer barrier.
- Descriptor-set caching (host record is only ~2.5 ms/token, so low value).
- RX 7600 validation of the fused matvec (expected ~19% traffic cut).
- Prefill still uses the N-column two-pass (`pq2_partial_n`/`pq2_rowsum_n`).
