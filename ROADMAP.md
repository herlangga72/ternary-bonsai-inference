# Migration roadmap: llama.cpp -> pure Rust

The compute engine was originally the PrismML llama.cpp fork (C/C++) reached
through a thin Rust FFI in `src/llama.rs`. This document tracked the incremental
migration to a pure-Rust engine. The migration is complete: `bonsai-run` is
standalone, and llama.cpp is only linked by the optional `llama-backend`
comparison tools. Rule followed: inference stayed working and verified after
every milestone; each milestone replaced one subsystem and validated against
the previous behavior before the next started.

## Strategy

Migrate from the outside in:

1. Replace pure-decision components that touch the FFI boundary first (cheap,
   immediately verifiable): sampler, detokenizer, GGUF parsing, tokenizer.
2. Replace compute components last, because each kernel needs a numerical
   reference. `llama.cpp` stays as the oracle until the last milestone.
3. Every compute kernel gets a reference probe: compute the same op on the same
   weights with ggml (via a tiny C++ harness or llama-cli artifacts) and compare
   outputs bit-exactly or within tolerance before wiring it into the graph.

## Milestones

| # | Milestone | Status | How it is verified |
| --- | --- | --- | --- |
| M1 | Rust sampler (top-k, top-p, min-p, temp, seeded dist) replaces the `llama_sampler_*` chain | done | `bonsai-run` generates coherent text without llama sampler calls |
| M2 | Rust GGUF reader: metadata, tensor index, block decode (F32, PQ2_0, Q4_1, TQ1_0) | done | header matches Python tool; Rust retag byte-identical to Python; PQ2_0 dequant matches C formula; layout chain verified on 6.7 GB file |
| M3 | Rust tokenizer: qwen35 pre-tokenizer + BPE merges + special tokens | done | token ids match `llama_tokenize` on 1581 corpus lines (0 mismatches); wired into bonsai-run |
| M4 | Rust detokenizer (vocab text + byte table) | done | output text identical to llama.cpp path; verified in end-to-end run |
| M5 | Rust kernels: dequant PQ2_0 matmul, RMSNorm, partial RoPE, gates | done (RMSNorm + PQ2_0 dequant; RoPE/gates folded into M6) | RMSNorm 2.3e-7 rel vs ggml; PQ2_0 row dequant bit-exact vs ggml `to_float` on 7 real tensor rows |
| M6 (prep) | golden logits captures (`bonsai-logits`, `golden/`) + architecture study (`notes/qwen35-arch.md`) | done | captured two llama.cpp logits rows as reference; documented real tensor layout and open graph questions |
| M6 | Rust forward pass for qwen35 blocks (attention layers every 4 + gated SSM layers), KV + recurrent state, LM head | done | greedy id matches llama.cpp golden logits on qa (8160) and code prompts; max logit rel diff 3.4-4.0e-3 |
| M7 | Standalone Rust engine (loader -> decode -> sample) | done | `bonsai-run` decodes + samples without llama.cpp; standalone `cargo build --release` (no BONSAI_LLAMA_DIR); llama link moved behind `llama-backend` feature |

## Done so far (context)

- PrismML llama.cpp fork builds (shared + static) in `../llama.cpp`.
- Legacy hand-rolled `llama.h` FFI (`src/llama.rs`), now used only by the
  `llama-backend` comparison tools.
- Legacy group-128 ternary GGUFs retagged 42 -> 142 (`bonsai-gguf retag`, Rust).
- Verified pure-Rust inference on Ternary-Bonsai-27B against llama.cpp golden
  logits (greedy ids match on both prompts).

## Done so far (migration)

| Subsystem | Status |
| --- | --- |
| chat template | Rust |
| sampler | Rust (M1) |
| GGUF read/dequant | Rust (M2) |
| tokenizer | Rust (M3), matches llama.cpp on 1581 corpus lines |
| detokenizer | Rust (M4) |
| kernels (RMSNorm, PQ2_0 dequant, activations) | Rust (M5/M6-2), verified vs ggml probes |
| forward pass (attn + GDN + FFN + head) | Rust (M6), greedy matches golden logits |
| decode / KV / EOG | Rust (M7), standalone, llama.cpp link removed |

## Next: inference performance (M8+) and GPU (RX 7600)

The migration is done. The CPU and GPU speedups planned here have landed:
mmap + threaded matvec + the AVX2 PQ2_0 row dot (CPU decode ~1.2-1.4 s/token),
and the Vulkan/RADV backend with device weights and single-submit decode
(GDev, ~1.4 s/token on the gfx902 iGPU, DRAM-parity with CPU). The PQ2_0
matvec now uses a fused single-pass kernel that drops the `partials` round
trip. See `notes/baseline.md` for the 2026-09-13 baseline and log.

What is left for the RX 7600 (288 GB/s, ~25 ms/token decode floor, ~20-40
tok/s projected): validate and tune the existing kernels on the discrete card,
and finish batched prefill (P1-P5 landed; on the APU a token loop wins, so
this is validated only when the card arrives).
