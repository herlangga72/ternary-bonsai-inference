# Migration roadmap: llama.cpp -> pure Rust

The compute engine today is the PrismML llama.cpp fork (C/C++) reached through a
thin Rust FFI in `src/llama.rs`. This document tracks the incremental migration
toward a pure-Rust engine. Rule: inference stays working and verified after
every milestone; each milestone replaces one subsystem and validates against the
previous behavior before the next starts.

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
| M5 | Rust kernels: dequant PQ2_0 matmul, RMSNorm, partial RoPE, gates | planned | probe against ggml reference graphs on real model tensors |
| M6 | Rust forward pass for qwen35 blocks (attention layers every 4 + gated SSM layers), KV + recurrent state, LM head | planned | greedy logits agree with llama.cpp decode on test prompts |
| M7 | Standalone Rust engine (loader -> decode -> sample) | planned | same prompts produce same/sane output; llama.cpp link removed |

## Done so far (context)

- PrismML llama.cpp fork builds (shared + static) in `../llama.cpp`.
- Hand-rolled `llama.h` FFI (`src/llama.rs`); Rust driver `bonsai-run`.
- Legacy group-128 ternary GGUFs retagged 42 -> 142 (`bonsai-gguf retag`, Rust).
- Verified inference on Ternary-Bonsai-27B (thinking + answer) at ~0.4-0.8 tok/s.

## Done so far (migration)

| Subsystem | Status |
| --- | --- |
| chat template | Rust |
| sampler | Rust (M1) |
| GGUF read/dequant | Rust (M2) |
| tokenizer | Rust (M3), matches llama.cpp on 1581 corpus lines |
| detokenizer | Rust (M4) |
| decode / KV / EOG | llama.cpp (M5-M7 remain) |
