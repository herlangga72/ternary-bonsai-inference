# Handoff: Ternary-Bonsai-27B Rust inference engine

Status as of 2026-09-06. All work is committed in this repo (git log below).
A fresh session should read this file, then ROADMAP.md and notes/*.md before
writing more code.

## Goal

Move the inference engine from the PrismML llama.cpp fork (C/C++) into pure
Rust incrementally, keeping inference working after every step, until no C/C++
code is required.

## Repo layout

```
tenarybonsai-fast/
  Cargo.toml            single package, several bins; `llama-backend` feature gates llama tools
  build.rs              links static PrismML llama.cpp libs only under `llama-backend`
  README.md             project + model explainer, build/run instructions
  ROADMAP.md            milestone table with verification notes
  notes/qwen35-arch.md       tensor layout, layer facts, open questions
  notes/qwen35-forward-spec.md  exact forward op spec + MRoPE/GDN notes
  golden/               llama.cpp reference logits for M6 validation
  src/
    llama.rs            hand-rolled FFI to llama.h (compute still in C)
    gguf.rs             GGUF v3 reader: metadata, tensors, F16/PQ2_0 decode, retag
    sampler.rs          pure-Rust sampler (M1)
    tokenizer.rs        pure-Rust qwen35 BPE tokenizer + detokenizer (M3/M4)
    kernels.rs          RMSNorm/PQ2_0 kernels + activation helpers (M5, M6-2)
    gdn.rs              gated-delta-net recurrent step (M6, validated)
    rope.rs             IMROPE multi-rope (M6, validated)
    weights.rs          weight-context engine: qwen35 hyperparams + tensor-name mapping (M6-1)
    forward.rs          qwen35 forward pieces: attn+recurrent layers, FFN, Decoder (M6-3..M6-5)
    main.rs             bonsai-run CLI (pure-Rust engine, M7)
    llama.rs            legacy FFI, only compiled by `llama-backend` feature bins
  src/bin/
    bonsai-gguf         inspect / probe / retag GGUF
    bonsai-tokcmp       Rust tokenizer vs llama_tokenize on corpus
    bonsai-kerncmp      kernels vs ggml reference (RMSNorm, dequant)
    bonsai-gdncmp       gdn.rs vs ggml_gated_delta_net
    bonsai-ropecmp      rope.rs vs ggml_rope_multi
    bonsai-actcmp       activation helpers vs ggml probes (M6-2)
    bonsai-logits       capture golden logits (M6 verification oracle)
    bonsai-matbench     scalar matvec throughput benchmark
    bonsai-weights      weight-context config / tensor-manifest check (M6-1)
    bonsai-attn         full-attention layer smoke on real model (M6-3)
    bonsai-ssm          recurrent (gated delta net) layer smoke on real model (M6-4)
    bonsai-decode       full single-token decoder smoke on real model (M6-5)
    bonsai-golden       M6-6 golden-logit validation runner
  tools/
    ggml_probe.c        C reference harness (rmsnorm, pq2row, pq2deq, gdn, rope, rmsrows, l2norm, unary, softmax)
    retag_gguf.py       superseded Python retag (kept for reference)
```

Model files (gitignored, stay in place): `Ternary-Bonsai-27B-Q2_0.gguf`
(original, legacy type 42), `Ternary-Bonsai-27B-PQ2_0.gguf` (retagged, loads
in current prism fork), plus the dspark sidecar pair.

External dependency: `/home/server/sdgs/llama.cpp` = PrismML fork clone with
`build-static/` (static libs). Probe tool needs headers from
`llama.cpp/ggml/include`.

## Build / verify commands

```sh
# pure engine (standalone, no llama.cpp needed)
cargo build --release

# llama.cpp-backed comparison tools (golden capture, tokenizer oracle)
export BONSAI_LLAMA_DIR=/home/server/sdgs/llama.cpp/build-static
cargo build --release --features llama-backend

cargo test --release

gcc -O2 tools/ggml_probe.c -I /home/server/sdgs/llama.cpp/ggml/include \
  -o target/ggml_probe -L /home/server/sdgs/llama.cpp/build-static/ggml/src \
  -lggml-base -lggml-cpu -lggml -lstdc++ -lgomp -lm -lpthread -ldl

# tokenizer corpus check (Rust vs llama.cpp, needs llama-backend build)
./target/release/bonsai-tokcmp Ternary-Bonsai-27B-PQ2_0.gguf < /tmp/corpus2.txt

# kernel / op reference checks
./target/release/bonsai-kerncmp Ternary-Bonsai-27B-PQ2_0.gguf ./target/ggml_probe
./target/release/bonsai-gdncmp ./target/ggml_probe
./target/release/bonsai-ropecmp ./target/ggml_probe
./target/release/bonsai-actcmp ./target/ggml_probe

# weight-context and forward smokes on the real model
./target/release/bonsai-weights check Ternary-Bonsai-27B-PQ2_0.gguf
./target/release/bonsai-attn Ternary-Bonsai-27B-PQ2_0.gguf 3 3
./target/release/bonsai-ssm Ternary-Bonsai-27B-PQ2_0.gguf 0 3

# golden-logit validation (pure Rust decode vs llama.cpp capture)
./target/release/bonsai-golden Ternary-Bonsai-27B-PQ2_0.gguf \
  golden/prompts/qa.txt golden/qa.logits.bin
```

## What works (verified)

| # | Item | Evidence |
|---|---|---|
| M1 | Rust sampler (top-k/top-p/min-p/temp/dist) | bonsai-run generates coherent text |
| M2 | Rust GGUF reader + dequant + retag | header/layout verified; retag byte-identical to Python |
| M3 | Rust qwen35 BPE tokenizer | 1581 corpus lines, 0 mismatches vs llama_tokenize |
| M4 | Rust detokenizer | end-to-end output identical to llama.cpp path |
| M5 | RMSNorm + PQ2_0 dequant kernels | vs ggml: 2e-7 and bit-exact |
| M6a | qwen35 forward spec | notes/qwen35-forward-spec.md (source-derived) |
| M6b | GDN recurrent step (gdn.rs) | vs ggml_gated_delta_net: < 5e-7 |
| M6c | IMROPE rope (rope.rs) | vs ggml_rope_multi: < 6e-8 |
| M6d | batched PQ2_0 matvec + bench | ~1 GMAC/s (4 threads); whole model ~25-30 s/token |
| M6e | golden logits captures | golden/*.logits.bin for qa + code prompts |
| M6-1 | weight context engine (weights.rs) | real file: 848/848 per-layer tensors verified; vec/matvec/row smoke; unit tests on synthetic mini-GGUF |
| M6-2 | activation helpers (kernels.rs) | row RMSNorm ~3e-7, L2/sigmoid/softplus 0, silu 1.2e-7, masked softmax 1.4e-7 vs ggml probes |
| M6-3 | full-attention layer (forward.rs) | smoke on blk.3 over 3 synthetic tokens: finite outputs, KV cache grows per pos; unit test on cache layout |
| M6-4 | recurrent layer (forward.rs) | smoke on blk.0 over 3 synthetic tokens: causal conv cache + GDN state evolve; finite outputs |
| M6-5 | full decoder (Decoder in forward.rs) | single-token decode through all 64 layers + LM head on real model: finite logits over 248k vocab; ~34 s/token |
| M6-6 | golden-logit validation | qa prompt: greedy id 8160 matches golden, logits rel diff 4.0e-3; code prompt: greedy 8160, rel diff 3.4e-3 (both first-run, no layer debugging) |
| M7 | standalone pure-Rust engine | bonsai-run decodes + samples with no llama.cpp link; cargo build --release works without BONSAI_LLAMA_DIR; llama tools behind llama-backend feature |

## Next work, structured (do in order)

None: the M1..M7 migration is complete. Remaining ideas are optimizations
(see "Later optimizations") and the dspark speculative sidecar.

Later optimizations (not required for correctness): mmap tensor data, cache
row decode buffers, SIMD dot via `std::arch`, fuse layers, speculative dspark.

## Pitfalls learned (avoid re-deriving)

- Legacy group-128 ternary GGUFs use type id 42; current prism branch wants
  id 142 (PQ2_0). Header-only retag is safe (`bonsai-gguf retag`).
- `llama_tokenize` crashes if passed text_len = -1; pass an explicit length.
- `llama_get_logits_ith(ctx, i)` indexes the batch position, not an output
  ordinal; after single-output decode use i = last batch position.
- ggml `mul_mat` quantizes activations (Q8-style); it is NOT an exact float
  oracle. Validate dequant against `to_float` / `ggml_get_type_traits`
  instead. GDN/rope probes above compare scalar math exactly.
- qwen35 rope type is IMROPE (interleaved), not plain MROPE.
- GDN state is stored transposed (M[j][i] = S[i][j]) per v-head.
- GGUF tensor offsets are relative to the aligned data section start, NOT
  absolute file positions. Always read tensor bytes at
  `data_start + info.offset` (see `gguf::GGUF::tensor_data_offset`); the
  reader used to skip `data_start`, silently decoding garbage for every
  tensor. Fixed 2026-09-06; synthetic writers must store relative offsets too.
