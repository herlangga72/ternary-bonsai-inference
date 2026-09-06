# tenarybonsai-fast

Rust-driven CPU inference for **Ternary-Bonsai-27B**, PrismML's ternary-weight
build of Qwen3.6-27B, using the PrismML llama.cpp fork as the compute engine.

## How Ternary Bonsai works

- **Base model**: Qwen3.6-27B. llama.cpp names its hybrid architecture
  `qwen35`: 64 layers, 262K context, GQA (24 q-heads / 4 kv-heads). Every 4th
  layer runs full attention; the others use a gated SSM branch (conv1d kernel 4,
  state 128, dt rank 48, inner width 6144) plus partial-split RoPE
  (`rope.dimension_sections = [11,11,10,0]`). 248,320-token BPE tokenizer with
  a thinking-mode chat template.
- **Ternary quantization**: every weight (embeddings, attention, MLPs, LM head)
  is one of {-1, 0, +1}, packed 2 bits each, with one FP16 group scale per 128
  weights (`PQ2_0`, ggml type 142). ~1.71 effective bits/weight, ~95% of the
  FP16 model's benchmark score, no higher-precision escape hatches.
- **Runtime**: llama.cpp is the reference runtime, but only the 1-bit variant is
  upstream. The ternary weights need the [PrismML llama.cpp fork]
  (https://github.com/PrismML-Eng/llama.cpp) (`prism` branch).

## What is in this repo

| File | Purpose |
| --- | --- |
| `Ternary-Bonsai-27B-Q2_0.gguf` | Original download: legacy group-128 ternary under ggml type id 42 |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | Header-retagged copy (type 42 -> 142) that the current prism branch loads |
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | DeepSpark speculator sidecar (3.6B, Q4_1 + TQ1_0, one legacy ternary tensor); not standalone |
| `Ternary-Bonsai-27B-dspark-PQ2_0.gguf` | Same, retagged so the speculative loader accepts it |
| `src/llama.rs` | Hand-rolled FFI to the fork's `llama.h` (structs transcribed) |
| `src/main.rs` | Rust driver: model load, chat template, tokenize, sample, stream, timing |
| `tools/retag_gguf.py` | Header-only 42 -> 142 retag for legacy ternary GGUFs |
| `build.rs` | Links the static PrismML llama.cpp libraries |

## The format gotcha (important)

The July 2026 `Ternary-Bonsai-27B-Q2_0.gguf` files store **group-128** ternary
weights under ggml type id 42. Current prism-branch builds read type 42 as the
official **group-64** Q2_0, so they refuse the old file with:

```
this file matches the legacy Prism Q2_0 layout (group size 128 stored as
ggml type id 42), but this build reads Q2_0 as the official group-64 format
```

Group-128 payloads now live under type id 142 (`PQ2_0`) with a byte-identical
codec, so a header-only retag fixes the file without touching weights:

```sh
python3 tools/retag_gguf.py Ternary-Bonsai-27B-Q2_0.gguf Ternary-Bonsai-27B-PQ2_0.gguf
```

The `dspark` file is a **DeepSpark speculator sidecar** (3.6B, 6 blocks, 40
heads, markov rank 256 + confidence + log-SNR heads) whose extra tensors are
fused into the main qwen35 graph by the fork's speculative path
(`common/speculative.cpp`, spec-type `draft-dspark`). It is deliberately not a
standalone model: `general.architecture = dspark` is unknown to the standalone
loader by design. It also carries one legacy ternary tensor and needs the same
retag before the speculative loader accepts it:

```sh
python3 tools/retag_gguf.py \
  Ternary-Bonsai-27B-dspark-Q4_1.gguf \
  Ternary-Bonsai-27B-dspark-PQ2_0.gguf
```

Wiring dspark speculation into this Rust driver is not exposed as a single
`llama.h` call; it lives in the fork's C++ acceptance loop, so it remains a
follow-up rather than part of the basic inference path.

## Build

1. Clone and build the PrismML fork (static libs):

   ```sh
   git clone -b prism https://github.com/PrismML-Eng/llama.cpp
   cd llama.cpp
   cmake -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
   cmake --build build-static -j
   ```

2. Build the Rust driver:

   ```sh
   BONSAI_LLAMA_DIR=/path/to/llama.cpp/build-static cargo build --release
   ```

## Run

```sh
./target/release/bonsai-run \
  -m Ternary-Bonsai-27B-PQ2_0.gguf \
  -p "What is the capital of France?" \
  -n 24 --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 -t 4
```

Sampling defaults are Bonsai-flavored (top-k 20, top-p 0.9, temp 0.6); the docs
recommend `--temp 0.5 --top-p 0.85 --top-k 20 --min-p 0`.

## Performance expectations

27B ternary weights stream from RAM every token. Measured on a 4-core Ryzen
3200G (15 GB RAM, DDR4):

| phase | rate |
| --- | --- |
| model load (mmap) | ~1 s |
| prefill | ~0.8 tok/s (first call also cold-maps the 6.8 GB file) |
| decode | ~0.4-0.8 tok/s (~1.7-2.3 s/token) |

The model is a reasoning model. The prompt is seeded with `<think>`, the model
streams its reasoning, closes `</think>`, then gives the final answer and stops
at `<|im_end|>`. A short factual prompt needs ~150-200 generated tokens, so
budget a few minutes per run on CPU-only hardware.

## Sample run

```
prompt: 104 chars -> 20 tokens
prefill done in 25.98s (0.8 tok/s)
Here's a thinking process:
1.  **Analyze User Input:**
    - Question: "What is the capital of France?"
    - Constraint: "Answer briefly."
...
</think>
Paris.
164 tokens in 371.53s (0.4 tok/s)
```
