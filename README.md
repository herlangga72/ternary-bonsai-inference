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
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | DeepSpark draft/speculator companion (Q4_1 + TQ1_0, one legacy ternary tensor) |
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

The `dspark` file also carries one legacy ternary tensor (`token_embd.weight`)
and needs the same retag before it can be loaded.

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

27B ternary weights stream from RAM every token. On a 4-core Ryzen 3200G with
~20 GB/s memory bandwidth, expect roughly **0.1-0.3 tokens/s** (3-10 s/token)
CPU-only. The model is a reasoning model, so responses start in thinking mode.
