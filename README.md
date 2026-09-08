# tenarybonsai-fast

Rust-driven CPU inference for **Ternary-Bonsai-27B**, PrismML's ternary-weight
build of Qwen3.6-27B. The engine is pure Rust: GGUF parsing, tokenizer,
64-layer qwen35 forward pass, sampler and detokenizer, with no llama.cpp link.
The [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp)
(`prism` branch) is only needed to regenerate the golden-logit captures used
for validation (`llama-backend` cargo feature).

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
- **Runtime**: llama.cpp was the reference while the forward pass was being
  ported. The migration is complete: `bonsai-run` decodes in pure Rust and the
  engine builds without the fork (see Build).

## What is in this repo

| File | Purpose |
| --- | --- |
| `Ternary-Bonsai-27B-Q2_0.gguf` | Original download: legacy group-128 ternary under ggml type id 42 |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | Header-retagged copy (type 42 -> 142) that the current prism branch loads |
| `Ternary-Bonsai-27B-dspark-Q4_1.gguf` | DeepSpark speculator sidecar (3.6B, Q4_1 + TQ1_0, one legacy ternary tensor); not standalone |
| `Ternary-Bonsai-27B-dspark-PQ2_0.gguf` | Same, retagged so the speculative loader accepts it |
| `src/llama.rs` | Legacy hand-rolled FFI to `llama.h`; only used by the `llama-backend` compare tools |
| `src/main.rs` | Pure-Rust `bonsai-run` driver: load, chat template, tokenize, decode, sample, stream |
| `src/sampler.rs` | Pure-Rust sampler (top-k/top-p/min-p/temp/dist) |
| `src/gguf.rs` | Pure-Rust GGUF reader: metadata, tensor index, PQ2_0/F16/F32 dequant, layout checks, retag |
| `src/tokenizer.rs` | Pure-Rust qwen35 BPE tokenizer + GPT-2 byte decoder (detokenizer) |
| `src/weights.rs` | Weight context: qwen35 hyperparams + name-indexed tensor access (M6-1) |
| `src/forward.rs` | Forward pass: full-attention, recurrent (GDN), FFN, full decoder (M6-3..M6-5) |
| `src/kernels.rs` | Pure-Rust kernels: RMSNorm, L2 norm, SiLU/softplus/sigmoid, softmax, PQ2_0 row dequant/dot |
| `src/rope.rs` / `src/gdn.rs` | IMROPE multi-rope / gated-delta-net step (validated vs ggml) |
| `src/bin/bonsai-gguf.rs` | CLI: `inspect`, `probe` (decode a tensor window), `retag` (42 -> 142) |
| `src/bin/bonsai-weights.rs` | Weight-context config / tensor-manifest check |
| `src/bin/bonsai-*.rs` (attn/ssm/decode/golden) | Layer + decoder smokes and M6-6 golden-logit validation |
| `src/bin/bonsai-tokcmp.rs` | Tokenizer verification: Rust vs `llama_tokenize` over a corpus (`llama-backend`) |
| `src/bin/bonsai-logits.rs` | Capture golden logits from llama.cpp (`llama-backend`) |
| `src/bin/bonsai-kerncmp.rs` / `bonsai-actcmp.rs` | Kernel/activation verification vs `tools/ggml_probe` |
| `tools/ggml_probe.c` | C reference harness running single ggml ops on real tensor data |
| `tools/retag_gguf.py` | Original Python retag (superseded by the Rust tool; kept for reference) |
| `build.rs` | No-op unless the `llama-backend` feature is enabled, then links the static fork libs |

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

## GPU (single-submit) decode

`bonsai-grun` / `bonsai-gdecode` run the full forward pass on the GPU as one
recorded command buffer per token (`gdev.rs`): every PQ2_0 matvec uses the
two-pass v3 kernel and every small op (rms, rope, gdn, attention, gates) is a
recorded dispatch. The LM head, sampling and embedding lookups stay on the
CPU. On the gfx902 iGPU this runs at ~1.4-1.6 s/token (DRAM-parity with the
AVX2 CPU path) and matches the golden prompt: greedy 8160, logit rel 4.4e-3.
The same engine is what targets the RX 7600 (288 GB/s) for the big speedup.
On APUs weights default to host-visible RAM (device-local heap < model); set
`BONSAI_VRAM=1` to force device-local (fits after per-layer cache reduction).

```sh
./target/release/bonsai-grun Ternary-Bonsai-27B-PQ2_0.gguf "What is the capital of France?" 24
./target/release/bonsai-gdecode Ternary-Bonsai-27B-PQ2_0.gguf golden/prompts/qa.txt golden/qa.logits.bin
```

## Resource restraint

Long decode/validation runs saturate memory bandwidth (the model streams
~7 GB per token), which can starve the rest of the machine. Set
`BONSAI_BW_PCT` (1-100, default 100) to cap the engine at roughly that
fraction of its normal bandwidth use: the CPU matvec fan-out is limited to
~pct% of the cores and decode loops pace themselves to pct% of the unpaced
token rate (the first token always runs unpaced to calibrate).

```sh
BONSAI_BW_PCT=75 ./target/release/bonsai-decode Ternary-Bonsai-27B-PQ2_0.gguf 4
BONSAI_BW_PCT=50 ./target/release/bonsai-vkdecode Ternary-Bonsai-27B-PQ2_0.gguf \
  golden/prompts/qa.txt golden/qa.logits.bin 4
```

## Build

The engine is pure Rust and builds standalone (no llama.cpp needed):

```sh
cargo build --release
```

The legacy llama.cpp-backed comparison tools (`bonsai-logits` golden capture,
`bonsai-tokcmp` oracle tokenizer) need the PrismML fork and the `llama-backend`
feature:

```sh
git clone -b prism https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
cmake -B build-static -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build-static -j
cd ..
export BONSAI_LLAMA_DIR=/path/to/llama.cpp/build-static
cargo build --release --features llama-backend
```

`tools/ggml_probe.c` is compiled separately against the fork's ggml headers
when you want to re-run the kernel/activation reference checks.

## Run

```sh
./target/release/bonsai-run \
  -m Ternary-Bonsai-27B-PQ2_0.gguf \
  -p "What is the capital of France?" \
  -n 24 --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0
```

Sampling defaults are Bonsai-flavored (top-k 20, top-p 0.9, temp 0.6); the docs
recommend `--temp 0.5 --top-p 0.85 --top-k 20 --min-p 0`. The scalar Rust
decoder is slow, so start with `-n 1..4` and a short prompt.

## Performance expectations

27B ternary weights stream from RAM every token. Measured on a 4-core Ryzen
3200G (15 GB RAM, DDR4) with the pure-Rust decoder (AVX2 PQ2_0 dot, 4 threads):

| phase | rate |
| --- | --- |
| model load (header + tensor index) | ~1 s |
| prefill | ~2.0-2.5 s/token |
| decode | ~1.9-2.5 s/token (LM head included; AVX2 PQ2_0 dot) |

The model is a reasoning model. The prompt is seeded with `<think>`, the model
streams its reasoning, closes `</think>`, then gives the final answer and stops
at `<|im_end|>`. A short factual prompt needs ~150-200 generated tokens, so
budget ~5-8 minutes per full answer on CPU-only hardware (or use the golden
logits + short prompts for validation).

## Sample run

```
prompt: 86 chars -> 20 tokens
prefill done in 649s (32.43 s/tok)
Here's a
3 tokens in 69s (23.09 s/tok)
```

(The model was mid-reasoning; a full answer needs many more tokens.)
