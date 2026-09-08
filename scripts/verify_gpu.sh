#!/usr/bin/env bash
# GPU engine verification: single-submit device decode vs llama golden captures
# plus the kernel self-checks. Usage: scripts/verify_gpu.sh <model.gguf>
set -euo pipefail
MODEL="${1:?usage: verify_gpu.sh <model.gguf>}"
cd "$(dirname "$0")/.."

echo "== Vulkan kernel self-checks =="
for m in rmsnorm elem normrows softmax rope gdn attn; do
  out=$(./target/release/bonsai-vk "$m" "$MODEL" 2>&1 || true)
  echo "$out" | grep -E "PASSED|FAILED" || echo "$m: no result"
done
echo "== matvec recorder smoke =="
./target/release/bonsai-vk rec "$MODEL" blk.0.ffn_up.weight 2>&1 | grep -E "PASSED|FAILED" || true

echo "== golden prompts through single-submit device decode =="
./target/release/bonsai-gdecode "$MODEL" golden/prompts/qa.txt golden/qa.logits.bin 2>&1 | tail -2
./target/release/bonsai-gdecode "$MODEL" golden/prompts/code.txt golden/code.logits.bin 2>&1 | tail -2
