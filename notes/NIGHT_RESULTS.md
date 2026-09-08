# Night build results (2026-09-08, ~00:37)

User goal: run Ternary-Bonsai-27B faster with CPU+GPU heterogeneous execution
and weights resident in memory, implementing until 07:00.

## Delivered

1. **CPU AVX2 PQ2_0 kernel** (128-bit then 256-bit): decode ~10.9 -> ~1.5 s/token.
   Both golden prompts pass (greedy 8160, logit rel ~4e-3). Commits 8637f38,
   8db350d.
2. **GPU kernel v3** (two-pass partial + row-sum, register-word block fetch):
   10-21 GMAC/s on the iGPU, all shapes match CPU <=1.1e-6. Commit 96f5180.
3. **Single-submit device decode engine (`gdev.rs`)**:
   - One recorded command buffer per token (host record ~2 ms), all matvecs and
     small ops as compute dispatches with barriers.
   - APUs use host-visible RAM buffers (device-local heap ~6.4 GB < model).
   - Golden qa AND code prompts pass through it: greedy 8160 MATCH, logit rel
     4.4e-3 / 3.2e-3; 10-token prefixes greedy-identical to CPU; interactive
     `bonsai-grun` streams structured reasoning at ~1.5 s/token.
   Commits dca2b41 ... ccf10d9.
4. Diagnostics: `bonsai-lat` (single-shot submits downclock the iGPU to
   0.3-0.5 GHz - the reason for the one-buffer design), GDEV_TIME phase timers,
   `scripts/verify_gpu.sh`.
5. Cache footprint cut to only the layers that use each cache (KV on 16
   full-attn, conv/state on 48 recurrent); runtime context length
   (BONSAI_CTX, default 2048); `BONSAI_VRAM=1` forces device-local weights
   (now fits after the cache cut; proven: same 1.5 s/token and qa golden
   PASSES on device-local). 200-token generation streamed coherent structured
   reasoning at 1.59 s/token. Full test suite: 20/20 pass.

## Speed today (this machine)

- CPU decode: ~1.5 s/token
- GPU single-submit decode: ~1.4-1.6 s/token (DRAM parity: both engines read
  ~7 GB/token from the same memory at ~5 GB/s effective)
- iGPU is at the shared-bus ceiling (~5 GB/s effective for the v3 kernel);
  further real speedup requires the RX 7600 (288 GB/s dedicated VRAM), which
  this exact engine targets (~0.03-0.05 s/token projected).

## Open items for after this session

- RX 7600 tuning (create_model_weight_buffer already picks device-local VRAM on
  discrete cards), larger N_CTX, batched prefill, optional CPU+GPU row-split
  (limited by the shared bus here).
