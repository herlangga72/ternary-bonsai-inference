# GPU decode measured numbers (2026-09-08 ~00:00)

Machine: Ryzen 3 3200G + Vega 8 iGPU (gfx902, RADV Mesa 25.3.5), 15 GB DDR4.

| item | value |
| --- | --- |
| CPU decode (AVX2, 4 threads) | ~1.5 s/token (best 10.0 s earlier, now 1.48-1.5) |
| GPU single-shot matvec (one submit each) | 6-34 ms small/ffn, 2.8 s LM head: iGPU downclocks between submits |
| GPU sustained kernel (multi-dispatch batch) | v3 two-pass: ffn_up 14.0, attn_q 15.1, LM head 21.0 GMAC/s |
| GDev full decode (one command buffer per token) | ~1.4-1.6 s/token (prefill 1.40, generation 1.51 measured) |
| GDev correctness | qa golden greedy 8160 MATCH, logit rel 4.4e-3 (CPU-vs-llama is 4.0e-3) |
| GDev multi-token | 10-token hidden rel ~1e-2 worst, greedy ids match CPU at every position |
| Load/upload (7 GB, 800 tensors) | ~30-40 s one-time |

Interpretation:
- Decode on this box is DRAM-parity between CPU and iGPU (~1.5 s/token; both
  read ~7 GB/token from the same memory). The single-submit GPU engine
  removes the downclocking pathology and is at the shared-bus ceiling here.
- The real headroom is the RX 7600: 288 GB/s dedicated VRAM vs ~18 GB/s
  shared now; the same GDev kernels/recorder target ~0.03-0.05 s/token.
- Remaining iGPU levers (measured small): descriptor-set churn, per-op
  barriers, small-op dispatch overhead. Hetero CPU+GPU row-splitting would
  only help until the shared bus saturates; not worth the complexity here.
