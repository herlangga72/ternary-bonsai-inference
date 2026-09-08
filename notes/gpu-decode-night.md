# GPU decode night build (2026-09-07 23:00 -> 06:00)

Goal: full single-command-buffer-per-token device decode so the iGPU stays
clock-boosted (measured: single-shot submits drop it to ~0.3-0.5 GHz; sustained
bursts reach 5.6+ GB/s) and the identical path is what the RX 7600 will run.

Architecture (one in-order command buffer per token, all activations on device):
- CPU per token: tokenize, then embed row from mmap -> upload to arena `cur`,
  submit the whole layer loop as one command buffer, wait once, read back the
  final hidden (5120), then LM-head + sampling stay on the CPU (head is 0.1-0.2 s
  and avoids moving output.weight/partials on-device).
- Device buffers: weight DevBufs (PQ2 + small f32 vectors), reusable partial
  buffer for the v3 two-pass matvec, per-token descriptor pool (1 set per
  dispatch, ~2000 sets), KV/conv/gdn-state caches, activation arena.
- Every dispatch writes a distinct arena region; a compute->compute buffer
  memory barrier after each op keeps ordering (no overlap intended inside a
  token).

Slice checklist (each: write shader, register, validate vs CPU, commit):
1. arena ops: embed copy, add residual, split q|gate, kv store/append.
2. recurrent layer device graph: qkv/gate/alpha/beta matvecs, conv1d+silu
   (+cache shift), l2 rows on q/k, gate prep (sigmoid/softplus/alpha*ssm_a),
   gdn_step, ssm_norm rows + silu(z), ssm_out matvec, residual.
3. full-attn layer device graph: q/k/v matvecs, q/k rms rows + rope, split
   q|gate, kv append, attn trio, gate multiply, output matvec, residual.
4. ffn device graph: gate/up matvecs, silu(g)*u, down matvec, residual.
5. Decoder orchestrator (record+barrier per layer), output norm readback.
6. Validate bonsai-vkdecode golden (greedy 8160) on iGPU; measure s/token.
7. Tune partials/LDS/occupancy; then the same build targets the RX 7600.

Status 2026-09-08 ~00:00: slices 1-4 + orchestrator + full-token validation
complete. Single-command-buffer device decode (GDev) matches the CPU decoder:
qa golden greedy 8160 MATCH, logit rel 4.4e-3, ~1.4-1.6 s/token on the iGPU
(bus/parity bound). Interactive bonsai-grun streams text. The identical
engine is the RX 7600 path; remaining: micro overhead reduction, optional
CPU+GPU row-split hetero on this box, and 7600 tuning.

Existing validated kernels reused: rms_norm, norm_rows, elem, rope_imrope,
softmax_inplace, attn_scores, attn_out, gdn_step, pq2 partial+rowsum.
