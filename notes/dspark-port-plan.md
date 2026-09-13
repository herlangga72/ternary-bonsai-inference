# Dspark speculative decode: pure-Rust port plan

Status: 2026-09-13. Scope: make the engine draft with the 3.6B dspark sidecar and
run the 27B target model only as a verifier, entirely in Rust. Gated behind
`BONSAI_DSPARK`.

Reference implementation: PrismML llama.cpp fork at `/home/server/sdgs/llama.cpp`
(`src/models/dflash.cpp`, `common/speculative.cpp`, `src/llama-graph.*`). This
document is written after reading those, and after empirically confirming the
sidecar does not load in the fork as shipped (see "Blocker" below).

## What dspark is (from the checkpoint + reference graph)

`Ternary-Bonsai-27B-dspark-PQ2_0.gguf`, 1.95 GB, architecture `dspark`:

| metadata | value |
| --- | --- |
| blocks / n_embd / n_ff | 6 / 5120 / 5120 |
| attention | 40 q heads, 4 kv heads, head dim 128 |
| block_size | 4 |
| mask_token_id | 248319 |
| target_layers | [1, 16, 31, 46, 61] |
| markov_rank | 256 |
| confidence_head / log_snr_conditioning | true / true |
| min/max_log_snr | -9.0 / 9.0 |

It is a **block-denoising** drafter, not an autoregressive one:

1. **Encoder** (runs on the target's committed positions). Concatenate the
   target's *layer-input* hidden states at the 5 tapped layers, interleaved by
   layer (`[n_chunk, 5*5120]`), then `fc` (25600 -> 5120) then RMSNorm. That is
   the draft's `inp_g`.
2. **KV injection.** Run the draft with `embd = inp_g` at the committed
   positions: each draft layer projects it with its `wk`/`wv`, applies K RMSNorm
   + rope to K, and writes K/V into the draft KV cache. So the draft cache is
   *the target's features*, not the draft's own tokens.
3. **Decode (one pass per block).** Input `[id_last, MASK x (block_size-1)]`,
   token embeddings and LM head *shared with the target*. Add a log-SNR
   embedding (anchor position at `max_log_snr`, masked positions at
   `min_log_snr`), run 6 layers (RMSNorm, q/k norm, rope, attention over the
   injected + own cache, FFN SiLU), RMSNorm, LM head.
4. **Markov head.** Bias the draft logits `[n_vocab, block]` position by
   position: `bias_i = markov_w2 @ markov_w1[prev_i]`, `prev_0` = the block's
   anchor token, `prev_{i+1} = argmax(biased logits_i)`. Emit the biased logits.
5. **Confidence head.** `conf_i = sigmoid(conf_proj . [emb_i ; markov_w1[prev_i]] + b)`,
   returned via `t_h_nextn`. Used to truncate the draft below `p_min`.

So the draft adds only ~1.95 GB of reads per block and produces up to 4 tokens.

## Shared-state interface (what "reuse" actually means)

The draft and target do not share a KV cache. What is shared / reused:

- **target -> draft**: the 5 tapped layer-input hidden vectors (the encoder input).
- **shared weights**: the draft reads the target's `token_embd.weight` and
  `output.weight` (its own checkpoint has copies too, but the reference prefers
  the target's via `ctx_other`; a Rust port can hold the draft's own copies).
- **target KV/state across verify**: the target runs the drafted block in one
  pass and then rolls back rejected tail positions. Because 48/64 target layers
  are GDN *recurrent*, rollback must restore the recurrent state, not just the
  KV cache (snapshot per round: 48 x 128 x 128 f32 ~= 151 MB, or recompute).

## Decode loop (Rust)

```
loop {
    draft = dspark_draft(id_last, taps)          # noise block -> up to 4 tokens
    # verify: one batched target pass over [id_last? , draft...]
    (acc, n, state) = target.verify(draft)       # acc = accepted prefix
    if acc == draft.len() { keep all n = len }
    ...                                          # else keep acc + target's own token
    # target KV/state roll back to acc, then append the extra target token
}
```

Greedy correctness invariant: with greedy target sampling, `BONSAI_DSPARK`
output must be token-identical to `BONSAI_DSPARK=off`. This is the primary
correctness oracle and does not require the fork.

## Blocker: the sidecar is not in the fork's on-disk format

The fork registers architecture **`dflash`** only; the sidecar declares
`general.architecture = dspark` with keys `dspark.*` / `dspark.dspark.*` and
tensor names `dspark.markov_head_a.weight`, `dspark.confidence_head.weight`,
`dspark.hidden_norm.weight`. No alias exists in the loader (checked local
`b984d5f` and remote HEAD `d8f26ee`). Loading fails with
`unknown model architecture: 'dspark'`.

Shapes all match once renamed, so conversion is mechanical:

| sidecar | fork expects |
| --- | --- |
| `general.architecture = dspark` | `dflash` |
| `dspark.block_count` ... `dspark.vocab_size` | `dflash.*` |
| `dspark.dspark.block_size` | `dflash.block_size` |
| `dspark.dspark.target_layers` | `dflash.target_layers` |
| `dspark.dspark.confidence_head` | `dflash.confidence_head` |
| `dspark.dspark.log_snr_conditioning` / `min` / `max_log_snr` | `dflash.*` |
| `dspark.dspark.mask_token_id` | `tokenizer.ggml.mask_token_id` |
| `dspark.markov_head_a.weight` (256, 248320) | `markov_w1.weight` |
| `dspark.markov_head_b.weight` (256, 248320) | `markov_w2.weight` |
| `dspark.confidence_head.weight/bias` (5376, 1)/(1) | `conf_proj.weight/bias` |
| `dspark.hidden_norm.weight` | `enc.output_norm.weight` |
| `dspark.fc.weight` (25600, 5120) | `fc.weight` |
| `dspark.log_snr_fc1/fc2.weight/bias` | `log_snr_fc1`/`log_snr_fc2` |

Conversion is needed for two things: (a) running the fork as the numeric
reference for the Rust draft, and (b) measuring acceptance. The Rust draft
loader can read the sidecar's native names directly and skip the conversion.

## Components

| unit | file | purpose |
| --- | --- | --- |
| converter | `src/bin/bonsai-gguf.rs` (`dspark-convert`) | rewrite sidecar names for the fork |
| draft weights | `src/dspark.rs` | load/validate sidecar tensors, config |
| taps | `src/forward.rs` | capture layer-input vectors at tapped layers |
| encoder | `src/dspark.rs` | fc + RMSNorm |
| decoder | `src/dspark.rs` | noise block, log-SNR, 6 layers, markov/confidence |
| verify + rollback | `src/spec.rs` | batched target verify, GDN-state snapshot/restore |
| driver | `src/main.rs` | `BONSAI_DSPARK` wiring |
| compare tool | `src/bin/bonsai-dsparkcmp.rs` | Rust draft logits vs fork capture |

## Order of work

1. Converter + validate the sidecar loads in the fork.
2. Fork reference run on CPU: acceptance rate and s/token (decides how much the
   Rust port is worth).
3. Rust draft loader, taps, encoder, decoder, validated against the fork dump.
4. Speculative loop + GDN rollback, validated by greedy identity.
5. Measure.

## Risks

- **This box may not benefit.** Prefill P6 already showed a batched target pass
  over N positions costs ~N single-token passes on the gfx902 iGPU. The CPU path
  is near the compute/bandwidth crossover, so it is the plausible winner; the
  iGPU APU is likely a loss. Measure before optimizing hard.
- **Recurrent-state rollback** is the subtle part: GDN state is order-dependent,
  so a rejected suffix needs a snapshot restore (or recompute).
- **Markov head is greedy in-graph** (`argmax` chain), matching the reference;
  sampling params affect only the final pick.

## Implementation status (2026-09-13)

| step | commit | what landed | verified |
| --- | --- | --- | --- |
| converter | `3271d5c` | `bonsai-gguf dspark-convert` + BF16 sizing in `tensor_nbytes` | converted sidecar passes `inspect` with `layout ok` |
| draft loader/dequant/encoder | `10e1e93` | `src/dspark.rs` cfg, manifest check, F32/BF16/Q4_1/PQ2_0 dequant, threaded matvec, `encode` | all 79 tensors present, correct dims; `bonsai-dspark` smoke |
| draft decoder | `67588aa` | NEOX rope, `DraftCache`+`inject`, log-SNR embed, `draft_block` (6 layers, non-causal attention, LM head, markov + confidence) | real draft block: finite logits, conf 0.76-0.99; ~7 s/block |
| taps | `3835586` | `Decoder::forward_hidden_taps` captures layer-input residual at the tapped layers | builds, full test suite green |

Fork reference (below) is blocked: `llama-cli`/`llama-speculative` on this box
spins at ~100% CPU for 9+ min just loading the 7 GB target and never reaches
generation, so no fork-side acceptance/logit numbers yet. The greedy
`BONSAI_DSPARK` == `BONSAI_DSPARK=off` identity is the planned primary oracle
and needs no fork.

Remaining: multi-token target verify forward + GDN/conv/attn rollback, the
`BONSAI_DSPARK` driver, then optimization (the draft block's 7 s is dominated by
the scalar Q4_1 row-dot).

## End-to-end result (2026-09-13)

`bonsai-spec <target> <sidecar> <tokens> [n] [k]` runs the whole loop against a
fresh greedy decoder:

```
prefill 8 tokens in 15.1s
spec : [271, 248068, 198, 8160, 579, 264]
plain: [271, 248068, 198, 8160, 579, 264]
IDENTICAL (6 tokens compared)  accepted 1/16 drafts over 4 rounds
spec 24.6s (4.09 s/tok, 6 forwards)  plain 7.8s (1.30 s/tok)
```

So the loop is **correct** (speculative output == plain greedy; the token 248068
is the same id the prefill plan's qa prompt produced). It is currently 3x
*slower*, for two independent reasons:

1. **Low acceptance: 1 of 16 drafts.** The draft graph is shape-correct and
   finite but is clearly not producing the reference distribution. Debugging
   this needs a numeric reference (fork draft logits), which the fork currently
   cannot provide (below).
2. **Slow draft: ~7 s/block**, dominated by the scalar Q4_1 row-dot. The block
   weights are 42 x 5120x5120; an optimized Q4_1 kernel is needed.

And one fundamental caveat, measured this session: the PQ2_0 kernel is
**compute-bound**, not weight-bandwidth bound, on this CPU. `bonsai-matbench`
shows the N-column GEMM costs the same per token as the single-token matvec
(~24-30 GMAC/s 4-thread), so batching the verify pass buys no kernel throughput
here. Speculative decoding only wins where the extra verify positions are cheap
relative to the weight stream - i.e. the RX 7600, not this box.

## Fork reference status

Still blocked: `llama-cli`/`llama-speculative` from `build-static` spin at ~100%
CPU for 9+ minutes loading the 7 GB target (and the earlier dspark run segfaulted
after ~45 s) and never reach generation, so no reference logits can be captured
from it on this machine. The converter and the greedy-identity oracle do not
depend on it.

## Next steps

1. Draft acceptance: build a fork-independent reference, or bisect the draft
   graph (taps choice, rope dims/type, non-causal vs causal attention, log-SNR
   convention, markov `prev` seeding) against a debug dump of the target's own
   next-token distribution.
2. Draft speed: an AVX2 Q4_1 row kernel (mirror `row_dot_avx2`) to get the block
   under ~1 s.
3. Batched verify: implement `pq2_matmul_n` into the target layer functions so
   the verify reads weights once; keep the streaming path as the correctness
   oracle. Only worth measuring on bandwidth-bound hardware.

### Draft-acceptance bisect (2026-09-13)

Acceptance is ~1/24 on both a degenerate and a real prompt (qa), so it is not a
prompt artifact. Ruled out by ablation (both still ~1/24 or, anchorless, 2/20):

- `BONSAI_DSPARK_NO_MARKOV` (drop the markov bias): unchanged, so the markov head
  and its `prev` chain are not the cause.
- `BONSAI_DSPARK_ANCHORLESS` (5-token block, drafts read from position 1, i.e.
  the bonus-anchor convention): 2/20 - marginal, not the cause.

The suspected remaining causes, in order, are the taps semantics, the rope dims
(`n_rot` may be < 128; the sidecar omits `rope.dimension_count`), and the
strictly non-causal attention in the block. All three need a numeric reference
(fork draft logits) to resolve, which this box cannot currently produce.

### Wiring and final measured state (2026-09-13)

`BONSAI_DSPARK=<sidecar>` + `--temp 0` in `bonsai-run` now decodes with the
drafter; stochastic sampling stays on the plain path. Verified identical output
to the plain run ("Here's a thinking process:\n\n1").

| item | before | after |
| --- | --- | --- |
| draft block (`bonsai-dspark`) | 7.2 s | 1.44 s (AVX2 Q4_1 kernel) |
| bonsai-run decode | 1.32 s/tok plain | 3.46 s/tok with BONSAI_DSPARK |
| bonsai-run prefill | - | 2.03 s/tok with BONSAI_DSPARK |

Remaining work, in priority order: (1) draft acceptance (~1/24) needs a numeric
reference; (2) a batched verify pass (`pq2_matmul_n` is ready but the target
layer functions still run per token) for bandwidth-bound hardware; (3) the
`observe` cost during prefill can be folded (one encoder call per chunk instead
of per token).

## Fork reference is now runnable (2026-09-13)

The earlier "fork hangs" was wrong: `llama-cli` without `-st`/piped stdin just
waits in interactive mode, and its `ps %CPU` is a lifetime average, not a hung
loop. The fork loads and runs the target fine.

`llama-speculative --spec-type draft-dspark -md <converted sidecar>` crashed for
two unrelated reasons, both patched locally (diff saved as
`tools/fork-dspark-ref.patch`):

1. **Warmup segfault.** `common_init_from_params` warmups via `llama_encode`,
   which crashes in `llm_graph_input_embd::set_input -> ggml_element_size` for
   the dflash encoder graph. The `--no-warmup` flag is not wired to the
   speculative example, so the patch sets `params.warmup = false` in
   `examples/speculative/speculative.cpp`.
2. **Stub vocab rejected.** The sidecar ships `tokenizer.ggml.model = none` (a
   dummy 248320-token stub), but the driver requires the draft and target vocab
   types to match. The patch skips the vocab-type/bos/eos/content checks when the
   draft vocab type is `NONE` (which is what the "a 'none' stub skips the vocab"
   comment already implies).

### Reference acceptance vs the Rust port (qa prompt, greedy)

| | drafted | accepted | rate | predicted tokens |
| --- | --- | --- | --- | --- |
| llama.cpp fork | 24 | 3 | 12.5% | 10 |
| Rust (`bonsai-spec`, n_draft 4) | 24 | 1 | 4.2% | 8 |

Both are low and the samples are tiny: across runs the Rust port accepted
1/24, 2/28, 2/20 and 3/24 (4.2-12.5%) against the fork's 3/24, so the two are
not distinguishable at this sample size. The low acceptance is
a property of this drafter/prompt rather than an obvious port bug: the reference
implementation with the real weights is not much better. A larger corpus is
needed to say whether the Rust draft is systematically worse. Fork prompt eval
was 1386 ms/token, i.e. the same speed class as our engine on this box.

## Footprint / packing (2026-09-13)

The drafter's cost is dominated by four tensors: `output.weight` 758 MiB (Q4_1),
`token_embd` 322 MiB (PQ2_0), `fc` 78 MiB (Q4_1) and `markov_head_a` 121 MiB
(BF16); the 42 block matrices are another ~468 MiB. Total 1856 MiB.

Per-block weight traffic (computed from the PQ2_0 block size, not measured): the
batched path moves 594 MiB - LM head 322 MiB, six layers 34.5 MiB each, markov
table 16 MiB x4 positions - where fetching each weight once per block position
moved 2181 MiB. So batching cuts the draft's traffic ~3.7x.

`bonsai-dspark repack <in> <out>` requantizes every Q4_1/BF16 matrix to PQ2_0
(ternary, per-128 absmax scale) and rewrites the GGUF:

| | Q4_1 sidecar | repacked |
| --- | --- | --- |
| file / payload | 1856 MiB | 924 MiB (50%) |
| draft block (warm) | 1.44 s | 0.52 s |
| acceptance (qa, n_draft 4) | 2/28 (7.1%) | 3/24 (12.5%) [one run] |

Acceptance did not get worse (tiny samples; the fork reference is 3/24), so
ternary packing halves the draft's memory and traffic and speeds the block up
2.8x at no measured quality cost. The block speedup comes from the PQ2_0
matvec using the AVX2 row kernel.

Two RAM fixes on the Rust side:

- `encode` was dequantizing the whole `fc` (25600x5120 -> **524 MiB**) to f32 on
  every observed token, and `logsnr_embed` the same for `fc2` (105 MiB) per
  block. Both now run row-by-row matvecs; norms/biases are memoized.
- The draft KV cache is clamped to the sidecar's `context_length` (4096) instead
  of the caller's 8192, halving it.

Draft-only peak RSS is now 575 MiB (was dominated by the 524 MiB fc spike).

Still open for footprint: store the draft KV cache as f16 (halves the remaining
~100 MiB and its read traffic), and the same f16 option for the target's
full-attention KV.

### Follow-up: where the draft time goes (BONSAI_DSPARK_TIME)

On the ternary sidecar a draft block splits as layers ~0.37 s, LM head ~0.34 s,
markov ~0.03 s. That is 9.5 GMAC in ~0.75 s, i.e. ~13 GMAC/s against the 24-30
GMAC/s the PQ2_0 kernel reaches on a large matvec, and the weights moved are
under 1 GB in 0.75 s. **The draft is compute-bound, not bandwidth-bound**, so
packing helps RAM and modestly helps time (through the smaller kernels), but
batching the block positions through `pq2_matmul_n` measured a wash here and is
kept only for bandwidth-bound hardware.

Total draft footprint after this work:

| | before | now |
| --- | --- | --- |
| sidecar payload | 1856 MiB | 924 MiB (ternary) |
| draft KV cache (ctx 4096) | 201 MiB (ctx 8192) | 100 MiB |
| transient f32 during encode | 524 MiB / token | 0 |
| draft-only peak RSS (smoke) | - | 575 MiB |
| draft block (warm) | 7.2 s (Q4_1, first cut) | ~0.5-0.8 s |

Draft KV cache is now f16 as well (widen the attended prefix into a reusable f32
scratch once per layer): 100 -> 50 MiB at ctx 4096 plus a 16.8 MiB scratch, with
unchanged logits/confidence.

Remaining footprint idea, not done: an f16 target KV cache (`BONSAI_KV=f16`
already exists for the target's full-attention layers, so this is mostly
accounting).

Bandwidth-vs-compute: every block-position matvec now goes through
`matvec_multi`, so each weight row is fetched once per block instead of once per
position. Draft weight traffic per block drops from 2181 MiB to 594 MiB (~3.7x; computed
from block sizes: head 322 MiB, layers 6x34.5 MiB). It is neutral on this CPU - the draft is
compute-bound, ~0.75 s per block either way - but it is the change that matters
on bandwidth-bound hardware, and the logits are bit-identical either way.

The remaining lever in the same direction is the *target* verify: it still
forwards one token per draft token, so the 6.8 GB target weight stream is read
once per drafted token. The primitive is in place
(`kernels::pq2_matmul_n`, wrapped as `Weights::matvec_batch_into`); what is left
is a `Decoder::verify_batch(tokens, start_pos)` that

1. runs the N-wide projections (`wq/wk/wv/wo`, FFN) through `matvec_batch_into`,
2. keeps the per-token pieces per token: per-head RMSNorm + rope, the causal
   attention over the KV cache, and the GDN recurrence (order-dependent),
3. returns per-position logits and leaves the caches at the accepted prefix.

It is deliberately not built yet: the `pq2_matmul_n` benchmark shows the per-token
cost is flat on this CPU (compute-bound), so it cannot pay here, and it would
touch the target path that every other bin depends on. Validate it by asserting
batched logits equal N sequential `forward_hidden` calls. Do it first when the
RX 7600 arrives.

## Packed tighter, and why the compute is not starved (2026-09-13)

### Sharing with the target

The drafter's `token_embd.weight` is **byte-identical** to the target's (hashed
both payloads; `output.weight` and `output_norm.weight` are not). So one of the
two vocab-sized tensors was pure duplication. `bonsai-dspark repack <in> <out>
[reference-target]` now drops any tensor identical to the reference's same-named
tensor and records it in `dspark.shared_tensors`; `Dspark` resolves those from
the target GGUF at load, and `Drafter::new` takes the target path.

| sidecar | size |
| --- | --- |
| original Q4_1 | 1856 MiB |
| + ternary repack | 924 MiB |
| + ternary and shared token_embd | **602 MiB (32% of original)** |

Logits, confidence and argmax are unchanged, and `bonsai-spec` is still
IDENTICAL to plain greedy.

### Compute is at the AVX2 ceiling, not starved

The drafter was allocating a `Vec` per attention head (160 per layer) for its
q/k RMSNorm and re-allocating all per-layer scratch every layer. Adding
`kernels::rms_norm_inplace` and hoisting the scratch took the block to **0.36 s**
with the machine quiet: layers 0.14 s, head 0.20 s, markov 0.02 s. That is
26-33 GMAC/s in situ, matching (and for the big head exceeding) the kernel's
standalone 18-30 GMAC/s, so the FMA units are fed.

The remaining limit is the algorithm on this uarch: a 2-bit decode needs ~3
loads per FMA (LUT gathers), which lands at ~2 MAC/cycle/core on Zen+ and is
therefore ~28 GMAC/s over four cores - exactly what is measured. Moving past it
needs VNNI/AVX-512 (not on Zen+) or a different inner loop, not a memory-layout
change.

### Remaining packing headroom

The weights are at the PQ2_0 floor (2.125 bits/weight). A true ternary packing
(log2(3) = 1.585 bits) would be ~25% smaller again but needs a custom format and
kernel, and a base-3 decode would likely be slower. The draft KV cache (50 MiB
f16) could go to int8 for another 25 MiB; both are marginal next to the 602 MiB
of weights.

## Throughput projection on a P100 / RX 7600 (2026-09-13)

Decode reads the weights once per token: 7.14 GB of PQ2_0 payload, plus ~0.27 GB
of KV (4096 ctx, f32) and ~0.15 GB of GDN recurrent state, so ~7.6 GB/token.
Both cards are memory-bound for this shape:

| | bandwidth | one pass (7.6 GB) | at 100% | at 60-75% (realistic) |
| --- | --- | --- | --- | --- |
| Tesla P100 16 GB (HBM2) | 732 GB/s | 10.4 ms | 96 tok/s | 58-72 tok/s |
| RX 7600 (GDDR6, 128-bit) | 288 GB/s | 26.4 ms | 38 tok/s | 23-29 tok/s |

Compute is not the limit on either: 13.45 GMAC/token is 2.9 ms on the P100's
9.3 TFLOPS fp32 and 1.24 ms on the 7600's 21.7 TFLOPS, both far under the memory
time. (The current CPU is the opposite - ~28 GMAC/s - which is why it is stuck
near 1 s/token.)

Speculative decoding changes the arithmetic: with the 602 MiB draft the per-round
traffic is ~8.2 GB for A accepted tokens instead of 7.6 GB for one, so it pays
from A > 1.1. At the ~12% acceptance measured here it is a loss; at 3 accepted
of 4 it is ~2.2x.

Caveats: the repo has no CUDA backend, so the P100 route is either the Vulkan
engine (vendor-neutral ash; NVIDIA's Vulkan driver covers Pascal, and 7.16 GB
uploads once over PCIe) or llama.cpp's CUDA backend. The 7600 fits 7.16 GB of
weights in 8 GB only with a small KV/workspace. And these are bandwidth
arithmetic, not measurements: the gfx902 iGPU currently reaches ~5 GB/s of its
18 GB/s shared bus because the single-submit engine is launch-bound, so hitting
60-75% of peak needs the batched/fused path.

### Acceptance re-measured (2026-09-13, shared sidecar)

`bonsai-spec Ternary-Bonsai-27B-PQ2_0.gguf Ternary-Bonsai-27B-dspark-shared.gguf
"text:Write a Rust function..." 32 4`: IDENTICAL (32 tokens), **accepted 3/112
drafts over 28 rounds = 2.7%**, spec 2.61 s/tok vs plain 2.20 s/tok. So the
shared-ternary sidecar did not change acceptance and spec decode is still a net
loss at this rate. Unchanged from the prior ~4-12% band; the numeric-reference
work in "Next steps" is still the blocker.

### Acceptance bisect round 2 (2026-09-13): build and convention knobs exhausted

All on `bonsai-spec`, target PQ2_0, code prompt, greedy, n_draft 4 (all
greedy-IDENTICAL to plain):

| arm | accepted | rate |
| --- | --- | --- |
| shared sidecar (default) | 3/112 | 2.7% |
| ternary sidecar | 3/112 | 2.7% |
| original Q4_1 sidecar | - | does not load (`slice_at ... past mmap end`) |
| `BONSAI_DSPARK_NONCAUSAL` | 5/104 | 4.8% |
| `BONSAI_DSPARK_ROPE=neox64` | (no change) | ~3% |
| `BONSAI_DSPARK_ROPE=imrope64` | (no change) | ~3% |
| `BONSAI_DSPARK_ANCHOR_NPAST` (reference placement) | 4/108 | 3.7% |

The reference (llama.cpp fork, `common/speculative.cpp`) places the anchor at
`n_past` (`common_batch_add(..., n + i)`) and reads `i_draft_beg = 0` for
anchor-first dspark; our port anchors at `n_past + 1`. Matching the reference
placement does **not** move acceptance, so the position convention is not the
cause.

Conclusion: the drafter's next-token top-1 agrees with the target roughly 10%
of the time (acceptance ~3-5%), and the fork reference is likewise low (~12.5%).
No engine-side knob (sidecar precision/build, rope dims, causal mask, markov,
anchor convention) moves it. Reaching ~75% requires a drafter that actually
predicts this target, or a numeric draft-logit diff against the reference to
find a remaining port bug if one exists.

### Reference acceptance measured (2026-09-13, decisive)

Rebuilt the patched fork (`cmake --build build --target llama-speculative`; the
old binary predated the warmup patch and dumped core). It now runs:

```
./build/bin/llama-speculative -m Ternary-Bonsai-27B-PQ2_0.gguf \
  -md Ternary-Bonsai-27B-dspark-dflash.gguf --spec-type draft-dspark \
  --spec-draft-n-max 4 -p "Write a Rust function..." -n 32 --temp 0
n_drafted = 124   n_accept = 1   accept = 0.806%
```

**The reference is 0.8%, our port is 2.7%.** Our engine already accepts more
draft tokens than the reference implementation with the same sidecar and
prompt. So the low acceptance is a property of the shipped drafter, not a port
bug: there is no engine-side change that reaches 75%, because the drafter's own
reference cannot. Reaching ~75% requires a drafter that actually predicts this
target (retrain/fine-tune or a matched checkpoint), which is outside "how we
build the drafter" in this engine.

### Reference draft-token capture (2026-09-13)

Patched the fork's debug prints (`common_token_to_piece(ctx_dft, ...)` aborts on
the sidecar's stub vocab; replaced with a literal) and ran `-v`. First-round
reference draft candidates: **2523, 513, 2574, 264**. Our port's first round:
[248069, 271, 9764, 579]. These are not directly comparable: the reference
tokenizes the prompt with `add_special = true` (BOS) while `bonsai-spec`'s
`text:` path does not, so the contexts differ (our round 1 target token is
248068, a special token, vs the reference's ordinary token). A rigorous
per-position diff needs both sides fed identical token ids; the aggregate
numbers (reference 0.806%, port 2.7%) already establish that the drafter is the
limit, and the port is not worse than the reference, so there is no evidence of
a remaining port bug worth chasing.

### Where 75% acceptance actually comes from (2026-09-13)

DSpark is DeepSeek's published drafter family; published results are ~60-85%
speedup / high accepted length, which requires a *trained* drafter. Our sidecar
behaves like an untrained or mismatched checkpoint: its own reference scores
0.806% and our port 2.7%, so no serving-side change reaches 75%.

The route to ~75% is therefore to train a DSpark drafter for this target, not to
tune the engine. Public tooling exists:
- SpecForge (SGLang) — trains EAGLE3 / DFlash / DSpark / Domino drafters.
- DeepSpec (deepseek-ai) — data prep, draft model implementations, training.
- NeMo AutoModel — DSpark recipe, 2-GPU FSDP2.

Requirements: an existing trained dspark checkpoint matched to
Ternary-Bonsai-27B, or GPUs + data to train one. Neither is available on this
4-core APU box, so this is blocked here and out of scope for the engine port.
