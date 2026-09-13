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

Both are low and the samples are tiny (3 vs 1 events), so the low acceptance is
a property of this drafter/prompt rather than an obvious port bug: the reference
implementation with the real weights is not much better. A larger corpus is
needed to say whether the Rust draft is systematically worse. Fork prompt eval
was 1386 ms/token, i.e. the same speed class as our engine on this box.

## Footprint / packing (2026-09-13)

The drafter's cost is dominated by four tensors: `output.weight` 758 MiB (Q4_1),
`token_embd` 322 MiB (PQ2_0), `fc` 78 MiB (Q4_1) and `markov_head_a` 121 MiB
(BF16); the 42 block matrices are another ~468 MiB. Total 1856 MiB.

`bonsai-dspark repack <in> <out>` requantizes every Q4_1/BF16 matrix to PQ2_0
(ternary, per-128 absmax scale) and rewrites the GGUF:

| | Q4_1 sidecar | repacked |
| --- | --- | --- |
| file / payload | 1856 MiB | 924 MiB (50%) |
| draft block (warm) | 1.44 s | 0.52 s |
| acceptance (qa, n_draft 4) | 2/28 (7.1%) | 3/24 (12.5%) |

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
position. Draft weight traffic per block drops from ~2.4 GB to ~0.6 GB (LM head
379 MiB x4, layer matrices ~211 MiB x4). It is neutral on this CPU - the draft is
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
