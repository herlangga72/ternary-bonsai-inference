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
