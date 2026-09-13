# KV cache: memory map and compaction options

Model: 64 layers, **16 full-attention** (every 4th), `n_head=24`, `n_head_kv=4`,
`head_dim=256`, `n_rep=6`. Byte costs below are per full-attention layer unless
stated.

## 1. Where the cache lives

| path | struct / buffer | layout |
| --- | --- | --- |
| CPU f32 | `forward.rs::AttnCache.k[il]`, `.v[il]` | `[token][kv_head][256 dims]`, f32 |
| CPU f16 | `.k16[il]`, `.v16[il]` | same, u16 |
| CPU planar | `.kq[il]` (indices) + `.kn[il]` (f32 norm) | `[token][kv_head][packed bytes]` + `[token][kv_head]` |
| GPU f32 | `gdev.rs::kcache[il]`/`vcache[il]` | `n_ctx * KV_STRIDE` f32, `KV_STRIDE = 4*256 = 1024` |
| GPU f16 | `k16[il]`/`v16[il]` | `n_ctx * 4 * 128` words |
| GPU planar | `kq[il]`/`vq[il]` | `n_ctx * 4 * (pwords + 1)` words, row = `pwords` index words **then one f32 norm word** |

The only "scale" stored is **one norm per (token, kv_head) vector** = one scale
per 256 coordinates (0.125 bit/coord at f16). It is not per-coordinate.

### GPU quantized row (per token, per kv head)

```
planar3:  [ 24 words packed 3-bit indices ][ 1 word f32 norm ]  = 100 B / 256 coords
planar4:  [ 32 words packed 4-bit indices ][ 1 word f32 norm ]  = 132 B / 256 coords
```

Packing is a continuous little-endian bitstream in `kvquant::pack_bits`. The
norm word is written by `kv_store_q.comp` (`words[nwords] = floatBitsToUint(n)`)
and read back per position in `attn_scores_q.comp` / `attn_out_q.comp`.

## 2. Bytes on disk (stored)

| mode | K row / token / layer | K+V pair | vs f32 |
| --- | --- | --- | --- |
| f32 / f32 | 4096 | 8192 | 1.0x |
| f16 / f16 | 2048 | 4096 | 2.0x |
| planar4 / f32 (K-only) | 528 | 4624 | 1.8x |
| planar3 / f32 (K-only) | 400 | 4496 | 1.8x |
| planar4 / planar4 | 528 | 1056 | 7.8x |
| **planar3 / planar3** | 400 | **800** | **10.2x** |
| planar2 / planar2 | 272 | 544 | 15.0x |

Whole cache over the 16 full-attention layers:

| ctx | f32 | f16 | planar3 sym | planar3 K-only |
| --- | --- | --- | --- | --- |
| 2048 | 256 MB | 128 MB | **25 MB** | 140 MB |
| 32768 | 4096 MB | 2048 MB | **400 MB** | 2248 MB |

## 3. Bytes *moved* per decode token (the part that is not obvious)

`attn_scores_q.comp` / `attn_out_q.comp` dispatch **one workgroup per query
head** (24 workgroups). Each query head reads its own kv head's rows for every
position, and 6 query heads share one kv head (`n_rep = 6`). So the K cache is
read **6x** per token, and V likewise. On the CPU path the same 6x happens in
the `for hq in 0..nh` loop.

Traffic per token per layer (planar3 symmetric) = 6 * (400 + 400) = 4.8 KB per
position, against 48 KB for f32. Shrinking the stored row shrinks all 6 reads,
so stored bytes and moved bytes scale together. But the 6x itself is a separate
multiplier that layout tuning does not fix.

## 4. What "grouped packing" would actually change

The upstream reference (`johndpope/llama-cpp-turboquant`) `block_planar3_0` is:

```
[ 2B f16 norm ][ 32B qs: 2-bit low plane ][ 16B signs: 1-bit plane ] = 50 B / 128 coords
```

= 3.125 bit/coord, with 128 coordinates per block (one norm per block).

Our planar3 row = 96 B indices + 4 B norm per 256 coords = **also 3.125
bit/coord**. So:

- The reference does **not** group tokens; it groups 128 contiguous coordinates
  and stores **one f16 norm per block**, exactly the 1-scale-per-vector idea we
  already have (ours is per 256 because `head_dim=256`).
- The compression difference is in the **bit arrangement**, not the scale count:
  the reference splits 3-bit into a byte-aligned 2-bit plane + 1-bit plane, so
  lane `i` reads its value with one 32-bit load and a shift; our bitstream can
  straddle word boundaries and needs the `off + bits > 8` fix-up branch.
- Byte savings from matching it: f16 norm instead of f32 = **2 B per vector**
  (~2% of the row). Not the 10x.

The 10x comes from **3-bit + symmetric (K and V both quantized)**, which we
already ship as `BONSAI_KV=planar3`.

## 5. Ranked compaction options (bytes first, then traffic)

| # | change | saves | cost / risk |
| --- | --- | --- | --- |
| 1 | use `planar3` not `planar4` | 1.32x row | quality (logit rel ~3e-2) |
| 2 | symmetric (quantize V too) | 2x | V inverse-rotation quality |
| 3 | grouped 2-bit+1-bit plane packing | ~0 bytes, fewer mem **instructions** (wide loads, no straddle branch) | rework `pack_bits` + 3 shaders |
| 4 | f16 norm, two per word | 2 B / vector (~2%) | align rows; small |
| 5 | drop the f32 norm word by folding it into the exponent of the centroids per row | 4 B / vector (~4%) | lossy, needs a per-row scale anyway |
| 6 | GQA-aware kernel: one workgroup per **kv** head, 6 query heads inside | **6x less K/V read traffic** | biggest traffic lever; restructure shaders + scores buffer |
| 7 | deferred prefill (keep f32/f16 during prefill, quantize on insert) | none (quality) | extra scratch (plan R5) |
| 8 | 2-bit + 1-bit sign (planar2) | 1.47x vs planar3 | quality cliff, untested |

## 6. Recommendation

Storage-wise the system is already near the reference: `planar3` symmetric is
3.125 bit/coord, the same as upstream, and is what makes 32K fit. The two real
remaining wins for **data moved through memory** are:

1. **GQA-aware attention kernel (option 6)** — read each K/V row once instead of
   6 times. This is a larger traffic reduction than any repacking.
2. **Grouped plane packing (option 3)** — same bytes, but byte-aligned groups so
   the inner dequant loop becomes wide loads + shift instead of a per-coordinate
   bitstream; this speeds up the 6x reads we keep.

Norm width (option 4) is a ~2% rounding error on top of these and is not worth
its own format change unless it rides along with option 3.
