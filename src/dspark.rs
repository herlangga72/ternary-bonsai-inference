//! Dspark speculator (PrismML `dflash`/`dspark` block-denoising drafter).
//!
//! This module ports the draft half of the speculative-decoding path: loading
//! the sidecar GGUF, the mixed-precision kernels it needs (F32, BF16, Q4_1,
//! PQ2_0), and the forward graph. The target half (tapped hidden states,
//! batched verify, recurrent-state rollback) lives in `spec.rs`.
//!
//! Graph (see `notes/dspark-port-plan.md`), reference `src/models/dflash.cpp`:
//!
//!   encoder:  taps[n_tok, 5*n_embd] --fc--> [n_tok, n_embd] --rmsnorm--> inp_g
//!   inject:   per draft layer, inp_g --wk/wv--> K/V (RMSNorm on K + rope),
//!             written into the draft KV cache at the committed positions
//!   decode:   [id_last, MASK x (block-1)] --token_embd--> (+ log-SNR embed)
//!             -> 6 layers -> output_norm -> LM head -> markov bias
//!
//! The sidecar's on-disk names are the `dspark.*` originals; nothing here
//! depends on the `bonsai-gguf dspark-convert` rewrite used for the fork.

#![allow(dead_code)]

use crate::gguf::{half_to_f32, GGUF, TensorInfo, TYPE_BF16, TYPE_F32, TYPE_PQ2_0, TYPE_Q4_1};
use crate::kernels;
use crate::rope::rope_neox;

/// Sidecar metadata resolved into the numbers the graph needs.
#[derive(Debug, Clone)]
pub struct DsparkCfg {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub eps: f32,
    pub rope_freq_base: f32,
    pub block_size: usize,
    pub context_length: usize,
    pub mask_token_id: u32,
    pub target_layers: Vec<u32>,
    pub markov_rank: usize,
    pub confidence_head: bool,
    pub log_snr: Option<(f32, f32)>,
}

impl DsparkCfg {
    pub fn from_gguf(g: &GGUF) -> Result<DsparkCfg, String> {
        let u32v = |k: &str| -> Result<u32, String> {
            g.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| format!("dspark cfg: missing/invalid '{k}'"))
        };
        let f32v = |k: &str| -> Result<f32, String> {
            g.get(k)
                .and_then(|v| v.as_f32())
                .ok_or_else(|| format!("dspark cfg: missing/invalid '{k}'"))
        };
        let arch = g
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch != "dspark" {
            return Err(format!("dspark cfg: architecture is '{arch}', expected 'dspark'"));
        }
        let conf = g
            .get("dspark.dspark.confidence_head")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let log_snr = if g
            .get("dspark.dspark.log_snr_conditioning")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            Some((
                f32v("dspark.dspark.min_log_snr")?,
                f32v("dspark.dspark.max_log_snr")?,
            ))
        } else {
            None
        };
        let target_layers = g
            .get("dspark.dspark.target_layers")
            .and_then(|v| v.as_array())
            .ok_or("dspark cfg: missing 'dspark.dspark.target_layers'")?
            .iter()
            .map(|v| v.as_u64().map(|x| x as u32))
            .collect::<Option<Vec<u32>>>()
            .ok_or("dspark cfg: target_layers must be integers")?;
        Ok(DsparkCfg {
            n_layer: u32v("dspark.block_count")? as usize,
            n_embd: u32v("dspark.embedding_length")? as usize,
            n_ff: u32v("dspark.feed_forward_length")? as usize,
            n_head: u32v("dspark.attention.head_count")? as usize,
            n_head_kv: u32v("dspark.attention.head_count_kv")? as usize,
            head_dim: u32v("dspark.attention.key_length")? as usize,
            n_vocab: u32v("dspark.vocab_size")? as usize,
            eps: f32v("dspark.attention.layer_norm_rms_epsilon")?,
            rope_freq_base: f32v("dspark.rope.freq_base")?,
            block_size: u32v("dspark.dspark.block_size")? as usize,
            context_length: u32v("dspark.context_length")? as usize,
            mask_token_id: u32v("dspark.dspark.mask_token_id")?,
            target_layers,
            markov_rank: u32v("dspark.dspark.markov_rank")? as usize,
            confidence_head: conf,
            log_snr,
        })
    }

    /// Width of the encoder input: `len(target_layers) * target n_embd`.
    pub fn n_embd_enc(&self) -> usize {
        self.target_layers.len() * self.n_embd
    }
}

// ---------------------------------------------------------------------------
// Mixed-precision dequant / matvec
// ---------------------------------------------------------------------------

const Q4_1_QK: usize = 32;
const Q4_1_BLOCK: usize = 20; // 2x fp16 (d, m) + 16 nibble bytes

/// Bytes in one row of a tensor with `ne0` columns.
pub fn row_bytes(ty: u32, ne0: usize) -> usize {
    match ty {
        TYPE_F32 => ne0 * 4,
        TYPE_BF16 => ne0 * 2,
        TYPE_Q4_1 => ne0.div_ceil(Q4_1_QK) * Q4_1_BLOCK,
        TYPE_PQ2_0 => kernels::pq2_row_bytes(ne0),
        _ => ne0 * 4, // caller validates the type first
    }
}

pub fn type_supported(ty: u32) -> bool {
    matches!(ty, TYPE_F32 | TYPE_BF16 | TYPE_Q4_1 | TYPE_PQ2_0)
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Dequantize one tensor row into `out` (`out.len() == ne0`).
pub fn dequant_row(ty: u32, raw: &[u8], ne0: usize, out: &mut [f32]) {
    match ty {
        TYPE_F32 => {
            for (i, o) in out.iter_mut().enumerate().take(ne0) {
                let b = &raw[i * 4..i * 4 + 4];
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        TYPE_BF16 => {
            for (i, o) in out.iter_mut().enumerate().take(ne0) {
                let b = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                *o = bf16_to_f32(b);
            }
        }
        TYPE_Q4_1 => {
            // ggml block_q4_1: elements 0..16 use the low nibble of qs[j],
            // elements 16..32 use the high nibble.
            for (blk, chunk) in raw.chunks_exact(Q4_1_BLOCK).enumerate() {
                let d = half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                let m = half_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
                let qs = &chunk[4..4 + 16];
                let base = blk * Q4_1_QK;
                for j in 0..Q4_1_QK / 2 {
                    if base + j < ne0 {
                        out[base + j] = d * (qs[j] & 0x0f) as f32 + m;
                    }
                    if base + j + Q4_1_QK / 2 < ne0 {
                        out[base + j + Q4_1_QK / 2] = d * (qs[j] >> 4) as f32 + m;
                    }
                }
                if base + Q4_1_QK >= ne0 {
                    break;
                }
            }
        }
        TYPE_PQ2_0 => {
            let dec = kernels::decode_pq2_0_row(raw, ne0);
            out[..ne0].copy_from_slice(&dec);
        }
        _ => {}
    }
}

/// Scalar `y = W @ x` for one mixed-precision tensor (rows contiguous).
/// Correctness-oriented; the speculative path runs this a few times per block.
pub fn matvec_scalar(
    ty: u32,
    payload: &[u8],
    ne0: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if !type_supported(ty) {
        return Err(format!("dspark matvec: unsupported tensor type {ty}"));
    }
    if x.len() != ne0 {
        return Err(format!("dspark matvec: x len {} != ne0 {ne0}", x.len()));
    }
    let rb = row_bytes(ty, ne0);
    let rows = y.len();
    if payload.len() < rows * rb {
        return Err(format!(
            "dspark matvec: payload {} < {} rows * {rb}",
            payload.len(),
            rows
        ));
    }
    let mut row = vec![0.0f32; ne0];
    for (r, y_r) in y.iter_mut().enumerate() {
        dequant_row(ty, &payload[r * rb..(r + 1) * rb], ne0, &mut row);
        let mut acc = 0.0f32;
        for (a, b) in x.iter().zip(row.iter()) {
            acc += a * b;
        }
        *y_r = acc;
    }
    Ok(())
}

/// Dot one quantized row with `x` (length ne0).
fn row_dot(ty: u32, raw: &[u8], ne0: usize, x: &[f32]) -> f32 {
    match ty {
        TYPE_F32 => {
            let mut acc = 0.0f32;
            for (i, xv) in x.iter().enumerate().take(ne0) {
                let b = &raw[i * 4..i * 4 + 4];
                acc += xv * f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            acc
        }
        TYPE_BF16 => {
            let mut acc = 0.0f32;
            for (i, xv) in x.iter().enumerate().take(ne0) {
                let b = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                acc += xv * bf16_to_f32(b);
            }
            acc
        }
        TYPE_Q4_1 => {
            let mut acc = 0.0f32;
            let nblk = ne0.div_ceil(Q4_1_QK);
            for blk in 0..nblk {
                let chunk = &raw[blk * Q4_1_BLOCK..(blk + 1) * Q4_1_BLOCK];
                let d = half_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                let m = half_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
                let qs = &chunk[4..4 + 16];
                let base = blk * Q4_1_QK;
                for j in 0..16 {
                    if base + j < ne0 {
                        acc += x[base + j] * (d * (qs[j] & 0x0f) as f32 + m);
                    }
                    if base + j + 16 < ne0 {
                        acc += x[base + j + 16] * (d * (qs[j] >> 4) as f32 + m);
                    }
                }
            }
            acc
        }
        TYPE_PQ2_0 => {
            let mut acc = 0.0f32;
            let nblk = ne0.div_ceil(128);
            for blk in 0..nblk {
                let base = blk * 34;
                let scale = half_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
                let qs = &raw[base + 2..base + 34];
                let off = blk * 128;
                let mut bs = 0.0f32;
                for j in 0..128 {
                    let idx = off + j;
                    if idx >= ne0 {
                        break;
                    }
                    let code = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
                    bs += x[idx] * (code as i32 - 1) as f32;
                }
                acc += bs * scale;
            }
            acc
        }
        _ => 0.0,
    }
}

/// Threaded `y[..n_rows] = W[base_row..base_row+n_rows] @ x`, matching the
/// row-chunking strategy of `kernels::pq2_matvec_range`. PQ2_0 tensors are
/// delegated to the AVX2 production kernel.
pub fn matvec_par(
    ty: u32,
    payload: &[u8],
    ne0: usize,
    base_row: u64,
    n_rows: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if ty == TYPE_PQ2_0 {
        return kernels::pq2_matvec_range(payload, ne0, base_row, n_rows, x, y);
    }
    if ty == TYPE_Q4_1 {
        return kernels::q4_1_matvec_range(payload, ne0, base_row, n_rows, x, y);
    }
    if !type_supported(ty) {
        return Err(format!("dspark matvec: unsupported tensor type {ty}"));
    }
    if x.len() != ne0 {
        return Err(format!("dspark matvec: x len {} != ne0 {ne0}", x.len()));
    }
    let rb = row_bytes(ty, ne0);
    let base = base_row as usize;
    if payload.len() < (base + n_rows) * rb {
        return Err("dspark matvec: payload too small".into());
    }
    let out = &mut y[..n_rows];
    const MIN_PARALLEL: usize = 512;
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let n_cores = kernels::scaled_threads(avail);
    if n_rows < MIN_PARALLEL || n_cores <= 1 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = row_dot(ty, &payload[(base + r) * rb..(base + r + 1) * rb], ne0, x);
        }
        return Ok(());
    }
    let n_workers = n_cores.min(n_rows);
    let chunk = n_rows.div_ceil(n_workers);
    let ranges: Vec<std::ops::Range<usize>> = (0..n_workers)
        .map(|w| {
            let s = w * chunk;
            s..(s + chunk).min(n_rows)
        })
        .collect();
    std::thread::scope(|scope| {
        let mut rest = out;
        for range in ranges {
            let (head, tail) = rest.split_at_mut(range.len());
            rest = tail;
            scope.spawn(move || {
                for (k, o) in head.iter_mut().enumerate() {
                    let r = base + range.start + k;
                    *o = row_dot(ty, &payload[r * rb..(r + 1) * rb], ne0, x);
                }
            });
        }
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// Draft weight context
// ---------------------------------------------------------------------------

pub struct Dspark {
    pub cfg: DsparkCfg,
    pub gguf: GGUF,
    /// Tensors the sidecar omits because they are byte-identical to the target
    /// model's (recorded as `dspark.shared_tensors`), plus that target GGUF.
    shared: Option<(GGUF, Vec<String>)>,
    /// Memoized small f32 tensors (norms, biases). Dequantizing them on every
    /// draft call is pure waste; they never change.
    f32_memo: std::cell::RefCell<std::collections::HashMap<String, Vec<f32>>>,
}

impl Dspark {
    pub fn open(path: &str) -> Result<Dspark, String> {
        Self::open_with_target(path, None)
    }

    /// Open the sidecar, resolving any tensor listed in `dspark.shared_tensors`
    /// from `target_path` (they were dropped at repack time because they are
    /// byte-identical to the target's, e.g. the 322 MiB token embedding).
    pub fn open_with_target(path: &str, target_path: Option<&str>) -> Result<Dspark, String> {
        let gguf = GGUF::open(path)?;
        let cfg = DsparkCfg::from_gguf(&gguf)?;
        let shared_names: Vec<String> = gguf
            .get("dspark.shared_tensors")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let shared = if shared_names.is_empty() {
            None
        } else {
            let tp = target_path.ok_or_else(|| {
                format!(
                    "{path}: needs the target model to resolve {} shared tensor(s); \
                     pass --model <target.gguf>",
                    shared_names.len()
                )
            })?;
            Some((GGUF::open(tp)?, shared_names))
        };
        Ok(Dspark {
            cfg,
            gguf,
            shared,
            f32_memo: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Locate a tensor and the GGUF that owns it: the sidecar first, then the
    /// target for names declared shared.
    fn resolve<'a>(&'a self, name: &str) -> Result<(&'a GGUF, TensorInfo), String> {
        if let Some(t) = self.gguf.tensors.iter().find(|t| t.name == name) {
            return Ok((&self.gguf, t.clone()));
        }
        if let Some((g, names)) = &self.shared {
            if names.iter().any(|n| n == name) {
                if let Some(t) = g.tensors.iter().find(|t| t.name == name) {
                    return Ok((g, t.clone()));
                }
            }
        }
        Err(format!("dspark: tensor '{name}' missing"))
    }

    pub fn tensor(&self, name: &str) -> Result<TensorInfo, String> {
        self.resolve(name).map(|(_, t)| t)
    }

    fn payload<'a>(&'a self, name: &str) -> Result<(&'a [u8], TensorInfo), String> {
        let (g, t) = self.resolve(name)?;
        let p = g.payload_slice(&t)?;
        Ok((p, t))
    }

    /// Fetch a full tensor as f32 (small tensors only).
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let (payload, t) = self.payload(name)?;
        if !type_supported(t.ty) {
            return Err(format!("dspark {name}: unsupported type {}", t.ty));
        }
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        let mut out = vec![0.0f32; rows * ne0];
        for r in 0..rows {
            let rb = row_bytes(t.ty, ne0);
            dequant_row(t.ty, &payload[r * rb..(r + 1) * rb], ne0, &mut out[r * ne0..(r + 1) * ne0]);
        }
        Ok(out)
    }

    /// Dequantize a small f32 tensor once and keep it (norms, biases).
    pub fn cached_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        if let Some(v) = self.f32_memo.borrow().get(name) {
            return Ok(v.clone());
        }
        let v = self.read_f32(name)?;
        self.f32_memo.borrow_mut().insert(name.to_string(), v.clone());
        Ok(v)
    }

    /// `y = W @ x` for a named matrix tensor. `x.len()` must equal `ne0`.
    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?.clone();
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        let payload = self.gguf.payload_slice(&t)?;
        let mut y = vec![0.0f32; rows];
        matvec_scalar(t.ty, payload, ne0, x, &mut y)?;
        Ok(y)
    }

    /// Encoder: fuse tapped target features into the draft hidden width.
    /// `features` is row-major `[n_tok, n_embd_enc]` (per token, the tapped
    /// layers' hidden vectors concatenated in `target_layers` order).
    pub fn encode(&self, features: &[f32], n_tok: usize) -> Result<Vec<f32>, String> {
        let n_enc = self.cfg.n_embd_enc();
        if features.len() != n_tok * n_enc {
            return Err(format!(
                "dspark encode: features len {} != {n_tok} * {n_enc}",
                features.len()
            ));
        }
        // Matvec row by row: never materialize fc (25600 x 5120 f32 = 524 MiB).
        let hidden_norm = self.cached_f32("dspark.hidden_norm.weight")?;
        let n_embd = self.cfg.n_embd;
        let mut out = vec![0.0f32; n_tok * n_embd];
        let mut y = vec![0.0f32; n_embd];
        for t in 0..n_tok {
            let x = &features[t * n_enc..(t + 1) * n_enc];
            self.matvec_range("dspark.fc.weight", 0, n_embd, x, &mut y)?;
            let dst = &mut out[t * n_embd..(t + 1) * n_embd];
            dst.copy_from_slice(&y);
            kernels::rms_norm_inplace(dst, &hidden_norm, self.cfg.eps);
        }
        Ok(out)
    }

    /// One tensor row (ne0 values) as f32.
    pub fn row_f32(&self, name: &str, row: u64) -> Result<Vec<f32>, String> {
        let (payload, t) = self.payload(name)?;
        if !type_supported(t.ty) {
            return Err(format!("dspark {name}: unsupported type {}", t.ty));
        }
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rb = row_bytes(t.ty, ne0);
        let start = row as usize * rb;
        if start + rb > payload.len() {
            return Err(format!("dspark {name}: row {row} out of range"));
        }
        let mut out = vec![0.0f32; ne0];
        dequant_row(t.ty, &payload[start..start + rb], ne0, &mut out);
        Ok(out)
    }

    /// `y = W[base..base+n_rows] @ x` for a named tensor.
    pub fn matvec_range(
        &self,
        name: &str,
        base_row: u64,
        n_rows: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        let (payload, t) = self.payload(name)?;
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        matvec_par(t.ty, payload, ne0, base_row, n_rows, x, y)
    }

    /// `y = W @ x` for a named tensor, allocating the output.
    pub fn matvec2(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?;
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        let mut y = vec![0.0f32; rows];
        self.matvec_range(name, 0, rows, x, &mut y)?;
        Ok(y)
    }

    /// `y[t] = W @ x[t]` for all `n_tok` rows of `xmat`. PQ2_0 uses the
    /// N-column GEMM (each weight row fetched once); other types fall back to
    /// per-token matvecs.
    pub fn matvec_multi(
        &self,
        name: &str,
        xmat: &[f32],
        n_tok: usize,
        y: &mut [f32],
    ) -> Result<(), String> {
        let (payload, t) = self.payload(name)?;
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        if xmat.len() != n_tok * ne0 || y.len() < n_tok * rows {
            return Err(format!("matvec_multi {name}: shape mismatch"));
        }
        if t.ty == TYPE_PQ2_0 {
            return kernels::pq2_matmul_n(payload, ne0, 0, rows, xmat, n_tok, y);
        }
        for tok in 0..n_tok {
            matvec_par(
                t.ty,
                payload,
                ne0,
                0,
                rows,
                &xmat[tok * ne0..(tok + 1) * ne0],
                &mut y[tok * rows..(tok + 1) * rows],
            )?;
        }
        Ok(())
    }

    /// Project committed target features into the draft KV cache.
    /// `inp_g` is `[n_tok, n_embd]` from `encode`, `positions[i]` the absolute
    /// position of token `i`. Mirrors the reference's "embd batch" decoder pass.
    pub fn inject(
        &self,
        cache: &mut DraftCache,
        inp_g: &[f32],
        positions: &[usize],
    ) -> Result<(), String> {
        let c = &self.cfg;
        let n_tok = positions.len();
        let n_kv_dim = c.n_head_kv * c.head_dim;
        if inp_g.len() != n_tok * c.n_embd {
            return Err("dspark inject: inp_g width mismatch".into());
        }
        for il in 0..c.n_layer {
            let wk = format!("blk.{il}.attn_k.weight");
            let wv = format!("blk.{il}.attn_v.weight");
            let kn = format!("blk.{il}.attn_k_norm.weight");
            let knorm = self.cached_f32(&kn)?;
            for (t, &pos) in positions.iter().enumerate() {
                if pos >= cache.n_ctx {
                    return Err(format!("dspark inject: pos {pos} >= ctx {}", cache.n_ctx));
                }
                let x = &inp_g[t * c.n_embd..(t + 1) * c.n_embd];
                let mut k = vec![0.0f32; n_kv_dim];
                let mut v = vec![0.0f32; n_kv_dim];
                self.matvec_range(&wk, 0, n_kv_dim, x, &mut k)?;
                self.matvec_range(&wv, 0, n_kv_dim, x, &mut v)?;
                for h in 0..c.n_head_kv {
                    let head = &mut k[h * c.head_dim..(h + 1) * c.head_dim];
                    kernels::rms_norm_inplace(head, &knorm, c.eps);
                    rope_neox(head, pos as f32, c.head_dim, c.rope_freq_base);
                }
                cache.put(il, pos, &k, &v);
            }
        }
        let end = positions.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        cache.filled = cache.filled.max(end);
        Ok(())
    }

    /// Log-SNR embedding for a block of `n_tok` tokens (anchor at offset 0 gets
    /// `max_log_snr`, the rest `min_log_snr`), `[n_tok, n_embd]`.
    pub fn logsnr_embed(&self, n_tok: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let (min_snr, max_snr) = c.log_snr.ok_or("dspark: log-SNR conditioning disabled")?;
        let n_freq = 128usize;
        let half = n_freq / 2;
        let b1 = self.cached_f32("dspark.log_snr_fc1.bias")?;
        let b2 = self.cached_f32("dspark.log_snr_fc2.bias")?;
        let ne = c.n_embd;
        let mut out = vec![0.0f32; n_tok * ne];
        let mut feat = vec![0.0f32; n_freq];
        let mut h = vec![0.0f32; ne];
        let mut e = vec![0.0f32; ne];
        for pos in 0..n_tok {
            let log_snr = if pos % c.block_size == 0 { max_snr } else { min_snr };
            let tt = (log_snr - min_snr) / (max_snr - min_snr) * 1000.0;
            for i in 0..half {
                let freq = (-(10000.0f32).ln() * i as f32 / half as f32).exp();
                let angle = tt * freq;
                feat[i] = angle.sin();
                feat[half + i] = angle.cos();
            }
            // row-by-row matvecs; nothing large is materialized
            self.matvec_range("dspark.log_snr_fc1.weight", 0, ne, &feat, &mut h)?;
            for r in 0..ne {
                h[r] = kernels::silu(h[r] + b1[r]);
            }
            self.matvec_range("dspark.log_snr_fc2.weight", 0, ne, &h, &mut e)?;
            for r in 0..ne {
                e[r] += b2[r];
            }
            out[pos * ne..(pos + 1) * ne].copy_from_slice(&e);
        }
        Ok(out)
    }

    /// Draft one noise block: `[id_last, MASK x (block_size-1)]` at positions
    /// `n_past..n_past+block_size`. Returns the markov-biased logits per block
    /// position (`[block_size * n_vocab]`), the confidence per position, and the
    /// normalized hidden states (`[block_size * n_embd]`, the confidence input).
    pub fn draft_block(
        &self,
        cache: &mut DraftCache,
        id_last: u32,
        n_past: usize,
    ) -> Result<DraftBlock, String> {
        let anchorless = std::env::var("BONSAI_DSPARK_ANCHORLESS").is_ok();
        let timeit = std::env::var("BONSAI_DSPARK_TIME").is_ok();
        let mut t_layers = 0.0f64;
        let mut t_head = 0.0f64;
        let mut t_markov = 0.0f64;
        let c = &self.cfg;
        let n_tok = c.block_size + if anchorless { 1 } else { 0 };
        let ne = c.n_embd;
        let hd = c.head_dim;
        let n_kv_dim = c.n_head_kv * hd;
        let group = c.n_head / c.n_head_kv;

        // --- embeddings ---------------------------------------------------
        let mut x = vec![0.0f32; n_tok * ne];
        for t in 0..n_tok {
            let tok = if t == 0 { id_last } else { c.mask_token_id };
            let e = self.row_f32("token_embd.weight", tok as u64)?;
            x[t * ne..(t + 1) * ne].copy_from_slice(&e);
        }
        if c.log_snr.is_some() {
            let snr = self.logsnr_embed(n_tok)?;
            for i in 0..n_tok * ne {
                x[i] += snr[i];
            }
        }

        let positions: Vec<usize> = (0..n_tok).map(|t| n_past + t).collect();
        let max_pos = n_past + n_tok;
        if max_pos > cache.n_ctx {
            return Err(format!("dspark: block end {max_pos} > ctx {}", cache.n_ctx));
        }

        // per-block scratch, allocated once and reused across layers
        let mut q = vec![0.0f32; n_tok * c.n_head * hd];
        let mut kk = vec![0.0f32; n_tok * n_kv_dim];
        let mut vv = vec![0.0f32; n_tok * n_kv_dim];
        let mut xn = vec![0.0f32; n_tok * ne];
        let mut attn = vec![0.0f32; n_tok * c.n_head * hd];
        let mut scores = vec![0.0f32; cache.n_ctx];
        let mut y = vec![0.0f32; n_tok * ne];
        let mut o = vec![0.0f32; n_tok * ne];
        let mut gate = vec![0.0f32; n_tok * c.n_ff];
        let mut up = vec![0.0f32; n_tok * c.n_ff];
        let mut down = vec![0.0f32; n_tok * ne];

        let t_lay = std::time::Instant::now();
        for il in 0..c.n_layer {
            let an = self.cached_f32(&format!("blk.{il}.attn_norm.weight"))?;
            let qn = self.cached_f32(&format!("blk.{il}.attn_q_norm.weight"))?;
            let kn = self.cached_f32(&format!("blk.{il}.attn_k_norm.weight"))?;
            let fnorm = self.cached_f32(&format!("blk.{il}.ffn_norm.weight"))?;

            // q/k/v projections for every block position in one pass, so each
            // weight row is fetched once per block rather than once per position
            for t in 0..n_tok {
                let dst = &mut xn[t * ne..(t + 1) * ne];
                dst.copy_from_slice(&x[t * ne..(t + 1) * ne]);
                kernels::rms_norm_inplace(dst, &an, c.eps);
            }
            self.matvec_multi(&format!("blk.{il}.attn_q.weight"), &xn, n_tok, &mut q)?;
            self.matvec_multi(&format!("blk.{il}.attn_k.weight"), &xn, n_tok, &mut kk)?;
            self.matvec_multi(&format!("blk.{il}.attn_v.weight"), &xn, n_tok, &mut vv)?;
            for t in 0..n_tok {
                for h in 0..c.n_head {
                    let head = &mut q[t * c.n_head * hd + h * hd..t * c.n_head * hd + (h + 1) * hd];
                    kernels::rms_norm_inplace(head, &qn, c.eps);
                    rope_neox(head, positions[t] as f32, hd, c.rope_freq_base);
                }
                for h in 0..c.n_head_kv {
                    let base = t * n_kv_dim + h * hd;
                    let head = &mut kk[base..base + hd];
                    kernels::rms_norm_inplace(head, &kn, c.eps);
                    rope_neox(head, positions[t] as f32, hd, c.rope_freq_base);
                }
                cache.put(
                    il,
                    positions[t],
                    &kk[t * n_kv_dim..(t + 1) * n_kv_dim],
                    &vv[t * n_kv_dim..(t + 1) * n_kv_dim],
                );
            }
            cache.filled = cache.filled.max(max_pos);

            // attention over the filled positions, causal like the reference
            // (`build_attn_inp_kq_mask` with hparams.causal_attn = true):
            // query at position p attends to cache positions <= p.
            let noncausal = std::env::var("BONSAI_DSPARK_NONCAUSAL").is_ok();
            let scale = 1.0f32 / (hd as f32).sqrt();
            let filled = cache.filled;
            let (kcache, vcache) = cache.prefix_f32(il);
            for t in 0..n_tok {
                let n_ctx_pos = if noncausal { filled } else { positions[t] + 1 };
                for h in 0..c.n_head {
                    let hkv = h / group;
                    let qh = &q[t * c.n_head * hd + h * hd..t * c.n_head * hd + (h + 1) * hd];
                    let mut maxs = f32::NEG_INFINITY;
                    for p in 0..n_ctx_pos {
                        let kb = p * n_kv_dim + hkv * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += qh[d] * kcache[kb + d];
                        }
                        let s = dot * scale;
                        scores[p] = s;
                        if s > maxs {
                            maxs = s;
                        }
                    }
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut().take(n_ctx_pos) {
                        let e = (*s - maxs).exp();
                        *s = e;
                        sum += e;
                    }
                    let out =
                        &mut attn[t * c.n_head * hd + h * hd..t * c.n_head * hd + (h + 1) * hd];
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for p in 0..n_ctx_pos {
                            acc += scores[p] * vcache[p * n_kv_dim + hkv * hd + d];
                        }
                        out[d] = acc / sum;
                    }
                }
            }

            // output projection (batched over positions) + residual
            self.matvec_multi(&format!("blk.{il}.attn_output.weight"), &attn, n_tok, &mut o)?;
            for i in 0..n_tok * ne {
                y[i] = x[i] + o[i];
            }

            // FFN, also batched
            for t in 0..n_tok {
                let dst = &mut xn[t * ne..(t + 1) * ne];
                dst.copy_from_slice(&y[t * ne..(t + 1) * ne]);
                kernels::rms_norm_inplace(dst, &fnorm, c.eps);
            }
            self.matvec_multi(&format!("blk.{il}.ffn_gate.weight"), &xn, n_tok, &mut gate)?;
            self.matvec_multi(&format!("blk.{il}.ffn_up.weight"), &xn, n_tok, &mut up)?;
            for i in 0..n_tok * c.n_ff {
                gate[i] = kernels::silu(gate[i]) * up[i];
            }
            self.matvec_multi(&format!("blk.{il}.ffn_down.weight"), &gate, n_tok, &mut down)?;
            for i in 0..n_tok * ne {
                x[i] = y[i] + down[i];
            }
        }

        t_layers += t_lay.elapsed().as_secs_f64();
        let t_hd = std::time::Instant::now();
        // --- final norm + LM head ----------------------------------------
        let on = self.cached_f32("output_norm.weight")?;
        let mut emb = vec![0.0f32; n_tok * ne];
        let mut base = vec![0.0f32; n_tok * c.n_vocab];
        for t in 0..n_tok {
            let dst = &mut emb[t * ne..(t + 1) * ne];
            dst.copy_from_slice(&x[t * ne..(t + 1) * ne]);
            kernels::rms_norm_inplace(dst, &on, c.eps);
        }
        // LM head once for every block position: the head is the largest tensor
        // (379 MiB), so reading it once instead of n_tok times is the main
        // traffic cut in the draft.
        self.matvec_multi("output.weight", &emb, n_tok, &mut base)?;

        t_head += t_hd.elapsed().as_secs_f64();
        let t_mk = std::time::Instant::now();
        // --- markov + confidence heads -----------------------------------
        let (mw1_payload, mw1) = self.payload("dspark.markov_head_a.weight")?;
        let mw1_ne0 = mw1.dims.first().copied().unwrap_or(0) as usize;
        let mw1_rb = row_bytes(mw1.ty, mw1_ne0);
        let conf = self.cached_f32("dspark.confidence_head.weight")?;
        let conf_b = self.cached_f32("dspark.confidence_head.bias")?;

        let mut logits = vec![0.0f32; n_tok * c.n_vocab];
        let mut conf_out = vec![0.0f32; n_tok];
        // ablation switches for localizing draft-quality issues
        let no_markov = std::env::var("BONSAI_DSPARK_NO_MARKOV").is_ok();
        let ignore_bias = no_markov;
        let i_beg = if anchorless { 1 } else { 0 };
        let mut prev = id_last as u64;
        for t in i_beg..n_tok {
            // markov_w1 row `prev` -> [rank]
            let start = prev as usize * mw1_rb;
            let mut w1_prev = vec![0.0f32; mw1_ne0];
            dequant_row(mw1.ty, &mw1_payload[start..start + mw1_rb], mw1_ne0, &mut w1_prev);
            let mut bias = vec![0.0f32; c.n_vocab];
            self.matvec_range("dspark.markov_head_b.weight", 0, c.n_vocab, &w1_prev, &mut bias)?;
            let col = &mut logits[t * c.n_vocab..(t + 1) * c.n_vocab];
            for i in 0..c.n_vocab {
                col[i] = base[t * c.n_vocab + i] + if ignore_bias { 0.0 } else { bias[i] };
            }
            // confidence: sigmoid(conf_proj . [emb ; w1_prev] + b)
            let mut feat = vec![0.0f32; ne + mw1_ne0];
            feat[..ne].copy_from_slice(&emb[t * ne..(t + 1) * ne]);
            feat[ne..].copy_from_slice(&w1_prev);
            let mut acc = conf_b[0];
            for (a, b) in feat.iter().zip(conf.iter()) {
                acc += a * b;
            }
            conf_out[t] = kernels::sigmoid(acc);
            // next position is conditioned on this position's argmax
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for i in 0..c.n_vocab {
                if col[i] > bv {
                    bv = col[i];
                    best = i;
                }
            }
            prev = best as u64;
        }

        t_markov += t_mk.elapsed().as_secs_f64();
        if timeit {
            eprintln!(
                "[dspark] layers {:.3}s  head {:.3}s  markov {:.3}s",
                t_layers, t_head, t_markov
            );
        }
        Ok(DraftBlock {
            logits,
            conf: conf_out,
            emb,
        })
    }
}

/// Per-layer draft K/V caches plus the number of filled positions.
pub struct DraftCache {
    pub n_ctx: usize,
    pub n_kv_dim: usize,
    /// K/V stored as f16: half the memory and half the read traffic. The
    /// attended prefix is widened into `ks`/`vs` once per layer.
    pub k16: Vec<Vec<u16>>,
    pub v16: Vec<Vec<u16>>,
    pub filled: usize,
    ks: Vec<f32>,
    vs: Vec<f32>,
}

impl DraftCache {
    pub fn new(cfg: &DsparkCfg, n_ctx: usize) -> DraftCache {
        let n_kv_dim = cfg.n_head_kv * cfg.head_dim;
        let len = n_ctx * n_kv_dim;
        DraftCache {
            n_ctx,
            n_kv_dim,
            k16: (0..cfg.n_layer).map(|_| vec![0u16; len]).collect(),
            v16: (0..cfg.n_layer).map(|_| vec![0u16; len]).collect(),
            filled: 0,
            ks: vec![0.0f32; len],
            vs: vec![0.0f32; len],
        }
    }

    /// Drop every position `>= len` (rejected draft tail).
    pub fn truncate(&mut self, len: usize) {
        self.filled = self.filled.min(len);
    }

    /// Widen layer `il`'s filled prefix into the f32 scratch and return it.
    fn prefix_f32(&mut self, il: usize) -> (&[f32], &[f32]) {
        let n = self.filled * self.n_kv_dim;
        for i in 0..n {
            self.ks[i] = crate::gguf::half_to_f32(self.k16[il][i]);
            self.vs[i] = crate::gguf::half_to_f32(self.v16[il][i]);
        }
        (&self.ks[..n], &self.vs[..n])
    }

    /// Append one position's K/V row (`n_kv_dim` f32 values) for layer `il`.
    fn put(&mut self, il: usize, pos: usize, k: &[f32], v: &[f32]) {
        let dst = pos * self.n_kv_dim;
        for i in 0..self.n_kv_dim {
            self.k16[il][dst + i] = crate::gguf::f32_to_half(k[i]);
            self.v16[il][dst + i] = crate::gguf::f32_to_half(v[i]);
        }
    }
}

/// Output of one draft block.
pub struct DraftBlock {
    /// markov-biased logits, `[block_size * n_vocab]`
    pub logits: Vec<f32>,
    /// confidence per block position
    pub conf: Vec<f32>,
    /// normalized hidden states `[block_size * n_embd]` (confidence input)
    pub emb: Vec<f32>,
}

/// Quantize one row (`len % 128 == 0`) to PQ2_0 blocks: fp16 scale + 32 code
/// bytes per 128 weights. Codes are ternary {-1,0,1} * scale with the scale set
/// to the block absmax (matching how the target model's weights are stored).
pub fn quantize_pq2_0_row(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % 128, 0);
    for blk in x.chunks_exact(128) {
        let absmax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = absmax;
        let bytes_at = out.len();
        out.extend_from_slice(&crate::gguf::f32_to_half(scale).to_le_bytes());
        out.extend_from_slice(&[0u8; 32]);
        if scale == 0.0 {
            for b in &mut out[bytes_at + 2..bytes_at + 34] {
                *b = 0x55; // all codes 1 -> value 0
            }
            continue;
        }
        let inv = 1.0 / scale;
        for j in 0..128 {
            if j % 4 == 0 {
                out[bytes_at + 2 + j / 4] = 0;
            }
            let q = (blk[j] * inv).round() as i32 + 1;
            let code = q.clamp(0, 2) as u8; // ternary: use -1, 0, 1
            out[bytes_at + 2 + j / 4] |= code << ((j % 4) * 2);
        }
    }
}

/// Requantize a sidecar's Q4_1 and BF16 matrices to PQ2_0 (ternary) and write
/// the result as a new GGUF. This trades drafter fidelity for roughly half the
/// bytes per draft pass, so the draft's memory traffic drops with it.
///
/// F32 norms and any tensor whose row width is not a multiple of 128 are kept
/// verbatim. Returns `(converted, bytes_in, bytes_out)`.
pub fn repack_ternary(
    src: &str,
    dst: &str,
    reference: Option<&str>,
) -> Result<(usize, usize, u64, u64), String> {
    use crate::gguf::{tensor_nbytes_for, write_gguf, TYPE_BF16, TYPE_PQ2_0, TYPE_Q4_1};
    let g = GGUF::open(src)?;
    let mut bytes_in = 0u64;
    for t in &g.tensors {
        bytes_in += g.tensor_nbytes(t);
    }
    // optional reference (the target model): tensors that are byte-identical to
    // the reference's same-named tensor are dropped and resolved from it at load
    let refg = match reference {
        Some(p) => Some(GGUF::open(p)?),
        None => None,
    };

    // data order (the writer lays payloads out contiguously, and llama.cpp
    // requires index order == data order)
    let mut order: Vec<usize> = (0..g.tensors.len()).collect();
    order.sort_by_key(|&i| g.tensors[i].offset);

    struct Item {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        src_ty: u32,
        src_off: u64,
        src_nbytes: u64,
        ne0: usize,
        rows: usize,
    }
    let mut items: Vec<Item> = Vec::with_capacity(order.len());
    let mut converted = 0usize;
    let mut shared: Vec<String> = Vec::new();
    for &i in &order {
        let t = &g.tensors[i];
        let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
        let rows = (t.n_elem() as usize) / ne0.max(1);
        // drop a tensor that is byte-identical to the reference's
        if let Some(rg) = &refg {
            if let Some(rt) = rg.tensors.iter().find(|r| r.name == t.name) {
                let same_shape = rt.ty == t.ty && rt.dims == t.dims;
                let same_bytes = same_shape && rt.n_elem() == t.n_elem() && {
                    let a = g.payload_slice(t)?;
                    let b = rg.payload_slice(rt)?;
                    a == b
                };
                if same_bytes {
                    shared.push(t.name.clone());
                    continue;
                }
            }
        }
        let convertible = (t.ty == TYPE_Q4_1 || t.ty == TYPE_BF16)
            && ne0 % 128 == 0
            && ne0 > 0
            && t.n_elem() as usize % 128 == 0;
        let ty = if convertible { TYPE_PQ2_0 } else { t.ty };
        if convertible {
            converted += 1;
        }
        items.push(Item {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ty,
            src_ty: t.ty,
            src_off: g.data_start + t.offset,
            src_nbytes: g.tensor_nbytes(t),
            ne0,
            rows,
        });
    }

    let mut meta: Vec<(String, crate::gguf::Value)> =
        g.meta.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    if !shared.is_empty() {
        meta.push((
            "dspark.shared_tensors".to_string(),
            crate::gguf::Value::Array {
                elem_type: 8, // GGUF string
                items: shared
                    .clone()
                    .into_iter()
                    .map(crate::gguf::Value::Str)
                    .collect(),
            },
        ));
    }
    meta.sort_by(|a, b| a.0.cmp(&b.0));
    let descs: Vec<(String, Vec<u64>, u32)> = items
        .iter()
        .map(|it| (it.name.clone(), it.dims.clone(), it.ty))
        .collect();

    let map = g.file_bytes();
    let mut row_buf: Vec<f32> = Vec::new();
    write_gguf(dst, g.version, &meta, &descs, |i, out| {
        let it = &items[i];
        if it.ty != TYPE_PQ2_0 || it.src_ty == TYPE_PQ2_0 {
            // verbatim copy
            let begin = it.src_off as usize;
            let end = begin + it.src_nbytes as usize;
            if end > map.len() {
                return Err(format!("{}: source data out of range", it.name));
            }
            out.extend_from_slice(&map[begin..end]);
            return Ok(());
        }
        row_buf.resize(it.ne0, 0.0);
        let src_rb = row_bytes(it.src_ty, it.ne0);
        for r in 0..it.rows {
            let begin = it.src_off as usize + r * src_rb;
            dequant_row(it.src_ty, &map[begin..begin + src_rb], it.ne0, &mut row_buf);
            quantize_pq2_0_row(&row_buf, out);
        }
        Ok(())
    })?;

    let bytes_out: u64 = items
        .iter()
        .map(|it| tensor_nbytes_for(it.ty, it.dims.iter().product()))
        .sum();
    Ok((converted, shared.len(), bytes_in, bytes_out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_quantizer_roundtrips_within_scale() {
        let x: Vec<f32> = (0..256)
            .map(|i| ((i as f32) / 256.0) * 2.0 - 1.0 + (i % 7) as f32 * 0.01)
            .collect();
        let mut packed = Vec::new();
        quantize_pq2_0_row(&x, &mut packed);
        assert_eq!(packed.len(), 2 * 34);
        let back = kernels::decode_pq2_0_row(&packed, 256);
        // ternary with the block absmax as scale: error is bounded by 0.5*scale
        for blk in 0..2 {
            let absmax = x[blk * 128..(blk + 1) * 128]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            for j in 0..128 {
                let e = (back[blk * 128 + j] - x[blk * 128 + j]).abs();
                assert!(e <= 0.5 * absmax + 1e-5, "blk {blk} j {j}: err {e}");
            }
        }
    }

    #[test]
    fn bf16_decode_roundtrip() {
        // 1.0f32 in bf16 is 0x3F80
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        // 2.0f32 = 0x4000
        assert_eq!(bf16_to_f32(0x4000), 2.0);
    }

    #[test]
    fn q4_1_block_decodes() {
        // one block: d=1.0, m=0.0, nibbles 0..15 low/high
        let mut raw = Vec::new();
        raw.extend_from_slice(&crate::gguf::f32_to_half(1.0).to_le_bytes());
        raw.extend_from_slice(&crate::gguf::f32_to_half(0.0).to_le_bytes());
        for b in 0..16u8 {
            raw.push(b | (b << 4));
        }
        let mut out = vec![0.0f32; 32];
        dequant_row(TYPE_Q4_1, &raw, 32, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], j as f32, "low element {j}");
            assert_eq!(out[16 + j], j as f32, "high element {j}");
        }
    }

    #[test]
    fn row_bytes_match_block_rules() {
        assert_eq!(row_bytes(TYPE_F32, 100), 400);
        assert_eq!(row_bytes(TYPE_BF16, 100), 200);
        assert_eq!(row_bytes(TYPE_Q4_1, 64), 2 * Q4_1_BLOCK);
        assert_eq!(row_bytes(TYPE_PQ2_0, 128), 34);
    }
}
