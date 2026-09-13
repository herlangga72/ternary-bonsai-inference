//! qwen35 forward-pass pieces (M6).
//!
//! M6-3: full-attention layers (il % 4 == 3). Given the pre-normed input of a
//! single token, compute the gated multi-head attention of llama.cpp's
//! `build_layer_attn`:
//!
//!   qfull = wq @ x                    # rows: per head [q(256) | gate(256)]
//!   q     = rms_norm_rows(q, q_norm)  # per q head, head-major
//!   gate  = second half of each head  # head-major, 6144
//!   k     = rms_norm_rows(wk @ x, k_norm); rope each kv head
//!   v     = wv @ x                    # no norm, no rope
//!   attn  = softmax(q·k^T / sqrt(256)) @ v     # GQA: kv = q / 6 (grouped)
//!   out   = wo @ (attn ⊙ sigmoid(gate))
//!
//! The KV cache is f32, one contiguous `[kv_head][head_dim]` block per token
//! per full-attention layer. Only every `interval`-th layer participates.

#![allow(dead_code)]

use crate::gdn::{gdn_step, zero_state};
use crate::kernels;
use crate::rope::rope_imrope;
use crate::weights::{Qwen35, Weights};

/// Per-layer KV caches for the full-attention layers.
///
/// `k[layer]` and `v[layer]` grow by `n_kv_elems` (n_head_kv * head_dim) floats
/// per appended token; token `t`'s block sits at `[t * n_kv_elems ..)`, laid
/// out as `[kv_head 0 head vector][kv_head 1 head vector]...`.
pub struct AttnCache {
    pub n_kv_elems: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    /// Some when the K cache is rotation-quantized (`BONSAI_KV=planarN[k]`).
    pub pq: Option<crate::kvquant::PlanarQuant>,
    /// Some when the V cache is rotation-quantized (symmetric mode).
    pub pqv: Option<crate::kvquant::PlanarQuant>,
    /// Packed rotated K indices: `n_pos * n_kv` rows of `pq.packed_len()` bytes.
    pub kq: Vec<Vec<u8>>,
    /// K norms: `n_pos * n_kv` floats per layer.
    pub kn: Vec<Vec<f32>>,
    /// Packed rotated V indices and norms (symmetric mode).
    pub vq: Vec<Vec<u8>>,
    pub vn: Vec<Vec<f32>>,
    /// f16 storage (`BONSAI_KV=f16`), same layout as `k`/`v` but 16-bit.
    pub f16_kv: bool,
    pub k16: Vec<Vec<u16>>,
    pub v16: Vec<Vec<u16>>,
}

impl AttnCache {
    pub fn new(cfg: &Qwen35) -> AttnCache {
        let n_kv_elems = cfg.n_head_kv * cfg.n_embd_head;
        let mode = crate::kvquant::mode_from_env();
        let (pq, pqv) = crate::kvquant::quantizers_for(mode, cfg.n_embd_head);
        let f16_kv = mode == crate::kvquant::KvMode::F16;
        AttnCache {
            n_kv_elems,
            n_kv: cfg.n_head_kv,
            hd: cfg.n_embd_head,
            k: vec![Vec::new(); cfg.n_layer],
            v: vec![Vec::new(); cfg.n_layer],
            pq,
            pqv,
            kq: vec![Vec::new(); cfg.n_layer],
            kn: vec![Vec::new(); cfg.n_layer],
            vq: vec![Vec::new(); cfg.n_layer],
            vn: vec![Vec::new(); cfg.n_layer],
            f16_kv,
            k16: vec![Vec::new(); cfg.n_layer],
            v16: vec![Vec::new(); cfg.n_layer],
        }
    }

    /// Number of cached tokens for one full-attention layer.
    pub fn n_pos(&self, il: usize) -> usize {
        if self.pq.is_some() {
            self.kn[il].len() / self.n_kv
        } else if self.f16_kv {
            self.k16[il].len() / self.n_kv_elems
        } else {
            self.k[il].len() / self.n_kv_elems
        }
    }

    /// Append the current token's k/v rows (each n_kv_elems floats, kv-head
    /// major) for layer `il`.
    pub fn append(&mut self, il: usize, krow: &[f32], vrow: &[f32]) {
        debug_assert_eq!(krow.len(), self.n_kv_elems);
        debug_assert_eq!(vrow.len(), self.n_kv_elems);
        self.k[il].extend_from_slice(krow);
        self.v[il].extend_from_slice(vrow);
    }

    /// Quantized append: store each kv head's k rotated + Lloyd-Max packed,
    /// with the separate norm. V still goes through `append_v`.
    pub fn append_k_quantized(&mut self, il: usize, krow: &[f32]) {
        let (n_kv, hd) = (self.n_kv, self.hd);
        debug_assert_eq!(krow.len(), n_kv * hd);
        let pq = self.pq.as_ref().expect("quantized append without quantizer");
        let plen = pq.packed_len();
        let mut packed = vec![0u8; plen];
        for kv in 0..n_kv {
            let mut norm = 0.0f32;
            pq.quantize(&krow[kv * hd..(kv + 1) * hd], &mut packed, &mut norm);
            self.kq[il].extend_from_slice(&packed);
            self.kn[il].push(norm);
        }
    }

    /// Append a v row to the (still f32) V cache.
    pub fn append_v(&mut self, il: usize, vrow: &[f32]) {
        debug_assert_eq!(vrow.len(), self.n_kv_elems);
        self.v[il].extend_from_slice(vrow);
    }

    /// f16 append for K and V (`BONSAI_KV=f16`).
    pub fn append_f16(&mut self, il: usize, krow: &[f32], vrow: &[f32]) {
        debug_assert_eq!(krow.len(), self.n_kv_elems);
        debug_assert_eq!(vrow.len(), self.n_kv_elems);
        self.k16[il].extend(krow.iter().map(|&v| crate::gguf::f32_to_half(v)));
        self.v16[il].extend(vrow.iter().map(|&v| crate::gguf::f32_to_half(v)));
    }

    /// Quantized V append (symmetric mode).
    pub fn append_v_quantized(&mut self, il: usize, vrow: &[f32]) {
        let (n_kv, hd) = (self.n_kv, self.hd);
        let pqv = self.pqv.as_ref().expect("quantized V append without quantizer");
        let plen = pqv.packed_len();
        let mut packed = vec![0u8; plen];
        for kv in 0..n_kv {
            let mut norm = 0.0f32;
            pqv.quantize(&vrow[kv * hd..(kv + 1) * hd], &mut packed, &mut norm);
            self.vq[il].extend_from_slice(&packed);
            self.vn[il].push(norm);
        }
    }
}

/// Scratch buffers reused across attention calls (avoids per-head allocs).
pub struct AttnScratch {
    pub qfull: Vec<f32>,
    pub q: Vec<f32>,
    pub gate: Vec<f32>,
    pub krow: Vec<f32>,
    pub vrow: Vec<f32>,
    pub scores: Vec<f32>,
    pub probs: Vec<f32>,
    pub out: Vec<f32>,
    pub head_out: Vec<f32>,
    /// Rotated query scratch (quantized-K path), sized to an even pair count.
    pub qrot: Vec<f32>,
    /// V dequant scratch (quantized-V path).
    pub vrot: Vec<f32>,
    pub vidx: Vec<u8>,
}

impl AttnScratch {
    pub fn new(cfg: &Qwen35) -> AttnScratch {
        let hd = cfg.n_embd_head;
        let nh = cfg.n_head;
        AttnScratch {
            qfull: vec![0.0; 2 * nh * hd],
            q: vec![0.0; nh * hd],
            gate: vec![0.0; nh * hd],
            krow: vec![0.0; cfg.n_head_kv * hd],
            vrow: vec![0.0; cfg.n_head_kv * hd],
            scores: Vec::new(),
            probs: Vec::new(),
            out: vec![0.0; nh * hd],
            head_out: vec![0.0; hd.div_ceil(2) * 2],
            qrot: vec![0.0; hd.div_ceil(2) * 2],
            vrot: vec![0.0; hd.div_ceil(2) * 2],
            vidx: vec![0u8; hd.div_ceil(2) * 2],
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot of f32 query with an f16 cache row, converting on the fly.
fn dot_f16(a: &[f32], b: &[u16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * crate::gguf::half_to_f32(*y))
        .sum()
}

/// One full-attention layer for a single token.
///
/// `x` is the already `attn_norm`-ed token embedding (length n_embd), `pos` is
/// the absolute token position (its KV block index equals `pos`).
///
/// Returns the layer output before the attention residual is added.
pub fn full_attention_layer(
    w: &mut Weights,
    cfg: &Qwen35,
    il: usize,
    x: &[f32],
    pos: usize,
    cache: &mut AttnCache,
    s: &mut AttnScratch,
) -> Result<Vec<f32>, String> {
    debug_assert!(cfg.is_full_attention(il), "layer {il} is not full attention");
    let n_embd = cfg.n_embd;
    let hd = cfg.n_embd_head;
    let nh = cfg.n_head;
    let n_kv = cfg.n_head_kv;
    let n_rep = nh / n_kv; // grouped GQA: kv head = q head / 6
    let eps = cfg.eps;

    let name = |suffix: &str| cfg.blk_name(il, suffix);

    // ---- projections -------------------------------------------------------
    let qfull = w.matvec(&name("attn_q.weight"), x)?;
    let kraw = w.matvec(&name("attn_k.weight"), x)?;
    let vraw = w.matvec(&name("attn_v.weight"), x)?;
    if qfull.len() != 2 * nh * hd || kraw.len() != n_kv * hd || vraw.len() != n_kv * hd {
        return Err(format!(
            "attention_layer({il}): unexpected projection rows ({} / {} / {})",
            qfull.len(),
            kraw.len(),
            vraw.len()
        ));
    }
    debug_assert_eq!(s.qfull.len(), 2 * nh * hd);
    s.qfull.copy_from_slice(&qfull);
    s.krow.copy_from_slice(&kraw);
    s.vrow.copy_from_slice(&vraw);

    // ---- split [q | gate] per head, then per-head RMSNorm + IMROPE ---------
    // wq rows are interleaved per head: [q(hd) | gate(hd)] head-major
    for h in 0..nh {
        let base = h * 2 * hd;
        s.q[h * hd..(h + 1) * hd].copy_from_slice(&s.qfull[base..base + hd]);
        s.gate[h * hd..(h + 1) * hd].copy_from_slice(&s.qfull[base + hd..base + 2 * hd]);
    }
    let q_norm_w = w.vec_f32(&name("attn_q_norm.weight"))?;
    let mut q = kernels::rms_norm_rows(&s.q, &q_norm_w, hd, eps)?;
    let k_norm_w = w.vec_f32(&name("attn_k_norm.weight"))?;
    let mut k = kernels::rms_norm_rows(&s.krow, &k_norm_w, hd, eps)?;

    for h in 0..nh {
        rope_imrope(
            &mut q[h * hd..(h + 1) * hd],
            pos as i64,
            pos as i64,
            pos as i64,
            pos as i64,
            cfg.n_rot,
            cfg.sections,
            cfg.freq_base,
        );
    }
    for kv in 0..n_kv {
        rope_imrope(
            &mut k[kv * hd..(kv + 1) * hd],
            pos as i64,
            pos as i64,
            pos as i64,
            pos as i64,
            cfg.n_rot,
            cfg.sections,
            cfg.freq_base,
        );
    }

    // ---- store in KV cache (token index == pos) ----------------------------
    if cache.pq.is_some() {
        cache.append_k_quantized(il, &k);
        if cache.pqv.is_some() {
            cache.append_v_quantized(il, &s.vrow);
        } else {
            cache.append_v(il, &s.vrow);
        }
    } else if cache.f16_kv {
        cache.append_f16(il, &k, &s.vrow);
    } else {
        cache.append(il, &k, &s.vrow);
    }
    debug_assert_eq!(cache.n_pos(il), pos + 1);

    // ---- GQA attention ------------------------------------------------------
    let scale = 1.0 / (hd as f32).sqrt();
    let n_pos = pos + 1;
    s.scores.resize(n_pos, 0.0);
    s.probs.resize(n_pos, 0.0);
    s.out.fill(0.0);
    let k_cache = &cache.k[il];
    let v_cache = &cache.v[il];
    let n_kv_elems = cache.n_kv_elems;

    for hq in 0..nh {
        let qh = &q[hq * hd..(hq + 1) * hd];
        let kv = hq / n_rep; // grouped GQA, matches ggml mul_mat broadcast
        let kv_off = kv * hd;
        if let Some(pq) = cache.pq.as_ref() {
            // K is cached rotated; score = ||k_j|| * (R q) . centroids
            let plen = pq.packed_len();
            pq.rotate(qh, &mut s.qrot);
            let kq = &cache.kq[il];
            let kn = &cache.kn[il];
            for j in 0..n_pos {
                // n_kv packed head-rows per token
                let r0 = (j * cache.n_kv + kv) * plen;
                let row = &kq[r0..r0 + plen];
                let norm = kn[j * cache.n_kv + kv];
                s.scores[j] = pq.dot_rotated_scratch(&s.qrot, row, norm, &mut s.vidx) * scale;
            }
        } else if cache.f16_kv {
            let k16 = &cache.k16[il];
            for j in 0..n_pos {
                let kj = &k16[j * n_kv_elems + kv_off..j * n_kv_elems + kv_off + hd];
                s.scores[j] = dot_f16(qh, kj) * scale;
            }
        } else {
            for j in 0..n_pos {
                let kj = &k_cache[j * n_kv_elems + kv_off..j * n_kv_elems + kv_off + hd];
                s.scores[j] = dot(qh, kj) * scale;
            }
        }
        let probs = kernels::softmax_rows(&s.scores, n_pos)?;
        s.probs.copy_from_slice(&probs);

        s.head_out.fill(0.0);
        if let Some(pqv) = cache.pqv.as_ref() {
            // V is cached rotated; accumulate the weighted sum in rotated
            // space (folding each position's norm) and inverse-rotate once.
            let plen = pqv.packed_len();
            let vq = &cache.vq[il];
            let vn = &cache.vn[il];
            for j in 0..n_pos {
                let pj = s.probs[j];
                if pj == 0.0 {
                    continue;
                }
                let r0 = (j * cache.n_kv + kv) * plen;
                let norm = vn[j * cache.n_kv + kv];
                pqv.unpack_centroids_into(&vq[r0..r0 + plen], &mut s.vidx, &mut s.vrot);
                let w = pj * norm;
                for d in 0..hd {
                    s.head_out[d] += w * s.vrot[d];
                }
            }
            pqv.rotate_inv_inplace(&mut s.head_out[..pqv.hd_padded]);
        } else if cache.f16_kv {
            let v16 = &cache.v16[il];
            for j in 0..n_pos {
                let pj = s.probs[j];
                if pj == 0.0 {
                    continue;
                }
                let vj = &v16[j * n_kv_elems + kv_off..j * n_kv_elems + kv_off + hd];
                for d in 0..hd {
                    s.head_out[d] += pj * crate::gguf::half_to_f32(vj[d]);
                }
            }
        } else {
            for j in 0..n_pos {
                let pj = s.probs[j];
                if pj == 0.0 {
                    continue;
                }
                let vj = &v_cache[j * n_kv_elems + kv_off..j * n_kv_elems + kv_off + hd];
                for d in 0..hd {
                    s.head_out[d] += pj * vj[d];
                }
            }
        }
        s.out[hq * hd..(hq + 1) * hd].copy_from_slice(&s.head_out[..hd]);
    }

    // ---- gate + output projection ------------------------------------------
    // out = wo @ (attn ⊙ sigmoid(gate))
    kernels::map_inplace(&mut s.gate, kernels::sigmoid);
    for i in 0..s.out.len() {
        s.out[i] *= s.gate[i];
    }
    let wo = w.tensor(&name("attn_output.weight"))?.clone();
    let mut y = vec![0.0f32; n_embd];
    w.matvec_into(&wo, 0, n_embd, &s.out, &mut y)?;
    Ok(y)
}

// ---------------------------------------------------------------------------
// M6-4: recurrent (gated delta net / SSM) layers
// ---------------------------------------------------------------------------

/// Per-sequence recurrent caches for the SSM branch.
///
/// `conv[il]` holds the last (conv_kernel - 1) pre-conv inputs per channel,
/// oldest first, channel-major (3 * conv_channels floats per layer). `state[il]`
/// is the gated-delta-net state, H_v transposed S x S matrices (48*128*128).
pub struct SsmCache {
    pub conv: Vec<Vec<f32>>,
    pub state: Vec<Vec<f32>>,
}

impl SsmCache {
    pub fn new(cfg: &Qwen35) -> SsmCache {
        let n_prev = cfg.ssm_conv_kernel - 1;
        let ch = cfg.ssm_conv_channels();
        let mut conv = vec![Vec::new(); cfg.n_layer];
        let mut state = vec![Vec::new(); cfg.n_layer];
        for il in 0..cfg.n_layer {
            if cfg.is_recurrent(il) {
                conv[il] = vec![0.0; n_prev * ch];
                state[il] = zero_state();
            }
        }
        SsmCache { conv, state }
    }
}

/// Scratch buffers for one recurrent layer step.
pub struct SsmScratch {
    pub qkv: Vec<f32>,
    pub z: Vec<f32>,
    pub beta: Vec<f32>,
    pub alpha: Vec<f32>,
    pub gate: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub window: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,
    pub normed: Vec<f32>,
}

impl SsmScratch {
    pub fn new(cfg: &Qwen35) -> SsmScratch {
        let ch = cfg.ssm_conv_channels();
        let dk = cfg.ssm_key_dim(); // q and k rows (2 * group_count * state)
        let di = cfg.ssm_inner;
        SsmScratch {
            qkv: vec![0.0; ch],
            z: vec![0.0; di],
            beta: vec![0.0; cfg.ssm_dt_rank],
            alpha: vec![0.0; cfg.ssm_dt_rank],
            gate: vec![0.0; cfg.ssm_dt_rank],
            conv_out: vec![0.0; ch],
            window: vec![0.0; cfg.ssm_conv_kernel],
            q: vec![0.0; dk],
            k: vec![0.0; dk],
            v: vec![0.0; di],
            attn: vec![0.0; di],
            normed: vec![0.0; di],
        }
    }
}

/// One recurrent layer for a single token.
///
/// `x` is the already `attn_norm`-ed token embedding (length n_embd).
/// Returns the layer output before the attention residual is added.
pub fn recurrent_layer(
    w: &mut Weights,
    cfg: &Qwen35,
    il: usize,
    x: &[f32],
    cache: &mut SsmCache,
    s: &mut SsmScratch,
) -> Result<Vec<f32>, String> {
    debug_assert!(cfg.is_recurrent(il), "layer {il} is not recurrent");
    let n_embd = cfg.n_embd;
    let ch = cfg.ssm_conv_channels();
    let dk = cfg.ssm_key_dim();
    let di = cfg.ssm_inner;
    let n_v = cfg.ssm_dt_rank; // 48 v-heads
    let sdim = cfg.ssm_state; // 128
    let n_prev = cfg.ssm_conv_kernel - 1; // 3
    let eps = cfg.eps;
    let name = |suffix: &str| cfg.blk_name(il, suffix);

    // ---- projections -------------------------------------------------------
    let qkv = w.matvec(&name("attn_qkv.weight"), x)?;
    let zraw = w.matvec(&name("attn_gate.weight"), x)?;
    let beta_raw = w.matvec(&name("ssm_beta.weight"), x)?;
    let alpha_raw = w.matvec(&name("ssm_alpha.weight"), x)?;
    if qkv.len() != ch || zraw.len() != di || beta_raw.len() != n_v || alpha_raw.len() != n_v {
        return Err(format!(
            "recurrent_layer({il}): unexpected projection rows ({} / {} / {} / {})",
            qkv.len(),
            zraw.len(),
            beta_raw.len(),
            alpha_raw.len()
        ));
    }
    s.qkv.copy_from_slice(&qkv);
    s.z.copy_from_slice(&zraw);

    // beta = sigmoid(ssm_beta @ x); alpha = softplus(ssm_alpha @ x + dt.bias)
    let dt_bias = w.vec_f32(&name("ssm_dt.bias"))?;
    for h in 0..n_v {
        s.beta[h] = kernels::sigmoid(beta_raw[h]);
        s.alpha[h] = kernels::softplus(alpha_raw[h] + dt_bias[h]);
    }
    // gate = alpha * ssm_a  (ssm_a < 0: negative log-decay per v-head)
    let ssm_a = w.vec_f32(&name("ssm_a"))?;
    for h in 0..n_v {
        s.gate[h] = s.alpha[h] * ssm_a[h];
    }

    // ---- causal conv1d over the qkv channels with cached state --------------
    // CPU ssm_conv semantics: out[c] = sum_j w[j][c] * win[j], win = [3 prev
    // inputs oldest..newest, current input]. New state = last 3 of win.
    let conv_w = w.vec_f32(&name("ssm_conv1d.weight"))?; // kernel x channels
    let conv_state = &cache.conv[il];
    for c in 0..ch {
        let base = c * n_prev;
        for j in 0..n_prev {
            s.window[j] = conv_state[base + j];
        }
        s.window[n_prev] = s.qkv[c];
        let mut acc = 0.0f32;
        for j in 0..cfg.ssm_conv_kernel {
            acc += conv_w[c * cfg.ssm_conv_kernel + j] * s.window[j];
        }
        s.conv_out[c] = kernels::silu(acc);
    }
    // rotate cache: drop the oldest sample, append current per channel
    for c in 0..ch {
        let base = c * n_prev;
        for j in 0..n_prev - 1 {
            cache.conv[il][base + j] = cache.conv[il][base + j + 1];
        }
        cache.conv[il][base + n_prev - 1] = s.qkv[c];
    }

    // ---- split q / k / v from the convolved channels, l2-normalize q, k -----
    s.q.copy_from_slice(&s.conv_out[..dk]);
    s.k.copy_from_slice(&s.conv_out[dk..2 * dk]);
    s.v.copy_from_slice(&s.conv_out[2 * dk..]);
    let q = kernels::l2_norm_rows(&s.q, sdim, eps)?;
    let k = kernels::l2_norm_rows(&s.k, sdim, eps)?;

    // ---- fused gated-delta-net recurrence -----------------------------------
    gdn_step(&q, &k, &s.v, &s.gate, &s.beta, &mut cache.state[il], &mut s.attn);

    // ---- gated output norm: rms_norm(out) * silu(z), then ssm_out -----------
    let ssm_norm_w = w.vec_f32(&name("ssm_norm.weight"))?;
    let normed = kernels::rms_norm_rows(&s.attn, &ssm_norm_w, sdim, eps)?;
    s.normed.copy_from_slice(&normed);
    for i in 0..di {
        s.normed[i] *= kernels::silu(s.z[i]);
    }
    let ssm_out = w.tensor(&name("ssm_out.weight"))?.clone();
    let mut y = vec![0.0f32; n_embd];
    w.matvec_into(&ssm_out, 0, n_embd, &s.normed, &mut y)?;
    let _ = n_v;
    Ok(y)
}

// ---------------------------------------------------------------------------
// M6-5: FFN, full layer loop and LM head
// ---------------------------------------------------------------------------

/// Dense FFN (LLM_FFN_PAR): down(silu(gate @ x) ⊙ (up @ x)).
pub fn ffn_layer(w: &mut Weights, cfg: &Qwen35, il: usize, x: &[f32]) -> Result<Vec<f32>, String> {
    let n_ff = cfg.n_ff;
    let name = |suffix: &str| cfg.blk_name(il, suffix);
    let gate = w.matvec(&name("ffn_gate.weight"), x)?;
    let up = w.matvec(&name("ffn_up.weight"), x)?;
    if gate.len() != n_ff || up.len() != n_ff {
        return Err(format!("ffn_layer({il}): unexpected rows ({} / {})", gate.len(), up.len()));
    }
    let mut h = vec![0.0f32; n_ff];
    for i in 0..n_ff {
        h[i] = kernels::silu(gate[i]) * up[i];
    }
    let down = w.tensor(&name("ffn_down.weight"))?.clone();
    let mut y = vec![0.0f32; cfg.n_embd];
    w.matvec_into(&down, 0, cfg.n_embd, &h, &mut y)?;
    Ok(y)
}

/// Full single-token decoder: embeddings -> 64 layers -> LM head logits.
pub struct Decoder {
    pub w: Weights,
    pub cfg: Qwen35,
    pub attn: AttnCache,
    pub attn_s: AttnScratch,
    pub ssm: SsmCache,
    pub ssm_s: SsmScratch,
}

impl Decoder {
    pub fn open(path: &str) -> Result<Decoder, String> {
        let w = Weights::open(path)?;
        Self::from_weights(w)
    }

    /// Like `open`, but routes every PQ2_0 matvec through a Vulkan device
    /// (weights uploaded to VRAM at load). Needs a working Vulkan GPU; the
    /// layer norms, attention, rope and gdn math still run on the CPU path.
    pub fn open_gpu(path: &str) -> Result<Decoder, String> {
        let mut w = Weights::open(path)?;
        w.enable_gpu()?;
        let dec = Self::from_weights(w)?;
        eprintln!("[gpu] decode accelerator active");
        Ok(dec)
    }

    fn from_weights(w: Weights) -> Result<Decoder, String> {
        let cfg = w.config().clone();
        let attn = AttnCache::new(&cfg);
        let attn_s = AttnScratch::new(&cfg);
        let ssm = SsmCache::new(&cfg);
        let ssm_s = SsmScratch::new(&cfg);
        Ok(Decoder {
            w,
            cfg,
            attn,
            attn_s,
            ssm,
            ssm_s,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.w
            .tensor("output.weight")
            .map(|t| {
                if t.dims.len() > 1 {
                    t.dims[1] as usize
                } else {
                    0
                }
            })
            .unwrap_or(0)
    }

    /// Drop all cached state so the decoder can start a fresh sequence (the
    /// server reuses one decoder across requests).
    pub fn reset(&mut self) {
        let n_layer = self.cfg.n_layer;
        for il in 0..n_layer {
            self.attn.k[il].clear();
            self.attn.v[il].clear();
            self.attn.kq[il].clear();
            self.attn.kn[il].clear();
            self.attn.vq[il].clear();
            self.attn.vn[il].clear();
            self.attn.k16[il].clear();
            self.attn.v16[il].clear();
            if self.cfg.is_recurrent(il) {
                self.ssm.conv[il].fill(0.0);
                self.ssm.state[il] = crate::gdn::zero_state();
            }
        }
    }

    /// Forward one token at absolute position `pos`, updating the recurrent,
    /// conv and attention caches (causal, single stream). Returns the
    /// output-normalized hidden vector (length n_embd).
    pub fn forward_hidden(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, String> {
        self.forward_hidden_taps(token, pos, &[], &mut [])
    }

    /// Forward one token at position `pos`, and additionally capture the layer
    /// *input* (the residual stream entering the layer, before its attention
    /// norm) for each layer index in `want`. `taps[i]` receives layer
    /// `want[i]`'s input. This is the "layer input" the dspark encoder consumes
    /// (reference: `llama_set_embeddings_layer_inp`).
    pub fn forward_hidden_taps(
        &mut self,
        token: u32,
        pos: usize,
        want: &[u32],
        taps: &mut [Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let Self {
            w,
            cfg,
            attn,
            attn_s,
            ssm,
            ssm_s,
        } = self;
        let eps = cfg.eps;
        let n_embd = cfg.n_embd;
        if taps.len() != want.len() {
            return Err("forward_hidden_taps: taps length != want length".into());
        }

        let mut cur = w.row_f32("token_embd.weight", token as u64)?;
        if cur.len() != n_embd {
            return Err(format!("decode_token: embedding len {} != n_embd", cur.len()));
        }

        for il in 0..cfg.n_layer {
            if let Some(k) = want.iter().position(|&x| x as usize == il) {
                taps[k].clear();
                taps[k].extend_from_slice(&cur);
            }
            let attn_norm_w = w.vec_f32(&cfg.blk_name(il, "attn_norm.weight"))?;
            let x_norm = kernels::rms_norm(&cur, &attn_norm_w, eps);
            let attn_out = if cfg.is_recurrent(il) {
                recurrent_layer(w, cfg, il, &x_norm, ssm, ssm_s)?
            } else {
                full_attention_layer(w, cfg, il, &x_norm, pos, attn, attn_s)?
            };
            for i in 0..n_embd {
                cur[i] += attn_out[i];
            }

            let post_w = w.vec_f32(&cfg.blk_name(il, "post_attention_norm.weight"))?;
            let ffn_in = kernels::rms_norm(&cur, &post_w, eps);
            let ffn_out = ffn_layer(w, cfg, il, &ffn_in)?;
            for i in 0..n_embd {
                cur[i] += ffn_out[i];
            }
        }

        let out_norm_w = w.vec_f32("output_norm.weight")?;
        Ok(kernels::rms_norm(&cur, &out_norm_w, eps))
    }

    /// LM head: logits over the full vocabulary for a normalized hidden vector.
    pub fn head_logits(&mut self, h: &[f32]) -> Result<Vec<f32>, String> {
        let Self { w, .. } = self;
        let head = w.tensor("output.weight")?.clone();
        let n_vocab = crate::kernels::n_rows(&head) as usize;
        let mut logits = vec![0.0f32; n_vocab];
        w.matvec_into(&head, 0, n_vocab, h, &mut logits)?;
        Ok(logits)
    }

    /// Forward one token and return its logits over the full vocabulary.
    pub fn decode_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, String> {
        let h = self.forward_hidden(token, pos)?;
        self.head_logits(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_cfg() -> Qwen35 {
        Qwen35 {
            n_layer: 4,
            n_embd: 128,
            n_ff: 256,
            n_head: 8,
            n_head_kv: 2,
            n_embd_head: 16,
            n_ctx_train: 4096,
            eps: 1e-6,
            interval: 4,
            n_rot: 8,
            sections: [2, 1, 1, 0],
            freq_base: 1e4,
            ssm_conv_kernel: 4,
            ssm_state: 16,
            ssm_group_count: 2,
            ssm_dt_rank: 8,
            ssm_inner: 128,
        }
    }

    #[test]
    fn cache_grows_per_token_and_indexes() {
        let cfg = mini_cfg();
        let mut c = AttnCache::new(&cfg);
        assert_eq!(c.n_kv_elems, 32);
        assert_eq!(c.n_pos(3), 0);
        let krow: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let vrow: Vec<f32> = (0..32).map(|i| -i as f32).collect();
        c.append(3, &krow, &vrow);
        c.append(3, &krow, &vrow);
        assert_eq!(c.n_pos(3), 2);
        // contiguous per-token blocks, kv-head major
        assert_eq!(c.k[3][0], 0.0);
        assert_eq!(c.k[3][31], 31.0);
        assert_eq!(c.k[3][32], 0.0); // token 1 starts here
        assert_eq!(c.v[3][32 + 16], -16.0); // kv head 1 of token 1
        assert_eq!(c.n_pos(0), 0);
    }
}
