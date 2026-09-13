//! gdev: full device decode mirror (single in-order command buffer per token).
//!
//! Every PQ2_0 matvec runs as the v3 two-pass kernel; every small op (rms,
//! row norms, l2, conv1d+silu, gdn, rope, attention, elementwise gates,
//! residual adds) runs as a recorded compute dispatch. All dispatches for one
//! token are recorded into one command buffer and submitted once (keeps the
//! iGPU clock boosted). Output: output-norm hidden (5120); LM head/sampling
//! stay on the CPU.
//!
//! Recording helpers are free functions that take `gpu: &mut vk::Gpu` plus
//! buffer args, so `GDev` methods can split field borrows (buffers are all
//! used immutably by the GPU; only `gpu` needs &mut).

#![allow(dead_code)]

use crate::gguf::{self, GGUF};
use crate::vk::{self, DevBuf};
use crate::weights::Qwen35;
use std::collections::HashMap;

const N_EMBD: usize = 5120;
const N_FF: usize = 17408;
const N_HEAD: usize = 24;
const N_KV: usize = 4;
const HEAD_D: usize = 256;
const N_ROT: usize = 64;
const FREQ_BASE: f32 = 1e7;
const SECTIONS: [u32; 4] = [11, 11, 10, 0];
const GDN_DK: usize = 2048;
const GDN_DI: usize = 6144;
const GDN_HV: usize = 48;
const GDN_CH: usize = 2 * GDN_DK + GDN_DI; // 10240
const STATE_SIZE: usize = 128;
const GDN_STATE_ELEMS: usize = GDN_HV * STATE_SIZE * STATE_SIZE;
const KV_STRIDE: usize = N_KV * HEAD_D;
const N_CTX_DEFAULT: usize = 2048;
const EPS: f32 = 1e-6;

fn n_ctx_env() -> usize {
    std::env::var("BONSAI_CTX")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        // Upper bound is the model's 262144-token training context; the KV
        // caches are allocated for the full value, so a too-large setting fails
        // at allocation rather than silently.
        .filter(|&n| n >= 16 && n <= 262_144)
        .unwrap_or(N_CTX_DEFAULT)
}

/// Default prefill window width (prompt columns processed per batched window)
/// when batching is enabled but `BONSAI_BATCH` does not name a width. Kept
/// modest so the per-window activation/partial buffers stay small; very long
/// prompts are then split into several windows (P5).
const DEFAULT_BATCH_WINDOW: usize = 64;

/// Resolve the batched-prefill configuration from `BONSAI_BATCH`, given whether
/// the compute device is a discrete (dedicated-VRAM) GPU.
///
/// Batching reads each weight block once across N prompt columns; it only pays
/// off where decode is weight-bandwidth bound, i.e. on a discrete high-bandwidth
/// GPU (e.g. the RX 7600). On a shared-bus APU prefill is small-op / launch
/// bound, so the batched path is measured no faster than the sequential token
/// loop (see `notes/prefill-plan.md` P6) and the default is the token loop.
///
/// * `"0"` or `"1"`  -> disabled (keep the sequential per-token loop);
/// * unset or `"auto"` -> enabled with [`DEFAULT_BATCH_WINDOW`] only on a
///   discrete GPU, else disabled;
/// * integer `>= 2`    -> enabled with that window width (explicit override);
/// * anything else     -> disabled (fall back safely).
///
/// Returns `(enabled, window_width)`.
pub fn batch_cfg(discrete: bool) -> (bool, usize) {
    match std::env::var("BONSAI_BATCH").ok().map(|v| v.trim().to_string()) {
        Some(v) if v == "0" || v == "1" => (false, 0),
        Some(v) if v == "auto" => (discrete, DEFAULT_BATCH_WINDOW),
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n >= 2 => (true, n),
            _ => (false, 0), // junk or < 2 -> fall back safely
        },
        None => (discrete, DEFAULT_BATCH_WINDOW), // default follows the device
    }
}

/// Whether the batched-prefill path should be used for a prompt of `n` tokens
/// (batch on and the batch is usable: more than one position). `window` is the
/// resolved column width from [`batch_cfg`].
pub fn prefill_batch_usable(n: usize, enabled: bool, _window: usize) -> bool {
    enabled && n >= 2
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn pc_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ---- recording helpers -----------------------------------------------------

fn rec_rms(
    gpu: &mut vk::Gpu,
    src: &DevBuf,
    w: &DevBuf,
    dst: &DevBuf,
    n: usize,
) -> Result<(), String> {
    gpu.ensure_kernel("rms_norm", vk::RMS_NORM_SPV)?;
    gpu.rec_dispatch("rms_norm", src, w, dst, pc_u32(&[n as u32]), 1)
}

fn rec_norm_rows(
    gpu: &mut vk::Gpu,
    src: &DevBuf,
    w: &DevBuf,
    dst: &DevBuf,
    total: usize,
    rowlen: usize,
    mode: u32,
) -> Result<(), String> {
    gpu.ensure_kernel("norm_rows", vk::NORM_ROWS_SPV)?;
    gpu.rec_dispatch(
        "norm_rows",
        src,
        w,
        dst,
        pc_u32(&[total as u32, rowlen as u32, mode, EPS.to_bits()]),
        (total / rowlen) as u32,
    )
}

fn rec_l2(gpu: &mut vk::Gpu, buf: &DevBuf, base: usize, rows: usize) -> Result<(), String> {
    gpu.ensure_kernel("l2_inplace", vk::L2_INPLACE_SPV)?;
    gpu.rec_dispatch("l2_inplace", buf, buf, buf, pc_u32(&[rows as u32, base as u32]), rows as u32)
}

fn rec_mat(
    gpu: &mut vk::Gpu,
    partials: &DevBuf,
    w: &DevBuf,
    ne0: usize,
    wrows: usize,
    x: &DevBuf,
    y: &DevBuf,
    ybase: usize,
) -> Result<(), String> {
    gpu.rec_matvec(w, x, y, ybase as u32, partials, ne0 as u32, 0, wrows as u32)
}

fn rec_add(gpu: &mut vk::Gpu, cur: &DevBuf, branch: &DevBuf) -> Result<(), String> {
    gpu.ensure_kernel("add_residual", vk::ADD_RESIDUAL_SPV)?;
    gpu.rec_dispatch(
        "add_residual",
        cur,
        branch,
        cur,
        pc_u32(&[N_EMBD as u32]),
        N_EMBD.div_ceil(256) as u32,
    )
}

/// Record a device-to-device block copy (dst[dbase+i] = src[sbase+i]).
fn rec_bcopy(
    gpu: &mut vk::Gpu,
    src: &DevBuf,
    dst: &DevBuf,
    sbase: usize,
    dbase: usize,
    len: usize,
) -> Result<(), String> {
    gpu.ensure_kernel("tile_copy", vk::TILE_COPY_SPV)?;
    let pc = [len as u32, sbase as u32, dbase as u32];
    gpu.rec_dispatch(
        "tile_copy",
        src,
        src,
        dst,
        pc_u32(&pc),
        (len as u32).div_ceil(256).max(1),
    )
}

fn rec_elem(
    gpu: &mut vk::Gpu,
    a: &DevBuf,
    b: &DevBuf,
    out: &DevBuf,
    n: usize,
    mode: u32,
) -> Result<(), String> {
    gpu.ensure_kernel("elem", vk::ELEM_SPV)?;
    gpu.rec_dispatch("elem", a, b, out, pc_u32(&[n as u32, mode]), n.div_ceil(256) as u32)
}

// ---- device decode context --------------------------------------------------

struct TensorW {
    buf: DevBuf,
    ne0: usize,
    rows: usize,
}

pub struct GDev {
    pub gpu: vk::Gpu,
    pub cfg: Qwen35,
    pub n_ctx: usize,
    tens: HashMap<String, TensorW>,

    cur: DevBuf,
    xnorm: DevBuf,
    branch: DevBuf,
    hidden: DevBuf,
    partials: DevBuf,

    qkv: DevBuf,
    z: DevBuf,
    smalls: Vec<DevBuf>,
    blob: DevBuf,
    attn_out: DevBuf,
    normed: DevBuf,

    qfull: DevBuf,
    q: DevBuf,
    qn: DevBuf,
    gate: DevBuf,
    kraw: DevBuf,
    kn: DevBuf,
    vraw: DevBuf,
    scores: DevBuf,
    attn_h: DevBuf,

    fg: DevBuf,
    fu: DevBuf,
    h1: DevBuf,

    kcache: Vec<Option<DevBuf>>,
    vcache: Vec<Option<DevBuf>>,
    /// Quantized KV path (BONSAI_KV): params needed to pack/unpack and the
    /// packed caches. `None` on every field means the f32 path is in use.
    kv_bits: Option<u32>,
    kv_pwords: usize,
    kv_cent_off: usize,
    kv_v_quant: bool,
    kq: Vec<Option<DevBuf>>,
    vq: Vec<Option<DevBuf>>,
    /// f16 KV mode (`BONSAI_KV=f16`).
    kv_f16: bool,
    k16: Vec<Option<DevBuf>>,
    v16: Vec<Option<DevBuf>>,
    conv_cache: Vec<Option<DevBuf>>,
    state: Vec<Option<DevBuf>>,
}

fn alloc_bytes(gpu: &mut vk::Gpu, bytes: usize) -> Result<DevBuf, String> {
    // BONSAI_VRAM=1 forces device-local even on APUs (experiment); otherwise
    // device-local on discrete GPUs, host-visible RAM on APUs.
    let force_vram = std::env::var("BONSAI_VRAM").map(|v| v == "1").unwrap_or(false);
    if force_vram || gpu.discrete {
        gpu.create_dev_buffer(bytes, vk::storage_usage())
    } else {
        gpu.create_host_dev_buffer(bytes, vk::storage_usage())
    }
}

fn make(gpu: &mut vk::Gpu, n: usize) -> Result<DevBuf, String> {
    alloc_bytes(gpu, n.max(1) * 4)
}

impl GDev {
    pub fn open(model: &str) -> Result<GDev, String> {
        let mut gpu = vk::Gpu::open()?;
        let mut gg = GGUF::open(model).map_err(|e| format!("gguf: {e}"))?;
        let cfg = Qwen35::from_gguf(&gg)?;
        let n_ctx = n_ctx_env();
        if cfg.n_embd != N_EMBD
            || cfg.n_ff != N_FF
            || cfg.n_head != N_HEAD
            || cfg.n_head_kv != N_KV
            || cfg.n_embd_head != HEAD_D
            || cfg.ssm_dt_rank != GDN_HV
        {
            return Err("gdev: model hyperparameters do not match gdev constants".into());
        }

        let mut tens: HashMap<String, TensorW> = HashMap::new();
        let mut order = gg.tensors.clone();
        order.sort_by(|a, b| a.name.cmp(&b.name));
        for t in &order {
            if t.ty != gguf::TYPE_PQ2_0 && t.ty != gguf::TYPE_F32 && t.ty != gguf::TYPE_F16 {
                continue;
            }
            // LM head and embeddings stay on the CPU path (head + token row are
            // host-side); uploading them would cost ~0.7 GB and a 322 MB staging
            // buffer we do not need.
            if t.name == "output.weight" || t.name == "token_embd.weight" {
                continue;
            }
            let nbytes = gg.tensor_nbytes(t) as usize;
            let buf = alloc_bytes(&mut gpu, nbytes)?;
            let payload = gg.payload_slice(t)?;
            gpu.upload(&buf, payload)?;
            let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
            let rows = (t.n_elem() as usize / ne0.max(1)).max(1);
            tens.insert(t.name.clone(), TensorW { buf, ne0, rows });
        }

        let cur = make(&mut gpu, N_EMBD)?;
        let xnorm = make(&mut gpu, N_EMBD)?;
        let branch = make(&mut gpu, N_EMBD)?;
        let hidden = make(&mut gpu, N_EMBD)?;
        let partials = make(&mut gpu, N_FF * (N_EMBD / 128))?;
        let qkv = make(&mut gpu, GDN_CH)?;
        let z = make(&mut gpu, GDN_DI)?;
        let blob = make(&mut gpu, 2 * GDN_DK + GDN_DI + 2 * GDN_HV)?;
        let attn_out = make(&mut gpu, GDN_DI)?;
        let normed = make(&mut gpu, GDN_DI)?;
        let qfull = make(&mut gpu, 2 * N_HEAD * HEAD_D)?;
        let q = make(&mut gpu, N_HEAD * HEAD_D)?;
        let qn = make(&mut gpu, N_HEAD * HEAD_D)?;
        let gate = make(&mut gpu, N_HEAD * HEAD_D)?;
        let kraw = make(&mut gpu, N_KV * HEAD_D)?;
        let kn = make(&mut gpu, N_KV * HEAD_D)?;
        let vraw = make(&mut gpu, N_KV * HEAD_D)?;
        let scores = make(&mut gpu, N_HEAD * n_ctx)?;
        let attn_h = make(&mut gpu, N_HEAD * HEAD_D)?;
        let fg = make(&mut gpu, N_FF)?;
        let fu = make(&mut gpu, N_FF)?;
        let h1 = make(&mut gpu, N_FF)?;

        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        let mut conv_cache = Vec::new();
        let mut state = Vec::new();
        let kv_bytes = vec![0u8; n_ctx * KV_STRIDE * 4];
        let conv_bytes = vec![0u8; 3 * GDN_CH * 4];
        let state_bytes = vec![0u8; GDN_STATE_ELEMS * 4];

        // KV quantization (BONSAI_KV): pack K (and optionally V) instead of f32.
        let kv_mode = crate::kvquant::mode_from_env();
        let f16_mode = kv_mode == crate::kvquant::KvMode::F16;
        let (kq_mode, vq_mode) = crate::kvquant::quantizers_for(kv_mode, HEAD_D);
        let mut kq: Vec<Option<DevBuf>> = Vec::new();
        let mut vq: Vec<Option<DevBuf>> = Vec::new();
        let mut k16: Vec<Option<DevBuf>> = Vec::new();
        let mut v16: Vec<Option<DevBuf>> = Vec::new();
        let kv_bits;
        let kv_pwords;
        let kv_cent_off;
        let kv_v_quant;
        match kq_mode.as_ref() {
            Some(pqm) => {
                let bits = pqm.bits;
                let pbytes = pqm.packed_len();
                if pbytes % 4 != 0 {
                    return Err(format!(
                        "gdev: packed KV row of {pbytes} bytes is not word aligned"
                    ));
                }
                kv_bits = Some(bits);
                kv_pwords = pbytes / 4;
                kv_cent_off = HEAD_D.div_ceil(2) * 2;
                kv_v_quant = vq_mode.is_some();
                let mut data: Vec<f32> = Vec::with_capacity(kv_cent_off + 256);
                for (c, s) in crate::kvquant::givens_table(HEAD_D) {
                    data.push(*c);
                    data.push(*s);
                }
                data.extend_from_slice(crate::kvquant::codebook(HEAD_D, bits));
                gpu.upload_kv_params(&data)?;
                let n_fa = (0..cfg.n_layer)
                    .filter(|&l| cfg.is_full_attention(l))
                    .count();
                let f32_bytes = n_fa * 2 * n_ctx * KV_STRIDE * 4;
                let q_bytes = n_fa * n_ctx * N_KV * (kv_pwords + 1) * 4
                    * if kv_v_quant { 2 } else { 1 }
                    + if kv_v_quant {
                        0
                    } else {
                        n_fa * n_ctx * KV_STRIDE * 4
                    };
                eprintln!(
                    "[gdev] KV quantized: {} bits, K{}: ctx {n_ctx} -> {:.1} MB (f32 {:.1} MB, {:.1}x)",
                    bits,
                    if kv_v_quant { "+V" } else { " only" },
                    q_bytes as f64 / 1e6,
                    f32_bytes as f64 / 1e6,
                    f32_bytes as f64 / q_bytes as f64
                );
            }
            None => {
                kv_bits = None;
                kv_pwords = 0;
                kv_cent_off = 0;
                kv_v_quant = false;
            }
        }

        if f16_mode {
            let n_fa = (0..cfg.n_layer)
                .filter(|&l| cfg.is_full_attention(l))
                .count();
            let f32_bytes = n_fa * 2 * n_ctx * KV_STRIDE * 4;
            let f16_bytes = n_fa * 2 * n_ctx * N_KV * (HEAD_D / 2) * 4;
            eprintln!(
                "[gdev] KV f16: ctx {n_ctx} -> {:.1} MB (f32 {:.1} MB, {:.1}x)",
                f16_bytes as f64 / 1e6,
                f32_bytes as f64 / 1e6,
                f32_bytes as f64 / f16_bytes as f64
            );
        }

        for il in 0..cfg.n_layer {
            let fa = cfg.is_full_attention(il);
            if fa {
                if f16_mode {
                    let words = n_ctx * N_KV * (HEAD_D / 2);
                    k16.push(Some(make(&mut gpu, words)?));
                    v16.push(Some(make(&mut gpu, words)?));
                    kcache.push(None);
                    vcache.push(None);
                    kq.push(None);
                    vq.push(None);
                } else if kv_bits.is_some() {
                    let words = n_ctx * N_KV * (kv_pwords + 1);
                    kq.push(Some(make(&mut gpu, words)?));
                    if kv_v_quant {
                        vq.push(Some(make(&mut gpu, words)?));
                        vcache.push(None);
                    } else {
                        // K-only: V stays f32, so the existing attn_out applies.
                        vq.push(None);
                        let v = make(&mut gpu, n_ctx * KV_STRIDE)?;
                        gpu.upload(&v, &kv_bytes)?;
                        vcache.push(Some(v));
                    }
                    kcache.push(None);
                    k16.push(None);
                    v16.push(None);
                } else {
                    let k = make(&mut gpu, n_ctx * KV_STRIDE)?;
                    let v = make(&mut gpu, n_ctx * KV_STRIDE)?;
                    gpu.upload(&k, &kv_bytes)?;
                    gpu.upload(&v, &kv_bytes)?;
                    kcache.push(Some(k));
                    vcache.push(Some(v));
                    kq.push(None);
                    vq.push(None);
                    k16.push(None);
                    v16.push(None);
                }
            } else {
                kcache.push(None);
                vcache.push(None);
                kq.push(None);
                vq.push(None);
                k16.push(None);
                v16.push(None);
            }
            if fa {
                conv_cache.push(None);
                state.push(None);
            } else {
                let c = make(&mut gpu, 3 * GDN_CH)?;
                let s = make(&mut gpu, GDN_STATE_ELEMS)?;
                gpu.upload(&c, &conv_bytes)?;
                gpu.upload(&s, &state_bytes)?;
                conv_cache.push(Some(c));
                state.push(Some(s));
            }
        }

        // per-layer smalls: [beta(48) alpha(48) ssa(48) dt(48)] staged once at
        // load for recurrent layers (ssa/dt constant; beta/alpha are matvec
        // outputs written per token)
        let mut smalls = Vec::new();
        for il in 0..cfg.n_layer {
            let sm = make(&mut gpu, 4 * GDN_HV)?;
            gpu.upload(&sm, &vec![0u8; 4 * GDN_HV * 4])?;
            if cfg.is_recurrent(il) {
                let mut v = vec![0.0f32; 4 * GDN_HV];
                let key = format!("blk.{il}.ssm_a");
                if let Some(t) = gg.tensors.iter().find(|t| t.name == key).cloned() {
                    let x = gg.read_tensor(&t)?;
                    let n = x.len().min(GDN_HV);
                    v[2 * GDN_HV..2 * GDN_HV + n].copy_from_slice(&x[..n]);
                }
                let key = format!("blk.{il}.ssm_dt.bias");
                if let Some(t) = gg.tensors.iter().find(|t| t.name == key).cloned() {
                    let x = gg.read_tensor(&t)?;
                    let n = x.len().min(GDN_HV);
                    v[3 * GDN_HV..3 * GDN_HV + n].copy_from_slice(&x[..n]);
                }
                gpu.upload(&sm, f32_bytes(&v))?;
            }
            smalls.push(sm);
        }

        Ok(GDev {
            gpu,
            cfg,
            n_ctx,
            tens,
            cur,
            xnorm,
            branch,
            hidden,
            partials,
            qkv,
            z,
            smalls,
            blob,
            attn_out,
            normed,
            qfull,
            q,
            qn,
            gate,
            kraw,
            kn,
            vraw,
            scores,
            attn_h,
            fg,
            fu,
            h1,
            kcache,
            vcache,
            kv_bits,
            kv_pwords,
            kv_cent_off,
            kv_v_quant,
            kq,
            vq,
            kv_f16: f16_mode,
            k16,
            v16,
            conv_cache,
            state,
        })
    }

    fn tw(&self, name: &str) -> Result<(DevBuf, usize, usize), String> {
        let t = self
            .tens
            .get(name)
            .ok_or_else(|| format!("gdev: tensor {name} missing"))?;
        Ok((t.buf, t.ne0, t.rows))
    }

    /// True when the device path uses a non-f32 KV cache.
    pub fn kv_quantized(&self) -> bool {
        self.kv_bits.is_some() || self.kv_f16
    }

    fn rec_attn_layer(&mut self, il: usize, pos: usize) -> Result<(), String> {
        let cur = &self.cur;
        let xnorm = &self.xnorm;
        let branch = &self.branch;
        let partials = &self.partials;

        let (wn, _, _) = self.tw(&format!("blk.{il}.attn_norm.weight"))?;
        rec_rms(&mut self.gpu, cur, &wn, xnorm, N_EMBD)?;

        for (suf, dst) in [
            ("attn_q.weight", &self.qfull),
            ("attn_k.weight", &self.kraw),
            ("attn_v.weight", &self.vraw),
        ] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            rec_mat(&mut self.gpu, partials, &w, ne0, rows, xnorm, dst, 0)?;
        }
        self.gpu.ensure_kernel("split_qgate", vk::SPLIT_QGATE_SPV)?;
        self.gpu.rec_dispatch(
            "split_qgate",
            &self.qfull,
            &self.q,
            &self.gate,
            pc_u32(&[HEAD_D as u32, 0]),
            (N_HEAD * HEAD_D / 256) as u32,
        )?;
        let (wq, _, _) = self.tw(&format!("blk.{il}.attn_q_norm.weight"))?;
        let (wk, _, _) = self.tw(&format!("blk.{il}.attn_k_norm.weight"))?;
        rec_norm_rows(&mut self.gpu, &self.q, &wq, &self.qn, N_HEAD * HEAD_D, HEAD_D, 0)?;
        rec_norm_rows(&mut self.gpu, &self.kraw, &wk, &self.kn, N_KV * HEAD_D, HEAD_D, 0)?;
        self.gpu.ensure_kernel("rope_imrope", vk::ROPE_IMROPE_SPV)?;
        let pc_rope = [
            HEAD_D as u32,
            N_ROT as u32,
            pos as u32,
            FREQ_BASE.to_bits(),
            SECTIONS[0],
            SECTIONS[1],
            SECTIONS[2],
            SECTIONS[3],
        ];
        self.gpu.rec_dispatch("rope_imrope", &self.qn, &self.qn, &self.qn, pc_u32(&pc_rope), N_HEAD as u32)?;
        self.gpu.rec_dispatch("rope_imrope", &self.kn, &self.kn, &self.kn, pc_u32(&pc_rope), N_KV as u32)?;

        let n_pos = pos + 1;
        let scale_bits = (1.0 / (HEAD_D as f32).sqrt()).to_bits();
        let pc_att = [
            N_HEAD as u32,
            n_pos as u32,
            HEAD_D as u32,
            KV_STRIDE as u32,
            scale_bits,
            0,
        ];

        if self.kv_f16 {
            let row_words = (HEAD_D / 2) as u32;
            let pc_k = [
                pos as u32,
                HEAD_D as u32,
                N_KV as u32,
                row_words,
                0,
                0,
                0,
                0,
            ];
            self.gpu.ensure_kernel("kv_store_f16", vk::KV_STORE_F16_SPV)?;
            let kb = self.k16[il].as_ref().unwrap();
            self.gpu
                .rec_dispatch("kv_store_f16", &self.kn, kb, kb, pc_u32(&pc_k), N_KV as u32)?;
            let vb = self.v16[il].as_ref().unwrap();
            self.gpu
                .rec_dispatch("kv_store_f16", &self.vraw, vb, vb, pc_u32(&pc_k), N_KV as u32)?;

            let pc_a = [
                N_HEAD as u32,
                n_pos as u32,
                HEAD_D as u32,
                N_KV as u32,
                row_words,
                scale_bits,
                0,
                0,
            ];
            self.gpu
                .ensure_kernel("attn_scores_f16", vk::ATTN_SCORES_F16_SPV)?;
            self.gpu.rec_dispatch(
                "attn_scores_f16",
                &self.qn,
                kb,
                &self.scores,
                pc_u32(&pc_a),
                N_HEAD as u32,
            )?;
            self.gpu
                .ensure_kernel("softmax_inplace", vk::SOFTMAX_INPLACE_SPV)?;
            self.gpu.rec_dispatch(
                "softmax_inplace",
                &self.scores,
                &self.scores,
                &self.scores,
                pc_u32(&pc_att),
                N_HEAD as u32,
            )?;
            self.gpu
                .ensure_kernel("attn_out_f16", vk::ATTN_OUT_F16_SPV)?;
            let pc_o = [
                N_HEAD as u32,
                n_pos as u32,
                HEAD_D as u32,
                N_KV as u32,
                row_words,
                0,
                0,
                0,
            ];
            self.gpu.rec_dispatch(
                "attn_out_f16",
                &self.scores,
                vb,
                &self.attn_h,
                pc_u32(&pc_o),
                N_HEAD as u32,
            )?;
        } else if let Some(bits) = self.kv_bits {
            // Quantized path: pack K (and V) then attend against the packed cache.
            let pwords = self.kv_pwords as u32;
            let cent_off = self.kv_cent_off as u32;
            let pc_q = [
                pos as u32,
                HEAD_D as u32,
                N_KV as u32,
                bits,
                pwords,
                cent_off,
                0,
                0,
            ];
            self.gpu.ensure_kernel("kv_store_q", vk::KV_STORE_Q_SPV)?;
            let kqb = self.kq[il].as_ref().unwrap();
            self.gpu
                .rec_dispatch("kv_store_q", &self.kn, kqb, kqb, pc_u32(&pc_q), N_KV as u32)?;
            if let Some(vqb) = self.vq[il].as_ref() {
                self.gpu.rec_dispatch(
                    "kv_store_q",
                    &self.vraw,
                    vqb,
                    vqb,
                    pc_u32(&pc_q),
                    N_KV as u32,
                )?;
            } else {
                // K-only: V still lands in the f32 cache for the f32 attn_out.
                self.gpu.ensure_kernel("kv_store", vk::KV_STORE_SPV)?;
                let pc_kv = [pos as u32, KV_STRIDE as u32];
                self.gpu.rec_dispatch(
                    "kv_store",
                    &self.vraw,
                    self.vcache[il].as_ref().unwrap(),
                    self.vcache[il].as_ref().unwrap(),
                    pc_u32(&pc_kv),
                    KV_STRIDE.div_ceil(256) as u32,
                )?;
            }

            let pc_a = [
                N_HEAD as u32,
                n_pos as u32,
                HEAD_D as u32,
                N_KV as u32,
                bits,
                pwords,
                cent_off,
                scale_bits,
            ];
            self.gpu
                .ensure_kernel("attn_scores_q", vk::ATTN_SCORES_Q_SPV)?;
            self.gpu.rec_dispatch(
                "attn_scores_q",
                &self.qn,
                kqb,
                &self.scores,
                pc_u32(&pc_a),
                N_HEAD as u32,
            )?;
            self.gpu
                .ensure_kernel("softmax_inplace", vk::SOFTMAX_INPLACE_SPV)?;
            self.gpu.rec_dispatch(
                "softmax_inplace",
                &self.scores,
                &self.scores,
                &self.scores,
                pc_u32(&pc_att),
                N_HEAD as u32,
            )?;
            if let Some(vqb) = self.vq[il].as_ref() {
                self.gpu.ensure_kernel("attn_out_q", vk::ATTN_OUT_Q_SPV)?;
                let pc_o = [
                    N_HEAD as u32,
                    n_pos as u32,
                    HEAD_D as u32,
                    N_KV as u32,
                    bits,
                    pwords,
                    cent_off,
                    0,
                ];
                self.gpu.rec_dispatch(
                    "attn_out_q",
                    &self.scores,
                    vqb,
                    &self.attn_h,
                    pc_u32(&pc_o),
                    N_HEAD as u32,
                )?;
            } else {
                self.gpu.ensure_kernel("attn_out", vk::ATTN_OUT_SPV)?;
                self.gpu.rec_dispatch(
                    "attn_out",
                    &self.scores,
                    self.vcache[il].as_ref().unwrap(),
                    &self.attn_h,
                    pc_u32(&pc_att),
                    N_HEAD as u32,
                )?;
            }
        } else {
            self.gpu.ensure_kernel("kv_store", vk::KV_STORE_SPV)?;
            let pc_kv = [pos as u32, KV_STRIDE as u32];
            self.gpu.rec_dispatch(
                "kv_store",
                &self.kn,
                self.kcache[il].as_ref().unwrap(),
                self.kcache[il].as_ref().unwrap(),
                pc_u32(&pc_kv),
                KV_STRIDE.div_ceil(256) as u32,
            )?;
            self.gpu.rec_dispatch(
                "kv_store",
                &self.vraw,
                self.vcache[il].as_ref().unwrap(),
                self.vcache[il].as_ref().unwrap(),
                pc_u32(&pc_kv),
                KV_STRIDE.div_ceil(256) as u32,
            )?;

            self.gpu.ensure_kernel("attn_scores", vk::ATTN_SCORES_SPV)?;
            self.gpu.rec_dispatch(
                "attn_scores",
                &self.qn,
                self.kcache[il].as_ref().unwrap(),
                &self.scores,
                pc_u32(&pc_att),
                N_HEAD as u32,
            )?;
            self.gpu
                .ensure_kernel("softmax_inplace", vk::SOFTMAX_INPLACE_SPV)?;
            self.gpu.rec_dispatch(
                "softmax_inplace",
                &self.scores,
                &self.scores,
                &self.scores,
                pc_u32(&pc_att),
                N_HEAD as u32,
            )?;
            self.gpu.ensure_kernel("attn_out", vk::ATTN_OUT_SPV)?;
            self.gpu.rec_dispatch(
                "attn_out",
                &self.scores,
                self.vcache[il].as_ref().unwrap(),
                &self.attn_h,
                pc_u32(&pc_att),
                N_HEAD as u32,
            )?;
        }

        // out = wo @ (sigmoid(gate) . attn_h); reuse qfull as the gated buffer
        rec_elem(&mut self.gpu, &self.gate, &self.attn_h, &self.qfull, N_HEAD * HEAD_D, 4)?;
        let (wout, ne0, rows) = self.tw(&format!("blk.{il}.attn_output.weight"))?;
        rec_mat(&mut self.gpu, partials, &wout, ne0, rows, &self.qfull, branch, 0)?;
        rec_add(&mut self.gpu, cur, branch)
    }

    fn rec_gdn_layer(&mut self, il: usize) -> Result<(), String> {
        let cur = &self.cur;
        let xnorm = &self.xnorm;
        let branch = &self.branch;
        let partials = &self.partials;

        let (wn, _, _) = self.tw(&format!("blk.{il}.attn_norm.weight"))?;
        rec_rms(&mut self.gpu, cur, &wn, xnorm, N_EMBD)?;
        // gdn_prep reads smalls as [beta_raw(48), alpha_raw(48), ssa(48), dt(48)]
        for (suf, dst, ybase) in [
            ("attn_qkv.weight", &self.qkv, 0usize),
            ("attn_gate.weight", &self.z, 0),
            ("ssm_beta.weight", &self.smalls[il], 0),
            ("ssm_alpha.weight", &self.smalls[il], GDN_HV),
        ] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            rec_mat(&mut self.gpu, partials, &w, ne0, rows, xnorm, dst, ybase)?;
        }
        let (cw, _, _) = self.tw(&format!("blk.{il}.ssm_conv1d.weight"))?;
        self.gpu.ensure_kernel("conv1d_silu", vk::CONV1D_SILU_SPV)?;
        self.gpu.rec_dispatch(
            "conv1d_silu",
            &self.qkv,
            self.conv_cache[il].as_ref().unwrap(),
            &cw,
            pc_u32(&[GDN_CH as u32, 0]),
            GDN_CH.div_ceil(256) as u32,
        )?;
        rec_l2(&mut self.gpu, &self.qkv, 0, GDN_DK / STATE_SIZE)?;
        rec_l2(&mut self.gpu, &self.qkv, GDN_DK, GDN_DK / STATE_SIZE)?;
        self.gpu.ensure_kernel("gdn_prep", vk::GDN_PREP_SPV)?;
        self.gpu.rec_dispatch(
            "gdn_prep",
            &self.qkv,
            &self.smalls[il],
            &self.blob,
            pc_u32(&[0, 0]),
            GDN_CH.div_ceil(256) as u32,
        )?;
        self.gpu.ensure_kernel("gdn_step", vk::GDN_STEP_SPV)?;
        self.gpu.rec_dispatch("gdn_step", &self.blob, self.state[il].as_ref().unwrap(), &self.attn_out, &[], GDN_HV as u32)?;
        let (wsn, _, _) = self.tw(&format!("blk.{il}.ssm_norm.weight"))?;
        rec_norm_rows(&mut self.gpu, &self.attn_out, &wsn, &self.normed, GDN_DI, STATE_SIZE, 0)?;
        rec_elem(&mut self.gpu, &self.z, &self.normed, &self.h1, GDN_DI, 3)?;
        let (wso, ne0, rows) = self.tw(&format!("blk.{il}.ssm_out.weight"))?;
        rec_mat(&mut self.gpu, partials, &wso, ne0, rows, &self.h1, branch, 0)?;
        rec_add(&mut self.gpu, cur, branch)
    }

    fn rec_ffn(&mut self, il: usize) -> Result<(), String> {
        let cur = &self.cur;
        let xnorm = &self.xnorm;
        let branch = &self.branch;
        let partials = &self.partials;
        let (wp, _, _) = self.tw(&format!("blk.{il}.post_attention_norm.weight"))?;
        rec_rms(&mut self.gpu, cur, &wp, xnorm, N_EMBD)?;
        for (suf, dst) in [("ffn_gate.weight", &self.fg), ("ffn_up.weight", &self.fu)] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            rec_mat(&mut self.gpu, partials, &w, ne0, rows, xnorm, dst, 0)?;
        }
        rec_elem(&mut self.gpu, &self.fg, &self.fu, &self.h1, N_FF, 3)?;
        let (wd, ne0, rows) = self.tw(&format!("blk.{il}.ffn_down.weight"))?;
        rec_mat(&mut self.gpu, partials, &wd, ne0, rows, &self.h1, branch, 0)?;
        rec_add(&mut self.gpu, cur, branch)
    }

    /// Record + run one full token. `embed` is the token embedding (host).
    pub fn forward_token(&mut self, pos: usize, embed: &[f32]) -> Result<Vec<f32>, String> {
        debug_assert_eq!(embed.len(), N_EMBD);
        if pos >= self.n_ctx {
            return Err(format!(
                "gdev: position {pos} exceeds context {}/{} (raise BONSAI_CTX)",
                self.n_ctx, self.n_ctx
            ));
        }
        let timing = std::env::var("GDEV_TIME").map(|v| v == "1").unwrap_or(false);
        let t_phase = std::time::Instant::now();
        // reset kv caches when pos==0 (fresh sequence)
        if pos == 0 {
            let kz = vec![0u8; self.n_ctx * KV_STRIDE * 4];
            let sz = vec![0u8; GDN_STATE_ELEMS * 4];
            let cz = vec![0u8; 3 * GDN_CH * 4];
            for il in 0..self.cfg.n_layer {
                if let Some(k) = &self.kcache[il] { self.gpu.upload(k, &kz)?; }
                if let Some(v) = &self.vcache[il] { self.gpu.upload(v, &kz)?; }
                if let Some(c) = &self.conv_cache[il] { self.gpu.upload(c, &cz)?; }
                if let Some(s) = &self.state[il] { self.gpu.upload(s, &sz)?; }
            }
        }
        self.gpu.upload(&self.cur, f32_bytes(embed))?;
        self.gpu.rec_begin()?;
        for il in 0..self.cfg.n_layer {
            if self.cfg.is_full_attention(il) {
                self.rec_attn_layer(il, pos)?;
            } else {
                self.rec_gdn_layer(il)?;
            }
            self.rec_ffn(il)?;
        }
        let (wout, _, _) = self.tw("output_norm.weight")?;
        rec_rms(&mut self.gpu, &self.cur, &wout, &self.hidden, N_EMBD)?;
        if timing {
            eprintln!("[gdev] record: {:.1}ms", t_phase.elapsed().as_secs_f64() * 1e3);
        }
        let t_sub = std::time::Instant::now();
        self.gpu.rec_end_submit()?;
        let bytes = self.gpu.read_dev(&self.hidden, N_EMBD * 4)?;
        if timing {
            eprintln!("[gdev] submit+read: {:.1}ms", t_sub.elapsed().as_secs_f64() * 1e3);
            eprintln!("[gdev] total: {:.1}ms", t_phase.elapsed().as_secs_f64() * 1e3);
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    // ------------------------------------------------------------------
    // P3+P4 - batched prefill over an N-token batch (all 64 layers).
    //
    // Processes the N prompt positions layer-by-layer (layer-major), which is
    // numerically equivalent to the sequential token loop: each position's
    // running activation is kept in an N-wide `[token][n_embd]` tile and every
    // layer's projections (full-attention + FFN + recurrent-GDN) run through
    // the P2 N-column GEMM (weight stream read once). The recurrent (GDN)
    // layers' causal conv + gated-delta-net state recurrence stays sequential
    // across positions (P4), which only touches the small resident conv
    // weights and the ~3 MB/layer state, never the weight stream. The
    // post-prefill KV / conv / state buffers are exactly what the token loop
    // leaves behind, so `forward_token` can append at pos = N.
    // ------------------------------------------------------------------
    fn reset_sequence(&mut self) -> Result<(), String> {
        let kz = vec![0u8; self.n_ctx * KV_STRIDE * 4];
        let sz = vec![0u8; GDN_STATE_ELEMS * 4];
        let cz = vec![0u8; 3 * GDN_CH * 4];
        for il in 0..self.cfg.n_layer {
            if let Some(k) = &self.kcache[il] { self.gpu.upload(k, &kz)?; }
            if let Some(v) = &self.vcache[il] { self.gpu.upload(v, &kz)?; }
            if let Some(c) = &self.conv_cache[il] { self.gpu.upload(c, &cz)?; }
            if let Some(s) = &self.state[il] { self.gpu.upload(s, &sz)?; }
        }
        Ok(())
    }

    /// Batched full-attention layer for all N positions (P3). Positions are the
    /// window-local range `[pos_base, pos_base + n)` (absolute sequence
    /// positions, appended to any KV/conv/state already present below `pos_base`
    /// from earlier windows). With `pos_base == 0` this is exactly the 
    /// single-batch behaviour of P3.
    fn record_battn(
        &mut self,
        il: usize,
        n: usize,
        pos_base: usize,
        bb: &P3Buf,
    ) -> Result<(), String> {
        let cur = &bb.xb;
        let xnorm = &bb.xnorm;
        let partials = &bb.partials;

        // attn_norm over the whole tile -> xnorm tile [token][n_embd]
        let (wn, _, _) = self.tw(&format!("blk.{il}.attn_norm.weight"))?;
        rec_norm_rows(&mut self.gpu, cur, &wn, xnorm, n * N_EMBD, N_EMBD, 0)?;

        // wq/wk/wv over the N columns.
        for (suf, dst) in [
            ("attn_q.weight", &bb.qfull),
            ("attn_k.weight", &bb.kraw),
            ("attn_v.weight", &bb.vraw),
        ] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            self.gpu.rec_matvec_batch(
                &w, xnorm, dst, partials, ne0 as u32, 0, rows as u32, n as u32,
            )?;
        }
        let (wq, _, _) = self.tw(&format!("blk.{il}.attn_q_norm.weight"))?;
        let (wk, _, _) = self.tw(&format!("blk.{il}.attn_k_norm.weight"))?;

        // Per-position small ops (split, head norm, rope, KV store, causal
        // attention, gate). These reuse the exact single-stream kernels so each
        // position is bit-identical to the token-loop path.
        for p in 0..n {
            let pos = pos_base + p; // absolute sequence position
            // bring this position's projection rows into the dense buffers
            rec_bcopy(&mut self.gpu, &bb.qfull, &self.qfull, p * 2 * N_HEAD * HEAD_D, 0, 2 * N_HEAD * HEAD_D)?;
            rec_bcopy(&mut self.gpu, &bb.kraw, &self.kraw, p * N_KV * HEAD_D, 0, N_KV * HEAD_D)?;
            rec_bcopy(&mut self.gpu, &bb.vraw, &self.vraw, p * N_KV * HEAD_D, 0, N_KV * HEAD_D)?;

            self.gpu.ensure_kernel("split_qgate", vk::SPLIT_QGATE_SPV)?;
            self.gpu.rec_dispatch(
                "split_qgate",
                &self.qfull,
                &self.q,
                &self.gate,
                pc_u32(&[HEAD_D as u32, 0]),
                (N_HEAD * HEAD_D / 256) as u32,
            )?;
            rec_norm_rows(&mut self.gpu, &self.q, &wq, &self.qn, N_HEAD * HEAD_D, HEAD_D, 0)?;
            rec_norm_rows(&mut self.gpu, &self.kraw, &wk, &self.kn, N_KV * HEAD_D, HEAD_D, 0)?;
            self.gpu.ensure_kernel("rope_imrope", vk::ROPE_IMROPE_SPV)?;
            let pc_rope = [
                HEAD_D as u32,
                N_ROT as u32,
                pos as u32,
                FREQ_BASE.to_bits(),
                SECTIONS[0],
                SECTIONS[1],
                SECTIONS[2],
                SECTIONS[3],
            ];
            self.gpu.rec_dispatch("rope_imrope", &self.qn, &self.qn, &self.qn, pc_u32(&pc_rope), N_HEAD as u32)?;
            self.gpu.rec_dispatch("rope_imrope", &self.kn, &self.kn, &self.kn, pc_u32(&pc_rope), N_KV as u32)?;

            self.gpu.ensure_kernel("kv_store", vk::KV_STORE_SPV)?;
            let pc_kv = [pos as u32, KV_STRIDE as u32];
            self.gpu.rec_dispatch(
                "kv_store",
                &self.kn,
                self.kcache[il].as_ref().unwrap(),
                self.kcache[il].as_ref().unwrap(),
                pc_u32(&pc_kv),
                KV_STRIDE.div_ceil(256) as u32,
            )?;
            self.gpu.rec_dispatch(
                "kv_store",
                &self.vraw,
                self.vcache[il].as_ref().unwrap(),
                self.vcache[il].as_ref().unwrap(),
                pc_u32(&pc_kv),
                KV_STRIDE.div_ceil(256) as u32,
            )?;

            let n_pos = pos + 1;
            let pc_att = [
                N_HEAD as u32,
                n_pos as u32,
                HEAD_D as u32,
                KV_STRIDE as u32,
                (1.0 / (HEAD_D as f32).sqrt()).to_bits(),
                0,
            ];
            self.gpu.ensure_kernel("attn_scores", vk::ATTN_SCORES_SPV)?;
            self.gpu.rec_dispatch("attn_scores", &self.qn, self.kcache[il].as_ref().unwrap(), &self.scores, pc_u32(&pc_att), N_HEAD as u32)?;
            self.gpu.ensure_kernel("softmax_inplace", vk::SOFTMAX_INPLACE_SPV)?;
            self.gpu.rec_dispatch("softmax_inplace", &self.scores, &self.scores, &self.scores, pc_u32(&pc_att), N_HEAD as u32)?;
            self.gpu.ensure_kernel("attn_out", vk::ATTN_OUT_SPV)?;
            self.gpu.rec_dispatch("attn_out", &self.scores, self.vcache[il].as_ref().unwrap(), &self.attn_h, pc_u32(&pc_att), N_HEAD as u32)?;

            // gated out = sigmoid(gate) * attn_h -> qfull[0..nhead*hd]
            rec_elem(&mut self.gpu, &self.gate, &self.attn_h, &self.qfull, N_HEAD * HEAD_D, 4)?;
            rec_bcopy(&mut self.gpu, &self.qfull, &bb.gated, 0, p * N_HEAD * HEAD_D, N_HEAD * HEAD_D)?;
        }

        // wo over the N columns -> branch tile
        let (wout, ne0, rows) = self.tw(&format!("blk.{il}.attn_output.weight"))?;
        self.gpu.rec_matvec_batch(
            &wout, &bb.gated, &bb.branch, partials, ne0 as u32, 0, rows as u32, n as u32,
        )?;
        // residual: xb += branch (flat over the tile)
        self.gpu.ensure_kernel("add_residual", vk::ADD_RESIDUAL_SPV)?;
        self.gpu.rec_dispatch(
            "add_residual",
            &bb.xb,
            &bb.branch,
            &bb.xb,
            pc_u32(&[N_EMBD as u32 * n as u32]),
            (N_EMBD * n).div_ceil(256) as u32,
        )
    }

    /// Batched FFN for all N positions (P3).
    fn record_bffn(&mut self, il: usize, n: usize, bb: &P3Buf) -> Result<(), String> {
        let xnorm = &bb.xnorm;
        let partials = &bb.partials;
        let (wp, _, _) = self.tw(&format!("blk.{il}.post_attention_norm.weight"))?;
        rec_norm_rows(&mut self.gpu, &bb.xb, &wp, xnorm, n * N_EMBD, N_EMBD, 0)?;
        for (suf, dst) in [("ffn_gate.weight", &bb.fg), ("ffn_up.weight", &bb.fu)] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            self.gpu.rec_matvec_batch(
                &w, xnorm, dst, partials, ne0 as u32, 0, rows as u32, n as u32,
            )?;
        }
        self.gpu.ensure_kernel("elem", vk::ELEM_SPV)?;
        self.gpu.rec_dispatch(
            "elem",
            &bb.fg,
            &bb.fu,
            &bb.h1,
            pc_u32(&[(N_FF * n) as u32, 3]),
            (N_FF * n).div_ceil(256) as u32,
        )?;
        let (wd, ne0, rows) = self.tw(&format!("blk.{il}.ffn_down.weight"))?;
        self.gpu.rec_matvec_batch(
            &wd, &bb.h1, &bb.branch, partials, ne0 as u32, 0, rows as u32, n as u32,
        )?;
        self.gpu.ensure_kernel("add_residual", vk::ADD_RESIDUAL_SPV)?;
        self.gpu.rec_dispatch(
            "add_residual",
            &bb.xb,
            &bb.branch,
            &bb.xb,
            pc_u32(&[(N_EMBD * n) as u32]),
            (N_EMBD * n).div_ceil(256) as u32,
        )
    }

    /// Batched recurrent (GDN) layer over all N positions (P4).
    ///
    /// The heavy work is batched across the N columns so the (once-streamed)
    /// weight blocks are read a single time:
    ///   1. attn_qkv / attn_gate / ssm_beta / ssm_alpha projections over the
    ///      whole N-wide rms-normed tile through the P2 N-column GEMM;
    ///   2. per position, the causal conv1d + q/k l2 + `gdn_step` state
    ///      recurrence + gated output norm (inherently sequential across
    ///      positions, and cheap - they touch only the small resident conv
    ///      weights and the ~3 MB/layer state, not the weight stream);
    ///   3. a single batched ssm_out GEMM over the N gated-norm outputs.
    ///
    /// The final conv_cache and GDN state equal exactly what the per-token
    /// `rec_gdn_layer` path leaves behind, because each position's conv/l2/
    /// gdn_step/gated-norm use the same kernels on the same per-position data
    /// (the batched projection result is bit-identical to the per-token
    /// matvec), in the same position order. Decode can therefore append at
    /// pos = N.
    fn record_bgdn(&mut self, il: usize, n: usize, bb: &P3Buf) -> Result<(), String> {
        let xnorm = &bb.xnorm;
        let partials = &bb.partials;

        // rms attn_norm for each position into the xnorm tile, reproducing the
        // exact single-stream rec_rms (rms_norm kernel) so the batched
        // projection input equals the token-loop input row-for-row.
        let (wn, _, _) = self.tw(&format!("blk.{il}.attn_norm.weight"))?;
        for p in 0..n {
            rec_bcopy(&mut self.gpu, &bb.xb, &self.cur, p * N_EMBD, 0, N_EMBD)?;
            rec_rms(&mut self.gpu, &self.cur, &wn, &self.xnorm, N_EMBD)?;
            rec_bcopy(&mut self.gpu, &self.xnorm, &bb.xnorm, 0, p * N_EMBD, N_EMBD)?;
        }

        // ---- batched projections over the N columns -------------------------
        for (suf, dst) in [
            ("attn_qkv.weight", &bb.gqkv),
            ("attn_gate.weight", &bb.gz),
            ("ssm_beta.weight", &bb.gbeta),
            ("ssm_alpha.weight", &bb.galpha),
        ] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            self.gpu.rec_matvec_batch(
                &w, xnorm, dst, partials, ne0 as u32, 0, rows as u32, n as u32,
            )?;
        }

        let (cw, _, _) = self.tw(&format!("blk.{il}.ssm_conv1d.weight"))?;
        let (wsn, _, _) = self.tw(&format!("blk.{il}.ssm_norm.weight"))?;

        // ---- sequential per-position conv + recurrence (cheap, exact) -------
        for p in 0..n {
            // bring this position's projection rows into the dense buffers the
            // per-position kernels consume.
            rec_bcopy(&mut self.gpu, &bb.gqkv, &self.qkv, p * GDN_CH, 0, GDN_CH)?;
            rec_bcopy(&mut self.gpu, &bb.gz, &self.z, p * GDN_DI, 0, GDN_DI)?;
            rec_bcopy(&mut self.gpu, &bb.gbeta, &self.smalls[il], p * GDN_HV, 0, GDN_HV)?;
            rec_bcopy(&mut self.gpu, &bb.galpha, &self.smalls[il], p * GDN_HV, GDN_HV, GDN_HV)?;

            // conv1d + silu (in place on qkv), shifting the conv cache
            self.gpu.ensure_kernel("conv1d_silu", vk::CONV1D_SILU_SPV)?;
            self.gpu.rec_dispatch(
                "conv1d_silu",
                &self.qkv,
                self.conv_cache[il].as_ref().unwrap(),
                &cw,
                pc_u32(&[GDN_CH as u32, 0]),
                GDN_CH.div_ceil(256) as u32,
            )?;
            rec_l2(&mut self.gpu, &self.qkv, 0, GDN_DK / STATE_SIZE)?;
            rec_l2(&mut self.gpu, &self.qkv, GDN_DK, GDN_DK / STATE_SIZE)?;
            self.gpu.ensure_kernel("gdn_prep", vk::GDN_PREP_SPV)?;
            self.gpu.rec_dispatch(
                "gdn_prep",
                &self.qkv,
                &self.smalls[il],
                &self.blob,
                pc_u32(&[0, 0]),
                GDN_CH.div_ceil(256) as u32,
            )?;
            self.gpu.ensure_kernel("gdn_step", vk::GDN_STEP_SPV)?;
            self.gpu.rec_dispatch("gdn_step", &self.blob, self.state[il].as_ref().unwrap(), &self.attn_out, &[], GDN_HV as u32)?;
            rec_norm_rows(&mut self.gpu, &self.attn_out, &wsn, &self.normed, GDN_DI, STATE_SIZE, 0)?;
            // gated output = rms_norm(attn_out) * silu(z) -> self.h1
            rec_elem(&mut self.gpu, &self.z, &self.normed, &self.h1, GDN_DI, 3)?;
            // store this position's gated output into the N-wide h1 tile
            rec_bcopy(&mut self.gpu, &self.h1, &bb.gh1, 0, p * GDN_DI, GDN_DI)?;
        }

        // ---- ssm_out batched over the N gated-output columns -> branch tile --
        let (wso, ne0, rows) = self.tw(&format!("blk.{il}.ssm_out.weight"))?;
        self.gpu.rec_matvec_batch(
            &wso, &bb.gh1, &bb.branch, partials, ne0 as u32, 0, rows as u32, n as u32,
        )?;
        // residual: xb += branch (flat over the tile)
        self.gpu.ensure_kernel("add_residual", vk::ADD_RESIDUAL_SPV)?;
        self.gpu.rec_dispatch(
            "add_residual",
            &bb.xb,
            &bb.branch,
            &bb.xb,
            pc_u32(&[(N_EMBD * n) as u32]),
            (N_EMBD * n).div_ceil(256) as u32,
        )
    }

    /// Run one layer of the batched prefill (attention + FFN) inside a single
    /// recorded command buffer, updating the N-wide tile in `bb`. The `n`
    /// positions are the absolute range `[pos_base, pos_base + n)`.
    fn pre_layer(&mut self, il: usize, n: usize, pos_base: usize, bb: &P3Buf) -> Result<(), String> {
        self.gpu.rec_begin()?;
        if self.cfg.is_full_attention(il) {
            self.record_battn(il, n, pos_base, bb)?;
        } else {
            self.record_bgdn(il, n, bb)?;
        }
        self.record_bffn(il, n, bb)?;
        self.gpu.rec_end_submit()
    }

    /// Allocate the N-wide device tiles + partials used by one batched prefill
    /// pass (sized for up to `w` columns). The partials buffer is sized for the
    /// largest matvec (`N_FF * (N_EMBD/128)` partials per column).
    fn alloc_bb(gpu: &mut vk::Gpu, w: usize) -> Result<P3Buf, String> {
        let xb = alloc_bytes(gpu, w * N_EMBD * 4)?;
        let xnorm = alloc_bytes(gpu, w * N_EMBD * 4)?;
        let branch = alloc_bytes(gpu, w * N_EMBD * 4)?;
        let qfull = alloc_bytes(gpu, w * 2 * N_HEAD * HEAD_D * 4)?;
        let kraw = alloc_bytes(gpu, w * N_KV * HEAD_D * 4)?;
        let vraw = alloc_bytes(gpu, w * N_KV * HEAD_D * 4)?;
        let gated = alloc_bytes(gpu, w * N_HEAD * HEAD_D * 4)?;
        let fg = alloc_bytes(gpu, w * N_FF * 4)?;
        let fu = alloc_bytes(gpu, w * N_FF * 4)?;
        let h1 = alloc_bytes(gpu, w * N_FF * 4)?;
        let gqkv = alloc_bytes(gpu, w * GDN_CH * 4)?;
        let gz = alloc_bytes(gpu, w * GDN_DI * 4)?;
        let gbeta = alloc_bytes(gpu, w * GDN_HV * 4)?;
        let galpha = alloc_bytes(gpu, w * GDN_HV * 4)?;
        let gh1 = alloc_bytes(gpu, w * GDN_DI * 4)?;
        let partials = alloc_bytes(gpu, N_FF * (N_EMBD / 128) * w * 4)?;
        Ok(P3Buf {
            xb, xnorm, branch, qfull, kraw, vraw, gated, fg, fu, h1,
            gqkv, gz, gbeta, galpha, gh1, partials,
        })
    }

    fn destroy_bb(gpu: &mut vk::Gpu, bb: P3Buf) {
        gpu.destroy_dev_buffer(bb.xb);
        gpu.destroy_dev_buffer(bb.xnorm);
        gpu.destroy_dev_buffer(bb.branch);
        gpu.destroy_dev_buffer(bb.qfull);
        gpu.destroy_dev_buffer(bb.kraw);
        gpu.destroy_dev_buffer(bb.vraw);
        gpu.destroy_dev_buffer(bb.gated);
        gpu.destroy_dev_buffer(bb.fg);
        gpu.destroy_dev_buffer(bb.fu);
        gpu.destroy_dev_buffer(bb.h1);
        gpu.destroy_dev_buffer(bb.gqkv);
        gpu.destroy_dev_buffer(bb.gz);
        gpu.destroy_dev_buffer(bb.gbeta);
        gpu.destroy_dev_buffer(bb.galpha);
        gpu.destroy_dev_buffer(bb.gh1);
        gpu.destroy_dev_buffer(bb.partials);
    }

    /// Run the batched 64-layer forward over `n` embeddings starting at the
    /// absolute position `pos_base`, updating the N-wide tile in `bb` (which is
    /// sized for >= n columns). The KV / conv / GDN-state caches are appended
    /// at positions `[pos_base, pos_base+n)` and must already hold the state
    /// for positions `< pos_base` (from `reset_sequence()` at the first window
    /// and earlier windows). Callers perform the reset and the output-norm read
    /// so that windowed prefill can append window after window.
    fn run_batch_at(
        &mut self,
        embeds: &[f32],
        n: usize,
        pos_base: usize,
        bb: &P3Buf,
    ) -> Result<(), String> {
        // upload the embedding tile (first n rows of the window slice)
        self.gpu.upload(&bb.xb, f32_bytes(embeds))?;
        for il in 0..self.cfg.n_layer {
            self.pre_layer(il, n, pos_base, bb)?;
        }
        Ok(())
    }

    /// Batched prefill of `n` prompt embeddings (each `N_EMBD` floats, one per
    /// prompt position in order). Returns the output-normalized hidden tile
    /// (`n * N_EMBD` floats, `[token][n_embd]`). All 64 layers run over the N
    /// columns: full-attention projections, attention, KV stores, all FFN
    /// GEMMs, and (P4) the recurrent GDN projections + ssm_out run across the
    /// N columns, with only the GDN causal conv + state recurrence sequential
    /// per position. KV / conv / state caches are left exactly as the
    /// sequential token loop leaves them, so `forward_token(pos = n, ...)` can
    /// append the first generated token.
    ///
    /// This is a single full-size window (`window == n`); see
    /// [`Self::prefill_batch_windowed`] for splitting very long prompts.
    pub fn prefill_batch(&mut self, embeds: &[f32], n: usize) -> Result<Vec<f32>, String> {
        self.prefill_batch_windowed(embeds, n, n)
    }

    /// Batched prefill of `n` prompt embeddings processed in **sequential
    /// windows of at most `window` columns** (P5). The full 64-layer pass runs
    /// over each window's positions in order (`reset_sequence()` once, then
    /// window 0 over positions `[0, w)`, window 1 over `[w, 2w)`, ...), so the
    /// KV / conv / GDN-state caches are appended window after window and decode
    /// can append at `pos = n`. Weight tensors stay resident across windows, so
    /// weight traffic is ~one pass per window regardless of N. Numerically each
    /// position is identical to a single full-size batch (per-column GEMM) and
    /// to the sequential token loop within the golden tolerance.
    ///
    /// Returns the output-normalized hidden tile of the **last** window
    /// (`len_last * N_EMBD` floats, `[token][n_embd]`); its final row is
    /// position `n - 1`, whose logits sample the first generated token. When
    /// `window >= n` this equals the full `n`-row tile of
    /// [`Self::prefill_batch`].
    pub fn prefill_batch_windowed(
        &mut self,
        embeds: &[f32],
        n: usize,
        window: usize,
    ) -> Result<Vec<f32>, String> {
        if self.kv_quantized() {
            return Err(
                "prefill_batch_windowed: not available with BONSAI_KV (the batched path \
                 writes f32 KV caches); use the token loop"
                    .into(),
            );
        }
        if n < 1 {
            return Err("prefill_batch_windowed: empty batch".into());
        }
        if n > self.n_ctx {
            return Err(format!(
                "prefill_batch_windowed: batch {n} exceeds context {}/{} (raise BONSAI_CTX)",
                self.n_ctx, self.n_ctx
            ));
        }
        let need = n * N_EMBD;
        if embeds.len() != need {
            return Err(format!(
                "prefill_batch_windowed: expected {need} embeddings, got {}",
                embeds.len()
            ));
        }
        let timing = std::env::var("GDEV_TIME").map(|v| v == "1").unwrap_or(false);
        let t_phase = std::time::Instant::now();

        let w = window.clamp(1, n); // window width, bounded to the batch
        let nwin = n.div_ceil(w);

        self.reset_sequence()?;
        let bb = Self::alloc_bb(&mut self.gpu, w)?;

        let r = (|| -> Result<Vec<f32>, String> {
            let mut last_hidden = Vec::new();
            for wi in 0..nwin {
                let w0 = wi * w;
                let w1 = (w0 + w).min(n);
                let len = w1 - w0;
                // slice this window's embeddings out of the contiguous tile
                let win_emb = &embeds[w0 * N_EMBD..w1 * N_EMBD];
                self.run_batch_at(win_emb, len, w0, &bb)?;
                // output norm over the window tile (positions w0..w1)
                let (wout, _, _) = self.tw("output_norm.weight")?;
                self.gpu.rec_begin()?;
                rec_norm_rows(&mut self.gpu, &bb.xb, &wout, &bb.xnorm, len * N_EMBD, N_EMBD, 0)?;
                self.gpu.rec_end_submit()?;
                last_hidden = self.gpu.read_dev(&bb.xnorm, len * N_EMBD * 4)?;
            }
            Ok(last_hidden
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        })();

        Self::destroy_bb(&mut self.gpu, bb);
        if timing {
            eprintln!(
                "[gdev] prefill_batch_windowed n={n} window={w} ({} windows): {:.1}ms",
                nwin,
                t_phase.elapsed().as_secs_f64() * 1e3
            );
        }
        r
    }

    /// Snapshot layer `il`'s KV caches for positions `0..n_pos`
    /// (k then v, `n_pos * KV_STRIDE` floats each) back to the host.
    pub fn dump_kv(&mut self, il: usize, n_pos: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let k = self.kcache.get(il).and_then(|o| o.as_ref()).ok_or("no kv")?;
        let v = self.vcache[il].as_ref().unwrap();
        let want = (n_pos * KV_STRIDE).min(k.len / 4) * 4;
        let kb = self.gpu.read_dev(k, want)?;
        let vb = self.gpu.read_dev(v, want)?;
        let tof = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        };
        Ok((tof(&kb), tof(&vb)))
    }

    /// Snapshot layer `il`'s GDN state (H_V transposed S x S matrices,
    /// `GDN_HV * STATE_SIZE * STATE_SIZE` floats) back to the host, if it is a
    /// recurrent layer (`None` for full-attention layers).
    pub fn dump_state(&mut self, il: usize) -> Option<Vec<f32>> {
        let s = self.state.get(il)?.as_ref()?;
        let bytes = self.gpu.read_dev(s, s.len).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Snapshot layer `il`'s conv cache (3 * GDN_CH floats, the last 3
    /// pre-conv inputs per channel) back to the host, if it is a recurrent
    /// layer (`None` for full-attention layers).
    pub fn dump_conv(&mut self, il: usize) -> Option<Vec<f32>> {
        let c = self.conv_cache.get(il)?.as_ref()?;
        let bytes = self.gpu.read_dev(c, c.len).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect(),
        )
    }
}

/// N-wide device tiles + partials for the batched (P3) prefill.
struct P3Buf {
    xb: DevBuf,     // running activation [token][n_embd]
    xnorm: DevBuf,  // rms scratch [token][n_embd]
    branch: DevBuf, // layer output [token][n_embd]
    qfull: DevBuf,  // [token][2*nhead*hd]
    kraw: DevBuf,   // [token][n_kv*hd]
    vraw: DevBuf,   // [token][n_kv*hd]
    gated: DevBuf,  // [token][nhead*hd]
    fg: DevBuf,     // [token][n_ff]
    fu: DevBuf,     // [token][n_ff]
    h1: DevBuf,     // [token][n_ff]
    gqkv: DevBuf,   // GDN attn_qkv projection [token][gdn_ch]  (P4)
    gz: DevBuf,     // GDN attn_gate projection [token][gdn_di] (P4)
    gbeta: DevBuf,  // GDN ssm_beta projection [token][gdn_hv]  (P4)
    galpha: DevBuf, // GDN ssm_alpha projection [token][gdn_hv] (P4)
    gh1: DevBuf,    // GDN gated output pre-ssm_out [token][gdn_di] (P4)
    partials: DevBuf,
}
