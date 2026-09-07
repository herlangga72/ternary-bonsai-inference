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
const N_CTX: usize = 2048;
const EPS: f32 = 1e-6;

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
    tens: HashMap<String, TensorW>,

    cur: DevBuf,
    xnorm: DevBuf,
    branch: DevBuf,
    hidden: DevBuf,
    partials: DevBuf,

    qkv: DevBuf,
    z: DevBuf,
    smalls: DevBuf,
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

    kcache: Vec<DevBuf>,
    vcache: Vec<DevBuf>,
    conv_cache: Vec<DevBuf>,
    state: Vec<DevBuf>,
}

fn make(gpu: &mut vk::Gpu, n: usize) -> Result<DevBuf, String> {
    // device-local on discrete GPUs; host-visible (RAM) on APUs so the whole
    // model + caches fit. n is in floats; buffers are sized in bytes.
    gpu.create_model_weight_buffer(n.max(1) * 4)
}

impl GDev {
    pub fn open(model: &str) -> Result<GDev, String> {
        let mut gpu = vk::Gpu::open()?;
        let gg = GGUF::open(model).map_err(|e| format!("gguf: {e}"))?;
        let cfg = Qwen35::from_gguf(&gg)?;
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
            let buf = gpu.create_model_weight_buffer(nbytes)?;
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
        let smalls = make(&mut gpu, 4 * GDN_HV)?;
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
        let scores = make(&mut gpu, N_HEAD * N_CTX)?;
        let attn_h = make(&mut gpu, N_HEAD * HEAD_D)?;
        let fg = make(&mut gpu, N_FF)?;
        let fu = make(&mut gpu, N_FF)?;
        let h1 = make(&mut gpu, N_FF)?;

        let mut kcache = Vec::new();
        let mut vcache = Vec::new();
        let mut conv_cache = Vec::new();
        let mut state = Vec::new();
        let kv_bytes = vec![0u8; N_CTX * KV_STRIDE * 4];
        let conv_bytes = vec![0u8; 3 * GDN_CH * 4];
        let state_bytes = vec![0u8; GDN_STATE_ELEMS * 4];
        for _ in 0..cfg.n_layer {
            let k = make(&mut gpu, N_CTX * KV_STRIDE)?;
            let v = make(&mut gpu, N_CTX * KV_STRIDE)?;
            gpu.upload(&k, &kv_bytes)?;
            gpu.upload(&v, &kv_bytes)?;
            let c = make(&mut gpu, 3 * GDN_CH)?;
            let s = make(&mut gpu, GDN_STATE_ELEMS)?;
            gpu.upload(&c, &conv_bytes)?;
            gpu.upload(&s, &state_bytes)?;
            kcache.push(k);
            vcache.push(v);
            conv_cache.push(c);
            state.push(s);
        }

        Ok(GDev {
            gpu,
            cfg,
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

    fn read_w(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let (buf, _ne0, _rows) = self.tw(name)?;
        let n = buf.len / 4; // f32 payload
        let bytes = self.gpu.read_dev(&buf, n * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn stage_smalls(&mut self, il: usize) -> Result<(), String> {
        let ssa = self.read_w(&format!("blk.{il}.ssm_a"))?;
        let dtb = self.read_w(&format!("blk.{il}.ssm_dt.bias"))?;
        let n = GDN_HV.min(ssa.len()).min(dtb.len());
        let mut sm = vec![0.0f32; 4 * GDN_HV];
        sm[2 * GDN_HV..2 * GDN_HV + n].copy_from_slice(&ssa[..n]);
        sm[3 * GDN_HV..3 * GDN_HV + n].copy_from_slice(&dtb[..n]);
        self.gpu.upload(&self.smalls, f32_bytes(&sm))?;
        Ok(())
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

        self.gpu.ensure_kernel("kv_store", vk::KV_STORE_SPV)?;
        let pc_kv = [pos as u32, KV_STRIDE as u32];
        self.gpu.rec_dispatch(
            "kv_store",
            &self.kn,
            &self.kcache[il],
            &self.kcache[il],
            pc_u32(&pc_kv),
            KV_STRIDE.div_ceil(256) as u32,
        )?;
        self.gpu.rec_dispatch(
            "kv_store",
            &self.vraw,
            &self.vcache[il],
            &self.vcache[il],
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
        self.gpu.rec_dispatch("attn_scores", &self.qn, &self.kcache[il], &self.scores, pc_u32(&pc_att), N_HEAD as u32)?;
        self.gpu.ensure_kernel("softmax_inplace", vk::SOFTMAX_INPLACE_SPV)?;
        self.gpu.rec_dispatch("softmax_inplace", &self.scores, &self.scores, &self.scores, pc_u32(&pc_att), N_HEAD as u32)?;
        self.gpu.ensure_kernel("attn_out", vk::ATTN_OUT_SPV)?;
        self.gpu.rec_dispatch("attn_out", &self.scores, &self.vcache[il], &self.attn_h, pc_u32(&pc_att), N_HEAD as u32)?;

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
            ("ssm_beta.weight", &self.smalls, 0),
            ("ssm_alpha.weight", &self.smalls, GDN_HV),
        ] {
            let (w, ne0, rows) = self.tw(&format!("blk.{il}.{suf}"))?;
            rec_mat(&mut self.gpu, partials, &w, ne0, rows, xnorm, dst, ybase)?;
        }
        let (cw, _, _) = self.tw(&format!("blk.{il}.ssm_conv1d.weight"))?;
        self.gpu.ensure_kernel("conv1d_silu", vk::CONV1D_SILU_SPV)?;
        self.gpu.rec_dispatch(
            "conv1d_silu",
            &self.qkv,
            &self.conv_cache[il],
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
            &self.smalls,
            &self.blob,
            pc_u32(&[0, 0]),
            GDN_CH.div_ceil(256) as u32,
        )?;
        self.gpu.ensure_kernel("gdn_step", vk::GDN_STEP_SPV)?;
        self.gpu.rec_dispatch("gdn_step", &self.blob, &self.state[il], &self.attn_out, &[], GDN_HV as u32)?;
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
        // reset kv caches when pos==0 (fresh sequence)
        if pos == 0 {
            let kz = vec![0u8; N_CTX * KV_STRIDE * 4];
            let sz = vec![0u8; GDN_STATE_ELEMS * 4];
            let cz = vec![0u8; 3 * GDN_CH * 4];
            for il in 0..self.cfg.n_layer {
                self.gpu.upload(&self.kcache[il], &kz)?;
                self.gpu.upload(&self.vcache[il], &kz)?;
                self.gpu.upload(&self.conv_cache[il], &cz)?;
                self.gpu.upload(&self.state[il], &sz)?;
            }
        }
        self.gpu.upload(&self.cur, f32_bytes(embed))?;
        self.gpu.rec_begin()?;
        for il in 0..self.cfg.n_layer {
            if self.cfg.is_full_attention(il) {
                self.rec_attn_layer(il, pos)?;
            } else {
                // stage per-layer constant smalls (ssm_a/dt.bias) first
                self.stage_smalls(il)?;
                self.rec_gdn_layer(il)?;
            }
            self.rec_ffn(il)?;
        }
        let (wout, _, _) = self.tw("output_norm.weight")?;
        rec_rms(&mut self.gpu, &self.cur, &wout, &self.hidden, N_EMBD)?;
        self.gpu.rec_end_submit()?;
        let bytes = self.gpu.read_dev(&self.hidden, N_EMBD * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}
