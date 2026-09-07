//! M6-1 weight context: qwen35 tensor-name mapping over the GGUF reader.
//!
//! `Weights` owns a `GGUF` handle plus a name -> tensor index and exposes the
//! three access patterns the forward pass (M6-3..M6-6) needs, so layer code
//! never has to reason about GGUF offsets or quant formats:
//!
//! * small f32 tensors (norm weights, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`)
//!   are decoded whole and cached (`vec_f32`);
//! * PQ2_0 matrices are never materialized: `matvec*` reads the rows it needs
//!   straight from the file and dots them with the activation, the pattern
//!   sized by bonsai-matbench (~1 GMAC/s across 4 threads);
//! * `row_f32` decodes one contiguous row (used later for token-embedding and
//!   LM-head lookups).
//!
//! Model hyperparameters are parsed from GGUF metadata into [`Qwen35`], and the
//! per-layer tensor set the architecture requires is derived from them, so
//! [`Weights::verify`] can prove a file matches the architecture before any
//! compute starts.

#![allow(dead_code)]

use crate::gguf::{GGUF, TensorInfo, TYPE_F16, TYPE_F32, TYPE_PQ2_0};
use crate::kernels;
use crate::vk;
use std::collections::HashMap;

/// Optional Vulkan matvec accelerator (G2c). Holds a Gpu plus one device-local
/// buffer per PQ2_0 tensor, uploaded once at load. `Weights::matvec_into`
/// routes matvecs to the GPU when the tensor is present here; all other reads
/// (norm vectors, embedding rows) stay on the CPU mmap path.
pub struct GpuAccel {
    pub gpu: vk::Gpu,
    pub dev: HashMap<String, vk::DevBuf>,
}

/// Largest whole-tensor decode allowed through `vec_f32` (16 MiB of f32).
/// Matrices must be touched via `matvec*` row access instead.
const MAX_VEC_ELEMS: u64 = 1 << 22;

/// qwen35 hyperparameters read from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35 {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head: usize,
    pub n_ctx_train: u64,
    pub eps: f32,
    /// full-attention layer spacing (every `interval`-th layer, the last slot)
    pub interval: usize,
    pub n_rot: usize,
    pub sections: [i32; 4],
    pub freq_base: f32,
    // gated-SSM (recurrent) branch hyperparameters
    pub ssm_conv_kernel: usize,
    pub ssm_state: usize,
    pub ssm_group_count: usize,
    pub ssm_dt_rank: usize,
    pub ssm_inner: usize,
}

impl Qwen35 {
    pub fn from_gguf(g: &GGUF) -> Result<Qwen35, String> {
        let arch = g
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if arch != "qwen35" {
            return Err(format!("Weights: expected qwen35 architecture, found {arch:?}"));
        }
        macro_rules! need {
            ($key:literal, $what:literal) => {
                g.get($key)
                    .and_then(|v| v.as_u32())
                    .map(|x| x as usize)
                    .ok_or_else(|| {
                        format!("Weights: missing {} metadata (GGUF key {})", $what, $key)
                    })?
            };
        }
        let sections = g
            .get("qwen35.rope.dimension_sections")
            .and_then(|v| v.as_array())
            .ok_or("Weights: missing rope.dimension_sections (GGUF key qwen35.rope.dimension_sections)")?;
        if sections.len() != 4 {
            return Err(format!(
                "Weights: rope.dimension_sections has {} entries, expected 4",
                sections.len()
            ));
        }
        let mut sec = [0i32; 4];
        for (dst, src) in sec.iter_mut().zip(sections.iter()) {
            *dst = src
                .as_u32()
                .ok_or("Weights: rope.dimension_sections must contain u32")?
                as i32;
        }
        let eps = g
            .get("qwen35.attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .ok_or("Weights: missing layer_norm_rms_epsilon metadata")?;
        let freq_base = g
            .get("qwen35.rope.freq_base")
            .and_then(|v| v.as_f32())
            .ok_or("Weights: missing rope.freq_base metadata")?;
        let n_ctx_train = g
            .get("qwen35.context_length")
            .and_then(|v| v.as_u64())
            .ok_or("Weights: missing context_length metadata")?;

        Ok(Qwen35 {
            n_layer: need!("qwen35.block_count", "block_count"),
            n_embd: need!("qwen35.embedding_length", "embedding_length"),
            n_ff: need!("qwen35.feed_forward_length", "feed_forward_length"),
            n_head: need!("qwen35.attention.head_count", "head_count"),
            n_head_kv: need!("qwen35.attention.head_count_kv", "head_count_kv"),
            n_embd_head: need!("qwen35.attention.key_length", "key_length"),
            n_ctx_train,
            eps,
            interval: need!("qwen35.full_attention_interval", "full_attention_interval"),
            n_rot: need!("qwen35.rope.dimension_count", "rope.dimension_count"),
            sections: sec,
            freq_base,
            ssm_conv_kernel: need!("qwen35.ssm.conv_kernel", "ssm.conv_kernel"),
            ssm_state: need!("qwen35.ssm.state_size", "ssm.state_size"),
            ssm_group_count: need!("qwen35.ssm.group_count", "ssm.group_count"),
            ssm_dt_rank: need!("qwen35.ssm.time_step_rank", "ssm.time_step_rank"),
            ssm_inner: need!("qwen35.ssm.inner_size", "ssm.inner_size"),
        })
    }

    /// Full-attention layers sit on the last slot of every interval
    /// (`il % 4 == 3` for the real 27B); the rest are recurrent (gated SSM).
    pub fn is_full_attention(&self, il: usize) -> bool {
        self.interval > 0 && il % self.interval == self.interval - 1
    }

    pub fn is_recurrent(&self, il: usize) -> bool {
        !self.is_full_attention(il)
    }

    pub fn blk_name(&self, il: usize, suffix: &str) -> String {
        format!("blk.{il}.{suffix}")
    }

    /// Total q (== k) projection rows of the SSM branch: group_count * state_size.
    pub fn ssm_key_dim(&self) -> usize {
        self.ssm_group_count * self.ssm_state
    }

    /// Rows of the fused SSM qkv projection: q + k + v (v is d_inner wide).
    pub fn ssm_conv_channels(&self) -> usize {
        2 * self.ssm_key_dim() + self.ssm_inner
    }

    /// Expected per-layer tensor set `(name, ggml type, dims)` for layer `il`,
    /// in GGUF dims order ([ne0, ne1, ...]: the contiguous input width first).
    pub fn expected_layer_tensors(&self, il: usize) -> Vec<(String, u32, Vec<u64>)> {
        let d2 = |ne0: usize, ne1: usize| vec![ne0 as u64, ne1 as u64];
        let mut out: Vec<(String, u32, Vec<u64>)> = Vec::new();
        let mut add = |suffix: &str, ty: u32, dims: Vec<u64>| {
            out.push((format!("blk.{il}.{suffix}"), ty, dims));
        };

        // shared by every layer
        add("attn_norm.weight", TYPE_F32, vec![self.n_embd as u64]);
        add("post_attention_norm.weight", TYPE_F32, vec![self.n_embd as u64]);
        add("ffn_gate.weight", TYPE_PQ2_0, d2(self.n_embd, self.n_ff));
        add("ffn_up.weight", TYPE_PQ2_0, d2(self.n_embd, self.n_ff));
        add("ffn_down.weight", TYPE_PQ2_0, d2(self.n_ff, self.n_embd));

        if self.is_full_attention(il) {
            // wo input = concatenated per-head attention results (24 * 256)
            let attn_in = self.n_head * self.n_embd_head;
            let kv_rows = self.n_head_kv * self.n_embd_head;
            // each q row block carries [q(head) | gate(head)] -> 2x attn_in
            add("attn_q.weight", TYPE_PQ2_0, d2(self.n_embd, 2 * attn_in));
            add("attn_q_norm.weight", TYPE_F32, vec![self.n_embd_head as u64]);
            add("attn_k.weight", TYPE_PQ2_0, d2(self.n_embd, kv_rows));
            add("attn_k_norm.weight", TYPE_F32, vec![self.n_embd_head as u64]);
            add("attn_v.weight", TYPE_PQ2_0, d2(self.n_embd, kv_rows));
            add("attn_output.weight", TYPE_PQ2_0, d2(attn_in, self.n_embd));
        } else {
            let ch = self.ssm_conv_channels();
            add("attn_qkv.weight", TYPE_PQ2_0, d2(self.n_embd, ch));
            add("attn_gate.weight", TYPE_PQ2_0, d2(self.n_embd, self.ssm_inner));
            add("ssm_a", TYPE_F32, vec![self.ssm_dt_rank as u64]);
            add("ssm_alpha.weight", TYPE_PQ2_0, d2(self.n_embd, self.ssm_dt_rank));
            add("ssm_beta.weight", TYPE_PQ2_0, d2(self.n_embd, self.ssm_dt_rank));
            add(
                "ssm_conv1d.weight",
                TYPE_F32,
                vec![self.ssm_conv_kernel as u64, ch as u64],
            );
            add("ssm_dt.bias", TYPE_F32, vec![self.ssm_dt_rank as u64]);
            add("ssm_norm.weight", TYPE_F32, vec![self.ssm_state as u64]);
            add("ssm_out.weight", TYPE_PQ2_0, d2(self.ssm_inner, self.n_embd));
        }
        out
    }
}

/// Weight context: an open GGUF plus a name index and decoded-vector cache.
pub struct Weights {
    cfg: Qwen35,
    gguf: GGUF,
    by_name: HashMap<String, TensorInfo>,
    f32_cache: HashMap<String, Vec<f32>>,
    accel: Option<GpuAccel>,
}

impl Weights {
    pub fn open(path: &str) -> Result<Weights, String> {
        let gguf = GGUF::open(path)?;
        let cfg = Qwen35::from_gguf(&gguf)?;
        let mut by_name = HashMap::with_capacity(gguf.tensors.len());
        for t in &gguf.tensors {
            by_name.insert(t.name.clone(), t.clone());
        }
        Ok(Weights {
            cfg,
            gguf,
            by_name,
            f32_cache: HashMap::new(),
            accel: None,
        })
    }

    /// Open a Vulkan device and upload every PQ2_0 tensor payload into
    /// device-local VRAM. Subsequent `matvec_into` calls for those tensors run
    /// on the GPU; everything else still reads the mmap. Fails with a clear
    /// error when no Vulkan GPU is available.
    pub fn enable_gpu(&mut self) -> Result<(), String> {
        if self.accel.is_some() {
            return Ok(());
        }
        let gpu = vk::Gpu::open()?;
        let mut dev: HashMap<String, vk::DevBuf> = HashMap::new();
        // clone the index so the gguf borrow ends before we need &mut gpu
        let tensors = self.gguf.tensors.clone();
        let mut gpu = gpu;
        let mut max_ne0 = 0usize;
        let mut max_rows = 0usize;
        for t in &tensors {
            if t.ty != TYPE_PQ2_0 {
                continue;
            }
            let len = self.gguf.tensor_nbytes(t) as usize;
            let buf = gpu.create_weight_buffer(len)?;
            let payload = self.gguf.payload_slice(t)?;
            gpu.upload(&buf, payload)?;
            let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
            let rows = t.n_elem() as usize / ne0.max(1);
            max_ne0 = max_ne0.max(ne0);
            max_rows = max_rows.max(rows);
            dev.insert(t.name.clone(), buf);
        }
        gpu.prep_matvec_cache(max_ne0, max_rows)?;
        eprintln!(
            "[gpu] uploaded {} PQ2_0 tensors to device-local memory",
            dev.len()
        );
        self.accel = Some(GpuAccel { gpu, dev });
        Ok(())
    }

    pub fn gpu_active(&self) -> bool {
        self.accel.is_some()
    }

    pub fn config(&self) -> &Qwen35 {
        &self.cfg
    }

    /// Number of tensors in the file's index.
    pub fn n_indexed(&self) -> usize {
        self.by_name.len()
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.by_name
            .get(name)
            .ok_or_else(|| format!("Weights: tensor '{name}' not in GGUF index"))
    }

    /// Fetch a tensor and check its ggml type + dims against expectations.
    pub fn expect_tensor(&self, name: &str, ty: u32, dims: &[u64]) -> Result<&TensorInfo, String> {
        let t = self.tensor(name)?;
        if t.ty != ty {
            return Err(format!(
                "Weights: {name}: type {} != expected {ty}",
                t.ty
            ));
        }
        if t.dims.as_slice() != dims {
            return Err(format!(
                "Weights: {name}: dims {:?} != expected {dims:?}",
                t.dims
            ));
        }
        Ok(t)
    }

    /// Decode a whole small f32 (or f16) tensor and cache it. Use for 1-d norm
    /// vectors and small 2-d weights (`ssm_conv1d`); matrices must go through
    /// `matvec*` so their huge payloads are never materialized.
    pub fn vec_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        if let Some(v) = self.f32_cache.get(name) {
            return Ok(v.clone());
        }
        let t = self.tensor(name)?.clone();
        if t.ty != TYPE_F32 && t.ty != TYPE_F16 {
            return Err(format!(
                "Weights: vec_f32({name}): tensor type {} is not f32/f16",
                t.ty
            ));
        }
        if t.n_elem() > MAX_VEC_ELEMS {
            return Err(format!(
                "Weights: vec_f32({name}): {} elements is too large, use matvec row access",
                t.n_elem()
            ));
        }
        let v = self.gguf.read_tensor(&t)?;
        self.f32_cache.insert(name.to_string(), v.clone());
        Ok(v)
    }

    /// Decode a single contiguous tensor row as f32 (PQ2_0, f32 or f16).
    /// PQ2_0 rows are the future token-embedding / LM-head lookup path.
    pub fn row_f32(&mut self, name: &str, row: u64) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?.clone();
        let ne0 = *t.dims.first().ok_or("Weights: row_f32 on empty-dims tensor")? as usize;
        let n_rows = t.n_elem() as u64 / ne0 as u64;
        if row >= n_rows {
            return Err(format!(
                "Weights: row_f32({name}): row {row} out of range (n_rows {n_rows})"
            ));
        }
        match t.ty {
            TYPE_PQ2_0 => {
                let row_bytes = kernels::pq2_row_bytes(ne0);
                let raw = self.gguf.slice_at(
                    self.gguf.tensor_data_offset(&t) + row * row_bytes as u64,
                    row_bytes,
                )?;
                Ok(kernels::decode_pq2_0_row(raw, ne0))
            }
            TYPE_F32 => {
                let raw = self.gguf.slice_at(
                    self.gguf.tensor_data_offset(&t) + row * ne0 as u64 * 4,
                    ne0 * 4,
                )?;
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            TYPE_F16 => {
                let raw = self.gguf.slice_at(
                    self.gguf.tensor_data_offset(&t) + row * ne0 as u64 * 2,
                    ne0 * 2,
                )?;
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| crate::gguf::half_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            ty => Err(format!(
                "Weights: row_f32({name}): unsupported tensor type {ty}"
            )),
        }
    }

    /// Full PQ2_0 matrix-vector product: y = W @ x over every row.
    pub fn matvec(&mut self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?.clone();
        let rows = kernels::n_rows(&t) as usize;
        let mut y = vec![0.0f32; rows];
        self.matvec_into(&t, 0, rows, x, &mut y)?;
        Ok(y)
    }

    /// `y[..n_rows] = W[base_row..base_row+n_rows] @ x` for a PQ2_0 matrix.
    /// Letting callers slice the fused projections (q|gate, q|k|v) by rows.
    pub fn matvec_into(
        &mut self,
        t: &TensorInfo,
        base_row: usize,
        n_rows: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        let t_name = t.name.clone();
        if t.ty != TYPE_PQ2_0 {
            return Err(format!(
                "Weights: matvec on '{t_name}': tensor type {} is not PQ2_0",
                t.ty
            ));
        }
        let ne0 = t.dims[0] as usize;
        if x.len() != ne0 {
            return Err(format!(
                "Weights: matvec {t_name}: x length {} != ne0 {ne0}",
                x.len()
            ));
        }
        let total_rows = kernels::n_rows(t) as usize;
        if base_row + n_rows > total_rows {
            return Err(format!(
                "Weights: matvec {t_name}: row range {base_row}..{} exceeds n_rows {total_rows}",
                base_row + n_rows
            ));
        }
        if y.len() < n_rows {
            return Err(format!(
                "Weights: matvec {t_name}: y buffer too small ({} < {n_rows})",
                y.len()
            ));
        }
        if let Some(a) = self.accel.as_mut() {
            if let Some(buf) = a.dev.get(&t_name) {
                let yv = a.gpu.matvec_on(buf, ne0, base_row as u32, n_rows, x)?;
                y[..n_rows].copy_from_slice(&yv);
                return Ok(());
            }
        }
        let payload = self.gguf.payload_slice(t)?;
        kernels::pq2_matvec_range(&payload, ne0, base_row as u64, n_rows, x, y)
    }

    /// Name-based `matvec_into`: fetch, validate, and run a row slice.
    pub fn matvec_named_into(
        &mut self,
        name: &str,
        base_row: usize,
        n_rows: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        let t = self.tensor(name)?.clone();
        self.matvec_into(&t, base_row, n_rows, x, y)
    }

    /// Check every expected per-layer tensor against the file index (type and
    /// dims). Returns the number of tensors verified.
    pub fn verify(&self) -> Result<usize, String> {
        let mut errs = Vec::new();
        let mut n = 0usize;
        for il in 0..self.cfg.n_layer {
            for (name, ty, dims) in self.cfg.expected_layer_tensors(il) {
                n += 1;
                match self.tensor(&name) {
                    Err(_) => errs.push(format!("missing tensor '{name}'")),
                    Ok(t) => {
                        if t.ty != ty {
                            errs.push(format!("{name}: type {} != expected {ty}", t.ty));
                        }
                        if t.dims != dims {
                            errs.push(format!(
                                "{name}: dims {:?} != expected {:?}",
                                t.dims, dims
                            ));
                        }
                    }
                }
            }
        }
        if errs.is_empty() {
            Ok(n)
        } else {
            let mut msg = format!("{}/{} tensor checks failed:", errs.len(), n);
            for e in errs.iter().take(8) {
                msg.push_str(&format!("\n  - {e}"));
            }
            if errs.len() > 8 {
                msg.push_str(&format!("\n  ... and {} more", errs.len() - 8));
            }
            Err(msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GGUF;
    use std::io::Write;

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

    // ---- tiny GGUF writer (test-only) --------------------------------------
    fn w_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn w_u64(out: &mut Vec<u8>, v: u64) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn w_f32(out: &mut Vec<u8>, v: f32) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn w_str(out: &mut Vec<u8>, s: &str) {
        w_u64(out, s.len() as u64);
        out.extend_from_slice(s.as_bytes());
    }
    fn kv(out: &mut Vec<u8>, key: &str, ty: u32, payload: &[u8]) {
        w_str(out, key);
        w_u32(out, ty);
        out.extend_from_slice(payload);
    }
    fn kv_u32_owned(key: &str, v: u32) -> Vec<u8> {
        let mut out = Vec::new();
        kv(&mut out, key, 4, &v.to_le_bytes()); // GGUF_VALUE_TYPE_UINT32
        out
    }
    fn kv_f32_owned(key: &str, v: f32) -> Vec<u8> {
        let mut out = Vec::new();
        kv(&mut out, key, 6, &v.to_le_bytes()); // GGUF_VALUE_TYPE_FLOAT32
        out
    }
    fn kv_str_owned(key: &str, v: &str) -> Vec<u8> {
        let mut out = Vec::new();
        w_str(&mut out, key);
        w_u32(&mut out, 8); // GGUF_VALUE_TYPE_STRING
        w_str(&mut out, v);
        out
    }
    fn kv_arr_u32_owned(key: &str, v: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        w_str(&mut out, key);
        w_u32(&mut out, 9); // GGUF_VALUE_TYPE_ARRAY
        w_u32(&mut out, 4); // element type uint32
        w_u64(&mut out, v.len() as u64);
        for &x in v {
            w_u32(&mut out, x);
        }
        out
    }
    fn kv_u64_owned(key: &str, v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        kv(&mut out, key, 10, &v.to_le_bytes()); // GGUF_VALUE_TYPE_UINT64
        out
    }

    /// Exact f32 -> f16 for the small power-of-two scales used in tests.
    fn f16(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        assert!(exp > 0 && exp < 31, "f16 test scale out of range: {v}");
        let mant = (b & 0x7f_ffff) >> 13;
        sign | (((exp as u16) << 10) | mant as u16)
    }

    fn payload_f32(seed: u64, n: usize) -> Vec<u8> {
        let mut b = Vec::with_capacity(n * 4);
        for i in 0..n {
            let x = (((i as u64 * 31 + seed * 7) % 251) as f32 - 125.0) / 64.0;
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    fn payload_pq2(seed: u64, ne0: usize, n_rows: usize) -> Vec<u8> {
        let blocks = ne0.div_ceil(128);
        let mut b = Vec::with_capacity(blocks * n_rows * 34);
        for r in 0..n_rows {
            for blk in 0..blocks {
                let scale = 0.5 * ((r as u64 + blk as u64 + seed) % 7 + 1) as f32;
                b.extend_from_slice(&f16(scale).to_le_bytes());
                let mut qs = [0u8; 32];
                for j in 0..128 {
                    let code = ((r * 128 + j + blk * 17) % 4) as u8;
                    qs[j / 4] |= code << ((j % 4) * 2);
                }
                b.extend_from_slice(&qs);
            }
        }
        b
    }

    fn tensor_nbytes(ty: u32, dims: &[u64]) -> u64 {
        match ty {
            TYPE_F32 => dims.iter().product::<u64>() * 4,
            TYPE_PQ2_0 => {
                let ne0 = dims[0];
                let rows: u64 = dims[1..].iter().product();
                rows * ne0.div_ceil(128) * 34
            }
            _ => unimplemented!("test writer only emits f32/pq2"),
        }
    }

    fn align32(v: u64) -> u64 {
        (v + 31) / 32 * 32
    }

    /// Serialize magic/version/counts/meta + tensor index. Returns the header
    /// bytes (padded to the aligned data start) and each tensor's data offset.
    fn build_header(
        specs: &[(String, u32, Vec<u64>)],
        meta: &[u8],
        kv_count: u64,
    ) -> (Vec<u8>, Vec<u64>) {
        let mut info_len = 0u64;
        for (name, _, dims) in specs {
            info_len += 8 + name.len() as u64 + 4 + 8 * dims.len() as u64 + 4 + 8;
        }
        let header_len = 4 + 4 + 8 + 8 + meta.len() as u64 + info_len;
        let data_start = align32(header_len);

        // GGUF tensor offsets are relative to the data section start.
        let mut off = 0u64;
        let mut offsets = Vec::with_capacity(specs.len());
        for (_, ty, dims) in specs {
            offsets.push(off);
            off += align32(tensor_nbytes(*ty, dims));
        }

        let mut h = Vec::new();
        h.extend_from_slice(b"GGUF");
        w_u32(&mut h, 3);
        w_u64(&mut h, specs.len() as u64);
        w_u64(&mut h, kv_count);
        h.extend_from_slice(meta);
        for ((name, ty, dims), offset) in specs.iter().zip(offsets.iter()) {
            w_str(&mut h, name);
            w_u32(&mut h, dims.len() as u32);
            for d in dims {
                w_u64(&mut h, *d);
            }
            w_u32(&mut h, *ty);
            w_u64(&mut h, *offset);
        }
        while h.len() < data_start as usize {
            h.push(0);
        }
        assert_eq!(h.len() as u64, data_start);
        (h, offsets)
    }

    fn write_mini(path: &std::path::Path, cfg: &Qwen35, skip_substr: Option<&str>) {
        let mut specs: Vec<(String, u32, Vec<u64>)> = Vec::new();
        for il in 0..cfg.n_layer {
            for s in cfg.expected_layer_tensors(il) {
                if let Some(sk) = skip_substr {
                    if s.0.contains(sk) {
                        continue;
                    }
                }
                specs.push(s);
            }
        }

        // metadata keys must round-trip through Qwen35::from_gguf
        let mut meta = Vec::new();
        let mut kv_count = 0u64;
        macro_rules! pm {
            ($bytes:expr) => {{
                kv_count += 1;
                meta.extend_from_slice(&$bytes);
            }};
        }
        pm!(kv_str_owned("general.architecture", "qwen35"));
        pm!(kv_u32_owned("qwen35.block_count", cfg.n_layer as u32));
        pm!(kv_u32_owned("qwen35.embedding_length", cfg.n_embd as u32));
        pm!(kv_u32_owned("qwen35.feed_forward_length", cfg.n_ff as u32));
        pm!(kv_u32_owned("qwen35.attention.head_count", cfg.n_head as u32));
        pm!(kv_u32_owned("qwen35.attention.head_count_kv", cfg.n_head_kv as u32));
        pm!(kv_u32_owned("qwen35.attention.key_length", cfg.n_embd_head as u32));
        pm!(kv_u32_owned(
            "qwen35.full_attention_interval",
            cfg.interval as u32
        ));
        pm!(kv_f32_owned(
            "qwen35.attention.layer_norm_rms_epsilon",
            cfg.eps
        ));
        pm!(kv_u64_owned("qwen35.context_length", cfg.n_ctx_train));
        pm!(kv_u32_owned("qwen35.rope.dimension_count", cfg.n_rot as u32));
        pm!(kv_arr_u32_owned(
            "qwen35.rope.dimension_sections",
            &cfg.sections.map(|x| x as u32)
        ));
        pm!(kv_f32_owned("qwen35.rope.freq_base", cfg.freq_base));
        pm!(kv_u32_owned(
            "qwen35.ssm.conv_kernel",
            cfg.ssm_conv_kernel as u32
        ));
        pm!(kv_u32_owned("qwen35.ssm.state_size", cfg.ssm_state as u32));
        pm!(kv_u32_owned(
            "qwen35.ssm.group_count",
            cfg.ssm_group_count as u32
        ));
        pm!(kv_u32_owned(
            "qwen35.ssm.time_step_rank",
            cfg.ssm_dt_rank as u32
        ));
        pm!(kv_u32_owned("qwen35.ssm.inner_size", cfg.ssm_inner as u32));

        let (header, _offsets) = build_header(&specs, &meta, kv_count);

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&header).unwrap();
        for (i, (_, ty, dims)) in specs.iter().enumerate() {
            let mut payload: Vec<u8> = match *ty {
                TYPE_F32 => payload_f32(1 + i as u64, dims.iter().product::<u64>() as usize),
                TYPE_PQ2_0 => {
                    let ne0 = dims[0] as usize;
                    let rows: usize = dims[1..].iter().map(|x| *x as usize).product();
                    payload_pq2(1 + i as u64, ne0, rows)
                }
                _ => unreachable!(),
            };
            let want = tensor_nbytes(*ty, dims) as usize;
            assert_eq!(payload.len(), want, "payload size mismatch");
            while (payload.len() as u64) < align32(want as u64) {
                payload.push(0);
            }
            f.write_all(&payload).unwrap();
        }
    }

    fn test_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = std::env::temp_dir().join(format!("bonsai-weights-{tag}-{nanos}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn mini_model_opens_and_verifies() {
        let dir = test_dir("ok");
        let path = dir.join("mini.gguf");
        let cfg = mini_cfg();
        write_mini(&path, &cfg, None);

        let mut w = Weights::open(path.to_str().unwrap()).unwrap();
        assert_eq!(*w.config(), cfg);

        let n = w.verify().expect("mini model should verify");
        // 4 layers: il3 full attention (11 tensors), il0..2 recurrent (14 each)
        assert_eq!(n, 11 + 3 * 14);

        // vec_f32 mapping: f32 tensor values reachable under the right name
        let norm = w.vec_f32("blk.0.attn_norm.weight").unwrap();
        assert_eq!(norm.len(), cfg.n_embd);
        // independent oracle: payload writer emits (seed 1) values directly
        // (not through the GGUF reader), so this guards data-start offsets.
        let expect: Vec<f32> = (0..cfg.n_embd)
            .map(|i| (((i as u64 * 31 + 7) % 251) as f32 - 125.0) / 64.0)
            .collect();
        assert_eq!(norm, expect, "vec_f32 must read real payload bytes");
        let mut raw = GGUF::open(path.to_str().unwrap()).unwrap();
        let t = raw
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_norm.weight")
            .unwrap()
            .clone();
        assert_eq!(norm, raw.read_tensor(&t).unwrap());

        // matvec mapping: full PQ2_0 matrix product under the right name
        let x = vec![0.5f32; cfg.n_embd];
        let y = w.matvec("blk.0.ffn_up.weight", &x).unwrap();
        assert_eq!(y.len(), cfg.n_ff);
        let t2 = raw
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_up.weight")
            .unwrap()
            .clone();
        let mut ref_y = vec![0.0f32; cfg.n_ff];
        let t2_ne0 = t2.dims[0] as usize;
        let payload2 = raw.payload_slice(&t2).unwrap();
        kernels::pq2_matvec_range(&payload2, t2_ne0, 0, cfg.n_ff, &x, &mut ref_y).unwrap();
        assert_eq!(y, ref_y);

        // row_f32 agrees with the raw kernel row path
        let row = w.row_f32("blk.3.attn_q.weight", 7).unwrap();
        assert_eq!(row.len(), cfg.n_embd);
        let t3 = raw
            .tensors
            .iter()
            .find(|t| t.name == "blk.3.attn_q.weight")
            .unwrap()
            .clone();
        let ne0 = t3.dims[0] as usize;
        let row_bytes = kernels::pq2_row_bytes(ne0);
        let mut raw_bytes = vec![0u8; row_bytes];
        raw.read_bytes(raw.tensor_data_offset(&t3) + 7 * row_bytes as u64, &mut raw_bytes)
            .unwrap();
        assert_eq!(row, kernels::decode_pq2_0_row(&raw_bytes, ne0));

        // expect_tensor catches a dims mismatch on a real entry
        assert!(
            w.expect_tensor("blk.0.ssm_norm.weight", TYPE_F32, &[cfg.ssm_state as u64]).is_ok()
        );
        assert!(
            w.expect_tensor(
                "blk.3.attn_output.weight",
                TYPE_PQ2_0,
                &[128, cfg.n_embd as u64]
            )
            .is_ok()
        );
        assert!(w.expect_tensor("blk.0.ssm_norm.weight", TYPE_F32, &[999]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_tensor_is_reported() {
        let dir = test_dir("missing");
        let path = dir.join("mini_bad.gguf");
        write_mini(&path, &mini_cfg(), Some("post_attention_norm"));
        let w = Weights::open(path.to_str().unwrap()).unwrap();
        let err = w.verify().unwrap_err();
        assert!(err.contains("missing tensor") && err.contains("post_attention_norm"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrong_arch_is_rejected() {
        let dir = test_dir("arch");
        let path = dir.join("mini_arch.gguf");
        write_mini(&path, &mini_cfg(), None);
        // patch the architecture string in place: "qwen35" -> "qwenXX" (same len)
        let mut out = std::fs::read(&path).unwrap();
        let needle = b"qwen35";
        let mut found = false;
        for i in 0..out.len() - 6 {
            if &out[i..i + 6] == needle {
                out[i + 4..i + 6].copy_from_slice(b"XX");
                found = true;
                break;
            }
        }
        assert!(found, "architecture string not found for patching");
        std::fs::write(&path, out).unwrap();
        let err = match Weights::open(path.to_str().unwrap()) {
            Ok(_) => panic!("wrong-arch model should not open"),
            Err(e) => e,
        };
        assert!(err.contains("qwen35"), "unexpected error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
