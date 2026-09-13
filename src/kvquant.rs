//! KV-cache rotation + scalar quantization (RotorQuant / PlanarQuant family).
//!
//! See `notes/kv-rotorquant-plan.md`. Per KV vector `x` (one head, `hd` dims):
//!
//!   1. split off the norm: `n = ||x||`, `xhat = x / n`
//!   2. rotate each adjacent pair by a fixed random Givens angle
//!      `(cos t, sin t)`: `[x0', x1'] = [c*x0 - s*x1, s*x0 + c*x1]`
//!   3. scalar-quantize each rotated coordinate to the Lloyd-Max codebook for
//!      `2^bits` levels (a coordinate of a rotated unit d-vector is
//!      `Beta((d-3)/2)` on [-1,1], approximated `N(0, 1/d)`)
//!   4. pack the indices at `bits` each and store `n`
//!
//! Reconstruction: centroid lookup -> inverse Givens -> `* n`.
//!
//! Two things make this cheap in attention:
//!   - K is stored in rotated space, so no inverse rotation is ever needed for
//!     scores: rotate q once and dot against the centroid values.
//!   - V is reconstructed by inverse-rotating the *weighted sum* once per
//!     token, not per cached position.

#![allow(dead_code)]

use std::sync::OnceLock;

/// Default head dimension this codebook is designed for (qwen35 full attn).
pub const DEFAULT_HD: usize = 256;

// ---------------------------------------------------------------------------
// Lloyd-Max codebook
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun 7.1.26, |error| <= 1.5e-7. Enough to design a
/// quantizer; not used on the hot path.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn norm_cdf(x: f64, sigma: f64) -> f64 {
    0.5 * (1.0 + erf(x / (sigma * std::f64::consts::SQRT_2)))
}

fn norm_pdf(x: f64, sigma: f64) -> f64 {
    (-(x * x) / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

/// Integral of `x * pdf(x)` over `[a, b]` for `N(0, sigma^2)`, using
/// `d/dx pdf = -(x/sigma^2) pdf`, so the antiderivative is `-sigma^2 pdf`.
fn gauss_first_moment(a: f64, b: f64, sigma: f64) -> f64 {
    sigma * sigma * (norm_pdf(a, sigma) - norm_pdf(b, sigma))
}

/// Solve the Lloyd-Max (1-D k-means) conditions for `N(0, sigma^2)`.
/// Returns `2^bits` centroids, ascending.
pub fn solve_lloyd_max(sigma: f64, bits: u32) -> Vec<f32> {
    let n = 1usize << bits;
    let lo = -3.5 * sigma;
    let hi = 3.5 * sigma;
    let mut c: Vec<f64> = (0..n)
        .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n as f64)
        .collect();
    // Wide outer edges stand in for +-inf (the tail mass there is ~1e-30).
    let outer = 12.0 * sigma;
    for _ in 0..300 {
        let mut edges = Vec::with_capacity(n + 1);
        edges.push(-outer);
        for i in 0..n - 1 {
            edges.push((c[i] + c[i + 1]) / 2.0);
        }
        edges.push(outer);
        let mut next = Vec::with_capacity(n);
        for i in 0..n {
            let (a, b) = (edges[i], edges[i + 1]);
            let mass = norm_cdf(b, sigma) - norm_cdf(a, sigma);
            if mass > 1e-15 {
                next.push(gauss_first_moment(a, b, sigma) / mass);
            } else {
                next.push(c[i]);
            }
        }
        let shift = next
            .iter()
            .zip(&c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        c = next;
        if shift < 1e-12 {
            break;
        }
    }
    c.iter().map(|&v| v as f32).collect()
}

/// Lloyd-Max centroids for a unit `d`-vector coordinate, dimension `d` and
/// `bits` per coordinate. Cached per `(d, bits)`.
pub fn codebook(d: usize, bits: u32) -> &'static [f32] {
    static CB: OnceLock<Vec<((usize, u32), Vec<f32>)>> = OnceLock::new();
    let all = CB.get_or_init(|| {
        let sigma = 1.0 / (d as f64).sqrt();
        (1u32..=8).map(|b| ((d, b), solve_lloyd_max(sigma, b))).collect()
    });
    &all
        .iter()
        .find(|(k, _)| *k == (d, bits))
        .expect("codebook not precomputed for this (d, bits)")
        .1
}

// ---------------------------------------------------------------------------
// Fixed Givens rotation table
// ---------------------------------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fixed random `(cos t, sin t)` per 2D group, generated once from a fixed
/// seed. Shared by every layer and head (decision R0.1 in the plan); escalate
/// to per-(layer, head) tables only if quality demands it.
pub fn givens_table(d: usize) -> &'static [(f32, f32)] {
    static T: OnceLock<Vec<(f32, f32)>> = OnceLock::new();
    T.get_or_init(|| {
        let n_groups = d.div_ceil(2);
        let mut state: u64 = 42;
        (0..n_groups)
            .map(|_| {
                let r = splitmix64(&mut state);
                let u = (r >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
                let t = u * std::f64::consts::TAU;
                (t.cos() as f32, t.sin() as f32)
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

/// Pack `bits`-wide indices little-endian into `out`.
pub fn pack_bits(idx: &[u8], bits: u32, out: &mut [u8]) {
    out.fill(0);
    let mut bitpos = 0usize;
    for &v in idx {
        let byte = bitpos >> 3;
        let off = (bitpos & 7) as u32;
        let v = (v as u32) & ((1u32 << bits) - 1);
        out[byte] |= ((v << off) & 0xff) as u8;
        if off + bits > 8 {
            out[byte + 1] |= (v >> (8 - off)) as u8;
        }
        bitpos += bits as usize;
    }
}

/// Inverse of `pack_bits`.
pub fn unpack_bits(packed: &[u8], bits: u32, n: usize, out: &mut [u8]) {
    let mask = (1u32 << bits) - 1;
    let mut bitpos = 0usize;
    for o in out.iter_mut().take(n) {
        let byte = bitpos >> 3;
        let off = (bitpos & 7) as u32;
        let mut v = (packed[byte] as u32) >> off;
        if off + bits > 8 {
            v |= (packed[byte + 1] as u32) << (8 - off);
        }
        *o = (v & mask) as u8;
        bitpos += bits as usize;
    }
}

/// Bytes needed to pack `n` indices at `bits` each.
pub fn packed_bytes(n: usize, bits: u32) -> usize {
    (n * bits as usize).div_ceil(8)
}

// ---------------------------------------------------------------------------
// The quantizer
// ---------------------------------------------------------------------------

/// Planar (2D Givens + Lloyd-Max) quantizer for one vector length.
#[derive(Clone)]
pub struct PlanarQuant {
    pub hd: usize,
    pub bits: u32,
    pub n_groups: usize,
    /// `hd` rounded up to an even count (pairs).
    pub hd_padded: usize,
}

impl PlanarQuant {
    pub fn new(hd: usize, bits: u32) -> PlanarQuant {
        assert!((1..=8).contains(&bits), "bits must be 1..8");
        let n_groups = hd.div_ceil(2);
        PlanarQuant {
            hd,
            bits,
            n_groups,
            hd_padded: n_groups * 2,
        }
    }

    /// Number of bytes to store the packed indices of one vector.
    pub fn packed_len(&self) -> usize {
        packed_bytes(self.hd_padded, self.bits)
    }

    fn centroids(&self) -> &'static [f32] {
        codebook(self.hd, self.bits)
    }

    /// Nearest centroid index for a rotated coordinate.
    fn nearest(&self, v: f32) -> u8 {
        let c = self.centroids();
        // centroids are ascending; linear scan is fine at 8/16 levels
        let mut best = 0usize;
        let mut bd = (v - c[0]).abs();
        for (i, &cv) in c.iter().enumerate().skip(1) {
            let d = (v - cv).abs();
            if d < bd {
                bd = d;
                best = i;
            }
        }
        best as u8
    }

    /// Forward Givens rotation, element 2i/2i+1 by table[i].
    pub fn rotate(&self, x: &[f32], out: &mut [f32]) {
        let t = givens_table(self.hd);
        for i in 0..self.n_groups {
            let (c, s) = t[i];
            let a = if 2 * i < x.len() { x[2 * i] } else { 0.0 };
            let b = if 2 * i + 1 < x.len() { x[2 * i + 1] } else { 0.0 };
            out[2 * i] = c * a - s * b;
            out[2 * i + 1] = s * a + c * b;
        }
    }

    /// Inverse Givens rotation (sin negated).
    pub fn rotate_inv(&self, x: &[f32], out: &mut [f32]) {
        let t = givens_table(self.hd);
        for i in 0..self.n_groups {
            let (c, s) = t[i];
            let a = if 2 * i < x.len() { x[2 * i] } else { 0.0 };
            let b = if 2 * i + 1 < x.len() { x[2 * i + 1] } else { 0.0 };
            out[2 * i] = c * a + s * b;
            out[2 * i + 1] = -s * a + c * b;
        }
    }

    /// Quantize `x` into `packed` (indices, rotated space) plus its norm.
    pub fn quantize(&self, x: &[f32], packed: &mut [u8], norm: &mut f32) {
        let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        *norm = n;
        let inv = if n > 1e-12 { 1.0 / n } else { 0.0 };
        let mut rot = vec![0.0f32; self.hd_padded];
        let mut unit = vec![0.0f32; self.hd_padded];
        for i in 0..self.hd {
            unit[i] = x[i] * inv;
        }
        self.rotate(&unit, &mut rot);
        let mut idx = vec![0u8; self.hd_padded];
        for (i, &v) in rot.iter().enumerate() {
            idx[i] = self.nearest(v);
        }
        pack_bits(&idx, self.bits, packed);
    }

    /// Reconstruct `xhat` from packed indices and the stored norm.
    pub fn dequantize(&self, packed: &[u8], norm: f32, out: &mut [f32]) {
        let c = self.centroids();
        let mut idx = vec![0u8; self.hd_padded];
        unpack_bits(packed, self.bits, self.hd_padded, &mut idx);
        let mut rot = vec![0.0f32; self.hd_padded];
        for (i, &j) in idx.iter().enumerate() {
            rot[i] = c[j as usize];
        }
        let mut un = vec![0.0f32; self.hd_padded];
        self.rotate_inv(&rot, &mut un);
        for i in 0..self.hd {
            out[i] = un[i] * norm;
        }
    }

    /// Unpack packed indices into centroid values, using caller scratch.
    pub fn unpack_centroids_into(&self, packed: &[u8], idx: &mut [u8], out: &mut [f32]) {
        let c = self.centroids();
        unpack_bits(packed, self.bits, self.hd_padded, idx);
        for i in 0..self.hd_padded {
            out[i] = c[idx[i] as usize];
        }
    }

    /// In-place inverse rotation. Safe because each output pair depends only on
    /// the same input pair.
    pub fn rotate_inv_inplace(&self, x: &mut [f32]) {
        let t = givens_table(self.hd);
        for i in 0..self.n_groups {
            let (c, s) = t[i];
            let a = x[2 * i];
            let b = x[2 * i + 1];
            x[2 * i] = c * a + s * b;
            x[2 * i + 1] = -s * a + c * b;
        }
    }

    /// `dot(q_rotated, dequantized)_` where the cache holds rotated K.
    /// `q_rot` must already be forward-rotated with the same table; the result
    /// equals `q . k_original` up to quantization error.
    pub fn dot_rotated(&self, q_rot: &[f32], packed: &[u8], norm: f32) -> f32 {
        let c = self.centroids();
        let mut idx = vec![0u8; self.hd_padded];
        unpack_bits(packed, self.bits, self.hd_padded, &mut idx);
        let mut acc = 0.0f32;
        for i in 0..self.hd {
            acc += q_rot[i] * c[idx[i] as usize];
        }
        acc * norm
    }
}

/// KV quantization selected by `BONSAI_KV`, returning `(k_quant, v_quant)`:
///   unset / `f32`     -> (None, None)   full-precision cache (the anchor)
///   `planar3`/`iso3`  -> both K and V (symmetric)
///   `planar4`/`iso4`  -> both K and V
///   `planarNk`        -> K only, V stays f32 (asymmetric, lower risk)
pub fn from_env(hd: usize) -> (Option<PlanarQuant>, Option<PlanarQuant>) {
    let Ok(v) = std::env::var("BONSAI_KV") else {
        return (None, None);
    };
    if v.is_empty() || v == "f32" {
        return (None, None);
    }
    let (name, k_only) = match v.strip_suffix('k') {
        Some(base) => (base, true),
        None => (v.as_str(), false),
    };
    let bits = match name {
        "planar3" | "iso3" => 3,
        "planar4" | "iso4" => 4,
        other => match other
            .strip_prefix("planar")
            .or_else(|| other.strip_prefix("iso"))
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|b| (1..=8).contains(b))
        {
            Some(b) => b,
            None => return (None, None),
        },
    };
    let k = PlanarQuant::new(hd, bits);
    let vq = if k_only {
        None
    } else {
        Some(PlanarQuant::new(hd, bits))
    };
    (Some(k), vq)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    #[test]
    fn centroids_are_sorted_symmetric_and_sized() {
        for bits in [3u32, 4u32] {
            let c = codebook(DEFAULT_HD, bits);
            assert_eq!(c.len(), 1 << bits);
            for w in c.windows(2) {
                assert!(w[0] < w[1], "centroids must ascend: {c:?}");
            }
            // symmetric about zero
            let n = c.len();
            for i in 0..n / 2 {
                assert!(
                    (c[i] + c[n - 1 - i]).abs() < 1e-6,
                    "not symmetric: {} vs {}",
                    c[i],
                    c[n - 1 - i]
                );
            }
            // lives in the +-3.5 sigma band (sigma = 1/16 for d=256)
            assert!(c[n - 1] < 4.0 / 16.0, "outermost centroid {}", c[n - 1]);
        }
    }

    #[test]
    fn lloyd_max_beats_uniform() {
        // 4-bit Lloyd-Max distortion should be below 3-bit, and both below the
        // variance of the source (sigma^2 = 1/256).
        let sigma2 = 1.0f32 / DEFAULT_HD as f32;
        let d3 = super::kvquant_expected_distortion(DEFAULT_HD, 3);
        let d4 = super::kvquant_expected_distortion(DEFAULT_HD, 4);
        if std::env::var("KVQ_DIAG").is_ok() {
            println!("KVQ_DIAG var={sigma2:.6} d3={d3:.6} ({:.1}%) d4={d4:.6} ({:.1}%)",
                100.0*(d3/sigma2).sqrt(), 100.0*(d4/sigma2).sqrt());
            println!("KVQ_DIAG c3={:?}", super::codebook(DEFAULT_HD,3));
            println!("KVQ_DIAG c4={:?}", super::codebook(DEFAULT_HD,4));
        }
        assert!(d3 < sigma2 && d4 < d3, "d3={d3} d4={d4} var={sigma2}");
    }

    #[test]
    fn givens_is_orthogonal() {
        let p = PlanarQuant::new(DEFAULT_HD, 4);
        let mut next = rng(7);
        let x: Vec<f32> = (0..p.hd).map(|_| next()).collect();
        let mut rot = vec![0.0f32; p.hd_padded];
        let mut back = vec![0.0f32; p.hd_padded];
        p.rotate(&x, &mut rot);
        p.rotate_inv(&rot, &mut back);
        for i in 0..p.hd {
            assert!((back[i] - x[i]).abs() < 1e-5, "i={i} {} vs {}", back[i], x[i]);
        }
    }

    #[test]
    fn quantize_roundtrip_error_is_small() {
        let p = PlanarQuant::new(DEFAULT_HD, 4);
        let mut next = rng(11);
        let mut worst_rel = 0.0f32;
        for _ in 0..64 {
            let x: Vec<f32> = (0..p.hd).map(|_| next()).collect();
            let mut packed = vec![0u8; p.packed_len()];
            let mut norm = 0.0f32;
            p.quantize(&x, &mut packed, &mut norm);
            let mut y = vec![0.0f32; p.hd];
            p.dequantize(&packed, norm, &mut y);
            let err: f32 = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let xn: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            worst_rel = worst_rel.max(err / xn.max(1e-9));
        }
        // 4-bit: relative L2 error should be a few percent
        assert!(worst_rel < 0.12, "worst relative L2 error {worst_rel}");
    }

    #[test]
    fn dot_rotated_matches_dequantized_dot() {
        let p = PlanarQuant::new(DEFAULT_HD, 4);
        let mut next = rng(23);
        let k: Vec<f32> = (0..p.hd).map(|_| next()).collect();
        let q: Vec<f32> = (0..p.hd).map(|_| next()).collect();
        let mut packed = vec![0u8; p.packed_len()];
        let mut norm = 0.0f32;
        p.quantize(&k, &mut packed, &mut norm);
        let mut khat = vec![0.0f32; p.hd];
        p.dequantize(&packed, norm, &mut khat);
        let mut qrot = vec![0.0f32; p.hd_padded];
        p.rotate(&q, &mut qrot);
        let a = p.dot_rotated(&qrot, &packed, norm);
        let b: f32 = q.iter().zip(&khat).map(|(a, b)| a * b).sum();
        assert!((a - b).abs() < 1e-4, "dot {a} vs {b}");
    }

    #[test]
    fn print_relative_error_per_bits() {
        if std::env::var("KVQ_BITS_DIAG").is_err() { return; }
        let mut next = rng(99);
        for bits in 1..=8u32 {
            let p = PlanarQuant::new(DEFAULT_HD, bits);
            let mut worst = 0.0f32;
            for _ in 0..32 {
                let x: Vec<f32> = (0..p.hd).map(|_| next()).collect();
                let xn: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
                let mut packed = vec![0u8; p.packed_len()];
                let mut norm = 0.0;
                p.quantize(&x, &mut packed, &mut norm);
                let mut y = vec![0.0f32; p.hd];
                p.dequantize(&packed, norm, &mut y);
                let err: f32 = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                worst = worst.max(err / xn.max(1e-9));
            }
            println!("KVQ_BITS_DIAG bits={bits} worst_rel_l2={worst:.6}");
        }
    }

    #[test]
    fn bit_packing_roundtrips_both_widths() {
        for bits in [3u32, 4u32] {
            let n = 256usize;
            let mut next = rng(5 + bits as u64);
            let idx: Vec<u8> = (0..n)
                .map(|_| ((next().abs() * 1000.0) as u32 % (1 << bits)) as u8)
                .collect();
            let mut packed = vec![0u8; packed_bytes(n, bits)];
            pack_bits(&idx, bits, &mut packed);
            let mut back = vec![0u8; n];
            unpack_bits(&packed, bits, n, &mut back);
            assert_eq!(idx, back, "bits={bits}");
        }
    }
}

/// Expected per-coordinate MSE of the Lloyd-Max codebook (test/diagnostic).
pub fn kvquant_expected_distortion(d: usize, bits: u32) -> f32 {
    let sigma = 1.0 / (d as f64).sqrt();
    let c = codebook(d, bits);
    let n = c.len();
    let mut edges = Vec::with_capacity(n + 1);
    let outer = 12.0 * sigma;
    edges.push(-outer);
    for i in 0..n - 1 {
        edges.push((c[i] as f64 + c[i + 1] as f64) / 2.0);
    }
    edges.push(outer);
    let mut dist = 0.0f64;
    for i in 0..n {
        let (a, b) = (edges[i], edges[i + 1]);
        let c0 = c[i] as f64;
        // E[(X-c)^2] over [a,b] = E[X^2] - 2c E[X] + c^2
        // E[X^2] = sigma^2 (Phi(b)-Phi(a)) + sigma^2 (a f(a) - b f(b)) ... use
        // the same antiderivative trick numerically via the second moment:
        let mass = norm_cdf(b, sigma) - norm_cdf(a, sigma);
        let m1 = gauss_first_moment(a, b, sigma);
        let m2 = sigma * sigma * (mass + a * norm_pdf(a, sigma) - b * norm_pdf(b, sigma));
        dist += m2 - 2.0 * c0 * m1 + c0 * c0 * mass;
    }
    dist as f32
}
