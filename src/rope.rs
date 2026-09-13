//! Rust port of ggml interleaved M-RoPE (IMROPE) used by qwen35 full-attention
//! layers. Scalar reference: ggml_mrope_cache_init + rotate_pairs in
//! ggml/src/ggml-cpu/ops.cpp with mode GGML_ROPE_TYPE_IMROPE.

/// Apply interleaved M-RoPE in place to one head vector.
///
/// qwen35: head dim 256, n_dims 64 (rotated dims), sections [11,11,10,0].
/// The four position ids are the text position for pure-text decoding.
pub fn rope_imrope(
    head: &mut [f32],
    p_t: i64,
    p_h: i64,
    p_w: i64,
    p_e: i64,
    n_dims: usize,
    sections: [i32; 4],
    freq_base: f32,
) {
    debug_assert!(n_dims <= head.len());
    debug_assert!(n_dims % 2 == 0);

    let sect_dims = (sections[0] + sections[1] + sections[2] + sections[3]) as i64;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);

    let mut th = p_t as f32;
    let mut hh = p_h as f32;
    let mut wh = p_w as f32;
    let mut eh = p_e as f32;

    let s0 = sections[0] as i64; // 11
    let s1 = sections[1] as i64; // 11
    let s2 = sections[2] as i64; // 10

    // precompute cos/sin cache for n_dims/2 pairs (n_dims entries)
    let mut cache = vec![0.0f32; n_dims];
    for i0 in (0..n_dims).step_by(2) {
        let sector = (i0 as i64 / 2) % sect_dims;

        let mut theta = th;
        // is_imrope band selection
        if sector % 3 == 1 && sector < 3 * s1 {
            theta = hh;
        } else if sector % 3 == 2 && sector < 3 * s2 {
            theta = wh;
        } else if sector % 3 == 0 && sector < 3 * s0 {
            theta = th;
        } else {
            theta = eh;
        }

        cache[i0] = theta.cos();
        cache[i0 + 1] = theta.sin();

        th *= theta_scale;
        hh *= theta_scale;
        wh *= theta_scale;
        eh *= theta_scale;
    }

    // NEOX-style application over the first n_dims dims: pair (j, j + n_dims/2)
    let half = n_dims / 2;
    for j in 0..half {
        let cos = cache[2 * j];
        let sin = cache[2 * j + 1];
        let x0 = head[j];
        let x1 = head[j + half];
        head[j] = x0 * cos - x1 * sin;
        head[j + half] = x0 * sin + x1 * cos;
    }
}

/// Plain NEOX RoPE (single position id), used by the dspark drafter's attention
/// (`rope_type = NEOX` for a DFlash backbone without DSV4 hyper-connections).
///
/// Rotates the first `n_dims` components of one head vector in place, pairing
/// `(j, j + n_dims/2)` with `theta_j = pos * freq_base^(-2j/n_dims)`.
pub fn rope_neox(head: &mut [f32], pos: f32, n_dims: usize, freq_base: f32) {
    debug_assert!(n_dims <= head.len() && n_dims % 2 == 0);
    let half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos;
    for j in 0..half {
        let (sin, cos) = theta.sin_cos();
        let x0 = head[j];
        let x1 = head[j + half];
        head[j] = x0 * cos - x1 * sin;
        head[j + half] = x0 * sin + x1 * cos;
        theta *= theta_scale;
    }
}

#[cfg(test)]
mod neox_tests {
    use super::rope_neox;

    #[test]
    fn neox_pos0_is_identity() {
        let mut h = vec![1.0, 2.0, 3.0, 4.0];
        let orig = h.clone();
        rope_neox(&mut h, 0.0, 4, 10000.0);
        for (a, b) in h.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn neox_preserves_pair_norm() {
        let mut h = vec![0.7, -0.3, 1.1, 0.2];
        let n0 = h[0] * h[0] + h[2] * h[2];
        let n1 = h[1] * h[1] + h[3] * h[3];
        rope_neox(&mut h, 3.5, 4, 10000.0);
        assert!((h[0] * h[0] + h[2] * h[2] - n0).abs() < 1e-5);
        assert!((h[1] * h[1] + h[3] * h[3] - n1).abs() < 1e-5);
    }
}
