//! Rust port of the qwen35 gated-delta-net (linear attention) recurrent step.
//!
//! Scalar reference: ggml_compute_forward_gated_delta_net_one_chunk in
//! ggml/src/ggml-cpu/ops.cpp (f32 path). Shapes for qwen35:
//!   S_k = S_v = 128, H_k = 16, H_v = 48, single sequence, decode token by token.
//!
//! State is stored per v-head as the transposed matrix M[j][i] = S[i][j]
//! (row j of M contiguous, matching ggml). All f32.

/// Per-token recurrent update for one sequence.
///
/// q, k: H_k x S vectors (flat, row per k-head)
/// v:    H_v x S vectors (flat)
/// gate, beta: length H_v scalars (per v-head)
/// state: H_v x (S*S) transposed matrices, in-place updated
/// attn:  H_v x S output buffer filled by this step
pub fn gdn_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    attn: &mut [f32],
) {
    const S: usize = 128;
    const H_K: usize = 16;
    const H_V: usize = 48;

    let scale = 1.0 / (S as f32).sqrt();

    for h in 0..H_V {
        let k_head = h % H_K;
        let q_head = h % H_K;
        let qv = &q[q_head * S..q_head * S + S];
        let kv = &k[k_head * S..k_head * S + S];
        let vv = &v[h * S..h * S + S];
        let s = &mut state[h * S * S..(h + 1) * S * S];

        // state matrix decay: M[j][i] *= exp(gate)
        let eg = gate[h].exp();
        for x in s.iter_mut() {
            *x *= eg;
        }

        // delta[j] = (v[j] - dot(row_j(M), k)) * beta
        let mut delta = [0.0f32; S];
        for j in 0..S {
            let mut sum = 0.0f32;
            let row = &s[j * S..(j + 1) * S];
            for i in 0..S {
                sum += row[i] * kv[i];
            }
            delta[j] = (vv[j] - sum) * beta[h];
        }

        // outer update: M[j][i] += k[i] * delta[j]
        for j in 0..S {
            let dj = delta[j];
            let row = &mut s[j * S..(j + 1) * S];
            for i in 0..S {
                row[i] += kv[i] * dj;
            }
        }

        // attn_out[j] = dot(row_j(M), q) * scale
        let out = &mut attn[h * S..(h + 1) * S];
        for j in 0..S {
            let mut sum = 0.0f32;
            let row = &s[j * S..(j + 1) * S];
            for i in 0..S {
                sum += row[i] * qv[i];
            }
            out[j] = sum * scale;
        }
    }
}

/// Initialize a zero state for one sequence.
pub fn zero_state() -> Vec<f32> {
    vec![0.0; 48 * 128 * 128]
}
