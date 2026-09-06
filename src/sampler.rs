//! Pure-Rust sampling for Bonsai-flavored generation.
//!
//! Replaces the llama.cpp `llama_sampler_*` chain: top-k, top-p (nucleus),
//! min-p, temperature, then seeded categorical sampling. Operates on the raw
//! logits row returned by `llama_get_logits_ith`.

/// SplitMix64: small, fast, seedable RNG (public domain).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    pub top_k: i32,   // <= 0 disables
    pub top_p: f32,   // 1.0 disables nucleus cut
    pub min_p: f32,   // 0.0 disables
    pub temp: f32,    // <= 0 uses greedy
    pub seed: u64,    // 0xFFFF_FFFF => random from time
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            top_k: 20,
            top_p: 0.9,
            min_p: 0.0,
            temp: 0.6,
            seed: 0xFFFF_FFFF,
        }
    }
}

pub struct Sampler {
    rng: Rng,
}

impl Sampler {
    pub fn new(cfg: &SamplerConfig) -> Self {
        let seed = if cfg.seed == 0xFFFF_FFFF {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9)
        } else {
            cfg.seed
        };
        Sampler { rng: Rng::new(seed) }
    }

    /// Pick the next token id from one row of logits (length n_vocab).
    pub fn sample(&mut self, logits: &[f32], cfg: &SamplerConfig) -> i32 {
        if logits.is_empty() {
            return 0;
        }
        if cfg.temp <= 0.0 {
            // greedy: argmax
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            return best as i32;
        }

        let inv_t = 1.0 / cfg.temp;

        // 1) collect scored candidates (all vocab), tracking top-k threshold.
        let k = if cfg.top_k > 0 {
            cfg.top_k as usize
        } else {
            usize::MAX
        };

        // Quick top-k threshold via partial select: iterate keeping a max-heap
        // of the k largest would be O(n log k); simpler: full arg-sort on id is
        // avoided here by keeping scores in a Vec and doing a partial selection.
        let mut order: Vec<usize> = (0..logits.len()).collect();
        // Reorder only enough for top-k: use select_nth_unstable_by for the k-th
        // largest, then keep the first k.
        if k < logits.len() {
            order.select_nth_unstable_by(k, |&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            order.truncate(k);
        }

        // 2) temperature scaling
        let mut scores: Vec<(usize, f32)> = order
            .iter()
            .map(|&i| (i, logits[i] * inv_t))
            .collect();

        // 3) softmax over candidates
        let max_s = scores
            .iter()
            .map(|&(_, s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for (_, s) in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s as f64;
        }
        let inv_sum = 1.0 / sum;
        for (_, s) in scores.iter_mut() {
            *s = (*s as f64 * inv_sum) as f32;
        }

        // 4) min-p: drop tokens with prob < min_p * max_prob
        if cfg.min_p > 0.0 && cfg.min_p < 1.0 {
            let max_p = scores.iter().map(|&(_, p)| p).fold(0.0f32, f32::max);
            let cutoff = cfg.min_p * max_p;
            scores.retain(|&(_, p)| p >= cutoff);
        }
        if scores.is_empty() {
            return order[0] as i32;
        }

        // 5) top-p nucleus: candidates are in descending probability order.
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep = scores.len();
        if cfg.top_p < 1.0 {
            let mut acc = 0.0f32;
            for (i, &(_, p)) in scores.iter().enumerate() {
                acc += p;
                if acc >= cfg.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        scores.truncate(keep);

        // 6) seeded categorical draw over the surviving candidates
        let total: f32 = scores.iter().map(|&(_, p)| p).sum();
        let r = self.rng.next_f64() * total as f64;
        let mut acc = 0.0f64;
        for &(id, p) in &scores {
            acc += p as f64;
            if acc >= r {
                return id as i32;
            }
        }
        scores.last().map(|&(id, _)| id as i32).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let s = SamplerConfig { temp: 0.0, ..Default::default() };
        let mut smp = Sampler::new(&s);
        let logits = vec![-3.0, 5.0, 1.0, 2.0];
        assert_eq!(smp.sample(&logits, &s), 1);
    }

    #[test]
    fn deterministic_with_seed() {
        let cfg = SamplerConfig { seed: 42, temp: 0.8, top_k: 20, ..Default::default() };
        // arbitrary but fixed logits
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0).sin()).collect();
        let mut a = Sampler::new(&cfg);
        let mut b = Sampler::new(&cfg);
        for _ in 0..5 {
            assert_eq!(a.sample(&logits, &cfg), b.sample(&logits, &cfg));
        }
    }

    #[test]
    fn result_is_in_vocab_range() {
        let cfg = SamplerConfig { seed: 1, ..Default::default() };
        let mut smp = Sampler::new(&cfg);
        let logits: Vec<f32> = (0..248_320).map(|i| (i % 997) as f32 / 997.0).collect();
        let id = smp.sample(&logits, &cfg);
        assert!((0..248_320).contains(&(id as usize)));
    }
}
