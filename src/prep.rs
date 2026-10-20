//! Weight preparation: the checkpoint's f32 weights in the layout the kernels expect. Runs once,
//! at load.

use half::f16;

/// A linear layer whose input LayerNorm (weights `γ`, no bias) is folded in, so the GEMM can run
/// on the raw residual stream:
///
/// ```text
/// LN(x) · Wᵀ = r · (x · W'ᵀ - μ · colsum)    W' = W · diag(γ),  colsum[j] = Σₖ W'[j, k]
/// ```
///
/// with `μ` and `r = 1 / √(var + ε)` the statistics of the row of `x`.
pub struct FoldedLinear {
    /// `W'`, `[out, in]` like the checkpoint.
    pub w: Vec<f16>,
    /// `Σₖ W'[j, k]` of the f16 `W'` (what the GEMM multiplies with), one per output.
    pub colsum: Vec<f32>,
}

/// Folds `gamma` into `w` (`[out, in]`, `in == gamma.len()`).
pub fn fold_layernorm(w: &[f32], gamma: &[f32]) -> FoldedLinear {
    let k = gamma.len();
    assert!(w.len().is_multiple_of(k));
    let w: Vec<f16> = (w.chunks_exact(k))
        .flat_map(|row| row.iter().zip(gamma).map(|(&w, &g)| f16::from_f32(w * g)))
        .collect();
    // From the rounded weights, so μ · colsum cancels the μ part of x · W'ᵀ as the GEMM
    // computes it. Julia's residual rows have |μ| < 0.1 σ, so f32 sums measure the same (see
    // the test); it would matter for a model whose rows have a large mean.
    let colsum = (w.chunks_exact(k))
        .map(|row| row.iter().map(|&w| f64::from(w)).sum::<f64>() as f32)
        .collect();
    FoldedLinear { w, colsum }
}

/// Wi's rows (`[n, k]`, n = 2·intermediate: the GELU inputs, then the gates) reordered for the
/// GEMM's fused GeGLU epilogue. In each 64-row tile, a warp's 32 rows are its four n8 fragments:
/// fragments 2q and 2q + 1 hold the same 8 output columns' input and gate rows. So packed row
/// p = 64t + 32h + 16q + 8s + e is source row 32t + 16h + 8q + e (+ intermediate for the gate,
/// s = 1), and output column 32t + 16h + 8q + e. `k = 1` reorders a per-row vector (colsum).
pub fn interleave_geglu<T: Copy>(w: &[T], k: usize) -> Vec<T> {
    let n = w.len() / k;
    assert!(w.len() == n * k && n.is_multiple_of(64));
    let inter = n / 2;
    let mut out = Vec::with_capacity(w.len());
    for p in 0..n {
        let (t, h, q, s, e) = (p / 64, (p % 64) / 32, (p % 32) / 16, (p % 16) / 8, p % 8);
        let src = t * 32 + h * 16 + q * 8 + e + if s == 1 { inter } else { 0 };
        out.extend_from_slice(&w[src * k..][..k]);
    }
    out
}

/// Dimensions per attention head.
pub const HEAD_DIM: usize = 64;

/// Wqkv's rows (`[3d, k]`: q, k and v, each `d` rows of heads of 64) reordered for the GEMM's
/// RoPE epilogue: in each q and k head, packed rows 2i and 2i + 1 are dimensions i and i + 32,
/// the pair RoPE rotates together (rotate-half), so one thread holds both. V stays in order.
/// q and k can stay permuted in memory: attention only takes their dot products, which the
/// same permutation of both doesn't change. `k = 1` reorders a per-row vector (colsum).
pub fn permute_rope<T: Copy>(w: &[T], k: usize) -> Vec<T> {
    let n = w.len() / k;
    assert!(w.len() == n * k && n.is_multiple_of(3 * HEAD_DIM));
    let mut out = Vec::with_capacity(w.len());
    for p in 0..n {
        let src = if p < 2 * n / 3 {
            let (head, i) = (p / HEAD_DIM, p % HEAD_DIM);
            head * HEAD_DIM + i / 2 + (i % 2) * (HEAD_DIM / 2)
        } else {
            p
        };
        out.extend_from_slice(&w[src * k..][..k]);
    }
    out
}

/// RoPE's (cos, sin) for positions `0..n` and pairs `0..32`, `[n][32][2]`: computed as julia-rs
/// (and HF's `ModernBertRotaryEmbedding`) does, in f32.
pub fn rope_table(theta: f32, n: usize) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let inv: Vec<f32> = (0..half)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / HEAD_DIM as f32))
        .collect();
    (0..n)
        .flat_map(|p| {
            inv.iter().flat_map(move |&f| {
                let a = p as f32 * f;
                [a.cos(), a.sin()]
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use julia_rs::{Model, sequence::sequence};

    use super::*;
    use crate::check::rel_err;

    /// The checkpoint, from `JULIA_MODEL` or ./model; `None` (test skipped) if it isn't there.
    fn model() -> Option<Model> {
        let dir = std::env::var_os("JULIA_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("model"));
        if !dir.join("model.safetensors").exists() {
            eprintln!(
                "skipped: no checkpoint at {} (set JULIA_MODEL)",
                dir.display()
            );
            return None;
        }
        Some(crate::check::reference_model(&dir).unwrap())
    }

    /// `x · wᵀ`, f64 accumulation (`x` is `[n, k]`, `w` `[out, k]`).
    fn matmul(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
        (x.chunks_exact(k))
            .flat_map(|row| {
                w.chunks_exact(k).map(|wr| {
                    row.iter()
                        .zip(wr)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>() as f32
                })
            })
            .collect()
    }

    fn f16_round(v: &[f32]) -> Vec<f32> {
        v.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
    }

    /// What the GPU computes for a folded linear: `x` stored as f16, row statistics as f32
    /// sums of the stored values (var = Σx²/K - μ²), an f16 GEMM with f32 accumulation, then
    /// the epilogue `r · (acc - μ · colsum)` rounded to f16. `colsum` is passed separately so a
    /// variant can be tried.
    fn folded_gpu(x: &[f32], f: &FoldedLinear, colsum: &[f32], eps: f32) -> Vec<f32> {
        let k = f.w.len() / colsum.len();
        let x = f16_round(x);
        let w: Vec<f32> = f.w.iter().map(|w| w.to_f32()).collect();
        let acc = matmul(&x, &w, k);
        let mut out = vec![0.0; acc.len()];
        for ((row, acc), out) in (x.chunks_exact(k))
            .zip(acc.chunks_exact(colsum.len()))
            .zip(out.chunks_exact_mut(colsum.len()))
        {
            let (s, s2) = row
                .iter()
                .fold((0f32, 0f32), |(s, s2), &v| (s + v, v.mul_add(v, s2)));
            let mean = s / k as f32;
            let r = 1.0 / (s2 / k as f32 - mean * mean + eps).sqrt();
            for ((o, &a), &c) in out.iter_mut().zip(acc).zip(colsum) {
                *o = r * (-mean).mul_add(c, a);
            }
        }
        f16_round(&out)
    }

    /// The fold against julia-rs's `LN(x) · Wᵀ` (in f64), on real residual streams at several
    /// depths, for both folded GEMMs (qkv after the attention LayerNorm, wi after the MLP one).
    /// "f16" is the same computation unfolded with f16 inputs and outputs: the error fp16 costs
    /// anyway. The fold should add little to it.
    #[test]
    fn fold_matches_layernorm_then_linear() {
        let Some(model) = model() else { return };
        let req = crate::sample_request();
        let seq = sequence(&model.tokenizer, &req, 1024, 512).unwrap();
        let (d, eps, data) = (model.config.hidden, model.config.norm_eps, model.data());
        let residuals: Vec<_> = (crate::check::trace(&model, &seq.ids).into_iter())
            .map(|t| [t.x_in, t.x_mid])
            .collect();
        eprintln!("{} tokens", seq.ids.len());
        eprintln!("layer  gemm  |mean|/std   f16       fold      fold, f32 colsum");
        for l in [1, 5, 11, 16, 21] {
            let layer = &model.weights.encoder.layers[l];
            let gemms = [
                (
                    "qkv",
                    &residuals[l][0],
                    layer.attn_norm.as_ref().unwrap(),
                    &layer.wqkv,
                ),
                ("wi", &residuals[l][1], &layer.mlp_norm, &layer.wi),
            ];
            for (name, x, gamma, w) in gemms {
                let (gamma, w) = (gamma.as_f32(data), w.as_f32(data));
                let mut h = vec![0.0; x.len()];
                crate::check::layernorm_rows(x, gamma, eps, &mut h);
                let reference = matmul(&h, w, d);

                let plain = f16_round(&matmul(&f16_round(&h), &f16_round(w), d));
                let folded = fold_layernorm(w, gamma);
                let fold = folded_gpu(x, &folded, &folded.colsum, eps);
                // The variant the comment in `fold_layernorm` warns about.
                let colsum_f32: Vec<f32> = (w.chunks_exact(d))
                    .map(|row| {
                        row.iter()
                            .zip(gamma)
                            .map(|(&w, &g)| f64::from(w * g))
                            .sum::<f64>() as f32
                    })
                    .collect();
                let fold_f32 = folded_gpu(x, &folded, &colsum_f32, eps);

                let (e_plain, e_fold) = (rel_err(&plain, &reference), rel_err(&fold, &reference));
                eprintln!(
                    "{l:>5}  {name:<4}  {:>10.2}   {e_plain:.2e}  {e_fold:.2e}  {:.2e}",
                    mean_over_std(x, d),
                    rel_err(&fold_f32, &reference),
                );
                assert!(
                    e_fold < 2.0 * e_plain,
                    "layer {l} {name}: fold error {e_fold:.2e} vs f16 {e_plain:.2e}"
                );
            }
        }
    }

    /// The interleave against the kernel's GeGLU epilogue (kernels/src/main.rs): for block column
    /// bx, warp column wc = warp & 1, lane pair e0 = (lane & 3)·2 and q < 2, the GELU input is
    /// packed row cw + 16q + e (fragment j = 2q), the gate cw + 16q + 8 + e (fragment 2q + 1),
    /// with cw = 64·bx + 32·wc + e0, and the output column 32·bx + 16·wc + e0 + 8q + e.
    #[test]
    fn interleave_matches_the_geglu_epilogue() {
        let (n, inter) = (2304, 1152);
        let rows: Vec<usize> = interleave_geglu(&(0..n).collect::<Vec<_>>(), 1);
        let mut seen = vec![false; inter];
        for bx in 0..n / 64 {
            for wc in 0..2 {
                for lane in 0..32 {
                    let e0 = (lane & 3) * 2;
                    let cw = 64 * bx + 32 * wc + e0;
                    for q in 0..2 {
                        for e in 0..2 {
                            let col = 32 * bx + 16 * wc + e0 + 8 * q + e;
                            assert_eq!(rows[cw + 16 * q + e], col, "input of column {col}");
                            assert_eq!(rows[cw + 16 * q + 8 + e], inter + col, "gate of {col}");
                            seen[col] = true;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every output column is produced");
    }

    /// The permutation against the RoPE epilogue: in q and k, packed columns 2i, 2i + 1 of each
    /// head are its dimensions i and i + 32; v is unchanged.
    #[test]
    fn rope_permutation_pairs_rotate_half_dims() {
        let n = 3 * 384;
        let rows = permute_rope(&(0..n).collect::<Vec<_>>(), 1);
        for p in (0..2 * 384).step_by(2) {
            let (head, i) = (p / 64, (p % 64) / 2);
            assert_eq!((rows[p], rows[p + 1]), (64 * head + i, 64 * head + i + 32));
        }
        assert!((2 * 384..n).all(|p| rows[p] == p));
    }

    /// Largest |mean| / std over the rows: how much the fold's cancellation has to cancel.
    fn mean_over_std(x: &[f32], k: usize) -> f64 {
        (x.chunks_exact(k))
            .map(|row| {
                let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / k as f64;
                let var = row
                    .iter()
                    .map(|&v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / k as f64;
                mean.abs() / var.sqrt()
            })
            .fold(0.0, f64::max)
    }
}
