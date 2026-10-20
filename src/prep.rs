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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use julia_rs::{
        Model,
        attn::{Rope, Scratch, attention},
        matmul::{linear, linear_add},
        mlp::geglu,
        sequence::sequence,
    };

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

    /// The residual stream entering each layer's attention and MLP LayerNorm, from julia-rs's
    /// layers (`Model::encode`, which keeps only the final output).
    fn residuals(model: &Model, tokens: &[u32]) -> Vec<[Vec<f32>; 2]> {
        let (n, d, inter, eps) = (
            tokens.len(),
            model.config.hidden,
            model.config.intermediate,
            model.config.norm_eps,
        );
        let data = model.data();
        let mut x = model.embed(tokens);
        let (mut h, mut o, mut qkv) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * 3 * d]);
        let (mut hh, mut m) = (vec![0.0; n * 2 * inter], vec![0.0; n * inter]);
        let mut s = Scratch::default();
        let rope_g = Rope::new(model.config.rope_theta_global, n);
        let rope_l = Rope::new(model.config.rope_theta_local, n);
        let mut out = vec![];
        for (l, p) in model
            .weights
            .encoder
            .layers
            .iter()
            .zip(&model.packed.encoder)
        {
            let before_attn = x.clone();
            match &l.attn_norm {
                Some(g) => layernorm_rows(&x, g.as_f32(data), eps, &mut h),
                None => h.copy_from_slice(&x),
            }
            linear(&h, &p.wqkv, None, &mut qkv);
            let (rope, window) = match l.global {
                true => (&rope_g, None),
                false => (&rope_l, Some(model.config.local_window / 2)),
            };
            rope.apply(&mut qkv, d);
            attention(&qkv, n, d, window, &mut o, &mut s);
            linear_add(&o, &p.wo, None, &mut x);
            let before_mlp = x.clone();
            layernorm_rows(&x, l.mlp_norm.as_f32(data), eps, &mut h);
            linear(&h, &p.wi, None, &mut hh);
            geglu(&hh, inter, &mut m);
            linear_add(&m, &p.wo_mlp, None, &mut x);
            out.push([before_attn, before_mlp]);
        }
        out
    }

    /// Bias-free LayerNorm of each `gamma.len()`-wide row, in f64.
    fn layernorm_rows(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
        let k = gamma.len();
        for (row, out) in x.chunks_exact(k).zip(out.chunks_exact_mut(k)) {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / k as f64;
            let var = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / k as f64;
            let r = 1.0 / (var + f64::from(eps)).sqrt();
            for ((o, &v), &g) in out.iter_mut().zip(row).zip(gamma) {
                *o = ((f64::from(v) - mean) * r * f64::from(g)) as f32;
            }
        }
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
        let residuals = residuals(&model, &seq.ids);
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
                layernorm_rows(x, gamma, eps, &mut h);
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
