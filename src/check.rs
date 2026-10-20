//! `--check`: compares the GPU's results with julia-rs, the f32 reference.
//!
//! Each kernel is compared twice: with julia-rs, to see what f16 costs (expect a relative error
//! around 1e-3 at most), and with a host emulation of the kernel's own arithmetic (f16 inputs,
//! f32 math, f16 output), to catch bugs: there, only summation order differs, so values may be off
//! by an f16 ulp but no more.

use std::path::Path;

use color_eyre::eyre::ensure;
use half::f16;
use julia_rs::{
    Model, ModelPaths, load_model,
    matmul::{Precision, linear},
};

use crate::{D, weights::HostWeights};

/// julia-rs's model, for its reference outputs.
pub fn reference_model(dir: &Path) -> color_eyre::Result<Model> {
    let path = |p: &str| dir.join(p);
    load_model(
        ModelPaths {
            safetensors_path: &path("model.safetensors"),
            encoder_config_json_path: &path("encoder/config.json"),
            julia_config_json_path: &path("julia_config.json"),
            tokenizer_json_path: &path("tokenizer/tokenizer.json"),
            tokenizer_config_json_path: &path("tokenizer/tokenizer_config.json"),
        },
        Precision::F32,
    )
}

/// ‖a - b‖ / ‖b‖ over the whole tensor.
pub fn rel_err(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&a, &b) in a.iter().zip(b) {
        num += (f64::from(a) - f64::from(b)).powi(2);
        den += f64::from(b).powi(2);
    }
    (num / den).sqrt()
}

/// How many f16 steps apart `a` and `b` are (0 for equal, ±0 counting as equal).
fn ulps(a: f16, b: f16) -> u32 {
    // f16 bit patterns ordered like the values: negatives below zero, mirrored.
    let ordered = |x: f16| {
        let b = x.to_bits();
        if b & 0x8000 != 0 {
            -i32::from(b & 0x7fff)
        } else {
            i32::from(b)
        }
    };
    ordered(a).abs_diff(ordered(b))
}

/// What `embed` computes, on the host: the f16 table row, a two-pass LayerNorm in f32, rounded to
/// f16. Only the summation order differs from the kernel's.
fn emulate_embed(host: &HostWeights, ids: &[u32], eps: f32) -> Vec<f16> {
    let table = &host.f16[host.layout.tok_embeddings.range()];
    let gamma = &host.f32[host.layout.emb_norm.range()];
    let mut x = Vec::with_capacity(ids.len() * D);
    for &id in ids {
        let v: Vec<f32> = table[id as usize * D..][..D]
            .iter()
            .map(|h| h.to_f32())
            .collect();
        let mean = v.iter().sum::<f32>() / D as f32;
        let var = v.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>() / D as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        x.extend(
            v.iter()
                .zip(gamma)
                .map(|(&a, &g)| f16::from_f32((a - mean) * rstd * g)),
        );
    }
    x
}

/// `embed`'s output `x` for `ids` against julia-rs's `Model::embed` and against the emulation.
pub fn embed(
    model: &Model,
    host: &HostWeights,
    ids: &[u32],
    eps: f32,
    x: &[f16],
) -> color_eyre::Result<()> {
    let got: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let rel = rel_err(&got, &model.embed(ids));
    let (mut off_by_one, mut worst) = (0, 0);
    for (&want, &got) in emulate_embed(host, ids, eps).iter().zip(x) {
        let u = ulps(want, got);
        off_by_one += usize::from(u == 1);
        worst = worst.max(u);
    }
    eprintln!(
        "embed: vs julia-rs {rel:.2e} relative; vs emulation {off_by_one} of {} values 1 ulp off, \
         worst {worst} ulp",
        x.len()
    );
    ensure!(rel < 2e-3, "embed is {rel:.2e} off julia-rs");
    ensure!(worst <= 1, "embed is {worst} ulp off the emulation");
    Ok(())
}

/// What `gemm` computes, on the host: `x · wᵀ` from the same f16 inputs, summed in f64 and rounded
/// to f16. `x` is `[m, k]`, `w` `[n, k]`.
fn emulate_gemm(x: &[f16], w: &[f16], k: usize) -> Vec<f16> {
    let w: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
    (x.chunks_exact(k))
        .flat_map(|row| {
            let row: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
            w.chunks_exact(k)
                .map(move |wr| f16::from_f64(row.iter().zip(wr).map(|(a, b)| a * b).sum()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Layer 0's qkv GEMM output `qkv` (from `embed`'s output `x`) against julia-rs (its f32 embedding
/// times its f32 Wqkv) and against the emulation on the same f16 `x`. The kernel sums 384 terms in
/// f32, the emulation in f64, so values near zero can be a few ulps apart; the relative error is
/// what passes or fails.
pub fn qkv0(
    model: &Model,
    host: &HostWeights,
    ids: &[u32],
    x: &[f16],
    qkv: &[f16],
) -> color_eyre::Result<()> {
    let got: Vec<f32> = qkv.iter().map(|v| v.to_f32()).collect();
    let mut want = vec![0.0; ids.len() * 3 * D];
    linear(
        &model.embed(ids),
        &model.packed.encoder[0].wqkv,
        None,
        &mut want,
    );
    let rel = rel_err(&got, &want);

    let emu = emulate_gemm(x, &host.f16[host.layout.layers[0].wqkv.range()], D);
    let emu_f32: Vec<f32> = emu.iter().map(|v| v.to_f32()).collect();
    let rel_emu = rel_err(&got, &emu_f32);
    let worst = emu.iter().zip(qkv).map(|(&a, &b)| ulps(a, b)).max();
    eprintln!(
        "qkv (layer 0): vs julia-rs {rel:.2e} relative; vs emulation {rel_emu:.2e} relative, \
         worst {} ulp",
        worst.unwrap_or(0)
    );
    ensure!(rel < 2e-3, "qkv is {rel:.2e} off julia-rs");
    ensure!(rel_emu < 1e-3, "qkv is {rel_emu:.2e} off the emulation");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use julia_rs::sequence::{DEFAULT_HEAD_LENGTH, sequence};

    use super::*;
    use crate::checkpoint::Checkpoint;

    /// The checks themselves: the emulation, standing in for a correct kernel, has to pass, and
    /// its distance from julia-rs is what f16 alone costs.
    #[test]
    fn embed_check_passes_the_emulation() {
        let dir = std::env::var_os("JULIA_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("model"));
        if !dir.join("model.safetensors").exists() {
            eprintln!(
                "skipped: no checkpoint at {} (set JULIA_MODEL)",
                dir.display()
            );
            return;
        }
        let (ck, model) = (
            Checkpoint::open(&dir).unwrap(),
            reference_model(&dir).unwrap(),
        );
        let host = HostWeights::prepare(&ck);
        let c = &ck.config;
        let seq = sequence(
            &ck.tokenizer,
            &crate::sample_request(),
            c.max_positions,
            DEFAULT_HEAD_LENGTH,
        )
        .unwrap();
        let x = emulate_embed(&host, &seq.ids, c.norm_eps);
        embed(&model, &host, &seq.ids, c.norm_eps, &x).unwrap();

        // And a wrong result fails: one value off by two f16 steps.
        let mut bad = x.clone();
        bad[100] = f16::from_bits(bad[100].to_bits() + 2);
        assert!(embed(&model, &host, &seq.ids, c.norm_eps, &bad).is_err());

        // The qkv check, the same way: the emulation passes, and so do plain f32 sums in k order,
        // which round like the kernel's f32 accumulators...
        let wqkv = &host.f16[host.layout.layers[0].wqkv.range()];
        let qkv = emulate_gemm(&x, wqkv, D);
        qkv0(&model, &host, &seq.ids, &x, &qkv).unwrap();
        let f32_sums: Vec<f16> = (x.as_chunks::<D>().0.iter())
            .flat_map(|row| {
                wqkv.as_chunks::<D>().0.iter().map(move |wr| {
                    let s: f32 = row
                        .iter()
                        .zip(wr)
                        .map(|(a, b)| a.to_f32() * b.to_f32())
                        .sum();
                    f16::from_f32(s)
                })
            })
            .collect();
        qkv0(&model, &host, &seq.ids, &x, &f32_sums).unwrap();
        // ...and the bugs a tiling mistake makes fail: a 64-row block of a tile left at zero, and
        // two columns swapped.
        let mut zeroed = qkv.clone();
        for r in 0..seq.ids.len() {
            zeroed[r * 3 * D + 64..][..64].fill(f16::ZERO);
        }
        assert!(qkv0(&model, &host, &seq.ids, &x, &zeroed).is_err());
        let mut swapped = qkv.clone();
        for r in 0..seq.ids.len() {
            swapped.swap(r * 3 * D + 3, r * 3 * D + 19);
        }
        assert!(qkv0(&model, &host, &seq.ids, &x, &swapped).is_err());
    }
}
