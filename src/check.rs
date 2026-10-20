//! `--check`: compares the GPU's results with julia-rs, the f32 reference.
//!
//! Each kernel is compared twice: with julia-rs, to see what f16 costs (expect a relative error
//! around 1e-3 at most), and with a host emulation of the kernel's own arithmetic (f16 inputs,
//! f32 math, f16 output), to catch bugs: there, only summation order differs, so values may be off
//! by an f16 ulp but no more.

use std::path::Path;

use color_eyre::eyre::ensure;
use half::f16;
use julia_rs::{Model, ModelPaths, load_model, matmul::Precision};

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
    }
}
