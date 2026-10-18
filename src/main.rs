use std::path::PathBuf;

use color_eyre::eyre::eyre;
use cudarc::driver::CudaContext;
use julia_rs::{ModelPaths, load_model, matmul::Precision};

const USAGE: &str = "usage: julia-gpu <checkpoint dir>";

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let dir = PathBuf::from(std::env::args().nth(1).ok_or_else(|| eyre!(USAGE))?);
    let path = |p: &str| dir.join(p);
    let model = load_model(
        ModelPaths {
            safetensors_path: &path("model.safetensors"),
            encoder_config_json_path: &path("encoder/config.json"),
            julia_config_json_path: &path("julia_config.json"),
            tokenizer_json_path: &path("tokenizer/tokenizer.json"),
            tokenizer_config_json_path: &path("tokenizer/tokenizer_config.json"),
        },
        Precision::F32,
    )?;
    let c = &model.config;
    eprintln!(
        "model: {} layers, hidden {}, intermediate {}",
        model.weights.encoder.layers.len(),
        c.hidden,
        c.intermediate
    );

    let ctx = CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    eprintln!("gpu: {} (sm_{major}{minor})", ctx.name()?);
    Ok(())
}
