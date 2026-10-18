use std::{path::PathBuf, sync::Arc};

use color_eyre::eyre::{Context, bail, ensure};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use julia_rs::{ModelPaths, load_model, matmul::Precision};

const USAGE: &str = "usage: julia-cuda [checkpoint dir]
  Without a checkpoint, only checks that the kernels load and run on GPU 0.";

/// Generated with `embed_ptx!`.
struct EmbeddedPtx {
    /// "sm_89"
    tier: &'static str,
    /// The oldest compute capability that can run it (89 for sm_89).
    sm: u32,
    /// The source of ptx/{tier}/kernels.ptx
    source: &'static str,
}

/// Returns 120 for "sm_120".
const fn sm_number(tier: &str) -> u32 {
    let b = tier.as_bytes();
    assert!(
        b.len() > 3 && b[0] == b's' && b[1] == b'm' && b[2] == b'_',
        "Feature must start with sm_"
    );
    let (mut i, mut n) = (3, 0);
    while i < b.len() {
        assert!(b[i].is_ascii_digit(), "SM number must be digits");
        n = n * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    n
}

/// Defines `PTX` as ptx/<tier>/kernels.ptx for the first tier whose feature is enabled, falling
/// back on the last one, and `PTX_TIERS` as all of them, newest first.
macro_rules! embed_ptx {
    ($($feature:literal),+ $(,)?) => {
        const PTX: EmbeddedPtx = embed_ptx!(@step $($feature),+);
        const PTX_TIERS: &[&str] = &[$($feature),+];
    };
    (@step $fallback:literal) => {
        EmbeddedPtx {
            tier: $fallback,
            sm: sm_number($fallback),
            source: include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/ptx/", $fallback, "/kernels.ptx")),
        }
    };
    (@step $head:literal, $($tail:literal),+) => {{
        #[cfg(feature = $head)]
        {
            embed_ptx!(@step $head)
        }
        #[cfg(not(feature = $head))]
        {
            embed_ptx!(@step $($tail),+)
        }
    }};
}

embed_ptx!("sm_120", "sm_89", "sm_80", "sm_75");

/// Loads the embedded PTX, after checking that this GPU can run its tier. The driver
/// JIT-compiles PTX forward (sm_80 PTX runs on sm_89) but never backward.
fn load_kernels(ctx: &Arc<CudaContext>) -> color_eyre::Result<Arc<CudaModule>> {
    let (major, minor) = ctx.compute_capability()?;
    let (name, sm) = (ctx.name()?, (major * 10 + minor) as u32);
    // The newest tier this GPU can run, i.e. what the binary should have been built with.
    let best = PTX_TIERS.iter().find(|&&t| sm_number(t) <= sm);
    let rebuild = |t: &str| format!("make dist ARCH={}", sm_number(t));
    ensure!(
        sm >= PTX.sm,
        "this binary has {} kernels, but {name} is sm_{sm}; {}",
        PTX.tier,
        match best {
            Some(t) => format!("rebuild with `{}`", rebuild(t)),
            None => format!("the oldest tier is {}", PTX_TIERS[PTX_TIERS.len() - 1]),
        }
    );
    let module = ctx
        .load_module(Ptx::from_src(PTX.source))
        .wrap_err_with(|| format!("loading the {} kernels on {name} (sm_{sm})", PTX.tier))?;
    eprintln!("gpu: {name} (sm_{sm}), {} kernels", PTX.tier);
    if let Some(&t) = best.filter(|&&t| t != PTX.tier) {
        eprintln!(
            "note: {t} kernels would suit this GPU better (`{}`)",
            rebuild(t)
        );
    }
    Ok(module)
}

/// Runs `axpy` on N elements (not a multiple of the block size, so the last block has idle
/// threads) and checks the result bit for bit against the CPU.
fn axpy_selftest(stream: &Arc<CudaStream>, kernels: &Arc<CudaModule>) -> color_eyre::Result<()> {
    const N: usize = 1000;
    const BLOCK: u32 = 256;
    // Fills y past N; the kernel is told y has N elements, so these must come back untouched.
    const GUARD: f32 = -1234.5;
    let grid = (N as u32).div_ceil(BLOCK);
    let threads = (grid * BLOCK) as usize;

    let a = 0.7f32;
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.37 - 100.0).collect();
    let mut y: Vec<f32> = (0..threads).map(|i| 1.0 / (i + 1) as f32).collect();
    y[N..].fill(GUARD);

    let axpy = kernels.load_function("axpy")?;
    let x_dev = stream.clone_htod(&x)?;
    let mut y_dev = stream.clone_htod(&y)?;
    // `arg` keeps references until the launch, so the lengths need names.
    let (x_len, y_len) = (N as u64, N as u64);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(&axpy);
    // axpy(a: f32, x: &[f32], y: DisjointSlice<f32>) in PTX: a, x ptr, x len, y ptr, y len.
    launch
        .arg(&a)
        .arg(&x_dev)
        .arg(&x_len)
        .arg(&mut y_dev)
        .arg(&y_len);
    // SAFETY: the arguments match that signature, and x and y hold at least N elements each.
    unsafe { launch.launch(cfg) }?;
    let out = stream.clone_dtoh(&y_dev)?; // waits for the kernel

    // The GPU computes a * x + y as one fma (one rounding); Rust's `a * x + y` rounds twice.
    for (i, &got) in out.iter().enumerate() {
        let want = if i < N { a.mul_add(x[i], y[i]) } else { GUARD };
        ensure!(
            got.to_bits() == want.to_bits(),
            "axpy: y[{i}] is {got}, expected {want}{}",
            if i < N {
                ""
            } else {
                " (past the end: the bounds check failed)"
            }
        );
    }
    let unfused = (0..N)
        .filter(|&i| a * x[i] + y[i] != a.mul_add(x[i], y[i]))
        .count();
    eprintln!("axpy ok: {N} elements bit-exact ({unfused} would differ from an unfused a * x + y)");
    Ok(())
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = match args.as_slice() {
        [] => None,
        [dir] if !dir.starts_with('-') => Some(PathBuf::from(dir)),
        _ => bail!(USAGE),
    };

    let ctx = CudaContext::new(0)?;
    let kernels = load_kernels(&ctx)?;
    axpy_selftest(&ctx.default_stream(), &kernels)?;
    let Some(dir) = dir else {
        return Ok(());
    };

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
    Ok(())
}
