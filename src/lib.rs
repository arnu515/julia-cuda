//! Julia-1 on the GPU: the kernels (embedded PTX), the weights, the forward pass. The
//! `julia-cuda` binary (`main.rs`) is its CLI; programs embedding it use `Engine`.

use std::sync::Arc;

use color_eyre::eyre::{Context, ensure};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, LaunchConfig,
        PushKernelArg,
    },
    nvrtc::Ptx,
};

pub mod attn;
pub mod capi;
pub mod check;
pub mod checkpoint;
pub mod engine;
pub mod flame;
pub mod forward;
pub mod gemm;
pub mod prep;
pub mod profile;
pub mod serve;
pub mod timing;
pub mod weights;

pub use engine::Engine;
/// The julia-rs this crate is built with (its `Request`, `Sequence` and `answer`).
pub use julia_rs;
use julia_rs::{head::QType, sequence::Request};
use weights::GpuWeights;

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

/// The PTX tier this build embeds ("sm_89").
pub fn ptx_tier() -> &'static str {
    PTX.tier
}

/// The kernels, looked up once (`load_function` is a lookup in the module: not something to
/// repeat per launch).
pub struct Kernels {
    embed: CudaFunction,
    /// The shared-memory tiled GEMM on CUDA cores (`gemm_tiled`), kept as a baseline.
    gemm_tiled: CudaFunction,
    /// The tensor-core GEMMs.
    gemm: gemm::GemmKernels,
    attention: CudaFunction,
    /// The persistent layer kernel, and how many of its 128-thread blocks
    /// fit on an SM at once.
    layer: (CudaFunction, usize),
    norm_add: CudaFunction,
    score: CudaFunction,
}

/// Loads the embedded PTX, after checking that this GPU can run its tier. The driver
/// JIT-compiles PTX forward (sm_80 PTX runs on sm_89) but never backward.
pub fn load_kernels(ctx: &Arc<CudaContext>) -> color_eyre::Result<Kernels> {
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
    Ok(Kernels {
        embed: module.load_function("embed")?,
        gemm_tiled: module.load_function("gemm_tiled")?,
        gemm: gemm::GemmKernels::load(&module)?,
        attention: module.load_function("attention")?,
        layer: layer_kernel(&module, "layer")?,
        norm_add: module.load_function("norm_add")?,
        score: module.load_function("score")?,
    })
}

/// A persistent layer kernel and its resident blocks per SM (for the cooperative launch's grid).
fn layer_kernel(module: &Arc<CudaModule>, name: &str) -> color_eyre::Result<(CudaFunction, usize)> {
    let f = module.load_function(name)?;
    let per_sm = f.occupancy_max_active_blocks_per_multiprocessor(128, 0, None)? as usize;
    ensure!(per_sm > 0, "{name} doesn't fit on an SM");
    Ok((f, per_sm))
}

/// Hidden size of the model.
pub const D: usize = 384;

/// Threads per block for `embed`: a multiple of 32 (its `warp_sum` needs whole warps), one warp
/// per token.
const EMBED_BLOCK: u32 = 128;

/// Runs the embedding kernel: the residual stream `x` (`[tokens.len(), D]`, f16) for `tokens`.
pub fn embed(
    stream: &Arc<CudaStream>,
    kernels: &Kernels,
    w: &GpuWeights,
    tokens: &CudaSlice<u32>,
    eps: f32,
) -> color_eyre::Result<CudaSlice<half::f16>> {
    let mut x = stream.alloc_zeros(tokens.len() * D)?;
    embed_into(stream, kernels, w, tokens, eps, &mut x)?;
    Ok(x)
}

/// `embed` into an existing `x`.
pub fn embed_into(
    stream: &Arc<CudaStream>,
    kernels: &Kernels,
    w: &GpuWeights,
    tokens: &CudaSlice<u32>,
    eps: f32,
    x: &mut CudaSlice<half::f16>,
) -> color_eyre::Result<()> {
    let table = w.f16.slice(w.layout.tok_embeddings.range());
    let table_len = table.len() as u64;
    let gamma = w.f32.slice(w.layout.emb_norm.range());
    let gamma_len = gamma.len() as u64;
    let n = tokens.len();
    ensure!(n > 0, "embed: no tokens");
    ensure!(
        x.len() == n * D,
        "embed: x has {} elements, not {n} × {D}",
        x.len()
    );
    let n = n as u64;

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(EMBED_BLOCK as u64 / 32) as u32, 1, 1),
        block_dim: (EMBED_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&kernels.embed);
    // embed(tokens: &[u32], table: &[f16], gamma: &[f32], eps: f32, x: *mut f16)
    launch
        .arg(tokens)
        .arg(&n)
        .arg(&table)
        .arg(&table_len)
        .arg(&gamma)
        .arg(&gamma_len)
        .arg(&eps)
        .arg(x);
    // SAFETY: the arguments match that signature, block size is a multiple of 32,
    //   and x's dimensions match properly.
    flame::around(stream, || "embed".into(), || unsafe { launch.launch(cfg) })?;
    Ok(())
}

/// Threads per block for `gemm_tiled`: 16 × 16, each computing 4 × 4 outputs of a 64 × 64 tile.
const GEMM_BLOCK: u32 = 256;
/// The tile `gemm_tiled` computes per block, and its K step: n has to be a multiple of `GEMM_TILE`
/// and k of `GEMM_BK` (rows of `a` past m are handled).
const GEMM_TILE: usize = 64;
const GEMM_BK: usize = 32;

/// Runs the CUDA-core tiled GEMM: `c = a · wᵀ`, with `a` `[m, k]`, `w` `[n, k]` and `c`
/// `[m, n]`, all f16. The baseline for the tensor-core GEMMs (`gemm::launch`).
pub fn gemm_tiled(
    stream: &Arc<CudaStream>,
    kernels: &Kernels,
    a: &CudaSlice<half::f16>,
    w: &CudaView<half::f16>,
    k: usize,
) -> color_eyre::Result<CudaSlice<half::f16>> {
    ensure!(k > 0, "gemm: k = 0");
    let mut c = stream.alloc_zeros(a.len() / k * (w.len() / k))?;
    gemm_tiled_into(stream, kernels, a, w, k, &mut c)?;
    Ok(c)
}

/// `gemm_tiled` into an existing `c`.
pub fn gemm_tiled_into(
    stream: &Arc<CudaStream>,
    kernels: &Kernels,
    a: &CudaSlice<half::f16>,
    w: &CudaView<half::f16>,
    k: usize,
    c: &mut CudaSlice<half::f16>,
) -> color_eyre::Result<()> {
    ensure!(
        k > 0 && k.is_multiple_of(GEMM_BK),
        "gemm: k = {k} is not a multiple of {GEMM_BK}"
    );
    ensure!(
        a.len().is_multiple_of(k) && w.len().is_multiple_of(k),
        "gemm: a ({}) or w ({}) is not a whole number of rows of {k}",
        a.len(),
        w.len()
    );
    let (m, n) = (a.len() / k, w.len() / k);
    ensure!(m > 0, "gemm: no rows");
    ensure!(
        n.is_multiple_of(GEMM_TILE),
        "gemm: n = {n} is not a multiple of {GEMM_TILE}"
    );
    ensure!(
        c.len() == m * n,
        "gemm: c has {} elements, not {m} × {n}",
        c.len()
    );

    let cfg = LaunchConfig {
        grid_dim: ((n / GEMM_TILE) as u32, m.div_ceil(GEMM_TILE) as u32, 1),
        block_dim: (GEMM_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // The kernel's k is a u32: `arg` copies as many bytes as the Rust type has, so a usize would
    // shift every parameter after it.
    let (a_len, w_len, k) = (a.len() as u64, w.len() as u64, k as u32);

    let mut launch = stream.launch_builder(&kernels.gemm_tiled);
    // gemm_tiled(a: &[f16], w: &[f16], k: u32, c: *mut f16)
    launch.arg(a).arg(&a_len).arg(w).arg(&w_len).arg(&k).arg(c);
    // SAFETY: the arguments match that signature, with the checks above: k is a multiple of 32,
    //   n of 64, and c has m·n elements. a is a cudarc allocation and w a weight span, so both
    //   are 16-byte aligned. The block is 256 threads, as the kernel's indexing assumes.
    flame::around(
        stream,
        || "gemm_tiled".into(),
        || unsafe { launch.launch(cfg) },
    )?;
    Ok(())
}

/// A request of 363 tokens, long enough that the sliding window (64 keys each side) leaves out
/// most of a local layer's keys (each query sees at most 129): the window check needs it.
pub fn long_request() -> Request<'static> {
    let state = (0..10)
        .map(|i| {
            format!(
                "Note {i}: the customer was charged twice for the same subscription renewal, once \
                 on the card on file and once on a gift card that should have been used up."
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    Request {
        state: state.into(),
        ..sample_request()
    }
}

/// The request the kernels run on until requests come from stdin.
pub fn sample_request() -> Request<'static> {
    Request {
        state: "Order #4417: the customer was charged twice for the same subscription renewal, \
                once on the card on file and once on a gift card that should have been used up. \
                They want the card charge refunded and the gift card restored."
            .into(),
        question: "Which team should handle this request?".into(),
        qtype: QType::Choice,
        options: ["Billing", "Shipping", "Account access", "Fraud"]
            .map(Into::into)
            .into(),
    }
}
