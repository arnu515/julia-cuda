use std::{path::PathBuf, sync::Arc, time::Instant};

use color_eyre::eyre::{Context, bail, ensure};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg,
        sys::CUdevice_attribute,
    },
    nvrtc::Ptx,
};

mod check;
mod checkpoint;
mod prep;
mod timing;
// The layout is read by the kernel launches (from the embedding kernel on); until then only by the upload.
#[allow(dead_code)]
mod weights;

use checkpoint::Checkpoint;
use julia_rs::{
    head::QType,
    sequence::{DEFAULT_HEAD_LENGTH, Request, sequence},
};
use weights::{GpuWeights, HostWeights};

const USAGE: &str = "usage: julia-cuda [checkpoint dir [--check] [--time]]
  Without a checkpoint, only checks that the kernels load on GPU 0. With one, runs the kernels on
  a sample request.
  --check also verifies the weight upload and compares each kernel's output with julia-rs.
  --time  also measures each kernel's GPU time (CUDA events, after warm-up).";

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

/// The kernels, looked up once (`load_function` is a lookup in the module: not something to
/// repeat per launch).
pub struct Kernels {
    embed: CudaFunction,
    gemm: CudaFunction,
}

/// Loads the embedded PTX, after checking that this GPU can run its tier. The driver
/// JIT-compiles PTX forward (sm_80 PTX runs on sm_89) but never backward.
fn load_kernels(ctx: &Arc<CudaContext>) -> color_eyre::Result<Kernels> {
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
        gemm: module.load_function("gemm")?,
    })
}

/// Hidden size of the model.
const D: usize = 384;

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
    unsafe { launch.launch(cfg) }?;
    Ok(())
}

/// Threads per block for `gemm`: 16 × 16, each computing 4 × 4 outputs of a 64 × 64 tile.
const GEMM_BLOCK: u32 = 256;
/// The tile `gemm` computes per block, and its K step: n has to be a multiple of `GEMM_TILE`
/// and k of `GEMM_BK` (rows of `a` past m are handled).
const GEMM_TILE: usize = 64;
const GEMM_BK: usize = 32;

/// Runs the tiled GEMM: `c = a · wᵀ`, with `a` `[m, k]`, `w` `[n, k]` and `c` `[m, n]`, all f16.
pub fn gemm(
    stream: &Arc<CudaStream>,
    kernels: &Kernels,
    a: &CudaSlice<half::f16>,
    w: &CudaView<half::f16>,
    k: usize,
) -> color_eyre::Result<CudaSlice<half::f16>> {
    ensure!(k > 0, "gemm: k = 0");
    let mut c = stream.alloc_zeros(a.len() / k * (w.len() / k))?;
    gemm_into(stream, kernels, a, w, k, &mut c)?;
    Ok(c)
}

/// `gemm` into an existing `c`.
pub fn gemm_into(
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

    let mut launch = stream.launch_builder(&kernels.gemm);
    // gemm(a: &[f16], w: &[f16], k: u32, c: *mut f16)
    launch.arg(a).arg(&a_len).arg(w).arg(&w_len).arg(&k).arg(c);
    // SAFETY: the arguments match that signature, with the checks above: k is a multiple of 32,
    //   n of 64, and c has m·n elements. a is a cudarc allocation and w a weight span, so both
    //   are 16-byte aligned. The block is 256 threads, as the kernel's indexing assumes.
    unsafe { launch.launch(cfg) }?;
    Ok(())
}

/// The request the kernels run on until requests come from stdin.
fn sample_request() -> Request<'static> {
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

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, flags) = match args.split_first() {
        None => (None, &[][..]),
        Some((dir, flags)) if !dir.starts_with('-') => (Some(PathBuf::from(dir)), flags),
        Some(_) => bail!(USAGE),
    };
    let (mut check, mut time) = (false, false);
    for flag in flags {
        match flag.as_str() {
            "--check" => check = true,
            "--time" => time = true,
            _ => bail!(USAGE),
        }
    }

    let ctx = CudaContext::new(0)?;
    let kernels = load_kernels(&ctx)?;
    let Some(dir) = dir else {
        return Ok(());
    };

    let t = Instant::now();
    let ck = Checkpoint::open(&dir)?;
    let c = &ck.config;
    ensure!(
        c.hidden == D,
        "the kernels are built for hidden size {D}, the model's is {}",
        c.hidden
    );
    eprintln!(
        "model: {} layers, hidden {}, intermediate {} ({:.0} ms to open)",
        c.n_layers,
        c.hidden,
        c.intermediate,
        ms(t)
    );
    let t = Instant::now();
    let host = HostWeights::prepare(&ck);
    eprintln!(
        "weights: {:.1} MB f16, {:.1} KB f32 ({:.0} ms to prepare)",
        (host.f16.len() * 2) as f64 / 1e6,
        (host.f32.len() * 4) as f64 / 1e3,
        ms(t)
    );
    let (stream, t) = (ctx.default_stream(), Instant::now());
    let gpu = GpuWeights::upload(&stream, &host)?;
    stream.synchronize()?;
    eprintln!("uploaded in {:.0} ms", ms(t));
    if check {
        gpu.verify(&stream, &host)?;
        eprintln!("upload verified: both buffers bit-exact");
    }

    let seq = sequence(
        &ck.tokenizer,
        &sample_request(),
        c.max_positions,
        DEFAULT_HEAD_LENGTH,
    )?;
    let ids = stream.clone_htod(&seq.ids)?;
    // Wall clock around the first calls: this includes allocating the outputs and loading each
    // kernel onto the GPU (CUDA does that on the first launch). --time measures the kernels.
    let t = Instant::now();
    let mut x = embed(&stream, &kernels, &gpu, &ids, c.norm_eps)?;
    // Layer 0 has no attention LayerNorm, so its qkv GEMM runs on x as is.
    let wqkv = gpu.f16.slice(gpu.layout.layers[0].wqkv.range());
    let mut qkv = gemm(&stream, &kernels, &x, &wqkv, D)?;
    stream.synchronize()?;
    eprintln!(
        "embed + qkv (layer 0) for {} tokens: {:.3} ms, first call (wall clock)",
        seq.ids.len(),
        ms(t)
    );

    if check {
        let model = check::reference_model(&dir)?;
        let x = stream.clone_dtoh(&x)?;
        check::embed(&model, &host, &seq.ids, c.norm_eps, &x)?;
        check::qkv0(&model, &host, &seq.ids, &x, &stream.clone_dtoh(&qkv)?)?;
    }

    if time {
        let n = seq.ids.len();
        let l2 = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as f64 / 1e6;
        let (warmup, batches, per_batch) = (20, 20, 100);
        let run = |launch: &mut dyn FnMut() -> color_eyre::Result<()>| {
            timing::time(&stream, warmup, batches, per_batch, launch)
        };
        eprintln!(
            "timing: GPU time per call, fastest / median of {batches} batches of {per_batch} \
             calls (after {warmup}); {l2:.1} MB L2"
        );

        let t = run(&mut || embed_into(&stream, &kernels, &gpu, &ids, c.norm_eps, &mut x))?;
        eprintln!(
            "  embed                    {n} tokens            {:>7.2} / {:>7.2} µs",
            t.min_us, t.median_us
        );

        let flop = 2.0 * (n * 3 * D * D) as f64;
        let wqkv_mb = (wqkv.len() * 2) as f64 / 1e6;
        let t = run(&mut || gemm_into(&stream, &kernels, &x, &wqkv, D, &mut qkv))?;
        eprintln!(
            "  qkv, layer 0's weights   {n} × {} × {D}    {:>7.2} / {:>7.2} µs   {:.2} TFLOP/s   \
             ({wqkv_mb:.2} MB of weights, stays in L2)",
            3 * D,
            t.min_us,
            t.median_us,
            flop / t.median_us / 1e6
        );

        // Every layer's Wqkv in turn: the same shape, but together more than fits in L2 (on
        // GPUs with less L2 than this), so the weights come from DRAM as in a forward pass.
        // Layers 1..21 hold the folded weights; the values don't matter for timing.
        let all: Vec<_> = (gpu.layout.layers.iter())
            .map(|l| gpu.f16.slice(l.wqkv.range()))
            .collect();
        let mut i = 0;
        let t = run(&mut || {
            i = (i + 1) % all.len();
            gemm_into(&stream, &kernels, &x, &all[i], D, &mut qkv)
        })?;
        eprintln!(
            "  qkv, {} layers' in turn  {n} × {} × {D}    {:>7.2} / {:>7.2} µs   {:.2} TFLOP/s   \
             ({:.1} MB of weights{})",
            all.len(),
            3 * D,
            t.min_us,
            t.median_us,
            flop / t.median_us / 1e6,
            wqkv_mb * all.len() as f64,
            if wqkv_mb * all.len() as f64 > l2 {
                ", more than L2"
            } else {
                ", fits in L2: still warm"
            }
        );
    }
    Ok(())
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
