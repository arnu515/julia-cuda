//! The tensor-core GEMM's host side (kernels/src/gemm.rs is the device side): which kernel
//! and launch shape to use, the split-K scratch buffers, and the launch itself.

use std::sync::Arc;

use color_eyre::eyre::{ensure, eyre};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use half::f16;

/// Tile shapes: BM rows × 64 columns per block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// 32 × 64 on 4 warps.
    S32,
    /// 64 × 64 on 4 warps.
    S64,
    /// 128 × 64 on 4 warps.
    S128,
    /// 128 × 64 on 8 warps.
    S128w8,
}

pub const SHAPES: [Shape; 4] = [Shape::S32, Shape::S64, Shape::S128, Shape::S128w8];

impl Shape {
    pub fn bm(self) -> usize {
        match self {
            Self::S32 => 32,
            Self::S64 => 64,
            Self::S128 | Self::S128w8 => 128,
        }
    }

    fn threads(self) -> u32 {
        match self {
            Self::S128w8 => 256,
            _ => 128,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::S32 => "32",
            Self::S64 => "64",
            Self::S128 => "128",
            Self::S128w8 => "128w8",
        }
    }
}

/// What the GEMM does around `a · wᵀ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    /// `c = a · wᵀ`, then RoPE on the q and k thirds of c, over `prep::permute_rope`d weights
    /// (needs `rope`).
    Rope,
    /// `c = LayerNorm(a) · wᵀ`, with γ folded into `w` (needs `ln`), then RoPE as `Rope`.
    LnRope,
    /// `c = GeGLU(LayerNorm(a) · wᵀ)` over GeGLU-interleaved `w`; `c` is `[m, n/2]`.
    LnGeglu,
    /// `c += a · wᵀ` (the residual stream, in place), and each row's Σ and Σ² of the result are
    /// added to `stats_out`.
    ResStats,
    /// `c = LayerNorm(a) · wᵀ + bias`, the LayerNorm with a bias folded too
    /// (`prep::fold_layernorm_bias`): the head's in_proj.
    LnBias,
    /// `LnBias`, then ReLU: the head's linear1.
    LnBiasRelu,
    /// `LnBias`, then GELU: the scorer's first linear.
    LnBiasGelu,
    /// `c += a · wᵀ + bias`, with `ResStats`'s statistics: the head's out_proj and linear2.
    BiasResStats,
}

const EPILOGUES: [Epilogue; 8] = [
    Epilogue::Rope,
    Epilogue::LnRope,
    Epilogue::LnGeglu,
    Epilogue::ResStats,
    Epilogue::LnBias,
    Epilogue::LnBiasRelu,
    Epilogue::LnBiasGelu,
    Epilogue::BiasResStats,
];

impl Epilogue {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rope => "rope",
            Self::LnRope => "lnrope",
            Self::LnGeglu => "lngeglu",
            Self::ResStats => "resstats",
            Self::LnBias => "lnbias",
            Self::LnBiasRelu => "lnbiasrelu",
            Self::LnBiasGelu => "lnbiasgelu",
            Self::BiasResStats => "biasresstats",
        }
    }

    fn index(self) -> usize {
        EPILOGUES.iter().position(|&e| e == self).unwrap()
    }

    fn ln(self) -> bool {
        matches!(
            self,
            Self::LnRope | Self::LnGeglu | Self::LnBias | Self::LnBiasRelu | Self::LnBiasGelu
        )
    }

    fn res(self) -> bool {
        matches!(self, Self::ResStats | Self::BiasResStats)
    }

    fn bias(self) -> bool {
        matches!(
            self,
            Self::LnBias | Self::LnBiasRelu | Self::LnBiasGelu | Self::BiasResStats
        )
    }
}

/// A launch configuration: tile shape and the number of K splits (1–4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub shape: Shape,
    pub splits: u32,
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x64 split {}", self.shape.name(), self.splits)
    }
}

/// Every configuration, for sweeps.
pub fn configs() -> impl Iterator<Item = Config> {
    SHAPES
        .into_iter()
        .flat_map(|shape| (1..=4).map(move |splits| Config { shape, splits }))
}

/// The configuration for an m × n × k GEMM on a GPU with `sms` SMs. A rough rule from
/// GEMM microbenchmark sweeps (2080 Ti and 4070, m = 160–2048) and `--time` at m = 72: the smallest
/// tiles until there are enough rows to fill the GPU with bigger ones; no split-K while there
/// are more than half as many tiles as SMs (qkv and wi at m = 72: splitting only adds the
/// reduction), else 4 splits (wo and wo_mlp at m = 72: within 3% of the best on the 2080 Ti,
/// the best on the 4070), keeping each split ≥ 3 K steps. `--time` sweeps every configuration
/// for the request at hand.
pub fn choose(m: usize, n: usize, k: usize, sms: usize) -> Config {
    let shape = match m {
        ..=200 => Shape::S32,
        201..=400 => Shape::S64,
        _ => Shape::S128w8,
    };
    Config {
        shape,
        splits: splits(m, n, k, shape.bm(), sms),
    }
}

/// `choose`'s split-K count for tiles of `bm` rows: none while there are more than half as many
/// tiles as SMs, else 4, keeping each split ≥ 3 K steps.
pub fn splits(m: usize, n: usize, k: usize, bm: usize, sms: usize) -> u32 {
    let tiles = (n / 64) * m.div_ceil(bm);
    // Strictly more: at 2·tiles == sms (wo and wo_mlp on the 4070's 36 SMs) splitting wins.
    let splits = if 2 * tiles > sms { 1 } else { 4 };
    splits.min((k / 32 / 3).max(1) as u32)
}

/// The 32 GEMM kernels (4 shapes × 8 epilogues).
pub struct GemmKernels {
    fns: Vec<CudaFunction>,
}

impl GemmKernels {
    pub fn load(module: &Arc<CudaModule>) -> color_eyre::Result<Self> {
        let mut fns = Vec::with_capacity(SHAPES.len() * EPILOGUES.len());
        for shape in SHAPES {
            for epi in EPILOGUES {
                fns.push(module.load_function(&format!("gemm_{}_{}", shape.name(), epi.name()))?);
            }
        }
        Ok(Self { fns })
    }

    fn get(&self, shape: Shape, epi: Epilogue) -> &CudaFunction {
        let s = SHAPES.iter().position(|&x| x == shape).unwrap();
        &self.fns[s * EPILOGUES.len() + epi.index()]
    }
}

/// The widest GEMM output (wi: 2·intermediate).
const MAX_N: usize = 2304;

/// Split-K scratch: a BM × 64 f32 partial tile per (tile, split), and an arrival counter per
/// tile (zero between launches: the last block of a tile resets it).
pub struct Scratch {
    pub(crate) ws: CudaSlice<f32>,
    pub(crate) counters: CudaSlice<u32>,
}

impl Scratch {
    /// Enough for any configuration with m ≤ `max_m` and n ≤ 2304.
    pub fn new(stream: &Arc<CudaStream>, max_m: usize) -> color_eyre::Result<Self> {
        // Over shapes, ⌈m/BM⌉·BM ≤ m + 127 rows of tiles; 4 splits.
        let rows = max_m + 127;
        Ok(Self {
            ws: stream.alloc_zeros(4 * rows * MAX_N)?,
            counters: stream.alloc_zeros((MAX_N / 64) * max_m.div_ceil(32))?,
        })
    }
}

/// LayerNorm inputs for the ln epilogues: the row statistics of `a` (Σ, Σ² per row, as a
/// `ResStats` GEMM writes them) and the folded weights' column sums.
pub struct LnInputs<'a, 'b> {
    pub stats: &'b CudaSlice<f32>,
    pub colsum: &'b CudaView<'a, f32>,
    pub eps: f32,
}

/// One GEMM: `a` is `[m, k]`, `w` `[n, k]`, `c` `[m, n]` (`[m, n/2]` for `LnGeglu`).
pub struct Gemm<'a, 'b> {
    pub epi: Epilogue,
    pub a: &'b CudaSlice<f16>,
    pub w: &'b CudaView<'a, f16>,
    pub k: usize,
    pub c: &'b mut CudaSlice<f16>,
    /// For the ln epilogues.
    pub ln: Option<LnInputs<'a, 'b>>,
    /// For `Rope` / `LnRope`: (cos, sin) per position (row) and pair, `[≥ m][32][2]`
    /// (`prep::rope_table`).
    pub rope: Option<&'b CudaView<'a, f32>>,
    /// For the bias epilogues: `n` values.
    pub bias: Option<&'b CudaView<'a, f32>>,
    /// For the resstats epilogues: `2m` floats, zero beforehand, that the row statistics are
    /// added to.
    pub stats_out: Option<&'b mut CudaSlice<f32>>,
    /// A buffer to clear on the way (the `stats_out` of a later `ResStats` GEMM).
    pub zero: Option<&'b mut CudaSlice<f32>>,
}

/// The raw device address of a buffer, for the kernels' pointer parameters (`res` aliases `c`,
/// which cudarc's typed arguments can't express).
pub(crate) fn ptr<T>(stream: &Arc<CudaStream>, x: &impl DevicePtr<T>) -> u64 {
    x.device_ptr(stream).0
}

/// Checks `g` against everything the kernel assumes, then launches it with `cfg`.
pub fn launch(
    stream: &Arc<CudaStream>,
    kernels: &GemmKernels,
    scratch: &Scratch,
    cfg: Config,
    g: Gemm,
) -> color_eyre::Result<()> {
    let k = g.k;
    ensure!(
        k > 0 && k.is_multiple_of(32),
        "gemm: k = {k} is not a multiple of 32"
    );
    ensure!(
        g.a.len().is_multiple_of(k) && g.w.len().is_multiple_of(k),
        "gemm: a or w is not whole rows of {k}"
    );
    let (m, n) = (g.a.len() / k, g.w.len() / k);
    ensure!(m > 0, "gemm: no rows");
    ensure!(
        n.is_multiple_of(64) && n <= MAX_N,
        "gemm: n = {n} is not a multiple of 64 up to {MAX_N}"
    );
    let n_out = if g.epi == Epilogue::LnGeglu { n / 2 } else { n };
    ensure!(
        g.c.len() == m * n_out,
        "gemm: c has {} elements, not {m} × {n_out}",
        g.c.len()
    );
    ensure!(
        (1..=4).contains(&cfg.splits) && cfg.splits as usize <= k / 32,
        "gemm: {} splits for {} K steps",
        cfg.splits,
        k / 32
    );
    let tiles = (n / 64) * m.div_ceil(cfg.shape.bm());
    ensure!(
        cfg.splits == 1
            || (tiles * cfg.splits as usize * cfg.shape.bm() * 64 <= scratch.ws.len()
                && tiles <= scratch.counters.len()),
        "gemm: the split-K scratch is too small for m = {m}"
    );
    ensure!(
        g.ln.is_some() == g.epi.ln(),
        "gemm: LayerNorm inputs go with the ln epilogues"
    );
    if let Some(ln) = &g.ln {
        ensure!(
            ln.stats.len() >= 2 * m && ln.colsum.len() == n,
            "gemm: LayerNorm statistics or column sums have the wrong size"
        );
    }
    let needs_rope = matches!(g.epi, Epilogue::Rope | Epilogue::LnRope);
    ensure!(
        g.rope.is_some() == needs_rope,
        "gemm: a RoPE table goes with the rope epilogues"
    );
    if let Some(rope) = g.rope {
        ensure!(
            rope.len() >= m * 64 && n.is_multiple_of(3 * 64),
            "gemm: the RoPE table is too short for {m} positions, or n = {n} isn't qkv"
        );
    }
    ensure!(
        g.stats_out.is_some() == g.epi.res(),
        "gemm: output statistics go with the resstats epilogues"
    );
    ensure!(
        g.bias.is_some() == g.epi.bias() && g.bias.is_none_or(|b| b.len() == n),
        "gemm: a bias of n values goes with the bias epilogues"
    );
    if let Some(s) = &g.stats_out {
        ensure!(
            s.len() >= 2 * m,
            "gemm: the output statistics buffer is too small"
        );
    }

    let null = 0u64;
    let (a, w, c) = (ptr(stream, g.a), ptr(stream, g.w), ptr(stream, &*g.c));
    let res = if g.epi.res() { c } else { null };
    let (ws, counters) = (ptr(stream, &scratch.ws), ptr(stream, &scratch.counters));
    let (stats_in, colsum, eps) = match &g.ln {
        Some(ln) => (ptr(stream, ln.stats), ptr(stream, ln.colsum), ln.eps),
        None => (null, null, 0.0),
    };
    let stats_out = g.stats_out.as_deref().map_or(null, |s| ptr(stream, s));
    let rope = g.rope.map_or(null, |r| ptr(stream, r));
    let bias = g.bias.map_or(null, |b| ptr(stream, b));
    let (zero, zero_len) = g
        .zero
        .as_deref()
        .map_or((null, 0), |z| (ptr(stream, z), z.len() as u32));
    ensure!(
        zero == null || (zero != stats_in && zero != stats_out),
        "gemm: the buffer to clear is one this GEMM uses"
    );
    let (m32, n32, k32) = (m as u32, n as u32, k as u32);

    let cfg_launch = LaunchConfig {
        grid_dim: (
            (n / 64) as u32,
            m.div_ceil(cfg.shape.bm()) as u32,
            cfg.splits,
        ),
        block_dim: (cfg.shape.threads(), 1, 1),
        shared_mem_bytes: 0,
    };
    let f = kernels.get(cfg.shape, g.epi);
    let mut launch = stream.launch_builder(f);
    // gemm_*(a, w, c, res, ws, counters, stats_in, colsum, stats_out, zero, zero_len, m, n, k,
    // eps, rope, bias): pointers are u64, the rest u32 / f32.
    launch
        .arg(&a)
        .arg(&w)
        .arg(&c)
        .arg(&res)
        .arg(&ws)
        .arg(&counters)
        .arg(&stats_in)
        .arg(&colsum)
        .arg(&stats_out)
        .arg(&zero)
        .arg(&zero_len)
        .arg(&m32)
        .arg(&n32)
        .arg(&k32)
        .arg(&eps)
        .arg(&rope)
        .arg(&bias);
    // SAFETY: the checks above are the kernel's requirements: the sizes of a, w, c, the LN
    // inputs, the RoPE table, the bias and the statistics, n % 64, k % 32, the scratch size for split-K, and a null pointer
    // only where the epilogue doesn't read it. The buffers are cudarc allocations or weight spans
    // (16-byte aligned, rows of k a multiple of 32 elements). res aliasing c is allowed; `zero`
    // is not read by this GEMM. The block size matches the kernel's shape.
    unsafe { launch.launch(cfg_launch) }
        .map_err(|e| eyre!("gemm_{}_{} ({cfg}): {e}", cfg.shape.name(), g.epi.name()))?;
    Ok(())
}
