//! `--check`: compares the GPU's results with julia-rs, the f32 reference.
//!
//! Each kernel is compared twice: with julia-rs, to see what f16 costs (expect a relative error
//! around 1e-3 at most), and with a host emulation of the kernel's own arithmetic (f16 inputs,
//! f32 math, f16 output), to catch bugs: there, only summation order differs, so values may be off
//! by an f16 ulp but no more.

use std::{path::Path, sync::Arc};

use color_eyre::eyre::ensure;
use cudarc::driver::{CudaSlice, CudaStream};
use half::f16;
use julia_rs::{
    Model, ModelPaths,
    attn::{Rope, Scratch as AttnScratch, attention},
    load_model,
    matmul::{Precision, linear, linear_add},
    mlp::geglu,
};

use crate::{
    D,
    gemm::{self, Config, Epilogue, Gemm, GemmKernels, LnInputs},
    weights::{GpuWeights, HostWeights},
};

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

/// One encoder layer of julia-rs's forward pass (f32, LayerNorms in f64).
pub struct LayerTrace {
    /// The residual stream entering the layer, `[n, D]`.
    pub x_in: Vec<f32>,
    /// LN(x_in) · Wqkvᵀ (no LayerNorm in layer 0), before RoPE.
    pub qkv: Vec<f32>,
    /// The attention output, before Wo.
    pub o: Vec<f32>,
    /// x_in + o · Woᵀ.
    pub x_mid: Vec<f32>,
    /// GeGLU(LN(x_mid) · Wiᵀ), `[n, intermediate]`.
    pub m: Vec<f32>,
    /// x_mid + m · Wo_mlpᵀ.
    pub x_out: Vec<f32>,
}

/// julia-rs's encoder layers on `tokens` (as `Model::encode`, keeping every intermediate).
pub fn trace(model: &Model, tokens: &[u32]) -> Vec<LayerTrace> {
    let (n, d, inter, eps) = (
        tokens.len(),
        model.config.hidden,
        model.config.intermediate,
        model.config.norm_eps,
    );
    let data = model.data();
    let mut x = model.embed(tokens);
    let mut h = vec![0.0; n * d];
    let mut s = AttnScratch::default();
    let rope_g = Rope::new(model.config.rope_theta_global, n);
    let rope_l = Rope::new(model.config.rope_theta_local, n);
    let mut out = vec![];
    for (l, p) in (model.weights.encoder.layers.iter()).zip(&model.packed.encoder) {
        let x_in = x.clone();
        match &l.attn_norm {
            Some(g) => layernorm_rows(&x, g.as_f32(data), eps, &mut h),
            None => h.copy_from_slice(&x),
        }
        let mut qkv = vec![0.0; n * 3 * d];
        linear(&h, &p.wqkv, None, &mut qkv);
        let pre_rope = qkv.clone();
        let (rope, window) = match l.global {
            true => (&rope_g, None),
            false => (&rope_l, Some(model.config.local_window / 2)),
        };
        rope.apply(&mut qkv, d);
        let mut o = vec![0.0; n * d];
        attention(&qkv, n, d, window, &mut o, &mut s);
        linear_add(&o, &p.wo, None, &mut x);
        let x_mid = x.clone();
        layernorm_rows(&x, l.mlp_norm.as_f32(data), eps, &mut h);
        let (mut hh, mut m) = (vec![0.0; n * 2 * inter], vec![0.0; n * inter]);
        linear(&h, &p.wi, None, &mut hh);
        geglu(&hh, inter, &mut m);
        linear_add(&m, &p.wo_mlp, None, &mut x);
        out.push(LayerTrace {
            x_in,
            qkv: pre_rope,
            o,
            x_mid,
            m,
            x_out: x.clone(),
        });
    }
    out
}

/// Bias-free LayerNorm of each `gamma.len()`-wide row, in f64.
pub fn layernorm_rows(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
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

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// Each row's Σ and Σ² (f32 sums of the f16 values, as the resstats epilogue makes them).
pub fn row_stats(x: &[f16], k: usize) -> Vec<f32> {
    (x.chunks_exact(k))
        .flat_map(|row| {
            let (s, q) = row.iter().fold((0f32, 0f32), |(s, q), v| {
                let v = v.to_f32();
                (s + v, v.mul_add(v, q))
            });
            [s, q]
        })
        .collect()
}

/// How far `stats` (Σ, Σ² per row) are from the f64 sums of `x`'s rows: the largest of
/// |ΔΣ| / √Σx² and |ΔΣ²| / Σx² over the rows.
fn stats_err(x: &[f16], stats: &[f32], k: usize) -> f64 {
    (x.chunks_exact(k))
        .zip(stats.as_chunks::<2>().0)
        .map(|(row, &[s, q])| {
            let s_ref: f64 = row.iter().map(|v| v.to_f64()).sum();
            let q_ref: f64 = row.iter().map(|v| v.to_f64().powi(2)).sum();
            let q_ref = q_ref.max(1e-30);
            ((f64::from(s) - s_ref).abs() / q_ref.sqrt()).max((f64::from(q) - q_ref).abs() / q_ref)
        })
        .fold(0.0, f64::max)
}

/// One encoder layer's four GEMMs on a residual stream the implementation keeps (on the GPU, or
/// emulated on the host). Attention isn't on the GPU yet, so its output comes from julia-rs.
pub trait LayerGemms {
    /// Replaces the stream with `x` (and the row statistics the next qkv reads with its own).
    fn set_stream(&mut self, x: &[f16]) -> color_eyre::Result<()>;
    /// The stream's qkv for layer `l` (LayerNorm folded in from layer 1).
    fn qkv(&mut self, l: usize) -> color_eyre::Result<Vec<f16>>;
    /// stream += o · Woᵀ; returns the stream and its row statistics.
    fn wo(&mut self, l: usize, o: &[f16]) -> color_eyre::Result<(Vec<f16>, Vec<f32>)>;
    /// m = GeGLU(LN(stream) · Wiᵀ).
    fn wi(&mut self, l: usize) -> color_eyre::Result<Vec<f16>>;
    /// stream += m · Wo_mlpᵀ; returns the stream and its row statistics.
    fn wo_mlp(&mut self, l: usize) -> color_eyre::Result<(Vec<f16>, Vec<f32>)>;
}

/// Relative errors of one layer's GEMMs against julia-rs.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerErrors {
    pub qkv: f64,
    pub x_mid: f64,
    pub m: f64,
    pub x_out: f64,
    /// The resstats epilogues' row statistics against their own output.
    pub stats: f64,
}

impl LayerErrors {
    fn max(self, o: Self) -> Self {
        Self {
            qkv: self.qkv.max(o.qkv),
            x_mid: self.x_mid.max(o.x_mid),
            m: self.m.max(o.m),
            x_out: self.x_out.max(o.x_out),
            stats: self.stats.max(o.stats),
        }
    }
}

impl std::fmt::Display for LayerErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "qkv {:.2e}, x+o·Wo {:.2e}, m {:.2e}, x+m·Wo2 {:.2e}, stats {:.1e}",
            self.qkv, self.x_mid, self.m, self.x_out, self.stats
        )
    }
}

/// Layer `l`'s four GEMMs, from whatever stream `g` holds, against `t`.
pub fn layer(g: &mut impl LayerGemms, l: usize, t: &LayerTrace) -> color_eyre::Result<LayerErrors> {
    let qkv = g.qkv(l)?;
    let (x_mid, s1) = g.wo(l, &to_f16(&t.o))?;
    let m = g.wi(l)?;
    let (x_out, s2) = g.wo_mlp(l)?;
    Ok(LayerErrors {
        qkv: rel_err(&to_f32(&qkv), &t.qkv),
        x_mid: rel_err(&to_f32(&x_mid), &t.x_mid),
        m: rel_err(&to_f32(&m), &t.m),
        x_out: rel_err(&to_f32(&x_out), &t.x_out),
        stats: stats_err(&x_mid, &s1, D).max(stats_err(&x_out, &s2, D)),
    })
}

/// Limits for each layer's GEMMs run from julia-rs's stream: about 3× what the host emulation
/// of the kernels' arithmetic reaches (≤ 6e-4, see the test), so a correct kernel passes and an
/// indexing bug (errors of order 1) can't.
const MAX_ERR: LayerErrors = LayerErrors {
    qkv: 2e-3,
    x_mid: 2e-3,
    m: 2e-3,
    x_out: 2e-3,
    stats: 1e-4,
};

/// Limits for the chain through all layers: f16's differences add up in the residual stream and
/// get amplified by Julia's massive activations (the emulation reaches 8e-3 in layer 18's GeGLU,
/// whose largest values are ~2000 against an rms of 12). Loose: this checks the data flow
/// between the GEMMs (statistics, clearing, the in-place residual), not precision.
const MAX_ERR_CHAINED: LayerErrors = LayerErrors {
    qkv: 3e-2,
    x_mid: 3e-2,
    m: 3e-2,
    x_out: 3e-2,
    stats: 1e-4,
};

fn within(e: LayerErrors, max: LayerErrors) -> bool {
    e.qkv < max.qkv
        && e.x_mid < max.x_mid
        && e.m < max.m
        && e.x_out < max.x_out
        && e.stats < max.stats
}

/// All layers, with julia-rs's attention outputs. `chained`: each layer on the stream the
/// previous layer left, as the forward pass will run them (starting from whatever `g` holds);
/// otherwise each layer from julia-rs's stream entering it. Prints a row per layer.
pub fn layers(
    g: &mut impl LayerGemms,
    traces: &[LayerTrace],
    chained: bool,
) -> color_eyre::Result<LayerErrors> {
    let how = match chained {
        true => "chained from embed's output",
        false => "each from julia-rs's stream",
    };
    eprintln!(
        "layer GEMMs vs julia-rs, {how} (relative error; stats: resstats row sums vs Σ of the output)"
    );
    eprintln!("  layer     qkv     x+o·Wo   GeGLU m  x+m·Wo2  stats");
    let mut worst = LayerErrors::default();
    for (l, t) in traces.iter().enumerate() {
        if !chained {
            g.set_stream(&to_f16(&t.x_in))?;
        }
        let e = layer(g, l, t)?;
        eprintln!(
            "  {l:5}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.1e}",
            e.qkv, e.x_mid, e.m, e.x_out, e.stats
        );
        worst = worst.max(e);
    }
    let max = if chained { MAX_ERR_CHAINED } else { MAX_ERR };
    ensure!(
        within(worst, max),
        "a layer GEMM is off julia-rs: worst {worst}; limits {max}"
    );
    Ok(worst)
}

/// The GEMMs on the GPU, with the forward pass's buffers: the stream `x`, and the two
/// statistics buffers in turn (wo writes `stats_attn` for wi, wo_mlp `stats_mlp` for the next
/// qkv; each is cleared by the GEMM before its writer: qkv clears `stats_attn`, wi `stats_mlp`).
pub struct GpuGemms<'a> {
    stream: &'a Arc<CudaStream>,
    kernels: &'a GemmKernels,
    scratch: &'a gemm::Scratch,
    w: &'a GpuWeights,
    eps: f32,
    /// Overrides `gemm::choose` for every GEMM (for sweeps).
    pub config: Option<Config>,
    sms: usize,
    pub x: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    o: CudaSlice<f16>,
    m: CudaSlice<f16>,
    stats_attn: CudaSlice<f32>,
    stats_mlp: CudaSlice<f32>,
}

impl<'a> GpuGemms<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: &'a Arc<CudaStream>,
        kernels: &'a GemmKernels,
        scratch: &'a gemm::Scratch,
        w: &'a GpuWeights,
        eps: f32,
        sms: usize,
        x: CudaSlice<f16>,
        inter: usize,
    ) -> color_eyre::Result<Self> {
        let n = x.len() / D;
        Ok(Self {
            stream,
            kernels,
            scratch,
            w,
            eps,
            config: None,
            sms,
            qkv: stream.alloc_zeros(n * 3 * D)?,
            o: stream.alloc_zeros(n * D)?,
            m: stream.alloc_zeros(n * inter)?,
            stats_attn: stream.alloc_zeros(2 * n)?,
            stats_mlp: stream.alloc_zeros(2 * n)?,
            x,
        })
    }

    fn rows(&self) -> usize {
        self.x.len() / D
    }

    pub fn config_for(&self, n: usize, k: usize) -> Config {
        self.config
            .unwrap_or_else(|| gemm::choose(self.rows(), n, k, self.sms))
    }

    /// Queues layer `l`'s qkv GEMM (the stream → `qkv`).
    pub fn launch_qkv(&mut self, l: usize) -> color_eyre::Result<()> {
        let lay = &self.w.layout.layers[l];
        let w = self.w.f16.slice(lay.wqkv.range());
        let colsum = lay.qkv_colsum.map(|c| self.w.f32.slice(c.range()));
        let cfg = self.config_for(3 * D, D);
        let g = Gemm {
            epi: if colsum.is_some() {
                Epilogue::Ln
            } else {
                Epilogue::Plain
            },
            a: &self.x,
            w: &w,
            k: D,
            c: &mut self.qkv,
            ln: colsum.as_ref().map(|colsum| LnInputs {
                stats: &self.stats_mlp,
                colsum,
                eps: self.eps,
            }),
            stats_out: None,
            zero: Some(&mut self.stats_attn),
        };
        gemm::launch(self.stream, self.kernels, self.scratch, cfg, g)
    }

    /// Queues layer `l`'s Wo GEMM (stream += o · Woᵀ).
    pub fn launch_wo(&mut self, l: usize) -> color_eyre::Result<()> {
        let w = self.w.f16.slice(self.w.layout.layers[l].wo.range());
        let cfg = self.config_for(D, D);
        let g = Gemm {
            epi: Epilogue::ResStats,
            a: &self.o,
            w: &w,
            k: D,
            c: &mut self.x,
            ln: None,
            stats_out: Some(&mut self.stats_attn),
            zero: None,
        };
        gemm::launch(self.stream, self.kernels, self.scratch, cfg, g)
    }

    /// Queues layer `l`'s Wi GEMM (m = GeGLU(LN(stream) · Wiᵀ)).
    pub fn launch_wi(&mut self, l: usize) -> color_eyre::Result<()> {
        let lay = &self.w.layout.layers[l];
        let w = self.w.f16.slice(lay.wi.range());
        let colsum = self.w.f32.slice(lay.wi_colsum.range());
        let inter = self.m.len() / self.rows();
        let cfg = self.config_for(2 * inter, D);
        let g = Gemm {
            epi: Epilogue::LnGeglu,
            a: &self.x,
            w: &w,
            k: D,
            c: &mut self.m,
            ln: Some(LnInputs {
                stats: &self.stats_attn,
                colsum: &colsum,
                eps: self.eps,
            }),
            stats_out: None,
            zero: Some(&mut self.stats_mlp),
        };
        gemm::launch(self.stream, self.kernels, self.scratch, cfg, g)
    }

    /// Queues layer `l`'s Wo_mlp GEMM (stream += m · Wo_mlpᵀ).
    pub fn launch_wo_mlp(&mut self, l: usize) -> color_eyre::Result<()> {
        let w = self.w.f16.slice(self.w.layout.layers[l].wo_mlp.range());
        let inter = self.m.len() / self.rows();
        let cfg = self.config_for(D, inter);
        let g = Gemm {
            epi: Epilogue::ResStats,
            a: &self.m,
            w: &w,
            k: inter,
            c: &mut self.x,
            ln: None,
            stats_out: Some(&mut self.stats_mlp),
            zero: None,
        };
        gemm::launch(self.stream, self.kernels, self.scratch, cfg, g)
    }
}

impl LayerGemms for GpuGemms<'_> {
    fn set_stream(&mut self, x: &[f16]) -> color_eyre::Result<()> {
        self.stream.memcpy_htod(x, &mut self.x)?;
        self.stream
            .memcpy_htod(&row_stats(x, D), &mut self.stats_mlp)?;
        Ok(())
    }

    fn qkv(&mut self, l: usize) -> color_eyre::Result<Vec<f16>> {
        self.launch_qkv(l)?;
        Ok(self.stream.clone_dtoh(&self.qkv)?)
    }

    fn wo(&mut self, l: usize, o: &[f16]) -> color_eyre::Result<(Vec<f16>, Vec<f32>)> {
        self.stream.memcpy_htod(o, &mut self.o)?;
        self.launch_wo(l)?;
        Ok((
            self.stream.clone_dtoh(&self.x)?,
            self.stream.clone_dtoh(&self.stats_attn)?,
        ))
    }

    fn wi(&mut self, l: usize) -> color_eyre::Result<Vec<f16>> {
        self.launch_wi(l)?;
        Ok(self.stream.clone_dtoh(&self.m)?)
    }

    fn wo_mlp(&mut self, l: usize) -> color_eyre::Result<(Vec<f16>, Vec<f32>)> {
        self.launch_wo_mlp(l)?;
        Ok((
            self.stream.clone_dtoh(&self.x)?,
            self.stream.clone_dtoh(&self.stats_mlp)?,
        ))
    }
}

/// Every configuration (shape × split-K) on one layer's GEMMs, each from julia-rs's stream
/// entering that layer: shape- and split-specific bugs show here, while `layers` runs only the
/// chosen configurations.
pub fn sweep(g: &mut GpuGemms, l: usize, t: &LayerTrace) -> color_eyre::Result<()> {
    eprintln!("every GEMM configuration on layer {l}:");
    let mut bad = vec![];
    for cfg in gemm::configs() {
        g.config = Some(cfg);
        g.set_stream(&to_f16(&t.x_in))?;
        let e = layer(g, l, t)?;
        eprintln!(
            "  {:<18} qkv {:.2e}  x+o·Wo {:.2e}  m {:.2e}  x+m·Wo2 {:.2e}  stats {:.1e}",
            cfg.to_string(),
            e.qkv,
            e.x_mid,
            e.m,
            e.x_out,
            e.stats
        );
        if !within(e, MAX_ERR) {
            bad.push(cfg.to_string());
        }
    }
    g.config = None;
    ensure!(
        bad.is_empty(),
        "these configurations are off: {}",
        bad.join(", ")
    );
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

/// The GEMMs' arithmetic on the host, for testing the checks: f16 inputs and outputs, exact
/// products summed in f64 (the GPU sums in f32), the epilogues as the kernels compute them,
/// including the GeGLU interleave.
#[cfg(test)]
struct HostGemms<'a> {
    host: &'a HostWeights,
    eps: f32,
    inter: usize,
    x: Vec<f16>,
    m: Vec<f16>,
    stats_attn: Vec<f32>,
    stats_mlp: Vec<f32>,
    /// Reads Wi's output in checkpoint order (as if it weren't interleaved): a bug to catch.
    naive_geglu: bool,
}

#[cfg(test)]
impl HostGemms<'_> {
    /// `a · wᵀ` with `a` `[m, k]` and `w` `[n, k]`.
    fn matmul(a: &[f16], w: &[f16], k: usize) -> Vec<f32> {
        let w: Vec<f32> = to_f32(w);
        (a.chunks_exact(k))
            .flat_map(|row| {
                let row = to_f32(row);
                w.chunks_exact(k)
                    .map(move |wr| {
                        let s: f64 = row.iter().zip(wr).map(|(&a, &b)| f64::from(a * b)).sum();
                        s as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The ln epilogue: acc → rstd · (acc − mean · colsum), from the row statistics.
    fn ln(acc: &mut [f32], stats: &[f32], colsum: &[f32], k: usize, eps: f32) {
        for (row, &[s, q]) in acc
            .chunks_exact_mut(colsum.len())
            .zip(stats.as_chunks::<2>().0)
        {
            let mean = s / k as f32;
            let rstd = 1.0 / (q / k as f32 - mean * mean + eps).max(eps).sqrt();
            for (v, &c) in row.iter_mut().zip(colsum) {
                *v = rstd * (*v - mean * c);
            }
        }
    }

    /// GELU by the same Abramowitz & Stegun formula as the kernel, in f64.
    fn gelu(x: f32) -> f32 {
        let x = f64::from(x);
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let poly = [
            1.061405429,
            -1.453152027,
            1.421413741,
            -0.284496736,
            0.254829592,
        ];
        let p = poly.iter().fold(0.0, |p, &a| p * t + a) * t;
        let q = 0.5 * p * (-x * x / 2.0).exp();
        (x * if x >= 0.0 { 1.0 - q } else { q }) as f32
    }

    fn residual(&mut self, acc: &[f32]) -> (Vec<f16>, Vec<f32>) {
        for (x, &a) in self.x.iter_mut().zip(acc) {
            *x = f16::from_f32(a + x.to_f32());
        }
        (self.x.clone(), row_stats(&self.x, D))
    }
}

#[cfg(test)]
impl LayerGemms for HostGemms<'_> {
    fn set_stream(&mut self, x: &[f16]) -> color_eyre::Result<()> {
        self.x = x.to_vec();
        self.stats_mlp = row_stats(x, D);
        Ok(())
    }

    fn qkv(&mut self, l: usize) -> color_eyre::Result<Vec<f16>> {
        let (lay, h) = (&self.host.layout.layers[l], self.host);
        let mut acc = Self::matmul(&self.x, &h.f16[lay.wqkv.range()], D);
        if let Some(c) = lay.qkv_colsum {
            Self::ln(&mut acc, &self.stats_mlp, &h.f32[c.range()], D, self.eps);
        }
        self.stats_attn.fill(0.0);
        Ok(to_f16(&acc))
    }

    fn wo(&mut self, l: usize, o: &[f16]) -> color_eyre::Result<(Vec<f16>, Vec<f32>)> {
        let acc = Self::matmul(o, &self.host.f16[self.host.layout.layers[l].wo.range()], D);
        let (x, s) = self.residual(&acc);
        self.stats_attn.copy_from_slice(&s);
        Ok((x, s))
    }

    fn wi(&mut self, l: usize) -> color_eyre::Result<Vec<f16>> {
        let (lay, h, inter) = (&self.host.layout.layers[l], self.host, self.inter);
        let mut acc = Self::matmul(&self.x, &h.f16[lay.wi.range()], D);
        Self::ln(
            &mut acc,
            &self.stats_attn,
            &h.f32[lay.wi_colsum.range()],
            D,
            self.eps,
        );
        self.m = (acc.chunks_exact(2 * inter))
            .flat_map(|row| {
                (0..inter).map(|c| {
                    // Output column c = 32t + 16h + 8q + e takes packed rows 64t + 32h + 16q + e
                    // (input) and that + 8 (gate): prep::interleave_geglu.
                    let (t, hh, q, e) = (c / 32, (c % 32) / 16, (c % 16) / 8, c % 8);
                    let p = 64 * t + 32 * hh + 16 * q + e;
                    let (x, g) = if self.naive_geglu {
                        (row[c], row[inter + c])
                    } else {
                        (row[p], row[p + 8])
                    };
                    f16::from_f32(Self::gelu(x) * g)
                })
            })
            .collect();
        self.stats_mlp.fill(0.0);
        Ok(self.m.clone())
    }

    fn wo_mlp(&mut self, l: usize) -> color_eyre::Result<(Vec<f16>, Vec<f32>)> {
        let w = &self.host.f16[self.host.layout.layers[l].wo_mlp.range()];
        let acc = Self::matmul(&self.m, w, self.inter);
        let (x, s) = self.residual(&acc);
        self.stats_mlp.copy_from_slice(&s);
        Ok((x, s))
    }
}

#[cfg(test)]
mod layer_tests {
    use std::path::{Path, PathBuf};

    use julia_rs::sequence::{DEFAULT_HEAD_LENGTH, sequence};

    use super::*;
    use crate::checkpoint::Checkpoint;

    /// The layer check on the host emulation of the kernels: it must pass (this is where
    /// the limits' margins come from), and a GeGLU read in the wrong order must fail.
    #[test]
    fn layer_check_passes_the_emulation() {
        let dir = std::env::var_os("JULIA_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("model"));
        if !dir.join("model.safetensors").exists() {
            eprintln!("skipped: no checkpoint at {}", dir.display());
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
        let n = seq.ids.len();
        let traces = trace(&model, &seq.ids);
        let gemms = |naive_geglu| HostGemms {
            host: &host,
            eps: c.norm_eps,
            inter: c.intermediate,
            x: emulate_embed(&host, &seq.ids, c.norm_eps),
            m: vec![],
            stats_attn: vec![0.0; 2 * n],
            stats_mlp: vec![0.0; 2 * n],
            naive_geglu,
        };
        let worst = layers(&mut gemms(false), &traces, false).unwrap();
        eprintln!("worst, per layer: {worst}");
        let worst = layers(&mut gemms(false), &traces, true).unwrap();
        eprintln!("worst, chained: {worst}");
        let e = layer(&mut gemms(true), 0, &traces[0]).unwrap();
        eprintln!("naive GeGLU order, layer 0: {e}");
        assert!(!within(e, MAX_ERR_CHAINED));
    }
}
