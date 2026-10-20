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
    Model, ModelPaths,
    attn::{Rope, Scratch as AttnScratch, attention, attention_q},
    head::{QType, answer},
    load_model,
    matmul::{Precision, linear, linear_add},
    mlp::{geglu, gelu_inplace},
};

use crate::{D, gemm, weights::HostWeights};

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
    // Wqkv's q and k rows are permuted for RoPE, which this kernel doesn't apply.
    let got = to_f32(&unpermute_rope(qkv));
    let mut want = vec![0.0; ids.len() * 3 * D];
    linear(
        &model.embed(ids),
        &model.packed.encoder[0].wqkv,
        None,
        &mut want,
    );
    let rel = rel_err(&got, &want);

    let emu = emulate_gemm(x, &host.f16[host.layout.layers[0].wqkv.range()], D);
    let rel_emu = rel_err(&to_f32(qkv), &to_f32(&emu));
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
    /// LN(x_in) · Wqkvᵀ (no LayerNorm in layer 0), after RoPE (on q and k).
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
            Some(g) => layernorm_rows(&x, g.as_f32(data), None, eps, &mut h),
            None => h.copy_from_slice(&x),
        }
        let mut qkv = vec![0.0; n * 3 * d];
        linear(&h, &p.wqkv, None, &mut qkv);
        let (rope, window) = match l.global {
            true => (&rope_g, None),
            false => (&rope_l, Some(model.config.local_window / 2)),
        };
        rope.apply(&mut qkv, d);
        let qkv = qkv;
        let mut o = vec![0.0; n * d];
        attention(&qkv, n, d, window, &mut o, &mut s);
        linear_add(&o, &p.wo, None, &mut x);
        let x_mid = x.clone();
        layernorm_rows(&x, l.mlp_norm.as_f32(data), None, eps, &mut h);
        let (mut hh, mut m) = (vec![0.0; n * 2 * inter], vec![0.0; n * inter]);
        linear(&h, &p.wi, None, &mut hh);
        geglu(&hh, inter, &mut m);
        linear_add(&m, &p.wo_mlp, None, &mut x);
        out.push(LayerTrace {
            x_in,
            qkv: qkv.clone(),
            o,
            x_mid,
            m,
            x_out: x.clone(),
        });
    }
    out
}

/// LayerNorm of each `gamma.len()`-wide row (with bias `beta`, if any), in f64.
pub fn layernorm_rows(x: &[f32], gamma: &[f32], beta: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let k = gamma.len();
    for (row, out) in x.chunks_exact(k).zip(out.chunks_exact_mut(k)) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / k as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / k as f64;
        let r = 1.0 / (var + f64::from(eps)).sqrt();
        for (i, (o, &v)) in out.iter_mut().zip(row).enumerate() {
            let b = beta.map_or(0.0, |b| f64::from(b[i]));
            *o = ((f64::from(v) - mean) * r * f64::from(gamma[i]) + b) as f32;
        }
    }
}

/// `qkv` (rows of `[q | k | v]`) from the GPU's order, with q and k permuted for RoPE
/// (`prep::permute_rope`), back to the checkpoint's.
pub fn unpermute_rope<T: Copy + Default>(qkv: &[T]) -> Vec<T> {
    let order = crate::prep::permute_rope(&(0..3 * D).collect::<Vec<_>>(), 1);
    let mut out = vec![T::default(); qkv.len()];
    let (rows, []) = qkv.as_chunks::<{ 3 * D }>() else {
        panic!("qkv is not whole rows")
    };
    for (src, dst) in rows.iter().zip(out.as_chunks_mut::<{ 3 * D }>().0) {
        for (p, &o) in order.iter().enumerate() {
            dst[o] = src[p];
        }
    }
    out
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
/// One encoder layer on a residual stream the implementation keeps (on the GPU, or emulated on
/// the host): the four GEMMs and attention.
pub trait EncoderSteps {
    /// Replaces the stream with `x` (and the row statistics the next qkv reads with its own).
    fn set_stream(&mut self, x: &[f16]) -> color_eyre::Result<()>;
    /// The stream's qkv for layer `l` (LayerNorm folded in from layer 1, RoPE applied), q and k
    /// in the permuted order.
    fn qkv(&mut self, l: usize) -> color_eyre::Result<Vec<f16>>;
    /// Attention over the last qkv (global or sliding-window as layer `l` is).
    fn attention(&mut self, l: usize) -> color_eyre::Result<Vec<f16>>;
    /// stream += o · Woᵀ, with `o` given or (`None`) from `attention`; returns the stream and its
    /// row statistics.
    fn wo(&mut self, l: usize, o: Option<&[f16]>) -> color_eyre::Result<(Vec<f16>, Vec<f32>)>;
    /// m = GeGLU(LN(stream) · Wiᵀ).
    fn wi(&mut self, l: usize) -> color_eyre::Result<Vec<f16>>;
    /// stream += m · Wo_mlpᵀ; returns the stream and its row statistics.
    fn wo_mlp(&mut self, l: usize) -> color_eyre::Result<(Vec<f16>, Vec<f32>)>;
}

/// Relative errors of one layer's steps against julia-rs.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerErrors {
    pub qkv: f64,
    /// The attention output (0 when julia-rs's is used).
    pub o: f64,
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
            o: self.o.max(o.o),
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
            "qkv {:.2e}, o {:.2e}, x+o·Wo {:.2e}, m {:.2e}, x+m·Wo2 {:.2e}, stats {:.1e}",
            self.qkv, self.o, self.x_mid, self.m, self.x_out, self.stats
        )
    }
}

/// Layer `l`, from whatever stream `g` holds, against `t`; attention from `g` if
/// `own_attention`, else julia-rs's.
pub fn layer(
    g: &mut impl EncoderSteps,
    l: usize,
    t: &LayerTrace,
    own_attention: bool,
) -> color_eyre::Result<LayerErrors> {
    let qkv = g.qkv(l)?;
    let (o_err, (x_mid, s1)) = if own_attention {
        let o = g.attention(l)?;
        (rel_err(&to_f32(&o), &t.o), g.wo(l, None)?)
    } else {
        (0.0, g.wo(l, Some(&to_f16(&t.o)))?)
    };
    let m = g.wi(l)?;
    let (x_out, s2) = g.wo_mlp(l)?;
    Ok(LayerErrors {
        qkv: rel_err(&to_f32(&unpermute_rope(&qkv)), &t.qkv),
        o: o_err,
        x_mid: rel_err(&to_f32(&x_mid), &t.x_mid),
        m: rel_err(&to_f32(&m), &t.m),
        x_out: rel_err(&to_f32(&x_out), &t.x_out),
        stats: stats_err(&x_mid, &s1, D).max(stats_err(&x_out, &s2, D)),
    })
}

/// Limits for a layer run from julia-rs's stream: about 3× what the host emulation of the
/// kernels' arithmetic reaches (≤ 6e-4 for the GEMMs, see the test), so a correct kernel passes
/// and an indexing bug (errors of order 1) can't. Attention's P is rounded to f16 for the P·V
/// product (as in FlashAttention), so its output is a little further off.
const MAX_ERR: LayerErrors = LayerErrors {
    qkv: 2e-3,
    o: 4e-3,
    x_mid: 2e-3,
    m: 2e-3,
    x_out: 2e-3,
    stats: 1e-4,
};

/// Limits for the chain through all layers: f16's differences add up in the residual stream and
/// get amplified by Julia's massive activations (the emulation reaches 8e-3 in layer 18's GeGLU,
/// whose largest values are ~2000 against an rms of 12). Loose: this checks the data flow
/// between the steps (statistics, clearing, the in-place residual), not precision.
const MAX_ERR_CHAINED: LayerErrors = LayerErrors {
    qkv: 3e-2,
    o: 3e-2,
    x_mid: 3e-2,
    m: 3e-2,
    x_out: 3e-2,
    stats: 1e-4,
};

fn within(e: LayerErrors, max: LayerErrors) -> bool {
    e.qkv < max.qkv
        && e.o < max.o
        && e.x_mid < max.x_mid
        && e.m < max.m
        && e.x_out < max.x_out
        && e.stats < max.stats
}

/// How `layers` runs the layers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Each layer from julia-rs's stream entering it, with julia-rs's attention output: the
    /// GEMMs alone.
    Gemms,
    /// Each layer from julia-rs's stream, attention included.
    Layers,
    /// Each layer on the stream the previous one left, from whatever `g` holds (embed's
    /// output): the encoder as the forward pass runs it.
    Encoder,
}

/// All layers in `mode`, against julia-rs. Prints a row per layer (or, `rows` false, only the
/// worst).
pub fn layers(
    g: &mut impl EncoderSteps,
    traces: &[LayerTrace],
    mode: Mode,
    rows: bool,
) -> color_eyre::Result<LayerErrors> {
    let how = match mode {
        Mode::Gemms => "the GEMMs, each layer from julia-rs's stream and attention output",
        Mode::Layers => "each layer from julia-rs's stream, attention on the GPU",
        Mode::Encoder => "the encoder, chained from embed's output",
    };
    let n = traces.first().map_or(0, |t| t.x_in.len() / D);
    eprintln!(
        "{how}, {n} tokens (relative error vs julia-rs; stats: resstats row sums vs Σ of the output)"
    );
    if rows {
        eprintln!("  layer     qkv      attn o   x+o·Wo   GeGLU m  x+m·Wo2  stats");
    }
    let mut worst = LayerErrors::default();
    for (l, t) in traces.iter().enumerate() {
        if mode != Mode::Encoder {
            g.set_stream(&to_f16(&t.x_in))?;
        }
        let e = layer(g, l, t, mode != Mode::Gemms)?;
        if rows {
            eprintln!(
                "  {l:5}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.2e}  {:.1e}",
                e.qkv, e.o, e.x_mid, e.m, e.x_out, e.stats
            );
        }
        worst = worst.max(e);
    }
    eprintln!("  worst: {worst}");
    let max = if mode == Mode::Encoder {
        MAX_ERR_CHAINED
    } else {
        MAX_ERR
    };
    ensure!(
        within(worst, max),
        "a layer is off julia-rs: worst {worst}; limits {max}"
    );
    Ok(worst)
}

/// julia-rs's head on the encoder output `x_enc` (`[n, D]`, before the final norm), keeping the
/// intermediates. As `Model::head`, but every head layer on all rows (the marker rows come out the
/// same) and LayerNorms in f64.
pub struct HeadTrace {
    /// After the final norm and the type embedding.
    pub x0: Vec<f32>,
    /// After each head layer.
    pub layers: Vec<Vec<f32>>,
    pub logits: Vec<f32>,
}

pub fn head_trace(model: &Model, x_enc: &[f32], qtype: QType, markers: &[usize]) -> HeadTrace {
    let (data, w, d) = (model.data(), &model.weights, D);
    let n = x_enc.len() / d;
    let eps = crate::forward::HEAD_LN_EPS;
    let mut x = vec![0.0; n * d];
    let final_norm = w.encoder.final_norm.as_f32(data);
    layernorm_rows(x_enc, final_norm, None, model.config.norm_eps, &mut x);
    let te = &w.type_emb.as_f32(data)[qtype as usize * d..][..d];
    for row in x.chunks_exact_mut(d) {
        for (v, &t) in row.iter_mut().zip(te) {
            *v += t;
        }
    }
    let x0 = x.clone();
    let mut layers = vec![];
    let (mut h, mut q, mut a) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
    let mut kv = vec![0.0; n * 2 * d];
    let mut s = AttnScratch::default();
    for (l, p) in w.head.iter().zip(&model.packed.head) {
        layernorm_rows(
            &x,
            l.norm1_w.as_f32(data),
            Some(l.norm1_b.as_f32(data)),
            eps,
            &mut h,
        );
        let (bq, bkv) = l.in_proj_b.as_f32(data).split_at(d);
        linear(&h, &p.q, Some(bq), &mut q);
        linear(&h, &p.kv, Some(bkv), &mut kv);
        attention_q(&q, d, &kv, 2 * d, n, d, None, &mut a, &mut s);
        linear_add(&a, &p.out_proj, Some(l.out_proj_b.as_f32(data)), &mut x);
        layernorm_rows(
            &x,
            l.norm2_w.as_f32(data),
            Some(l.norm2_b.as_f32(data)),
            eps,
            &mut h,
        );
        let mut f = vec![0.0; n * model.config.head_ff];
        linear(&h, &p.linear1, Some(l.linear1_b.as_f32(data)), &mut f);
        f.iter_mut().for_each(|v| *v = v.max(0.0));
        linear_add(&f, &p.linear2, Some(l.linear2_b.as_f32(data)), &mut x);
        layers.push(x.clone());
    }
    let sc = &w.scorer;
    let z: Vec<f32> = markers
        .iter()
        .flat_map(|&r| x[r * d..][..d].to_vec())
        .collect();
    let mut zn = vec![0.0; z.len()];
    layernorm_rows(
        &z,
        sc.norm_w.as_f32(data),
        Some(sc.norm_b.as_f32(data)),
        eps,
        &mut zn,
    );
    let [lin1, lin2] = &model.packed.scorer;
    let mut y = vec![0.0; z.len()];
    linear(&zn, lin1, Some(sc.lin1_b.as_f32(data)), &mut y);
    gelu_inplace(&mut y);
    let mut logits = vec![0.0; markers.len()];
    linear(&y, lin2, Some(sc.lin2_b.as_f32(data)), &mut logits);
    HeadTrace { x0, layers, logits }
}

/// The largest |Δ| between two logit vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    (a.iter().zip(b))
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0, f64::max)
}

/// Logits within this of julia-rs's: an f16 CPU emulation of Julia (julia-rs rounding to f16
/// where the kernels do) moved them by 0.024 on average over the 2000 typed-decisions
/// questions, and changed no answer.
const MAX_LOGIT_DIFF: f64 = 0.15;

/// The head on the GPU (`g`, sized for `x_enc`'s rows and `markers`) from julia-rs's encoder
/// output `x_enc` (before the final norm), step by step against `head_trace`.
pub fn head(
    g: &mut crate::forward::GpuHead,
    model: &Model,
    x_enc: &[f32],
    qtype: QType,
    markers: &[usize],
) -> color_eyre::Result<()> {
    let t = head_trace(model, x_enc, qtype, markers);
    let x = g.stream.clone_htod(&to_f16(x_enc))?;
    g.launch_input(&x, qtype as usize)?;
    let mut errs = vec![rel_err(&to_f32(&g.stream.clone_dtoh(&g.x)?), &t.x0)];
    for (l, want) in t.layers.iter().enumerate() {
        g.launch_layer(l)?;
        errs.push(rel_err(&to_f32(&g.stream.clone_dtoh(&g.x)?), want));
    }
    g.launch_score()?;
    let logits = g.stream.clone_dtoh(&g.logits)?;
    let diff = max_abs_diff(&logits, &t.logits);
    eprintln!(
        "head from julia-rs's encoder output: stream {} (input, then after each layer); logits \
         {logits:.3?} vs {:.3?}, max |Δ| {diff:.1e}",
        errs.iter()
            .map(|e| format!("{e:.2e}"))
            .collect::<Vec<_>>()
            .join(", "),
        t.logits
    );
    ensure!(
        errs.iter().all(|&e| e < MAX_ERR.x_out) && diff < MAX_LOGIT_DIFF,
        "the head is off julia-rs"
    );
    Ok(())
}

/// The whole forward pass's logits (`logits`, from the GPU) against julia-rs's on `ids`.
pub fn forward(
    model: &Model,
    ids: &[u32],
    qtype: QType,
    markers: &[usize],
    logits: &[f32],
) -> color_eyre::Result<()> {
    let want = model.forward(ids, qtype, markers);
    let diff = max_abs_diff(logits, &want);
    let (a, pa) = answer(qtype, logits);
    let (b, pb) = answer(qtype, &want);
    eprintln!(
        "forward pass, {} tokens: logits {logits:.3?} vs julia-rs {want:.3?}, max |Δ| {diff:.1e}; \
         answer {a:?} vs {b:?} (probabilities {pa:.3?} vs {pb:.3?})",
        ids.len()
    );
    ensure!(
        diff < MAX_LOGIT_DIFF && a == b,
        "the forward pass is off julia-rs"
    );
    Ok(())
}

impl EncoderSteps for crate::forward::GpuEncoder<'_> {
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

    fn attention(&mut self, l: usize) -> color_eyre::Result<Vec<f16>> {
        self.launch_attention(l)?;
        Ok(self.stream.clone_dtoh(&self.o)?)
    }

    fn wo(&mut self, l: usize, o: Option<&[f16]>) -> color_eyre::Result<(Vec<f16>, Vec<f32>)> {
        if let Some(o) = o {
            self.stream.memcpy_htod(o, &mut self.o)?;
        }
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
pub fn sweep(
    g: &mut crate::forward::GpuEncoder,
    l: usize,
    t: &LayerTrace,
) -> color_eyre::Result<()> {
    eprintln!("every GEMM configuration on layer {l}:");
    let mut bad = vec![];
    for cfg in gemm::configs() {
        g.config = Some(cfg);
        g.set_stream(&to_f16(&t.x_in))?;
        let e = layer(g, l, t, false)?;
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
struct HostEncoder<'a> {
    host: &'a HostWeights,
    eps: f32,
    inter: usize,
    /// The sliding-window layers' reach.
    window: usize,
    x: Vec<f16>,
    qkv: Vec<f16>,
    o: Vec<f16>,
    m: Vec<f16>,
    stats_attn: Vec<f32>,
    stats_mlp: Vec<f32>,
    /// Reads Wi's output in checkpoint order (as if it weren't interleaved): a bug to catch.
    naive_geglu: bool,
    /// Leaves RoPE out: another.
    no_rope: bool,
    /// Lets every layer attend to all keys: another.
    no_window: bool,
}

#[cfg(test)]
impl HostEncoder<'_> {
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
impl EncoderSteps for HostEncoder<'_> {
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
        // RoPE on the permuted pairs (2i, 2i + 1) of each q and k head; row r is position r.
        let table = match lay.global {
            true => h.layout.rope_global,
            false => h.layout.rope_local,
        };
        let table = &h.f32[table.range()];
        let rows = if self.no_rope {
            &mut [][..]
        } else {
            acc.as_chunks_mut::<{ 3 * D }>().0
        };
        for (r, row) in rows.iter_mut().enumerate() {
            for (p, pair) in row[..2 * D].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let t = (r * 32 + p % 32) * 2;
                let (cs, sn) = (table[t], table[t + 1]);
                let [x1, x2] = *pair;
                *pair = [x1 * cs - x2 * sn, x2 * cs + x1 * sn];
            }
        }
        self.stats_attn.fill(0.0);
        self.qkv = to_f16(&acc);
        Ok(self.qkv.clone())
    }

    /// Softmax in f32 from f16 q, k (the dot products don't see the RoPE permutation), P rounded
    /// to f16 for P·V as the kernel does, f32 sums.
    fn attention(&mut self, l: usize) -> color_eyre::Result<Vec<f16>> {
        let n = self.qkv.len() / (3 * D);
        let global = self.no_window || self.host.layout.layers[l].global;
        let row = |i: usize| &self.qkv[i * 3 * D..][..3 * D];
        let mut o = vec![f16::ZERO; n * D];
        for h in 0..D / 64 {
            for i in 0..n {
                let q = to_f32(&row(i)[h * 64..][..64]);
                let keys: Vec<usize> = (0..n)
                    .filter(|&j| global || i.abs_diff(j) <= self.window)
                    .collect();
                let s: Vec<f32> = (keys.iter())
                    .map(|&j| {
                        let k = &row(j)[D + h * 64..][..64];
                        q.iter().zip(k).map(|(a, b)| a * b.to_f32()).sum::<f32>() / 8.0
                    })
                    .collect();
                let mx = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let p: Vec<f32> = s.iter().map(|&v| (v - mx).exp()).collect();
                let sum: f32 = p.iter().sum();
                let mut acc = [0f32; 64];
                for (&j, &p) in keys.iter().zip(&p) {
                    let p = f16::from_f32(p).to_f32();
                    for (a, v) in acc.iter_mut().zip(&row(j)[2 * D + h * 64..][..64]) {
                        *a += p * v.to_f32();
                    }
                }
                for (dst, a) in o[i * D + h * 64..][..64].iter_mut().zip(acc) {
                    *dst = f16::from_f32(a / sum);
                }
            }
        }
        self.o = o.clone();
        Ok(o)
    }

    fn wo(&mut self, l: usize, o: Option<&[f16]>) -> color_eyre::Result<(Vec<f16>, Vec<f32>)> {
        let o = o.map_or_else(|| self.o.clone(), <[f16]>::to_vec);
        let acc = Self::matmul(&o, &self.host.f16[self.host.layout.layers[l].wo.range()], D);
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

    /// The layer checks on the host emulation of the kernels: they must pass (this is where the
    /// limits' margins come from), and a GeGLU read in the wrong order, a missing RoPE or a
    /// missing attention window must fail.
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
        let encoder = |ids: &[u32], bugs: [bool; 3]| HostEncoder {
            host: &host,
            eps: c.norm_eps,
            inter: c.intermediate,
            window: c.local_window / 2,
            x: emulate_embed(&host, ids, c.norm_eps),
            qkv: vec![],
            o: vec![],
            m: vec![],
            stats_attn: vec![0.0; 2 * ids.len()],
            stats_mlp: vec![0.0; 2 * ids.len()],
            naive_geglu: bugs[0],
            no_rope: bugs[1],
            no_window: bugs[2],
        };
        let ok = [false; 3];
        let traces = trace(&model, &seq.ids);
        for mode in [Mode::Gemms, Mode::Layers, Mode::Encoder] {
            layers(&mut encoder(&seq.ids, ok), &traces, mode, false).unwrap();
        }
        let e = layer(
            &mut encoder(&seq.ids, [true, false, false]),
            0,
            &traces[0],
            false,
        )
        .unwrap();
        eprintln!("naive GeGLU order, layer 0: {e}");
        assert!(!within(e, MAX_ERR_CHAINED));
        let e = layer(
            &mut encoder(&seq.ids, [false, true, false]),
            0,
            &traces[0],
            false,
        )
        .unwrap();
        eprintln!("no RoPE, layer 0: {e}");
        assert!(!within(e, MAX_ERR_CHAINED));

        // The window, on a request long enough for it to matter: layer 0 is global, layer 1
        // sliding-window.
        let long = sequence(
            &ck.tokenizer,
            &crate::long_request(),
            c.max_positions,
            DEFAULT_HEAD_LENGTH,
        )
        .unwrap();
        let traces = trace(&model, &long.ids);
        assert!(host.layout.layers[0].global && !host.layout.layers[1].global);
        eprintln!("{} tokens:", long.ids.len());
        for (l, t) in traces[..2].iter().enumerate() {
            let mut g = encoder(&long.ids, ok);
            g.set_stream(&to_f16(&t.x_in)).unwrap();
            let e = layer(&mut g, l, t, true).unwrap();
            eprintln!("  layer {l}: {e}");
            assert!(within(e, MAX_ERR));
        }
        // The head reference (all rows, f64 LayerNorms) against julia-rs's own forward pass.
        let t = head_trace(&model, &traces[21].x_out, long.qtype, &long.markers);
        let want = model.forward(&long.ids, long.qtype, &long.markers);
        let diff = max_abs_diff(&t.logits, &want);
        eprintln!(
            "head reference: logits {:.4?} vs {want:.4?}, max |Δ| {diff:.1e}",
            t.logits
        );
        assert!(diff < 1e-3);

        let mut g = encoder(&long.ids, [false, false, true]);
        g.set_stream(&to_f16(&traces[1].x_in)).unwrap();
        let e = layer(&mut g, 1, &traces[1], true).unwrap();
        eprintln!("  no window, layer 1: {e}");
        assert!(!within(e, MAX_ERR));
    }
}
