//! The forward pass on the GPU, as separate launches: the encoder layers (`GpuEncoder`) and the
//! head (`GpuHead`), each with the buffers it runs on.

use std::sync::Arc;

use color_eyre::eyre::ensure;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::{
    D,
    gemm::{self, Config, Epilogue, Gemm, LnInputs},
    weights::GpuWeights,
};

/// The encoder layers on the GPU, with the forward pass's buffers: the stream `x`, and the two
/// statistics buffers in turn (wo writes `stats_attn` for wi, wo_mlp `stats_mlp` for the next
/// qkv; each is cleared by the GEMM before its writer: qkv clears `stats_attn`, wi `stats_mlp`).
pub struct GpuEncoder<'a> {
    pub stream: &'a Arc<CudaStream>,
    pub kernels: &'a crate::Kernels,
    pub scratch: &'a gemm::Scratch,
    pub w: &'a GpuWeights,
    pub eps: f32,
    /// The sliding-window layers' reach: keys with |i − j| <= window.
    pub window: usize,
    /// Overrides `gemm::choose` for every GEMM (for sweeps).
    pub config: Option<Config>,
    pub sms: usize,
    pub x: CudaSlice<f16>,
    pub qkv: CudaSlice<f16>,
    pub o: CudaSlice<f16>,
    pub m: CudaSlice<f16>,
    pub stats_attn: CudaSlice<f32>,
    pub stats_mlp: CudaSlice<f32>,
    /// Run each layer as one persistent launch (`layer`)
    /// instead of five.
    pub fused: bool,
    /// The layer kernel's grid-barrier counter, and the arrivals counted so far.
    pub bar: CudaSlice<u32>,
    pub bar_count: u32,
    /// With `fused`: the layer kernel as five plain launches per layer, one phase each (same
    /// grid and tasks, launch boundaries in place of the grid barriers), for measurements.
    pub split_phases: bool,
    /// With `fused`: at most this many blocks (for measurements).
    pub grid_cap: Option<u32>,
    /// With `fused`: overrides `default_plan` (for sweeps).
    pub plan: Option<LayerPlan>,
    /// With `fused`: the original grid barrier, with two fences on each side (for comparison).
    pub two_fence_barrier: bool,
    /// With `fused`: the layer kernel's timestamps (`prof`), `PROF_WORDS` u64 per block of the
    /// largest grid, per layer; None: no profiling.
    pub prof: Option<CudaSlice<u64>>,
}

impl<'a> GpuEncoder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: &'a Arc<CudaStream>,
        kernels: &'a crate::Kernels,
        scratch: &'a gemm::Scratch,
        w: &'a GpuWeights,
        eps: f32,
        window: usize,
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
            window,
            config: None,
            sms,
            qkv: stream.alloc_zeros(n * 3 * D)?,
            o: stream.alloc_zeros(n * D)?,
            m: stream.alloc_zeros(n * inter)?,
            stats_attn: stream.alloc_zeros(2 * n)?,
            stats_mlp: stream.alloc_zeros(2 * n)?,
            fused: false,
            bar: stream.alloc_zeros(1)?,
            bar_count: 0,
            split_phases: false,
            grid_cap: None,
            plan: None,
            two_fence_barrier: false,
            prof: None,
            x,
        })
    }

    pub fn rows(&self) -> usize {
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
        let table = match lay.global {
            true => self.w.layout.rope_global,
            false => self.w.layout.rope_local,
        };
        let rope = self.w.f32.slice(table.range());
        let cfg = self.config_for(3 * D, D);
        let g = Gemm {
            epi: if colsum.is_some() {
                Epilogue::LnRope
            } else {
                Epilogue::Rope
            },
            rope: Some(&rope),
            bias: None,
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
        gemm::launch(self.stream, &self.kernels.gemm, self.scratch, cfg, g)
    }

    /// Queues layer `l`'s attention (qkv → o).
    pub fn launch_attention(&mut self, l: usize) -> color_eyre::Result<()> {
        let window = if self.w.layout.layers[l].global {
            0
        } else {
            self.window
        };
        crate::attn::launch(
            self.stream,
            &self.kernels.attention,
            &self.qkv,
            &mut self.o,
            window,
        )
    }

    /// Queues all of layer `l`: qkv, attention, wo, wi, wo_mlp; as one launch if `fused`.
    pub fn launch_layer(&mut self, l: usize) -> color_eyre::Result<()> {
        if self.fused {
            return self.launch_layer_fused(l);
        }
        self.launch_qkv(l)?;
        self.launch_attention(l)?;
        self.launch_wo(l)?;
        self.launch_wi(l)?;
        self.launch_wo_mlp(l)
    }

    /// The layer kernel's plan for this token count and GPU: for each GEMM, the tile height and
    /// split-K with the lowest `phase_cost`; for global attention, key ranges while the tasks
    /// still fit on the GPU at once, at least two key blocks each (on the 2080 Ti at 363 tokens,
    /// 3 ranges: 15.4 instead of 20.8 µs per global layer; at 72-607 tokens:
    /// the best count or within 0.4 µs per layer of it). Sliding-window layers keep whole key
    /// ranges: a tile reaches only 3 key blocks, and splitting them was never faster.
    pub fn default_plan(&self) -> LayerPlan {
        let n = self.rows();
        let inter = self.m.len() / n;
        let (sms, per_sm) = (self.sms, self.kernels.layer.1);
        let mut plan = LayerPlan {
            bm: [32; 4],
            splits: [1; 4],
            kv: [1; 2],
        };
        let shapes = [(3 * D, D), (D, D), (2 * inter, D), (D, inter)];
        for (g, &(cols, k)) in shapes.iter().enumerate() {
            (plan.bm[g], plan.splits[g]) = gemm_plan(n, cols, k, sms, per_sm);
        }
        let (blocks, tiles) = (n.div_ceil(64), n.div_ceil(64) * (D / 64));
        plan.kv[0] = (sms * per_sm / tiles).min(blocks.div_ceil(2)).clamp(1, 6) as u32;
        plan
    }

    pub fn plan(&self) -> LayerPlan {
        self.plan.unwrap_or_else(|| self.default_plan())
    }

    /// The tasks of each of the layer kernel's phases under `plan`, in a global (or
    /// sliding-window) layer.
    pub fn layer_tasks(&self, plan: LayerPlan, global: bool) -> [usize; 5] {
        let n = self.rows();
        let inter = self.m.len() / n;
        let gemm =
            |g: usize, cols: usize| (cols / 64) * n.div_ceil(plan.bm[g]) * plan.splits[g] as usize;
        let kv = plan.kv[if global { 0 } else { 1 }] as usize;
        [
            gemm(0, 3 * D),
            n.div_ceil(64) * (D / 64) * kv,
            gemm(1, D),
            gemm(2, 2 * inter),
            gemm(3, D),
        ]
    }

    /// The layer kernel's grid for layer `l`: as many blocks as fit on the GPU at once, but no
    /// more than the largest phase has tasks (and `grid_cap`).
    pub fn layer_grid(&self, l: usize) -> u32 {
        let tasks = self.layer_tasks(self.plan(), self.w.layout.layers[l].global);
        let grid = self.max_grid().min(tasks.into_iter().max().unwrap()) as u32;
        self.grid_cap.map_or(grid, |c| grid.min(c))
    }

    /// The most blocks the layer kernel can have.
    pub fn max_grid(&self) -> usize {
        self.sms * self.kernels.layer.1
    }

    /// Queues layer `l` as one cooperative launch of the persistent layer kernel.
    pub fn launch_layer_fused(&mut self, l: usize) -> color_eyre::Result<()> {
        let n = self.rows();
        let inter = self.m.len() / n;
        let (f, plan, grid) = (&self.kernels.layer.0, self.plan(), self.layer_grid(l));
        let sc = self.scratch;
        let lay = &self.w.layout.layers[l];
        let kv = plan.kv[if lay.global { 0 } else { 1 }];
        ensure!(
            plan.bm.iter().all(|&b| b == 32 || b == 64)
                && plan.splits.iter().all(|&z| (1..=4).contains(&z))
                && (1..=6).contains(&kv),
            "layer: no such plan: {plan}"
        );
        // Split-K partials (BM × 64 floats per tile and split) and key-range partials
        // (attn::PART floats per thread per tile and chunk), one counter per tile.
        let cols = [3 * D, D, 2 * inter, D];
        for g in 0..4 {
            let tiles = (cols[g] / 64) * n.div_ceil(plan.bm[g]);
            ensure!(
                tiles * plan.splits[g] as usize * plan.bm[g] * 64 <= sc.ws.len()
                    && tiles <= sc.counters.len(),
                "layer: the split-K scratch is too small for {n} tokens ({plan})"
            );
        }
        let attn_tiles = n.div_ceil(64) * (D / 64);
        ensure!(
            attn_tiles * kv as usize * crate::attn::PART * 128 <= sc.ws.len()
                && attn_tiles <= sc.counters.len(),
            "layer: the key-range scratch is too small for {n} tokens ({plan})"
        );
        let s = self.stream;
        let rope = match lay.global {
            true => self.w.layout.rope_global,
            false => self.w.layout.rope_local,
        };
        ensure!(
            rope.len >= n * 64,
            "layer: the RoPE table is too short for {n} positions"
        );
        let (f16s, f32s) = (&self.w.f16, &self.w.f32);
        let wptr = |r: std::ops::Range<usize>| gemm::ptr(s, &f16s.slice(r));
        let vptr = |r: std::ops::Range<usize>| gemm::ptr(s, &f32s.slice(r));
        let args: [u64; 16] = [
            gemm::ptr(s, &self.x),
            gemm::ptr(s, &self.qkv),
            gemm::ptr(s, &self.o),
            gemm::ptr(s, &self.m),
            gemm::ptr(s, &self.stats_attn),
            gemm::ptr(s, &self.stats_mlp),
            gemm::ptr(s, &sc.ws),
            gemm::ptr(s, &sc.counters),
            gemm::ptr(s, &self.bar),
            wptr(lay.wqkv.range()),
            lay.qkv_colsum.map_or(0, |c| vptr(c.range())),
            vptr(rope.range()),
            wptr(lay.wo.range()),
            wptr(lay.wi.range()),
            vptr(lay.wi_colsum.range()),
            wptr(lay.wo_mlp.range()),
        ];
        let window = if lay.global { 0 } else { self.window as u32 };
        let (n32, inter32) = (n as u32, inter as u32);
        let prof = match &self.prof {
            Some(p) => {
                let words = crate::profile::PROF_WORDS * self.max_grid();
                ensure!(
                    p.len() >= (l + 1) * words,
                    "layer: the profile buffer is too small"
                );
                gemm::ptr(s, p) + (l * words * 8) as u64
            }
            None => 0,
        };
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let pack = |v: [u32; 4]| v.iter().rev().fold(0, |a, &x| (a << 4) | x);
        let mis = pack(plan.bm.map(|b| (b / 32) as u32));
        let splits = pack(plan.splits);
        let old_barrier = if self.two_fence_barrier { 1 << 8 } else { 0 };
        // All five phases with grid barriers, or each on its own (no barriers).
        let launches: &[u32] = match self.split_phases {
            false => &[0x1f],
            true => &[1, 2, 4, 8, 16],
        };
        for &phases in launches {
            let phases = phases | old_barrier;
            let base = self.bar_count;
            let mut launch = s.launch_builder(f);
            // layer(x, qkv, o, mbuf, stats_attn, stats_mlp, ws, counters, bar, bar_base, wqkv,
            // qkv_colsum, rope, wo, wi, wi_colsum, wo_mlp, m, inter, eps, window, mis, splits,
            // kv_chunks, phases, prof)
            for a in &args[..9] {
                launch.arg(a);
            }
            launch.arg(&base);
            for a in &args[9..] {
                launch.arg(a);
            }
            launch
                .arg(&n32)
                .arg(&inter32)
                .arg(&self.eps)
                .arg(&window)
                .arg(&mis)
                .arg(&splits)
                .arg(&kv)
                .arg(&phases)
                .arg(&prof);
            // SAFETY: the buffers are the encoder's ([n, D] x and o, [n, 3D] qkv, [n, inter] m,
            // 2n statistics each) and the weights' spans for layer l, all 16-byte aligned; the
            // split-K / key-range scratch and the RoPE table are big enough for the plan (checked;
            // tile heights 32 or 64, 1-4 splits, 1-6 key ranges); qkv_colsum is null
            // only in layer 0, which has no LayerNorm; prof is null or has PROF_WORDS per block
            // for this layer (checked, grid <= max_grid). With barriers (more than one phase), a
            // cooperative launch with at most the resident number of blocks, so they can't
            // deadlock; one phase has none. 128 threads, as the kernel assumes.
            unsafe {
                match (phases & 0x1f).count_ones() {
                    1 => launch.launch(cfg),
                    _ => launch.launch_cooperative(cfg),
                }
            }?;
            let barriers = (phases & 0x1f).count_ones() - 1;
            self.bar_count = self.bar_count.wrapping_add(barriers * grid);
        }
        Ok(())
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
            rope: None,
            bias: None,
        };
        gemm::launch(self.stream, &self.kernels.gemm, self.scratch, cfg, g)
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
            rope: None,
            bias: None,
        };
        gemm::launch(self.stream, &self.kernels.gemm, self.scratch, cfg, g)
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
            rope: None,
            bias: None,
        };
        gemm::launch(self.stream, &self.kernels.gemm, self.scratch, cfg, g)
    }
}

/// `nn.LayerNorm`'s default eps: Julia's head and scorer LayerNorms don't set one (julia-rs's
/// `TORCH_LN_EPS`).
pub const HEAD_LN_EPS: f32 = 1e-5;

/// The head on the GPU: the encoder's final norm and the question type's embedding (`norm_add`),
/// the head layers (pre-LayerNorm transformer layers: in_proj, attention, out_proj, linear1,
/// ReLU, linear2), then the scorer (LayerNorm, lin1, GELU, then lin2 on the marker rows only:
/// `score`). Every head layer runs on all rows: julia-rs computes only the marker rows in the
/// last one, but rows don't interact after its attention, so theirs are the same.
///
/// The row statistics alternate between two buffers as in the encoder: `stats_a` feeds in_proj
/// (written by norm_add, then by linear2) and the scorer, `stats_b` feeds linear1 (written by
/// out_proj); in_proj clears `stats_b`, linear1 `stats_a`.
pub struct GpuHead<'a> {
    pub stream: &'a Arc<CudaStream>,
    pub kernels: &'a crate::Kernels,
    pub scratch: &'a gemm::Scratch,
    pub w: &'a GpuWeights,
    /// The encoder's final norm eps.
    pub eps: f32,
    pub sms: usize,
    /// The head's residual stream, `[n, D]`.
    pub x: CudaSlice<f16>,
    pub qkv: CudaSlice<f16>,
    pub o: CudaSlice<f16>,
    /// linear1's output, `[n, head_ff]`.
    pub f: CudaSlice<f16>,
    /// The scorer's hidden layer, `[n, D]`.
    pub y: CudaSlice<f16>,
    pub stats_a: CudaSlice<f32>,
    pub stats_b: CudaSlice<f32>,
    pub markers: CudaSlice<u32>,
    pub logits: CudaSlice<f32>,
}

impl<'a> GpuHead<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stream: &'a Arc<CudaStream>,
        kernels: &'a crate::Kernels,
        scratch: &'a gemm::Scratch,
        w: &'a GpuWeights,
        eps: f32,
        sms: usize,
        n: usize,
        markers: &[u32],
    ) -> color_eyre::Result<Self> {
        let ff = w.layout.head.first().map_or(0, |l| l.linear1.len / D);
        Ok(Self {
            stream,
            kernels,
            scratch,
            w,
            eps,
            sms,
            x: stream.alloc_zeros(n * D)?,
            qkv: stream.alloc_zeros(n * 3 * D)?,
            o: stream.alloc_zeros(n * D)?,
            f: stream.alloc_zeros(n * ff)?,
            y: stream.alloc_zeros(n * D)?,
            stats_a: stream.alloc_zeros(2 * n)?,
            stats_b: stream.alloc_zeros(2 * n)?,
            markers: stream.clone_htod(markers)?,
            logits: stream.alloc_zeros(markers.len())?,
        })
    }

    fn rows(&self) -> usize {
        self.x.len() / D
    }

    fn config(&self, n: usize, k: usize) -> Config {
        gemm::choose(self.rows(), n, k, self.sms)
    }

    /// The head's input from the encoder's output `x_enc` (before its final norm): the final
    /// norm, plus the embedding of question type `qtype`.
    pub fn launch_input(&mut self, x_enc: &CudaSlice<f16>, qtype: usize) -> color_eyre::Result<()> {
        let n = self.rows();
        color_eyre::eyre::ensure!(
            x_enc.len() == n * D,
            "head: the encoder output isn't [{n}, {D}]"
        );
        let lay = &self.w.layout;
        color_eyre::eyre::ensure!(
            (qtype + 1) * D <= lay.type_emb.len,
            "head: no type embedding {qtype}"
        );
        let gamma = self.w.f32.slice(lay.final_norm.range());
        let te = self
            .w
            .f32
            .slice(lay.type_emb.start + qtype * D..lay.type_emb.start + (qtype + 1) * D);
        let (gamma_len, te_len, x_len) = (gamma.len() as u64, te.len() as u64, x_enc.len() as u64);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(4) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.kernels.norm_add);
        // norm_add(x: &[f16], gamma: &[f32], add: &[f32], eps: f32, out: *mut f16, stats: *mut f32)
        launch
            .arg(x_enc)
            .arg(&x_len)
            .arg(&gamma)
            .arg(&gamma_len)
            .arg(&te)
            .arg(&te_len)
            .arg(&self.eps)
            .arg(&mut self.x)
            .arg(&mut self.stats_a);
        // SAFETY: x_enc and out are [n, D] and stats has 2n values; gamma and the type embedding
        // are D-long weight spans (16-byte aligned: the type embedding's rows are 1536 bytes into
        // a 128-byte aligned span); blocks of 128 threads, one warp per row.
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }

    /// Queues head layer `l`.
    pub fn launch_layer(&mut self, l: usize) -> color_eyre::Result<()> {
        let lay = &self.w.layout.head[l];
        let (f16s, f32s) = (&self.w.f16, &self.w.f32);
        let ff = self.f.len() / self.rows();
        let gemm =
            |g: Gemm, cfg| gemm::launch(self.stream, &self.kernels.gemm, self.scratch, cfg, g);

        let (w, colsum, bias) = (
            f16s.slice(lay.in_proj.range()),
            f32s.slice(lay.in_proj_colsum.range()),
            f32s.slice(lay.in_proj_b.range()),
        );
        let cfg = self.config(3 * D, D);
        gemm(
            Gemm {
                epi: Epilogue::LnBias,
                a: &self.x,
                w: &w,
                k: D,
                c: &mut self.qkv,
                ln: Some(LnInputs {
                    stats: &self.stats_a,
                    colsum: &colsum,
                    eps: HEAD_LN_EPS,
                }),
                rope: None,
                bias: Some(&bias),
                stats_out: None,
                zero: Some(&mut self.stats_b),
            },
            cfg,
        )?;
        crate::attn::launch(
            self.stream,
            &self.kernels.attention,
            &self.qkv,
            &mut self.o,
            0,
        )?;

        let (w, bias) = (
            f16s.slice(lay.out_proj.range()),
            f32s.slice(lay.out_proj_b.range()),
        );
        let cfg = self.config(D, D);
        gemm(
            Gemm {
                epi: Epilogue::BiasResStats,
                a: &self.o,
                w: &w,
                k: D,
                c: &mut self.x,
                ln: None,
                rope: None,
                bias: Some(&bias),
                stats_out: Some(&mut self.stats_b),
                zero: None,
            },
            cfg,
        )?;

        let (w, colsum, bias) = (
            f16s.slice(lay.linear1.range()),
            f32s.slice(lay.linear1_colsum.range()),
            f32s.slice(lay.linear1_b.range()),
        );
        let cfg = self.config(ff, D);
        gemm(
            Gemm {
                epi: Epilogue::LnBiasRelu,
                a: &self.x,
                w: &w,
                k: D,
                c: &mut self.f,
                ln: Some(LnInputs {
                    stats: &self.stats_b,
                    colsum: &colsum,
                    eps: HEAD_LN_EPS,
                }),
                rope: None,
                bias: Some(&bias),
                stats_out: None,
                zero: Some(&mut self.stats_a),
            },
            cfg,
        )?;

        let (w, bias) = (
            f16s.slice(lay.linear2.range()),
            f32s.slice(lay.linear2_b.range()),
        );
        let cfg = self.config(D, ff);
        gemm(
            Gemm {
                epi: Epilogue::BiasResStats,
                a: &self.f,
                w: &w,
                k: ff,
                c: &mut self.x,
                ln: None,
                rope: None,
                bias: Some(&bias),
                stats_out: Some(&mut self.stats_a),
                zero: None,
            },
            cfg,
        )
    }

    /// Queues the scorer: y = GELU(LN(x) · lin1ᵀ + b) on every row, then the logits of the marker
    /// rows.
    pub fn launch_score(&mut self) -> color_eyre::Result<()> {
        let sc = &self.w.layout.scorer;
        let (w, colsum, bias) = (
            self.w.f16.slice(sc.lin1.range()),
            self.w.f32.slice(sc.lin1_colsum.range()),
            self.w.f32.slice(sc.lin1_b.range()),
        );
        let cfg = self.config(D, D);
        gemm::launch(
            self.stream,
            &self.kernels.gemm,
            self.scratch,
            cfg,
            Gemm {
                epi: Epilogue::LnBiasGelu,
                a: &self.x,
                w: &w,
                k: D,
                c: &mut self.y,
                ln: Some(LnInputs {
                    stats: &self.stats_a,
                    colsum: &colsum,
                    eps: HEAD_LN_EPS,
                }),
                rope: None,
                bias: Some(&bias),
                stats_out: None,
                zero: None,
            },
        )?;

        let n = self.rows();
        let k = self.markers.len();
        let lin2 = self.w.f16.slice(sc.lin2.range());
        color_eyre::eyre::ensure!(lin2.len() == D, "head: the scorer's lin2 isn't [1, {D}]");
        let (y_len, w_len, k_len) = ((n * D) as u64, D as u64, k as u64);
        let cfg = LaunchConfig {
            grid_dim: (k.div_ceil(4).max(1) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.stream.launch_builder(&self.kernels.score);
        // score(y: &[f16], w: &[f16], b: f32, markers: &[u32], logits: *mut f32)
        launch
            .arg(&self.y)
            .arg(&y_len)
            .arg(&lin2)
            .arg(&w_len)
            .arg(&sc.lin2_b)
            .arg(&self.markers)
            .arg(&k_len)
            .arg(&mut self.logits);
        // SAFETY: y is [n, D], lin2 D values, logits one per marker; the markers are rows < n
        // (checked by `new`'s caller, see `run`); blocks of 128 threads, one warp per marker.
        unsafe { launch.launch(cfg) }?;
        Ok(())
    }

    /// The whole head from the encoder's output: the logits, one per marker.
    pub fn run(&mut self, x_enc: &CudaSlice<f16>, qtype: usize) -> color_eyre::Result<Vec<f32>> {
        self.launch_all(x_enc, qtype)?;
        Ok(self.stream.clone_dtoh(&self.logits)?)
    }

    /// Queues the whole head.
    pub fn launch_all(&mut self, x_enc: &CudaSlice<f16>, qtype: usize) -> color_eyre::Result<()> {
        self.launch_input(x_enc, qtype)?;
        for l in 0..self.w.layout.head.len() {
            self.launch_layer(l)?;
        }
        self.launch_score()
    }
}

/// How the layer kernel cuts a layer into tasks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    /// The tile height (32 or 64 rows) of qkv, wo, wi and wo_mlp.
    pub bm: [usize; 4],
    /// Their split-K (1-4).
    pub splits: [u32; 4],
    /// Key ranges per attention tile (64 queries, head; 1-6), in global and in sliding-window
    /// layers.
    pub kv: [u32; 2],
}

impl std::fmt::Display for LayerPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = ["qkv", "wo", "wi", "wo_mlp"];
        for (g, name) in names.iter().enumerate() {
            write!(f, "{name} {}/{} ", self.bm[g], self.splits[g])?;
        }
        write!(f, "kv {}/{}", self.kv[0], self.kv[1])
    }
}

/// The tile height (32 or 64) and split-K (1-4, at least 3 K steps per split as
/// `gemm::splits`) with the lowest `phase_cost` for a layer-kernel GEMM phase.
pub fn gemm_plan(n: usize, cols: usize, k: usize, sms: usize, per_sm: usize) -> (usize, u32) {
    let cost = |&(bm, z): &(usize, u32)| phase_cost(n, cols, k, bm, z, sms, per_sm);
    [32, 64]
        .into_iter()
        .flat_map(|bm| (1..=4).map(move |z| (bm, z)))
        .filter(|&(_, z)| k / 32 / z as usize >= 3)
        .min_by(|a, b| cost(a).total_cmp(&cost(b)))
        .unwrap()
}

/// The estimated time (µs) of a layer-kernel GEMM phase, `n` rows × `cols` × `k`, in tiles of
/// `bm` rows with `z` K splits, on `sms` SMs with `per_sm` blocks each: the slowest SM's.
///
/// Fitted to plan sweeps on the 2080 Ti (each GEMM's 8 tile heights × splits, at 9
/// lengths from 72 to 607 tokens): it picks the measured best (or a tie) in 30 of the 36
/// cases, and is at most 1.6 µs per layer slower in the others (2.7 µs per layer summed over
/// the 36); also an earlier sweep's best in 7 of 8. A task (one tile, or one split of it) takes 5 µs
/// (10 with 64-row tiles: twice the epilogue and the first loads) plus 0.35 µs per 32-deep K
/// step, the steps waiting mostly on L2 latency; c tasks at once on one SM take
/// 1 + 0.2·(c − 1) times one task (the second block per SM mostly fills the first one's
/// waits); tiles split z ways add 0.4 µs per split and 32 rows for the block that sums the
/// partials. Tasks go to SMs round-robin, so the busiest SM has ⌈tasks / SMs⌉.
pub fn phase_cost(
    n: usize,
    cols: usize,
    k: usize,
    bm: usize,
    z: u32,
    sms: usize,
    per_sm: usize,
) -> f64 {
    let tasks = (cols / 64) * n.div_ceil(bm) * z as usize;
    let steps = (k / 32) as f64 / z as f64;
    let task = if bm == 32 { 5.0 } else { 10.0 } + 0.35 * steps;
    let at_once = |c: usize| 1.0 + 0.2 * (c as f64 - 1.0);
    let on_sm = tasks.div_ceil(sms);
    let (rounds, rest) = (on_sm / per_sm, on_sm % per_sm);
    let mut t = rounds as f64 * at_once(per_sm) * task;
    if rest > 0 {
        t += at_once(rest) * task;
    }
    if z > 1 {
        t += 0.4 * z as f64 * (bm / 32) as f64;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `gemm_plan` on the 2080 Ti (68 SMs, 2 blocks each) against the plan sweeps: the
    /// measured best of each GEMM at 72, 363 and 512 tokens, except wo_mlp at 72 (measured 32/4,
    /// 0.3 µs per layer faster than this 32/3).
    #[test]
    fn plan_matches_the_sweeps() {
        let (inter, sms, per_sm) = (1152, 68, 2);
        let shapes = [(3 * D, D), (D, D), (2 * inter, D), (D, inter)];
        let picks = |n: usize| shapes.map(|(cols, k)| gemm_plan(n, cols, k, sms, per_sm));
        assert_eq!(picks(72), [(32, 1), (32, 3), (32, 1), (32, 3)]);
        assert_eq!(picks(363), [(64, 1), (32, 1), (64, 1), (64, 3)]);
        assert_eq!(picks(512), [(64, 1), (32, 1), (64, 1), (32, 1)]);
    }
}
