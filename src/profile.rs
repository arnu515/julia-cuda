//! `--time`'s profile of the layer kernel, from the timestamps its blocks write when given a
//! `prof` buffer (kernels/src/prof.rs): per phase, how long it spans on the GPU, how long each
//! block works in it, what the hand-over to the next phase costs (a grid barrier, or a launch
//! boundary with `split_phases`), and how the working blocks spread over the SMs.
//!
//! `%globaltimer` (ns, one clock for the whole GPU) times spans and hand-overs; on some GPUs it
//! advances only every µs or so, so single values are coarse but averages over many launches
//! aren't biased. `%clock64` (cycles of the block's own SM) times what happens within a block:
//! its busy time in a phase and its wait at a barrier.

use cudarc::driver::CudaSlice;
use half::f16;

use crate::forward::{GpuEncoder, LayerPlan};

/// u64 words per block in the profile buffer (kernels/src/prof.rs's PROF_WORDS).
pub const PROF_WORDS: usize = 32;

const PHASES: [&str; 5] = ["qkv", "attention", "wo", "wi", "wo_mlp"];

/// One block's record of one launch.
struct Block {
    smid: [u32; 5],
    /// Stamps 2p (phase p's start) and 2p + 1 (its end): globaltimer ns, clock64 cycles.
    gt: [u64; 10],
    ck: [u64; 10],
}

fn blocks(words: &[u64], grid: usize) -> Vec<Block> {
    (0..grid)
        .map(|b| {
            let r = &words[b * PROF_WORDS..][..PROF_WORDS];
            Block {
                smid: std::array::from_fn(|p| r[p] as u32),
                gt: std::array::from_fn(|s| r[8 + 2 * s]),
                ck: std::array::from_fn(|s| r[9 + 2 * s]),
            }
        })
        .collect()
}

/// Running sums, averaged at the end.
#[derive(Default, Clone, Copy)]
struct Mean {
    sum: f64,
    n: usize,
}

impl Mean {
    fn add(&mut self, x: f64) {
        self.sum += x;
        self.n += 1;
    }

    fn get(self) -> f64 {
        if self.n == 0 {
            f64::NAN
        } else {
            self.sum / self.n as f64
        }
    }
}

#[derive(Default, Clone, Copy)]
struct PhaseStats {
    /// First start to last end over all blocks (µs).
    span: Mean,
    /// Busy time of the blocks with tasks: mean over them, and the slowest one (µs).
    busy: Mean,
    busy_max: Mean,
    /// Last end of this phase to the first start of the next (µs).
    gap: Mean,
    /// A block's wait at the barrier after the phase (same SM, clock64; µs): the shortest (the
    /// last block to arrive: what the barrier itself costs) and the mean.
    wait_min: Mean,
    wait: Mean,
    /// SMs with a working block, and the most working blocks on one SM.
    sms: Mean,
    per_sm: Mean,
}

/// Runs the encoder `runs` times from `x0` with profiling on (`g` set up for the layer kernel:
/// `fused`, and `split_phases` / `grid_cap` as wanted), and prints the summary under `label`.
pub fn profile(
    g: &mut GpuEncoder,
    x0: &CudaSlice<f16>,
    runs: usize,
    label: &str,
) -> color_eyre::Result<()> {
    let s = g.stream;
    let n_layers = g.w.layout.layers.len();
    let words = PROF_WORDS * g.max_grid();
    g.prof = Some(s.alloc_zeros(n_layers * words)?);
    let plan = g.plan();
    let layers = &g.w.layout.layers;
    let tasks: Vec<[usize; 5]> = (0..n_layers)
        .map(|l| g.layer_tasks(plan, layers[l].global))
        .collect();
    let grids: Vec<usize> = (0..n_layers).map(|l| g.layer_grid(l) as usize).collect();
    let (tasks_of, grid_of) = (|l: usize| tasks[l], |l: usize| grids[l]);

    let mut ph = [PhaseStats::default(); 5];
    let (mut layer, mut between) = (Mean::default(), Mean::default());
    let (mut attn_global, mut attn_local) = (Mean::default(), Mean::default());
    let (mut cycles, mut ns) = (0f64, 0f64);
    let mut step = u64::MAX;
    let mut all = Vec::with_capacity(runs * n_layers);
    for _ in 0..runs {
        s.memcpy_dtod(x0, &mut g.x)?;
        s.memset_zeros(g.prof.as_mut().unwrap())?;
        for l in 0..n_layers {
            g.launch_layer(l)?;
        }
        let host = s.clone_dtoh(g.prof.as_ref().unwrap())?;
        for l in 0..n_layers {
            all.push((l, blocks(&host[l * words..], grid_of(l))));
        }
    }
    g.prof = None;

    for (i, (l, bl)) in all.iter().enumerate() {
        let tasks = tasks_of(*l);
        // The SM clock, from every block's busy stretches (same SM within a phase).
        for b in bl {
            for p in 0..5 {
                cycles += (b.ck[2 * p + 1] - b.ck[2 * p]) as f64;
                ns += (b.gt[2 * p + 1] - b.gt[2 * p]) as f64;
            }
        }
        // The timer's step: the smallest positive difference between any two stamps.
        let mut t: Vec<u64> = bl.iter().flat_map(|b| b.gt).collect();
        t.sort_unstable();
        t.windows(2)
            .filter(|w| w[1] > w[0])
            .for_each(|w| step = step.min(w[1] - w[0]));
        let first = |s: usize| bl.iter().map(|b| b.gt[s]).min().unwrap();
        let last = |s: usize| bl.iter().map(|b| b.gt[s]).max().unwrap();
        layer.add((last(9) - first(0)) as f64 / 1e3);
        if let Some((nl, next)) = all.get(i + 1)
            && *nl == l + 1
        {
            let start = next.iter().map(|b| b.gt[0]).min().unwrap();
            between.add(start as f64 / 1e3 - last(9) as f64 / 1e3);
        }
        for p in 0..5 {
            let st = &mut ph[p];
            let span = (last(2 * p + 1) - first(2 * p)) as f64 / 1e3;
            st.span.add(span);
            if p == 1 {
                match g.w.layout.layers[*l].global {
                    true => attn_global.add(span),
                    false => attn_local.add(span),
                }
            }
            if p < 4 {
                st.gap
                    .add(first(2 * p + 2) as f64 / 1e3 - last(2 * p + 1) as f64 / 1e3);
            }
            let working = &bl[..tasks[p].min(bl.len())];
            let busy: Vec<u64> = working
                .iter()
                .map(|b| b.ck[2 * p + 1] - b.ck[2 * p])
                .collect();
            // Cycles to µs at the end (the clock is only known then): keep cycles for now.
            st.busy
                .add(busy.iter().sum::<u64>() as f64 / busy.len() as f64);
            st.busy_max.add(*busy.iter().max().unwrap() as f64);
            if p < 4 && !g.split_phases {
                let waits: Vec<u64> = bl
                    .iter()
                    .map(|b| b.ck[2 * p + 2] - b.ck[2 * p + 1])
                    .collect();
                st.wait_min.add(*waits.iter().min().unwrap() as f64);
                st.wait
                    .add(waits.iter().sum::<u64>() as f64 / waits.len() as f64);
            }
            let mut per_sm = std::collections::HashMap::new();
            for b in working {
                *per_sm.entry(b.smid[p]).or_insert(0usize) += 1;
            }
            st.sms.add(per_sm.len() as f64);
            st.per_sm.add(*per_sm.values().max().unwrap() as f64);
        }
    }

    let ghz = cycles / ns;
    let us = |c: Mean| c.get() / ghz / 1e3;
    let how = match g.split_phases {
        false => "one cooperative launch per layer",
        true => "one launch per phase",
    };
    // Tasks and grid: a global layer's / a sliding-window layer's where they differ.
    let (tg, tl) = (tasks_of(0), tasks_of(1));
    let both = |a: usize, b: usize| match a == b {
        true => a.to_string(),
        false => format!("{a}/{b}"),
    };
    eprintln!(
        "  layer kernel profile, {} tokens, {label} ({how}, {} blocks; {plan}; {runs} runs × \
         {n_layers} layers): SM clock {ghz:.2} GHz, globaltimer step {step} ns",
        g.rows(),
        both(grid_of(0), grid_of(1)),
    );
    eprintln!(
        "    phase        tasks  SMs used  most/SM   span µs   busy µs mean / slowest   \
         hand-over µs   barrier wait µs least / mean"
    );
    let mut sums = (0f64, 0f64);
    for (p, st) in ph.iter().enumerate() {
        sums.0 += st.span.get();
        let mut row = format!(
            "    {:<10} {:>7}  {:>8.1}  {:>7.1}   {:>7.2}   {:>7.2} / {:>7.2}",
            PHASES[p],
            both(tg[p], tl[p]),
            st.sms.get(),
            st.per_sm.get(),
            st.span.get(),
            us(st.busy),
            us(st.busy_max),
        );
        if p < 4 {
            sums.1 += st.gap.get();
            row += &format!("          {:>7.2}", st.gap.get());
            if !g.split_phases {
                row += &format!("        {:>7.2} / {:>7.2}", us(st.wait_min), us(st.wait));
            }
        }
        eprintln!("{row}");
    }
    eprintln!(
        "    layer {:.2} µs (spans {:.2} + hand-overs {:.2}); {:.2} µs from one layer's end to the \
         next's start; attention span global {:.2}, sliding-window {:.2} µs",
        layer.get(),
        sums.0,
        sums.1,
        between.get(),
        attn_global.get(),
        attn_local.get()
    );
    Ok(())
}

/// The encoder's GPU time (median, ms) with the layer kernel as `g` is set up.
fn time_encoder(g: &mut GpuEncoder) -> color_eyre::Result<f64> {
    let n_layers = g.w.layout.layers.len();
    let t = crate::timing::time(g.stream, 2, 10, 5, &mut || {
        (0..n_layers).try_for_each(|l| g.launch_layer(l))
    })?;
    Ok(t.median_us / 1e3)
}

/// The plan the per-step kernels' rule gives (`gemm::choose`'s tile height for qkv, of the two
/// the layer kernel has, for every GEMM; `gemm::splits` for wo and wo_mlp; whole key ranges):
/// the layer kernel's first default.
pub fn per_step_plan(g: &GpuEncoder) -> LayerPlan {
    let n = g.rows();
    let inter = g.m.len() / n;
    let bm = match crate::gemm::choose(n, 3 * crate::D, crate::D, g.sms).shape {
        crate::gemm::Shape::S32 => 32,
        _ => 64,
    };
    LayerPlan {
        bm: [bm; 4],
        splits: [
            1,
            crate::gemm::splits(n, crate::D, crate::D, bm, g.sms),
            1,
            crate::gemm::splits(n, crate::D, inter, bm, g.sms),
        ],
        kv: [1, 1],
    }
}

/// Sweeps the layer kernel's plan one phase at a time from `default_plan` (each GEMM's tile
/// height × split-K with `phase_cost`'s estimate beside it, attention's key ranges in global and
/// in sliding-window layers), times the combination of each phase's best, and
/// returns it. The phases are separated by barriers, so their times add up, and the best of
/// each is (nearly: the grid follows the largest phase) the best combination. Also times
/// `per_step_plan`.
pub fn sweep(g: &mut GpuEncoder) -> color_eyre::Result<LayerPlan> {
    let base = g.default_plan();
    let old = per_step_plan(g);
    g.fused = true;
    let timed = |g: &mut GpuEncoder, p: LayerPlan| -> color_eyre::Result<f64> {
        g.plan = Some(p);
        time_encoder(g)
    };
    let t_base = timed(g, base)?;
    let t_old = timed(g, old)?;
    eprintln!(
        "  layer kernel plans, {} tokens, encoder ms (median); default {base}: {t_base:.3}; the \
         per-step rule {old}: {t_old:.3}; per GEMM: tile/split ms (estimated phase µs)",
        g.rows()
    );
    let n = g.rows();
    let inter = g.m.len() / n;
    let (d, sms, per_sm) = (crate::D, g.sms, g.kernels.layer.1);
    let shapes = [(3 * d, d), (d, d), (2 * inter, d), (d, inter)];
    let mut best = base;
    for (gi, name) in PHASES.iter().filter(|&&p| p != "attention").enumerate() {
        let mut row = format!("    {name:<7}");
        let mut top = (t_base, base.bm[gi], base.splits[gi]);
        for bm in [32, 64] {
            for z in 1..=4 {
                let mut p = base;
                (p.bm[gi], p.splits[gi]) = (bm, z);
                let t = if p == base { t_base } else { timed(g, p)? };
                let (cols, k) = shapes[gi];
                let est = crate::forward::phase_cost(n, cols, k, bm, z, sms, per_sm);
                row += &format!("  {bm}/{z} {t:.3} ({est:.1})");
                if t < top.0 {
                    top = (t, bm, z);
                }
            }
        }
        (best.bm[gi], best.splits[gi]) = (top.1, top.2);
        eprintln!("{row}");
    }
    for (k, (kind, options)) in [
        ("global", &[1, 2, 3, 4, 6][..]),
        ("sliding", &[1, 2, 3][..]),
    ]
    .into_iter()
    .enumerate()
    {
        let mut row = format!("    kv {kind:<7}");
        let mut top = (t_base, base.kv[k]);
        for &c in options {
            let mut p = base;
            p.kv[k] = c;
            let t = if p == base { t_base } else { timed(g, p)? };
            row += &format!("  {c} {t:.3}");
            if t < top.0 {
                top = (t, c);
            }
        }
        best.kv[k] = top.1;
        eprintln!("{row}");
    }
    let t_best = timed(g, best)?;
    g.two_fence_barrier = true;
    let (t_base2, t_best2) = (timed(g, base)?, timed(g, best)?);
    g.two_fence_barrier = false;
    eprintln!(
        "    each phase's best together: {best}: {t_best:.3} (default {t_base:.3}); with the \
         two-fence barrier: {t_best2:.3} (default {t_base2:.3})"
    );
    (g.fused, g.plan) = (false, None);
    Ok(best)
}
