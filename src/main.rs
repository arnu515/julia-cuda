use std::{path::PathBuf, time::Instant};

use color_eyre::eyre::{Context, bail, ensure};
use cudarc::driver::{CudaContext, CudaSlice, sys::CUdevice_attribute};
use julia_cuda::{
    D, check,
    checkpoint::Checkpoint,
    embed, embed_into, forward, gemm, gemm_tiled, gemm_tiled_into,
    julia_rs::sequence::{DEFAULT_HEAD_LENGTH, sequence},
    load_kernels, long_request, profile, sample_request, serve, timing,
    weights::{GpuWeights, HostWeights},
};

const USAGE: &str =
    "usage: julia-cuda [checkpoint dir [--max-length N] [--head-length N] [--layer-fused]] < requests.jsonl
       julia-cuda checkpoint dir [--check] [--time]
  Without a checkpoint, only checks that the kernels load on GPU 0. With one, answers the requests
  on stdin, one JSON object per line, as julia-rs does (same fields, same order), and reports the
  latency on stderr. --max-length and --head-length are julia-rs's (default 8192 and 512).
  --layer-fused runs each encoder layer as one persistent launch instead of five.
  --check runs the kernels on two sample requests instead, and compares every step with julia-rs
  (the weight upload, each kernel, each layer, the head, the whole forward pass).
  --time  measures each kernel's GPU time on them (CUDA events, after warm-up).";

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, flags) = match args.split_first() {
        None => (None, &[][..]),
        Some((dir, flags)) if !dir.starts_with('-') => (Some(PathBuf::from(dir)), flags),
        Some(_) => bail!(USAGE),
    };
    let (mut check, mut time) = (false, false);
    let (mut max_length, mut head_length) = (None, DEFAULT_HEAD_LENGTH);
    let mut layer_fused = false;
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let mut number = || -> color_eyre::Result<usize> {
            let v = flags.next().ok_or_else(|| color_eyre::eyre::eyre!(USAGE))?;
            v.parse().wrap_err_with(|| format!("{flag} {v}"))
        };
        match flag.as_str() {
            "--check" => check = true,
            "--time" => time = true,
            "--max-length" => max_length = Some(number()?),
            "--head-length" => head_length = number()?,
            "--layer-fused" => layer_fused = true,
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
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize;
    if !check && !time {
        let max_length = max_length.unwrap_or(c.max_positions);
        return serve::serve(
            &stream,
            &kernels,
            &gpu,
            &ck,
            sms,
            max_length,
            head_length,
            layer_fused,
        );
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
    let mut qkv = gemm_tiled(&stream, &kernels, &x, &wqkv, D)?;
    stream.synchronize()?;
    eprintln!(
        "embed + qkv (layer 0) for {} tokens: {:.3} ms, first call (wall clock)",
        seq.ids.len(),
        ms(t)
    );

    let n = seq.ids.len();
    // The long request's (for checking the attention window, and timing at 363 tokens).
    let long = sequence(
        &ck.tokenizer,
        &long_request(),
        c.max_positions,
        DEFAULT_HEAD_LENGTH,
    )?;
    let long_ids = stream.clone_htod(&long.ids)?;
    let x_long = embed(&stream, &kernels, &gpu, &long_ids, c.norm_eps)?;
    let scratch = gemm::Scratch::new(&stream, n.max(long.ids.len()))?;
    let window = c.local_window / 2;
    let encoder = |x: &CudaSlice<half::f16>| {
        forward::GpuEncoder::new(
            &stream,
            &kernels,
            &scratch,
            &gpu,
            c.norm_eps,
            window,
            sms,
            stream.clone_dtod(x)?,
            c.intermediate,
        )
    };
    let mut layers = encoder(&x)?;
    let mut layers_long = encoder(&x_long)?;
    let head = |seq: &julia_rs::sequence::Sequence| {
        let n = seq.ids.len();
        ensure!(
            seq.markers.iter().all(|&m| m < n),
            "a marker is past the sequence"
        );
        let markers: Vec<u32> = seq.markers.iter().map(|&m| m as u32).collect();
        forward::GpuHead::new(
            &stream, &kernels, &scratch, &gpu, c.norm_eps, sms, n, &markers,
        )
    };
    let mut head_short = head(&seq)?;
    let mut head_long = head(&long)?;
    let inter = c.intermediate;
    let shapes = [
        ("qkv", 3 * D, D),
        ("wo", D, D),
        ("wi", 2 * inter, D),
        ("wo_mlp", D, inter),
    ];
    eprintln!(
        "gemm configurations for {n} tokens on {sms} SMs: {}",
        (shapes.iter())
            .map(|&(name, nn, k)| format!("{name} {}", layers.config_for(nn, k)))
            .collect::<Vec<_>>()
            .join(", ")
    );

    if check {
        let model = check::reference_model(&dir)?;
        let x_host = stream.clone_dtoh(&x)?;
        check::embed(&model, &host, &seq.ids, c.norm_eps, &x_host)?;
        check::qkv0(&model, &host, &seq.ids, &x_host, &stream.clone_dtoh(&qkv)?)?;
        let traces = check::trace(&model, &seq.ids);
        use check::Mode;
        // The GEMMs alone, then with attention, each layer from julia-rs's stream; then the whole
        // encoder from embed's output.
        check::layers(&mut layers, &traces, Mode::Gemms, true)?;
        check::layers(&mut layers, &traces, Mode::Layers, true)?;
        stream.memcpy_htod(&x_host, &mut layers.x)?;
        check::layers(&mut layers, &traces, Mode::Encoder, true)?;
        check::sweep(&mut layers, 1, &traces[1])?;
        // The long request, where the sliding window leaves out most keys.
        let traces = check::trace(&model, &long.ids);
        check::layers(&mut layers_long, &traces, Mode::Layers, false)?;
        stream.memcpy_htod(&stream.clone_dtoh(&x_long)?, &mut layers_long.x)?;
        check::layers(&mut layers_long, &traces, Mode::Encoder, false)?;

        // The head, from julia-rs's encoder output, then the whole forward pass on the GPU (embed,
        // 22 layers, head) against julia-rs's, for both requests.
        let traces_short = check::trace(&model, &seq.ids);
        for (s, t, enc, h, x0) in [
            (&seq, &traces_short, &mut layers, &mut head_short, &x),
            (&long, &traces, &mut layers_long, &mut head_long, &x_long),
        ] {
            check::head(h, &model, &t[t.len() - 1].x_out, s.qtype, &s.markers)?;
            stream.memcpy_dtod(x0, &mut enc.x)?;
            for l in 0..gpu.layout.layers.len() {
                enc.launch_layer(l)?;
            }
            let logits = h.run(&enc.x, s.qtype as usize)?;
            check::forward(&model, &s.ids, s.qtype, &s.markers, &logits)?;
        }

        // The persistent layer kernel, then the forward pass with it.
        for (s, t, enc, h, x0) in [
            (&seq, &traces_short, &mut layers, &mut head_short, &x),
            (&long, &traces, &mut layers_long, &mut head_long, &x_long),
        ] {
            check::layer_kernel(enc, t, x0)?;
            enc.fused = true;
            stream.memcpy_dtod(x0, &mut enc.x)?;
            for l in 0..gpu.layout.layers.len() {
                enc.launch_layer(l)?;
            }
            enc.fused = false;
            let logits = h.run(&enc.x, s.qtype as usize)?;
            eprint!("with the layer kernel: ");
            check::forward(&model, &s.ids, s.qtype, &s.markers, &logits)?;
        }
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

        // Half a second of GEMMs first, so the clocks are up before anything is measured.
        timing::warm_up(&stream, std::time::Duration::from_millis(500), || {
            gemm_tiled_into(&stream, &kernels, &x, &wqkv, D, &mut qkv)
        })?;
        let t = run(&mut || embed_into(&stream, &kernels, &gpu, &ids, c.norm_eps, &mut x))?;
        eprintln!(
            "  embed                    {n} tokens            {:>7.2} / {:>7.2} µs",
            t.min_us, t.median_us
        );

        let flop = 2.0 * (n * 3 * D * D) as f64;
        let wqkv_mb = (wqkv.len() * 2) as f64 / 1e6;
        let t = run(&mut || gemm_tiled_into(&stream, &kernels, &x, &wqkv, D, &mut qkv))?;
        eprintln!(
            "  qkv (layer 0), CUDA-core tiled GEMM: {:.2} / {:.2} µs, {:.2} TFLOP/s \
             ({wqkv_mb:.2} MB of weights, in L2)",
            t.min_us,
            t.median_us,
            flop / t.median_us / 1e6
        );

        // The tensor-core GEMMs, as one layer runs them. The in-place residual GEMMs keep
        // adding to the stream, so its values are meaningless from here on: timing comes last.
        // Each configuration with layer 1's weights (in L2), then the chosen one rotating
        // through every layer's weights (from DRAM when they don't fit in L2).
        let quick = |launch: &mut dyn FnMut() -> color_eyre::Result<()>| {
            timing::time(&stream, 5, 7, 30, launch)
        };
        let n_layers = gpu.layout.layers.len();
        for (layers, n) in [(&mut layers, n), (&mut layers_long, long.ids.len())] {
            let mut layer_total = 0.0;
            for (gi, &(name, nn, k)) in shapes.iter().enumerate() {
                let flop = 2.0 * (n * nn * k) as f64;
                let launch = |g: &mut forward::GpuEncoder, l: usize| match gi {
                    0 => g.launch_qkv(l),
                    1 => g.launch_wo(l),
                    2 => g.launch_wi(l),
                    _ => g.launch_wo_mlp(l),
                };
                eprintln!(
                    "  {name} ({n} × {nn} × {k}), µs per call (median), by tile and split-K:"
                );
                let mut best: Option<(f64, gemm::Config)> = None;
                for shape in gemm::SHAPES {
                    let mut row = format!("    {:>5}x64", shape.name());
                    for splits in 1..=4u32 {
                        let cfg = gemm::Config { shape, splits };
                        if splits as usize > k / 32 {
                            continue;
                        }
                        layers.config = Some(cfg);
                        let t = quick(&mut || launch(layers, 1))?;
                        row += &format!("   split {splits} {:>6.2}", t.median_us);
                        if best.is_none_or(|(b, _)| t.median_us < b) {
                            best = Some((t.median_us, cfg));
                        }
                    }
                    eprintln!("{row}");
                }
                layers.config = None;
                let chosen = layers.config_for(nn, k);
                let mut l = 0;
                let t = run(&mut || {
                    // Layers 1..=21 (layer 0's qkv has no LayerNorm).
                    l = l % (gpu.layout.layers.len() - 1) + 1;
                    launch(layers, l)
                })?;
                layer_total += t.median_us;
                let (bt, bc) = best.unwrap();
                eprintln!(
                    "    chosen {chosen}: {:.2} µs rotating through 21 layers' weights, {:.2} TFLOP/s; \
                 fastest here {bc} ({bt:.2} µs, in L2)",
                    t.median_us,
                    flop / t.median_us / 1e6
                );
            }
            eprintln!(
                "  one layer's four GEMMs: {layer_total:.1} µs; × {n_layers} layers: {:.2} ms",
                layer_total * n_layers as f64 / 1e3
            );
        }

        // Attention (layer 0 is global, layer 1 sliding-window), and the whole encoder (22 ×
        // qkv, attention, wo, wi, wo_mlp: 110 launches), for both requests.
        for (g, tokens, x0) in [
            (&mut layers, n, &x),
            (&mut layers_long, long.ids.len(), &x_long),
        ] {
            for l in [0, 1] {
                let kind = if gpu.layout.layers[l].global {
                    "global"
                } else {
                    "sliding-window"
                };
                let t = run(&mut || g.launch_attention(l))?;
                eprintln!(
                    "  attention, {kind}, {tokens} tokens: {:.2} / {:.2} µs",
                    t.min_us, t.median_us
                );
            }
            // The per-step kernels, the layer kernel, and the layer kernel with launch boundaries in
            // place of its grid barriers (same grid and tasks).
            let variants = [
                ("per-step kernels", false, false),
                ("the layer kernel", true, false),
                ("the layer kernel, a launch per phase", true, true),
            ];
            for (how, fused, split) in variants {
                (g.fused, g.split_phases) = (fused, split);
                let launches = match (fused, split) {
                    (true, false) => n_layers,
                    _ => 5 * n_layers,
                };
                let t = timing::time(&stream, 2, 10, 5, &mut || {
                    (0..n_layers).try_for_each(|l| g.launch_layer(l))
                })?;
                eprintln!(
                    "  encoder, {how} ({launches} launches), {tokens} tokens: {:.3} ms \
                     (fastest {:.3})",
                    t.median_us / 1e3,
                    t.min_us / 1e3
                );
            }
            (g.fused, g.split_phases) = (false, false);
            // The layer kernel's plan, phase by phase; then where the time goes with the default
            // plan and with the best one.
            let best = profile::sweep(g)?;
            g.fused = true;
            profile::profile(g, x0, 10, "default plan")?;
            g.plan = Some(best);
            profile::profile(g, x0, 10, "best plan")?;
            // Kept for the forward pass below.
            g.fused = false;
        }
        // The head, and the whole forward pass: embed, the encoder, the head (from token ids on
        // the GPU to logits on the GPU; no copies).
        let ids_short = &ids;
        for (s, ids, enc, h) in [
            (&seq, ids_short, &mut layers, &mut head_short),
            (&long, &long_ids, &mut layers_long, &mut head_long),
        ] {
            let tokens = s.ids.len();
            let launches = 5 * gpu.layout.head.len() + 3;
            let t = run(&mut || h.launch_all(&enc.x, s.qtype as usize))?;
            eprintln!(
                "  head ({launches} launches), {tokens} tokens: {:.2} / {:.2} µs",
                t.min_us, t.median_us
            );
            let mut x_in = stream.clone_dtod(&enc.x)?;
            let best = enc.plan.take();
            for (fused, plan) in [(false, None), (true, None), (true, best)] {
                (enc.fused, enc.plan) = (fused, plan);
                let t = timing::time(&stream, 2, 10, 5, &mut || {
                    embed_into(&stream, &kernels, &gpu, ids, c.norm_eps, &mut x_in)?;
                    stream.memcpy_dtod(&x_in, &mut enc.x)?;
                    (0..n_layers).try_for_each(|l| enc.launch_layer(l))?;
                    h.launch_all(&enc.x, s.qtype as usize)
                })?;
                let (encoder, how) = match (fused, plan) {
                    (false, _) => (5 * n_layers, "per-step kernels".to_string()),
                    (true, None) => (n_layers, "the layer kernel".to_string()),
                    (true, Some(p)) => (n_layers, format!("the layer kernel, {p}")),
                };
                eprintln!(
                    "  forward pass, {how} ({} launches + a copy), {tokens} tokens: {:.3} ms \
                     (fastest {:.3})",
                    1 + encoder + launches,
                    t.median_us / 1e3,
                    t.min_us / 1e3
                );
            }
            enc.fused = false;
        }

        // The plan sweep at more token counts (the 2000 questions have 123-607 tokens, 293 on
        // average): the long request's tokens repeated to each length.
        let lengths = [128, 192, 256, 320, 448, 512, 607];
        let scratch = gemm::Scratch::new(&stream, lengths[lengths.len() - 1])?;
        for tokens in lengths {
            let more: Vec<u32> = long.ids.iter().cycle().take(tokens).copied().collect();
            let x0 = embed(
                &stream,
                &kernels,
                &gpu,
                &stream.clone_htod(&more)?,
                c.norm_eps,
            )?;
            let mut g = forward::GpuEncoder::new(
                &stream,
                &kernels,
                &scratch,
                &gpu,
                c.norm_eps,
                window,
                sms,
                stream.clone_dtod(&x0)?,
                c.intermediate,
            )?;
            let t = timing::time(&stream, 2, 10, 5, &mut || {
                (0..n_layers).try_for_each(|l| g.launch_layer(l))
            })?;
            eprintln!(
                "  encoder, per-step kernels, {tokens} tokens: {:.3} ms",
                t.median_us / 1e3
            );
            profile::sweep(&mut g)?;
        }
    }
    Ok(())
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}
