//! Requests from stdin, results to stdout: julia-rs's JSONL interface (one request per line,
//! results in the same order with the same fields), so the two engines run the same files and
//! `tools/score.py` compares their outputs. Latency goes to stderr at the end.

use std::{
    io::{BufRead, Write},
    sync::Arc,
    time::Instant,
};

use color_eyre::eyre::{Context, ensure};
use cudarc::driver::{CudaStream, sys::CUevent_flags};
use julia_rs::{
    head::{Answer, answer},
    sequence::{Request, sequence},
};
use serde_json::json;

use crate::{
    D, Kernels, checkpoint::Checkpoint, embed_into, forward::GpuEncoder, forward::GpuHead, gemm,
    weights::GpuWeights,
};

/// Answers every request on stdin, one at a time: tokenize, upload the ids, embed, the encoder
/// and the head (124 launches), download the logits. Buffers are allocated per request (stream-
/// ordered allocations, outside the timed GPU span).
pub fn serve(
    stream: &Arc<CudaStream>,
    kernels: &Kernels,
    gpu: &GpuWeights,
    ck: &Checkpoint,
    sms: usize,
    max_length: usize,
    head_length: usize,
) -> color_eyre::Result<()> {
    let c = &ck.config;
    ensure!(
        max_length <= c.max_positions,
        "--max-length {max_length} is over the model's {}",
        c.max_positions
    );
    let scratch = gemm::Scratch::new(stream, max_length)?;
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);
    // (tokens, wall ms, GPU ms) per request.
    let mut stats: Vec<(usize, f64, f64)> = vec![];
    let mut out = std::io::stdout().lock();
    for (i, line) in std::io::stdin().lock().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("line {}", i + 1);
        let t = Instant::now();
        let req: Request = serde_json::from_str(&line).wrap_err_with(at)?;
        let seq = sequence(&ck.tokenizer, &req, max_length, head_length).wrap_err_with(at)?;
        let n = seq.ids.len();
        ensure!(
            seq.markers.iter().all(|&m| m < n),
            "{}: a marker is past the sequence",
            at()
        );
        let markers: Vec<u32> = seq.markers.iter().map(|&m| m as u32).collect();
        let ids = stream.clone_htod(&seq.ids)?;
        let mut enc = GpuEncoder::new(
            stream,
            kernels,
            &scratch,
            gpu,
            c.norm_eps,
            c.local_window / 2,
            sms,
            stream.alloc_zeros(n * D)?,
            c.intermediate,
        )?;
        let mut head = GpuHead::new(stream, kernels, &scratch, gpu, c.norm_eps, sms, n, &markers)?;

        let start = stream.record_event(timing)?;
        embed_into(stream, kernels, gpu, &ids, c.norm_eps, &mut enc.x)?;
        for l in 0..gpu.layout.layers.len() {
            enc.launch_layer(l)?;
        }
        head.launch_all(&enc.x, seq.qtype as usize)?;
        let end = stream.record_event(timing)?;
        let logits = stream.clone_dtoh(&head.logits)?;
        let gpu_ms = f64::from(start.elapsed_ms(&end)?);

        ensure!(
            logits.iter().all(|z| z.is_finite()),
            "{}: inference returned nonfinite logits",
            at()
        );
        let (ans, probabilities) = answer(seq.qtype, &logits);
        let mut result = json!({
            "logits": logits,
            "probabilities": probabilities,
            "truncated": seq.truncated,
        });
        match ans {
            Answer::Choice(i) => result["choice"] = i.into(),
            Answer::Score(s) => result["score"] = s.into(),
            Answer::Noul(t) => result["noul"] = t.into(),
        }
        writeln!(out, "{result}")?;
        stats.push((n, t.elapsed().as_secs_f64() * 1e3, gpu_ms));
    }
    out.flush()?;
    report(&stats);
    Ok(())
}

/// Latency percentiles, over the requests after the first (which also loads every kernel).
fn report(stats: &[(usize, f64, f64)]) {
    let Some((first, rest)) = stats.split_first() else {
        eprintln!("no requests");
        return;
    };
    let pct = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        format!(
            "mean {:.3}, p50 {:.3}, p90 {:.3}, p99 {:.3}, max {:.3} ms",
            v.iter().sum::<f64>() / v.len() as f64,
            at(0.5),
            at(0.9),
            at(0.99),
            at(1.0)
        )
    };
    let tokens: Vec<usize> = stats.iter().map(|s| s.0).collect();
    eprintln!(
        "{} requests, {} tokens on average ({}–{}); the first took {:.1} ms (wall, loading the kernels)",
        stats.len(),
        tokens.iter().sum::<usize>() / tokens.len(),
        tokens.iter().min().unwrap(),
        tokens.iter().max().unwrap(),
        first.1
    );
    if rest.is_empty() {
        return;
    }
    eprintln!(
        "  per request, wall (parse, tokenize, upload, run, download): {}",
        pct(rest.iter().map(|s| s.1).collect())
    );
    eprintln!(
        "  per request, GPU (embed to logits):                         {}",
        pct(rest.iter().map(|s| s.2).collect())
    );
}
