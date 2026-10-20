//! `--time`: GPU time per kernel call, measured with CUDA events.
//!
//! An event is a marker on the stream; the GPU writes a timestamp when it reaches it, so the time
//! between two events is GPU time, without the host's overhead around the launches. Each batch
//! queues `per_batch` calls between two events, so the GPU stays busy as long as the host queues
//! calls faster than the GPU runs them. For a kernel shorter than a launch (a few µs), it can't,
//! and the batch average is the launch rate instead.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use cudarc::driver::{CudaStream, sys::CUevent_flags};

pub struct Timing {
    /// The fastest batch's average, µs per call.
    pub min_us: f64,
    /// The median batch's average, µs per call.
    pub median_us: f64,
}

/// Keeps the GPU busy with `launch` for at least `at_least`. GPUs (laptop ones especially) leave
/// their power-saving clocks only under sustained load, over hundreds of milliseconds: a 4070
/// Laptop measured at 1500 MHz in P3 after a few ms of warm-up, against ~1.9 GHz under load.
pub fn warm_up(
    stream: &Arc<CudaStream>,
    at_least: Duration,
    mut launch: impl FnMut() -> color_eyre::Result<()>,
) -> color_eyre::Result<()> {
    let start = Instant::now();
    while start.elapsed() < at_least {
        for _ in 0..100 {
            launch()?;
        }
        stream.synchronize()?;
    }
    Ok(())
}

/// Times `launch`, which queues one call: `warmup` calls first (the first launch loads the kernel,
/// and the clocks rise under load), then `batches` batches of `per_batch` calls.
pub fn time(
    stream: &Arc<CudaStream>,
    warmup: usize,
    batches: usize,
    per_batch: usize,
    mut launch: impl FnMut() -> color_eyre::Result<()>,
) -> color_eyre::Result<Timing> {
    for _ in 0..warmup {
        launch()?;
    }
    // cudarc's default is CU_EVENT_DISABLE_TIMING (events for synchronization only).
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let mut events = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = stream.record_event(timing)?;
        for _ in 0..per_batch {
            launch()?;
        }
        events.push((start, stream.record_event(timing)?));
    }
    // Once, at the end: synchronizing per batch would let the queue run dry.
    stream.synchronize()?;
    let mut us = events
        .iter()
        .map(|(start, end)| Ok(f64::from(start.elapsed_ms(end)?) * 1e3 / per_batch as f64))
        .collect::<color_eyre::Result<Vec<f64>>>()?;
    us.sort_by(f64::total_cmp);
    Ok(Timing {
        min_us: us[0],
        median_us: us[us.len() / 2],
    })
}
