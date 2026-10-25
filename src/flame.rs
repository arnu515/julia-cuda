//! GPU flame graphs: GPU time per kernel and per Rust call stack that launched it.
//!
//! With `enable(true)`, every kernel launch (`around`) records the call stack that made it (the
//! instruction pointers only; symbols are resolved once per distinct stack in `collect`) and is
//! bracketed by two CUDA events on its stream. `collect` waits for the GPU and returns the
//! time of each (stack, kernel) pair, as folded stacks for flamegraph.pl. Off (the default), a
//! launch costs one relaxed atomic load.
//!
//! The events serialize nothing (a stream runs in order anyway) but the bookkeeping slows the
//! host by a few µs per launch, which can show up as GPU idle time; the times recorded are the
//! kernels' own.

use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
};

use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags};

static ON: AtomicBool = AtomicBool::new(false);

struct Launch {
    ips: Vec<usize>,
    kernel: String,
    start: CudaEvent,
    end: CudaEvent,
}

static LAUNCHES: Mutex<Vec<Launch>> = Mutex::new(Vec::new());

/// Starts (or stops) recording launches.
pub fn enable(on: bool) {
    ON.store(on, Relaxed);
}

/// Runs `launch` (one kernel launch on `stream`), recording it if enabled. `kernel` names it.
pub fn around<T, E>(
    stream: &CudaStream,
    kernel: impl FnOnce() -> String,
    launch: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    if !ON.load(Relaxed) {
        return launch();
    }
    let mut ips = Vec::with_capacity(32);
    backtrace::trace(|frame| {
        ips.push(frame.ip() as usize);
        true
    });
    let timing = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let start = stream.record_event(timing);
    let out = launch();
    if let (Ok(start), Ok(end)) = (start, stream.record_event(timing)) {
        let l = Launch {
            ips,
            kernel: kernel(),
            start,
            end,
        };
        LAUNCHES.lock().unwrap_or_else(|e| e.into_inner()).push(l);
    }
    out
}

/// The frames of a stack (instruction pointers, innermost first), root first, from the
/// program's `main` to the caller of `around`, inlined functions included.
fn frames(ips: &[usize]) -> Vec<String> {
    let mut names = vec![];
    for &ip in ips {
        let mut here = vec![];
        backtrace::resolve(ip as *mut c_void, |sym| {
            here.push(
                sym.name()
                    .map_or_else(|| format!("{ip:#x}"), |n| format!("{n:#}")),
            );
        });
        if here.is_empty() {
            here.push(format!("{ip:#x}"));
        }
        names.extend(here); // innermost first, as resolve gives inlined frames
    }
    // Drop the recording machinery: everything up to and including `around` itself.
    if let Some(i) = names
        .iter()
        .rposition(|n| n.starts_with("julia_cuda::flame::around"))
    {
        names.drain(..=i);
    }
    names.reverse();
    // Drop the runtime below the program's main.
    if let Some(i) = names
        .iter()
        .position(|n| n.ends_with("::main") && !n.starts_with("std::") && !n.starts_with("core::"))
    {
        names.drain(..i);
    }
    names
}

/// Waits for the recorded launches and returns their GPU time as folded stacks ("frame;…;frame;
/// kernel nanoseconds"), summed per distinct line; clears the record.
pub fn collect() -> color_eyre::Result<Vec<String>> {
    let launches = std::mem::take(&mut *LAUNCHES.lock().unwrap_or_else(|e| e.into_inner()));
    let mut resolved: HashMap<Vec<usize>, String> = HashMap::new();
    let mut ns: HashMap<String, u64> = HashMap::new();
    for l in launches {
        l.end.synchronize()?;
        let t = (f64::from(l.start.elapsed_ms(&l.end)?) * 1e6).round() as u64;
        let stack = resolved
            .entry(l.ips)
            .or_insert_with_key(|ips| {
                frames(ips)
                    .iter()
                    .map(|f| f.replace([';', ' '], "_"))
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .clone();
        *ns.entry(format!("{stack};{}", l.kernel)).or_default() += t;
    }
    let mut lines: Vec<String> = ns.into_iter().map(|(k, v)| format!("{k} {v}")).collect();
    lines.sort();
    Ok(lines)
}
