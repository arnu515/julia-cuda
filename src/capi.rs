//! A C interface to `Engine`, for other languages (megaLaya's Python harness loads the shared
//! library with ctypes): load a checkpoint, run sequences already tokenized, free it.
//!
//! Functions returning a pointer give NULL on failure, and those returning an int give -1; either
//! writes the error into `err` (`err_len` bytes, NUL-terminated, truncated to fit). Nothing
//! unwinds across the boundary: a panic is reported as an error.

use std::{
    ffi::{CStr, c_char, c_int},
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    slice,
};

use julia_rs::{head::QType, sequence::Sequence};

use crate::Engine;

/// `msg` into the caller's buffer, NUL-terminated.
///
/// # Safety
/// `err` is NULL or points to `len` writable bytes.
unsafe fn report(err: *mut c_char, len: usize, msg: &str) {
    if err.is_null() || len == 0 {
        return;
    }
    let n = msg.len().min(len - 1);
    // SAFETY: n + 1 <= len bytes at err, as the caller promises.
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), err.cast(), n);
        *err.add(n) = 0;
    }
}

/// `f`'s result, or its error or panic reported into `err`.
fn guarded<T>(
    err: *mut c_char,
    len: usize,
    f: impl FnOnce() -> color_eyre::Result<T>,
) -> Option<T> {
    let msg = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => return Some(v),
        Ok(Err(e)) => format!("{e:#}"),
        Err(p) => {
            let what = (p.downcast_ref::<&str>().map(|s| s.to_string()))
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            format!("julia-cuda panicked: {what}")
        }
    };
    // SAFETY: the caller's promise about err, passed on.
    unsafe { report(err, len, &msg) };
    None
}

/// The PTX tier this library was built with ("sm_89"), NUL-terminated, static.
#[unsafe(no_mangle)]
pub extern "C" fn julia_cuda_tier() -> *const c_char {
    static TIER: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    TIER.get_or_init(|| std::ffi::CString::new(crate::ptx_tier()).unwrap())
        .as_ptr()
}

/// Opens the checkpoint in `dir` and uploads it to GPU 0 (`Engine::load`). `max_length` is the
/// longest sequence `julia_cuda_run` will get (it sizes the scratch); 0 for the model's maximum.
///
/// # Safety
/// `dir` is a NUL-terminated path; `err` NULL or `err_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn julia_cuda_load(
    dir: *const c_char,
    max_length: usize,
    err: *mut c_char,
    err_len: usize,
) -> *mut Engine {
    guarded(err, err_len, || {
        // SAFETY: a NUL-terminated string, as the caller promises.
        let dir = unsafe { CStr::from_ptr(dir) }.to_str()?;
        let max = (max_length > 0).then_some(max_length);
        let engine = Engine::load(Path::new(dir), max, julia_rs::sequence::DEFAULT_HEAD_LENGTH)?;
        Ok(Box::into_raw(Box::new(engine)))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Frees an engine from `julia_cuda_load` (NULL is fine).
///
/// # Safety
/// `engine` came from `julia_cuda_load` and isn't used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn julia_cuda_free(engine: *mut Engine) {
    if !engine.is_null() {
        // SAFETY: a Box from julia_cuda_load, freed once.
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// The GPU's name into `buf` (`len` bytes, NUL-terminated).
///
/// # Safety
/// `engine` is live; `buf` has `len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn julia_cuda_device(engine: *const Engine, buf: *mut c_char, len: usize) {
    // SAFETY: as promised by the caller.
    unsafe { report(buf, len, &(*engine).device) };
}

/// Runs `b` sequences back to back (`Engine::run`): sequence i is `lens[i]` token ids (all of
/// them concatenated in `ids`), with `n_markers[i]` marker positions (concatenated in `markers`)
/// and question type `qtypes[i]` (0 choice, 1 score, 2 noul, as Julia's `QTYPES`). Writes each
/// sequence's logits, one per marker, concatenated into `logits`, and its GPU time (ms, embed to
/// logits) into `gpu_ms[i]`. `layer_fused`: the layer kernel instead of the per-step kernels.
/// Returns 0, or -1 with the error in `err`.
///
/// # Safety
/// `engine` is live and not used by another thread meanwhile; the arrays have the lengths
/// described (`logits`: Σ n_markers, `gpu_ms`: b); `err` NULL or `err_len` writable bytes.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn julia_cuda_run(
    engine: *const Engine,
    b: usize,
    ids: *const u32,
    lens: *const usize,
    markers: *const u32,
    n_markers: *const usize,
    qtypes: *const u32,
    layer_fused: c_int,
    logits: *mut f32,
    gpu_ms: *mut f32,
    err: *mut c_char,
    err_len: usize,
) -> c_int {
    let ran = guarded(err, err_len, || {
        // SAFETY: the lengths and liveness the caller promises.
        let (engine, lens, n_markers, qtypes) = unsafe {
            (
                &*engine,
                slice::from_raw_parts(lens, b),
                slice::from_raw_parts(n_markers, b),
                slice::from_raw_parts(qtypes, b),
            )
        };
        let (n_ids, n_out) = (lens.iter().sum::<usize>(), n_markers.iter().sum::<usize>());
        // SAFETY: as above.
        let (ids, markers, logits, gpu_ms) = unsafe {
            (
                slice::from_raw_parts(ids, n_ids),
                slice::from_raw_parts(markers, n_out),
                slice::from_raw_parts_mut(logits, n_out),
                slice::from_raw_parts_mut(gpu_ms, b),
            )
        };
        let (mut at_id, mut at_marker) = (0, 0);
        let mut seqs = Vec::with_capacity(b);
        for i in 0..b {
            let qtype = match qtypes[i] {
                0 => QType::Choice,
                1 => QType::Score,
                2 => QType::Noul,
                q => color_eyre::eyre::bail!("sequence {i}: no question type {q}"),
            };
            color_eyre::eyre::ensure!(
                lens[i] > 0 && lens[i] <= engine.max_length,
                "sequence {i}: {} tokens, not 1 to {}",
                lens[i],
                engine.max_length
            );
            seqs.push(Sequence {
                ids: ids[at_id..at_id + lens[i]].to_vec(),
                markers: (markers[at_marker..at_marker + n_markers[i]].iter())
                    .map(|&m| m as usize)
                    .collect(),
                qtype,
                truncated: false,
            });
            at_id += lens[i];
            at_marker += n_markers[i];
        }
        let out = engine.run(&seqs, layer_fused != 0)?;
        let mut at = 0;
        for (i, (l, ms)) in out.into_iter().enumerate() {
            logits[at..at + l.len()].copy_from_slice(&l);
            at += l.len();
            gpu_ms[i] = ms;
        }
        Ok(())
    });
    if ran.is_some() { 0 } else { -1 }
}
