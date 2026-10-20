//! The attention kernel's host side (kernels/src/attn.rs is the device side).

use std::sync::Arc;

use color_eyre::eyre::ensure;
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;

use crate::D;

/// Heads of 64 dimensions; the kernel's grid has one block row per head.
const HEADS: usize = D / 64;

/// f32 values per thread in a key-range partial of the layer kernel's attention
/// (kernels/src/attn.rs's PART).
pub const PART: usize = 36;

/// `o = attention(qkv)`: `qkv` is `[n, 3·D]` (q and k in the RoPE-permuted order the qkv GEMM
/// writes), `o` is `[n, D]`. `window`: keys with |i − j| <= window, or 0 for all (global).
pub fn launch(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    qkv: &CudaSlice<f16>,
    o: &mut CudaSlice<f16>,
    window: usize,
) -> color_eyre::Result<()> {
    let n = qkv.len() / (3 * D);
    ensure!(
        n > 0 && qkv.len() == n * 3 * D,
        "attention: qkv has {} elements, not whole rows of {}",
        qkv.len(),
        3 * D
    );
    ensure!(
        o.len() == n * D,
        "attention: o has {} elements, not {n} × {D}",
        o.len()
    );
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(64) as u32, HEADS as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n32, w32) = (n as u32, window as u32);
    let mut launch = stream.launch_builder(f);
    // attention(qkv: *const u16, o: *mut u16, n: u32, window: u32)
    launch.arg(qkv).arg(o).arg(&n32).arg(&w32);
    // SAFETY: qkv is [n, 3·D] and o [n, D] (checked), both cudarc allocations (16-byte aligned);
    // n > 0; the grid covers ⌈n/64⌉ query blocks × 6 heads with the kernel's 128 threads.
    unsafe { launch.launch(cfg) }?;
    Ok(())
}
