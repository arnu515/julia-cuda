//! Julia-1's GPU kernels (cuda-oxide). `make` (../Makefile) compiles them to
//! ../ptx/sm_XX/kernels.ptx, which the host crate (..) embeds and loads with cudarc.

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    // Lesson 1: write `axpy` here.
    #[kernel]
    pub fn axpy(a: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let i = thread::index_1d();
        let idx = i.get();
        if let Some(y) = y.get_mut(i) {
            *y += a * x[idx];
        }
    }
}

// needed for binary crate; never ran.
fn main() {}
