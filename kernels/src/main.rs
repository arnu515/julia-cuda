//! Julia-1's GPU kernels (cuda-oxide). `make` (../Makefile) compiles them to
//! ../ptx/sm_XX/kernels.ptx, which the host crate (..) embeds and loads with cudarc.

#![feature(f16)]

use cuda_device::vector::{F32x4, U16x4};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

mod helpers;

// Hidden size of the model
const D: usize = 384;
// 12 values per lane (32 threads in a warp)
const PER_LANE: usize = D / 32;
// 3 (PER_LANE/4) chunks of 4 (i.e. 32 * 4 = 128 f16 values = 256 bytes per fetch -- yay cache!)
const CHUNK_SIZE: usize = 4;

#[cuda_module]
mod kernels {
    use crate::helpers::warp_sum;

    use super::*;

    /// tokens -- the request's tokens
    /// table -- the embedding table (\[vocab, D] size)
    ///   should be passed at the span's 128-byte aligned start.
    /// gamma -- emb_norm (\[D] size)
    /// x -- output array, (\[tokens.len(), D] size)
    #[kernel]
    pub fn embed(tokens: &[u32], table: &[f16], gamma: &[f32], eps: f32, x: *mut f16) {
        let tid = thread::index_1d();
        let tid_usz = tid.get();
        let warp_idx = tid_usz / 32;
        let lane_idx = tid_usz % 32;

        if warp_idx >= tokens.len() {
            return;
        }

        let src = &table[tokens[warp_idx] as usize * D..][..D];

        let mut v = [0f32; PER_LANE];
        {
            let (chunks, []) = v.as_chunks_mut::<CHUNK_SIZE>() else {
                unreachable!()
            };
            for (j, chunk) in chunks.iter_mut().enumerate() {
                let c = j * 32 + lane_idx;
                // Reading them as U16x4 because this type is aligned to 8 bytes (becomes
                // one 8-byte load), while [f16; 4] is aligned to 2 bytes (becomes four
                // 2-byte loads).
                // SAFETY: c <= 95, so elements 4c..4c+4 are inside the row. 8-byte aligned: the
                // table's span starts on a 128-byte boundary (weights::ALIGN_BYTES), rows are 768
                // bytes, chunks 8.
                let q = unsafe { *(src.as_ptr().add(c * 4) as *const U16x4) }
                    .to_array()
                    .map(|f| f16::from_bits(f) as f32);
                *chunk = q;
            }
        }
        // SAFETY: all 32 lanes get here: the host launches blocks of a multiple of 32 threads,
        // so every warp is whole, and the early return is the same for every lane of a warp.
        let mean = unsafe { warp_sum(v.iter().sum::<f32>()) } / D as f32;
        // SAFETY: as for the mean.
        let var =
            unsafe { warp_sum(v.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>()) } / D as f32;
        let rstd = 1f32 / (var + eps).sqrt();

        let (chunks, []) = v.as_chunks_mut::<CHUNK_SIZE>() else {
            unreachable!()
        };
        for (j, chunk) in chunks.iter_mut().enumerate() {
            let c = j * 32 + lane_idx;
            // SAFETY: 4c..4c+4 is inside gamma's D values. 16-byte aligned: gamma's span starts
            // on a 128-byte boundary, and element 4c is 16c bytes in.
            let gammas = unsafe { *(gamma.as_ptr().add(c * 4) as *const F32x4) }.to_array();
            let out: [_; 4] =
                core::array::from_fn(|i| (((chunk[i] - mean) * rstd * gammas[i]) as f16).to_bits());
            // SAFETY: each warp writes only row warp_idx and each lane only its own chunks, so no
            // two threads write the same bytes. In bounds: warp_idx < tokens.len() and x has
            // tokens.len() * D elements. 8-byte aligned: x's allocation is (cudarc allocations
            // are at least 256-byte aligned), rows are 768 bytes, chunks 8.
            unsafe { *(x.add(warp_idx * D + c * 4) as *mut U16x4) = U16x4::new(out) }
        }
    }
}

// needed for binary crate; never ran.
fn main() {}
