//! Julia-1's GPU kernels (cuda-oxide). `make` (../Makefile) compiles them to
//! ../ptx/sm_XX/kernels.ptx, which the host crate (..) embeds and loads with cudarc.

#![feature(f16)]

use cuda_device::vector::{F32x4, U16x4};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

mod helpers;

// Embedding constants
/// Hidden size of the model
const D: usize = 384;
/// 12 values per lane (32 threads in a warp)
const PER_LANE: usize = D / 32;
/// 3 (PER_LANE/4) chunks of 4 (i.e. 32 * 4 = 128 f16 values = 256 bytes per fetch -- yay cache!)
const CHUNK_SIZE: usize = 4;

// GEMM constants
/// rows of c per block
const BR: usize = 64;
/// cols of c per block
const BC: usize = 64;
/// k values per slice
const BK: usize = 32;

#[cuda_module]
mod kernels {
    use cuda_device::{SharedArray, vector::U16x8};

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

    #[kernel]
    pub fn gemm(a: &[f16], w: &[f16], k: u32, c: *mut f16) {
        let k = k as usize;
        let (m, n) = (a.len() / k, w.len() / k);

        let (r0, c0) = (
            thread::blockIdx_y() as usize * BR,
            thread::blockIdx_x() as usize * BC,
        );
        let tid = thread::threadIdx_x() as usize;
        let (tx, ty) = (tid % 16, tid / 16);
        let (lr, lc) = (tid / 4, (tid % 4) * 8);

        static mut AS: SharedArray<f32, { BK * (BR + 1) }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { BK * (BC + 1) }> = SharedArray::UNINIT;

        let mut acc = [[0f32; 4]; 4];
        let mut k0 = 0;
        while k0 < k {
            let av = if r0 + lr < m {
                // SAFETY: in bounds: row r0 + lr < m, and k0 + lc + 8 <= k because k is a multiple
                //   of BK (the host checks). 16-byte aligned: the element offset
                //   (r0 + lr)·k + k0 + lc is a multiple of 8 (k, k0 and lc all are), and a starts
                //   on a 256-byte boundary (a cudarc allocation).
                unsafe { *(a.as_ptr().add((r0 + lr) * k + k0 + lc) as *const U16x8) }.to_array()
            } else {
                [0; _]
            };
            let wv =
                // SAFETY: in bounds without a check: n is a multiple of BC (the host checks), so
                //   row c0 + lr < n; k0 + lc + 8 <= k as for A. 16-byte aligned: the offset is a
                //   multiple of 8 elements as for A, and w is a weight span, which starts on a
                //   128-byte boundary (weights::ALIGN_BYTES).
                unsafe { *(w.as_ptr().add((c0 + lr) * k + k0 + lc) as *const U16x8) }.to_array();
            #[unroll]
            for i in 0..8 {
                // SAFETY: in bounds: lc + i < BK and lr < BR. No data race: each thread has its
                //   own (lr, lc), so no two threads write the same element, and barrier (2) of the
                //   previous slice ensures no thread is still reading the tile.
                unsafe { AS[(lc + i) * (BR + 1) + lr] = f16::from_bits(av[i]) as f32 };
                // SAFETY: as for AS (BC = BR, so WS has the same padded row length).
                unsafe { WS[(lc + i) * (BR + 1) + lr] = f16::from_bits(wv[i]) as f32 };
            }

            thread::sync_threads();

            #[unroll]
            for kk in 0..BK {
                // SAFETY: in bounds: kk < BK and ty + 16·i < BR. No data race: barrier (1) above
                //   ensures the whole tile is written, and no thread writes it again until every
                //   thread has passed barrier (2).
                let av: [f32; 4] =
                    core::array::from_fn(|i| unsafe { AS[kk * (BR + 1) + ty + 16 * i] });
                // SAFETY: as for av, with tx + 16·j < BC.
                let wv: [f32; 4] =
                    core::array::from_fn(|i| unsafe { WS[kk * (BC + 1) + tx + 16 * i] });
                #[unroll]
                for i in 0..4 {
                    #[unroll]
                    for j in 0..4 {
                        acc[i][j] += av[i] * wv[j];
                    }
                }
            }

            thread::sync_threads();
            k0 += BK;
        }
        #[unroll]
        for i in 0..4 {
            let r = r0 + ty + 16 * i;
            if r < m {
                #[unroll]
                for j in 0..4 {
                    // SAFETY: in bounds: r < m, and the column c0 + tx + 16·j < c0 + BC <= n, and
                    //   c has m·n elements (the host checks). No two threads write the same element:
                    //   each (r, column) has one owner (tx, ty). A single f16 write needs only 2-byte
                    //   alignment, which c has.
                    unsafe { *c.add(r * n + c0 + tx + 16 * j) = acc[i][j] as f16 };
                }
            }
        }
    }
}

// needed for binary crate; never ran.
fn main() {}
