//! Julia-1's GPU kernels (cuda-oxide). `make` (../Makefile) compiles them to
//! ../ptx/sm_XX/kernels.ptx, which the host crate (..) embeds and loads with cudarc.

#![feature(f16)]

use cuda_device::vector::{F32x4, U16x4};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

mod gemm;
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
    pub fn gemm_tiled(a: &[f16], w: &[f16], k: u32, c: *mut f16) {
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

    // -----------------------------------------------------------------------------------------
    // The tensor-core GEMM (see gemm.rs), with fused epilogues:
    //
    // - plain:    c = a · wᵀ.
    // - ln:       c = LayerNorm(a) · wᵀ, with γ folded into w and colsum[j] = Σₖ w[j, k]:
    //             LN(a) · wᵀ = rstd · (a · wᵀ − mean · colsum). The row statistics (Σa, Σa² per
    //             row, `stats_in`) come from the GEMM that wrote a (resstats).
    // - lngeglu:  ln, then GeGLU: c[r, j] = GELU(input) · gate, over weights interleaved so that
    //             each warp's n8 fragments 2q and 2q + 1 are the same 8 input and gate columns
    //             (weights::interleave_geglu). c is [m, n/2].
    // - resstats: c = a · wᵀ + res (`res` may alias c), and each output row's Σ and Σ² (of the
    //             stored f16 values) are added to `stats_out`, which must be zero beforehand.
    //
    // Any GEMM can also clear `zero_len` floats at `zero` (the statistics buffer the next
    // resstats GEMM accumulates into). Pointers an epilogue doesn't use may be null.

    const EPI_RES: u32 = 1;
    const EPI_GEGLU: u32 = 2;
    const EPI_LN: u32 = 4;
    const EPI_STATS: u32 = 8;

    /// gemm_{shape}_{epilogue}(a, w, c, res, ws, counters, stats_in, colsum, stats_out, zero,
    /// zero_len, m, n, k, eps): grid (n/64, ⌈m/BM⌉, splits), 64·WR threads.
    macro_rules! gemm {
        ($name:ident, $mi:literal, $wr:literal, $epi:expr) => {
            #[kernel]
            pub fn $name(
                a: *const u16,
                w: *const u16,
                c: *mut u16,
                res: *const u16,
                ws: *mut f32,
                counters: *mut u32,
                stats_in: *const f32,
                colsum: *const f32,
                stats_out: *mut f32,
                zero: *mut f32,
                zero_len: u32,
                m: u32,
                n: u32,
                k: u32,
                eps: f32,
            ) {
                // A const, not `$epi` inline: #[kernel] re-tokenizes the body, so `A | B` pasted
                // into `$epi & C` would parse as `A | (B & C)`.
                const EPI: u32 = $epi;
                let tid = thread::threadIdx_x();
                let flat = (thread::blockIdx_z() * thread::gridDim_y() + thread::blockIdx_y())
                    * thread::gridDim_x()
                    + thread::blockIdx_x();
                let mut zi = flat * 64 * $wr + tid;
                while zi < zero_len {
                    // SAFETY: zi < zero_len, the host's length for `zero`.
                    unsafe { *zero.add(zi as usize) = 0.0 };
                    zi += thread::gridDim_x() * thread::gridDim_y() * thread::gridDim_z() * 64 * $wr;
                }
                let mut acc = [[[0f32; 4]; 4]; $mi];
                gemm::mainloop::<$mi, $wr>(&mut acc, a, w, m, k);
                // SAFETY: the host sizes ws and zeroes counters for split-K launches.
                if !unsafe { gemm::split_reduce::<$mi, $wr>(&mut acc, ws, counters) } {
                    return;
                }
                // The epilogue's loops live here: #[unroll] works only in kernels and #[device]
                // functions, and LLVM won't inline a #[device] function this large.
                let (lane, warp) = (tid & 31, tid >> 5);
                let r0 = thread::blockIdx_y() * 16 * $mi * $wr + (warp >> 1) * 16 * $mi + (lane >> 2);
                // The (packed) W row, i.e. output column, of fragment j = 0.
                let cw = thread::blockIdx_x() * 64 + (warp & 1) * 32 + (lane & 3) * 2;
                #[unroll]
                for i in 0..$mi {
                    #[unroll]
                    for h in 0..2usize {
                        let r = r0 + i as u32 * 16 + h as u32 * 8;
                        let ok = r < m;
                        let mut v = [[0f32; 2]; 4];
                        #[unroll]
                        for j in 0..4usize {
                            v[j] = [acc[i][j][2 * h], acc[i][j][2 * h + 1]];
                        }
                        if (EPI & EPI_LN) != 0 && ok {
                            // SAFETY: stats_in has 2m values and r < m.
                            let (mean, rstd) = unsafe { gemm::ln_stats(stats_in, r, k, eps) };
                            #[unroll]
                            for j in 0..4usize {
                                let col = (cw + j as u32 * 8) as usize;
                                // SAFETY: colsum has n values and col + 1 < n.
                                let (s0, s1) = unsafe { (*colsum.add(col), *colsum.add(col + 1)) };
                                v[j] = [rstd * (v[j][0] - mean * s0), rstd * (v[j][1] - mean * s1)];
                            }
                        }
                        if (EPI & EPI_GEGLU) != 0 {
                            let c0 = thread::blockIdx_x() * 32 + (warp & 1) * 16 + (lane & 3) * 2;
                            #[unroll]
                            for q in 0..2usize {
                                if ok {
                                    let (x, g) = (v[2 * q], v[2 * q + 1]);
                                    let idx = r as usize * (n / 2) as usize + (c0 + q as u32 * 8) as usize;
                                    let out = gemm::pack(gemm::gelu(x[0]) * g[0], gemm::gelu(x[1]) * g[1]);
                                    // SAFETY: r < m and the column < n/2; 4-byte aligned (even
                                    // column); each (r, column pair) has one owner.
                                    unsafe { *(c.add(idx) as *mut u32) = out };
                                }
                            }
                        } else {
                            let (mut s, mut s2) = (0f32, 0f32);
                            #[unroll]
                            for j in 0..4usize {
                                if ok {
                                    let idx = r as usize * n as usize + (cw + j as u32 * 8) as usize;
                                    let (mut lo, mut hi) = (v[j][0], v[j][1]);
                                    if (EPI & EPI_RES) != 0 {
                                        // SAFETY: res is [m, n] like c; this thread reads only
                                        // the pair it writes next, so aliasing c is fine.
                                        let (x, y) = gemm::unpack(unsafe { *(res.add(idx) as *const u32) });
                                        lo += x;
                                        hi += y;
                                    }
                                    let packed = gemm::pack(lo, hi);
                                    // SAFETY: r < m and the column < n; 4-byte aligned (even
                                    // column); each (r, column pair) has one owner.
                                    unsafe { *(c.add(idx) as *mut u32) = packed };
                                    if (EPI & EPI_STATS) != 0 {
                                        let (x, y) = gemm::unpack(packed);
                                        s += x + y;
                                        s2 += x * x + y * y;
                                    }
                                }
                            }
                            if (EPI & EPI_STATS) != 0 {
                                // The 4 lanes of a row hold its 32 columns of this warp.
                                // SAFETY: the whole warp is here (`ok` only guards stores).
                                unsafe {
                                    s += gemm::shfl_xor(s, 1);
                                    s2 += gemm::shfl_xor(s2, 1);
                                    s += gemm::shfl_xor(s, 2);
                                    s2 += gemm::shfl_xor(s2, 2);
                                }
                                if ok && (lane & 3) == 0 {
                                    // SAFETY: stats_out has 2m values and r < m.
                                    unsafe {
                                        gemm::red_add(stats_out.add(2 * r as usize), s);
                                        gemm::red_add(stats_out.add(2 * r as usize + 1), s2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    gemm!(gemm_32_plain, 1, 2, 0);
    gemm!(gemm_32_ln, 1, 2, EPI_LN);
    gemm!(gemm_32_lngeglu, 1, 2, EPI_LN | EPI_GEGLU);
    gemm!(gemm_32_resstats, 1, 2, EPI_RES | EPI_STATS);
    gemm!(gemm_64_plain, 2, 2, 0);
    gemm!(gemm_64_ln, 2, 2, EPI_LN);
    gemm!(gemm_64_lngeglu, 2, 2, EPI_LN | EPI_GEGLU);
    gemm!(gemm_64_resstats, 2, 2, EPI_RES | EPI_STATS);
    gemm!(gemm_128_plain, 4, 2, 0);
    gemm!(gemm_128_ln, 4, 2, EPI_LN);
    gemm!(gemm_128_lngeglu, 4, 2, EPI_LN | EPI_GEGLU);
    gemm!(gemm_128_resstats, 4, 2, EPI_RES | EPI_STATS);
    gemm!(gemm_128w8_plain, 2, 4, 0);
    gemm!(gemm_128w8_ln, 2, 4, EPI_LN);
    gemm!(gemm_128w8_lngeglu, 2, 4, EPI_LN | EPI_GEGLU);
    gemm!(gemm_128w8_resstats, 2, 4, EPI_RES | EPI_STATS);
}

// needed for binary crate; never ran.
fn main() {}
