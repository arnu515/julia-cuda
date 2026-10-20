//! Julia-1's GPU kernels (cuda-oxide). `make` (../Makefile) compiles them to
//! ../ptx/sm_XX/kernels.ptx, which the host crate (..) embeds and loads with cudarc.

#![feature(f16)]

use cuda_device::vector::{F32x4, U16x4};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

mod attn;
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

    /// The head's input: out = LayerNorm(x)·γ + add per row (the encoder's final norm, then the
    /// question type's embedding), and each output row's Σ and Σ² (of the stored f16 values) into
    /// `stats` (written, not added). One warp per row, as `embed`.
    #[kernel]
    pub fn norm_add(x: &[f16], gamma: &[f32], add: &[f32], eps: f32, out: *mut f16, stats: *mut f32) {
        let tid = thread::index_1d().get();
        let (row, lane) = (tid / 32, tid % 32);
        if row >= x.len() / D {
            return;
        }
        let src = &x[row * D..][..D];
        let mut v = [0f32; PER_LANE];
        #[unroll]
        for j in 0..PER_LANE / CHUNK_SIZE {
            let c = j * 32 + lane;
            // SAFETY: c < 96, so elements 4c..4c + 4 are inside the row; 8-byte aligned: x is a
            // cudarc allocation and rows are 768 bytes.
            let q = unsafe { *(src.as_ptr().add(c * 4) as *const U16x4) }.to_array();
            #[unroll]
            for i in 0..4 {
                v[j * 4 + i] = f16::from_bits(q[i]) as f32;
            }
        }
        // SAFETY: all 32 lanes get here: whole warps (the host's block size is a multiple of 32)
        // and the early return is the same for a warp's lanes.
        let mean = unsafe { warp_sum(v.iter().sum::<f32>()) } / D as f32;
        // SAFETY: as for the mean.
        let var = unsafe { warp_sum(v.iter().map(|&a| (a - mean) * (a - mean)).sum::<f32>()) } / D as f32;
        let rstd = 1f32 / (var + eps).sqrt();
        let (mut s, mut s2) = (0f32, 0f32);
        #[unroll]
        for j in 0..PER_LANE / CHUNK_SIZE {
            let c = j * 32 + lane;
            // SAFETY: 4c..4c + 4 is inside gamma's and add's D values; 16-byte aligned (weight
            // spans start on 128-byte boundaries, element 4c is 16c bytes in).
            let (g, a) = unsafe {
                (
                    (*(gamma.as_ptr().add(c * 4) as *const F32x4)).to_array(),
                    (*(add.as_ptr().add(c * 4) as *const F32x4)).to_array(),
                )
            };
            let mut o = [0u16; 4];
            #[unroll]
            for i in 0..4 {
                let y = ((v[j * 4 + i] - mean) * rstd * g[i] + a[i]) as f16;
                o[i] = y.to_bits();
                let y = y as f32;
                s += y;
                s2 += y * y;
            }
            // SAFETY: row < rows of x, and out is as long as x; each warp writes its own row,
            // each lane its own chunks; 8-byte aligned.
            unsafe { *(out.add(row * D + c * 4) as *mut U16x4) = U16x4::new(o) };
        }
        // SAFETY: as for the mean.
        let (s, s2) = unsafe { (warp_sum(s), warp_sum(s2)) };
        if lane == 0 {
            // SAFETY: stats has 2 values per row of x.
            unsafe {
                *stats.add(2 * row) = s;
                *stats.add(2 * row + 1) = s2;
            }
        }
    }

    /// The scorer's last linear (D → 1) on the marker rows: logits[i] = y[markers[i]] · w + b.
    /// One warp per marker.
    #[kernel]
    pub fn score(y: &[f16], w: &[f16], b: f32, markers: &[u32], logits: *mut f32) {
        let tid = thread::index_1d().get();
        let (i, lane) = (tid / 32, tid % 32);
        if i >= markers.len() {
            return;
        }
        let row = &y[markers[i] as usize * D..][..D];
        let mut acc = 0f32;
        #[unroll]
        for j in 0..PER_LANE {
            let c = j * 32 + lane;
            acc += (row[c] as f32) * (w[c] as f32);
        }
        // SAFETY: whole warps, and the early return is the same for a warp's lanes.
        let acc = unsafe { warp_sum(acc) };
        if lane == 0 {
            // SAFETY: logits has one value per marker.
            unsafe { *logits.add(i) = acc + b };
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
    // Attention (see attn.rs): o = softmax(q·kᵀ / √64) · v per head, keys limited to
    // |i − j| <= window when window > 0 (the sliding-window layers). Grid (⌈n/64⌉, 6 heads),
    // 128 threads.

    #[kernel]
    pub fn attention(qkv: *const u16, o: *mut u16, n: u32, window: u32) {
        static mut QS: SharedArray<u16, { attn::BQ * attn::HD }, 128> = SharedArray::UNINIT;
        static mut KS: SharedArray<u16, { attn::BQ * attn::HD }, 128> = SharedArray::UNINIT;
        static mut VS: SharedArray<u16, { attn::BQ * attn::HD }, 128> = SharedArray::UNINIT;
        let (qs, ks, vs) = (
            core::ptr::addr_of_mut!(QS) as *mut u16,
            core::ptr::addr_of_mut!(KS) as *mut u16,
            core::ptr::addr_of_mut!(VS) as *mut u16,
        );
        let tid = thread::threadIdx_x();
        let (lane, warp) = (tid & 31, tid >> 5);
        let (q0, head) = (thread::blockIdx_x() * attn::BQ as u32, thread::blockIdx_y());

        attn::load_tile(qkv, qs, n, q0, head, 0);
        thread::sync_threads();
        // This warp's 16 queries as A fragments, one per 16 dimensions.
        let mut qf = [[0u32; 4]; 4];
        #[unroll]
        for ks_ in 0..4u32 {
            qf[ks_ as usize] = gemm::ldsm_x4(attn::at(qs, attn::swz64(warp * 16 + (lane & 15), ks_ * 2 + (lane >> 4))));
        }

        // Key blocks: all, or those the window of any of this block's queries reaches.
        let nb = n.div_ceil(attn::BQ as u32);
        let (kb0, kb1) = if window == 0 {
            (0, nb)
        } else {
            let lo = q0.saturating_sub(window) / attn::BQ as u32;
            let hi = ((q0 + attn::BQ as u32 - 1 + window).min(n - 1)) / attn::BQ as u32 + 1;
            (lo, hi)
        };
        // 1/√64 and log2(e), so the softmax is a bare exp2.
        let scale = core::f32::consts::LOG2_E / 8.0;
        let (g, t) = (lane >> 2, lane & 3);
        let qrow = [q0 + warp * 16 + g, q0 + warp * 16 + g + 8];
        let mut mx = [f32::NEG_INFINITY; 2];
        let mut sum = [0f32; 2];
        let mut acc = [[0f32; 4]; 8];

        let mut kb = kb0;
        while kb < kb1 {
            let k0 = kb * attn::BQ as u32;
            attn::load_tile(qkv, ks, n, k0, head, 1);
            attn::load_tile(qkv, vs, n, k0, head, 2);
            thread::sync_threads();

            // S = Q·Kᵀ for this warp's 16 queries × 64 keys: 8 n8 tiles.
            let mut s = [[0f32; 4]; 8];
            let nrow = (lane & 7) + ((lane >> 4) << 3);
            #[unroll]
            for ks_ in 0..4u32 {
                let kc = ks_ * 2 + ((lane >> 3) & 1);
                #[unroll]
                for jp in 0..4u32 {
                    let b = gemm::ldsm_x4(attn::at(ks, attn::swz64(16 * jp + nrow, kc)));
                    gemm::mma(&mut s[2 * jp as usize], &qf[ks_ as usize], b[0], b[1]);
                    gemm::mma(&mut s[2 * jp as usize + 1], &qf[ks_ as usize], b[2], b[3]);
                }
            }

            // Scale, mask (keys past n, outside the window), and the online softmax: each row's
            // values sit in the 4 lanes of a quad (same g), elements 2h, 2h + 1 of every tile.
            #[unroll]
            for h in 0..2usize {
                let mut m = f32::NEG_INFINITY;
                #[unroll]
                for jt in 0..8u32 {
                    #[unroll]
                    for e in 0..2usize {
                        let key = k0 + 8 * jt + 2 * t + e as u32;
                        let q = qrow[h];
                        let near = window == 0 || (if q > key { q - key } else { key - q }) <= window;
                        let v = &mut s[jt as usize][2 * h + e];
                        *v = if key < n && near { *v * scale } else { f32::NEG_INFINITY };
                        m = if *v > m { *v } else { m };
                    }
                }
                // SAFETY: the whole warp runs this (no divergence in this kernel).
                unsafe {
                    let o1 = gemm::shfl_xor(m, 1);
                    m = if o1 > m { o1 } else { m };
                    let o2 = gemm::shfl_xor(m, 2);
                    m = if o2 > m { o2 } else { m };
                }
                let m_new = if m > mx[h] { m } else { mx[h] };
                // A row with no valid key yet keeps -inf; exp2(-inf - 0) = 0 then.
                let base = if m_new == f32::NEG_INFINITY { 0.0 } else { m_new };
                let alpha = attn_exp2(mx[h] - base);
                mx[h] = m_new;
                let mut rs = 0f32;
                #[unroll]
                for jt in 0..8usize {
                    #[unroll]
                    for e in 0..2usize {
                        let p = attn_exp2(s[jt][2 * h + e] - base);
                        s[jt][2 * h + e] = p;
                        rs += p;
                    }
                    acc[jt][2 * h] *= alpha;
                    acc[jt][2 * h + 1] *= alpha;
                }
                // Per-lane partial sums: alpha is the same across the quad, so they reduce at the end.
                sum[h] = sum[h] * alpha + rs;
            }

            // O += P·V: P's accumulators for keys 16kk..16kk + 15 are the A fragment; V via
            // ldmatrix.trans, two dimension tiles per load.
            #[unroll]
            for kk in 0..4usize {
                let a = [
                    gemm::pack(s[2 * kk][0], s[2 * kk][1]),
                    gemm::pack(s[2 * kk][2], s[2 * kk][3]),
                    gemm::pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                    gemm::pack(s[2 * kk + 1][2], s[2 * kk + 1][3]),
                ];
                #[unroll]
                for dp in 0..4u32 {
                    let v = attn::ldsm_x4_trans(attn::at(vs, attn::swz64(16 * kk as u32 + (lane & 15), 2 * dp + (lane >> 4))));
                    gemm::mma(&mut acc[2 * dp as usize], &a, v[0], v[1]);
                    gemm::mma(&mut acc[2 * dp as usize + 1], &a, v[2], v[3]);
                }
            }
            thread::sync_threads();
            kb += 1;
        }

        #[unroll]
        for h in 0..2usize {
            let mut l = sum[h];
            // SAFETY: the whole warp is here.
            unsafe {
                l += gemm::shfl_xor(l, 1);
                l += gemm::shfl_xor(l, 2);
            }
            let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
            if qrow[h] < n {
                #[unroll]
                for jt in 0..8u32 {
                    let col = head * attn::HD as u32 + 8 * jt + 2 * t;
                    let out = gemm::pack(acc[jt as usize][2 * h] * inv, acc[jt as usize][2 * h + 1] * inv);
                    // SAFETY: qrow < n and col + 1 < D: inside o ([n, D]); 4-byte aligned (even
                    // col); each (row, column pair) has one owner.
                    unsafe { *(o.add(qrow[h] as usize * attn::D + col as usize) as *mut u32) = out };
                }
            }
        }
    }

    #[inline(always)]
    fn attn_exp2(x: f32) -> f32 {
        let r: f32;
        // SAFETY: register-only.
        unsafe { cuda_device::ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x) };
        r
    }

    // -----------------------------------------------------------------------------------------
    // The tensor-core GEMM (see gemm.rs), with fused epilogues:
    //
    // - rope:     c = a · wᵀ, then RoPE on the q and k thirds of c (qkv): the weights' q and k
    //             rows are permuted (weights::permute_rope) so that each head's output columns
    //             2i and 2i + 1, which a thread holds as a pair, are its dimensions i and i + 32,
    //             a rotation pair; `rope` is the [position][32] table of (cos, sin), and row r is
    //             position r.
    // - lnrope:   LayerNorm first, with γ folded into w and colsum[j] = Σₖ w[j, k]:
    //             LN(a) · wᵀ = rstd · (a · wᵀ − mean · colsum). The row statistics (Σa, Σa² per
    //             row, `stats_in`) come from the GEMM that wrote a (resstats). Then RoPE.
    // - lngeglu:  LayerNorm as in lnrope, then GeGLU: c[r, j] = GELU(input) · gate, over weights interleaved so that
    //             each warp's n8 fragments 2q and 2q + 1 are the same 8 input and gate columns
    //             (weights::interleave_geglu). c is [m, n/2].
    // - resstats: c = a · wᵀ + res (`res` may alias c), and each output row's Σ and Σ² (of the
    //             stored f16 values) are added to `stats_out`, which must be zero beforehand.
    //
    // The head's GEMMs (LayerNorms with a bias, folded too: b' = b + W·β, see prep):
    // - lnbias:       LayerNorm, + bias (in_proj).
    // - lnbiasrelu:   LayerNorm, + bias, ReLU (linear1).
    // - lnbiasgelu:   LayerNorm, + bias, GELU (the scorer's first linear).
    // - biasresstats: + bias, then resstats (out_proj, linear2).
    //
    // Any GEMM can also clear `zero_len` floats at `zero` (the statistics buffer the next
    // resstats GEMM accumulates into). Pointers an epilogue doesn't use may be null.

    const EPI_RES: u32 = 1;
    const EPI_GEGLU: u32 = 2;
    const EPI_LN: u32 = 4;
    const EPI_STATS: u32 = 8;
    const EPI_ROPE: u32 = 16;
    const EPI_BIAS: u32 = 32;
    const EPI_RELU: u32 = 64;
    const EPI_GELU: u32 = 128;

    /// gemm_{shape}_{epilogue}(a, w, c, res, ws, counters, stats_in, colsum, stats_out, zero,
    /// zero_len, m, n, k, eps, rope, bias): grid (n/64, ⌈m/BM⌉, splits), 64·WR threads.
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
                rope: *const f32,
                bias: *const f32,
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
                        if (EPI & EPI_BIAS) != 0 {
                            #[unroll]
                            for j in 0..4usize {
                                let col = (cw + j as u32 * 8) as usize;
                                // SAFETY: bias has n values and col + 1 < n.
                                let (b0, b1) = unsafe { (*bias.add(col), *bias.add(col + 1)) };
                                v[j] = [v[j][0] + b0, v[j][1] + b1];
                            }
                        }
                        if (EPI & EPI_RELU) != 0 {
                            #[unroll]
                            for j in 0..4usize {
                                v[j] = [v[j][0].max(0.0), v[j][1].max(0.0)];
                            }
                        }
                        if (EPI & EPI_GELU) != 0 {
                            #[unroll]
                            for j in 0..4usize {
                                v[j] = [gemm::gelu(v[j][0]), gemm::gelu(v[j][1])];
                            }
                        }
                        if (EPI & EPI_ROPE) != 0 && ok {
                            #[unroll]
                            for j in 0..4usize {
                                let col = cw + j as u32 * 8;
                                // q and k are the first two thirds of qkv; a warp's 32 columns
                                // lie in one 64-column head, so this is the same for the warp.
                                if col < 2 * (n / 3) {
                                    let pair = ((col % 64) / 2) as usize;
                                    // SAFETY: the table has (cos, sin) for 32 pairs of at least
                                    // m positions (the host checks), and r < m.
                                    let (cs, sn) = unsafe {
                                        let t = rope.add((r as usize * 32 + pair) * 2);
                                        (*t, *t.add(1))
                                    };
                                    // julia-rs: x1·c − x2·s, x2·c + x1·s (x1 = dim i, x2 = i + 32).
                                    let [x1, x2] = v[j];
                                    v[j] = [x1 * cs - x2 * sn, x2 * cs + x1 * sn];
                                }
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
    gemm!(gemm_32_rope, 1, 2, EPI_ROPE);
    gemm!(gemm_32_lnrope, 1, 2, EPI_LN | EPI_ROPE);
    gemm!(gemm_32_lngeglu, 1, 2, EPI_LN | EPI_GEGLU);
    gemm!(gemm_32_resstats, 1, 2, EPI_RES | EPI_STATS);
    gemm!(gemm_32_lnbias, 1, 2, EPI_LN | EPI_BIAS);
    gemm!(gemm_32_lnbiasrelu, 1, 2, EPI_LN | EPI_BIAS | EPI_RELU);
    gemm!(gemm_32_lnbiasgelu, 1, 2, EPI_LN | EPI_BIAS | EPI_GELU);
    gemm!(gemm_32_biasresstats, 1, 2, EPI_BIAS | EPI_RES | EPI_STATS);
    gemm!(gemm_64_rope, 2, 2, EPI_ROPE);
    gemm!(gemm_64_lnrope, 2, 2, EPI_LN | EPI_ROPE);
    gemm!(gemm_64_lngeglu, 2, 2, EPI_LN | EPI_GEGLU);
    gemm!(gemm_64_resstats, 2, 2, EPI_RES | EPI_STATS);
    gemm!(gemm_64_lnbias, 2, 2, EPI_LN | EPI_BIAS);
    gemm!(gemm_64_lnbiasrelu, 2, 2, EPI_LN | EPI_BIAS | EPI_RELU);
    gemm!(gemm_64_lnbiasgelu, 2, 2, EPI_LN | EPI_BIAS | EPI_GELU);
    gemm!(gemm_64_biasresstats, 2, 2, EPI_BIAS | EPI_RES | EPI_STATS);
    gemm!(gemm_128_rope, 4, 2, EPI_ROPE);
    gemm!(gemm_128_lnrope, 4, 2, EPI_LN | EPI_ROPE);
    gemm!(gemm_128_lngeglu, 4, 2, EPI_LN | EPI_GEGLU);
    gemm!(gemm_128_resstats, 4, 2, EPI_RES | EPI_STATS);
    gemm!(gemm_128_lnbias, 4, 2, EPI_LN | EPI_BIAS);
    gemm!(gemm_128_lnbiasrelu, 4, 2, EPI_LN | EPI_BIAS | EPI_RELU);
    gemm!(gemm_128_lnbiasgelu, 4, 2, EPI_LN | EPI_BIAS | EPI_GELU);
    gemm!(gemm_128_biasresstats, 4, 2, EPI_BIAS | EPI_RES | EPI_STATS);
    gemm!(gemm_128w8_rope, 2, 4, EPI_ROPE);
    gemm!(gemm_128w8_lnrope, 2, 4, EPI_LN | EPI_ROPE);
    gemm!(gemm_128w8_lngeglu, 2, 4, EPI_LN | EPI_GEGLU);
    gemm!(gemm_128w8_resstats, 2, 4, EPI_RES | EPI_STATS);
    gemm!(gemm_128w8_lnbias, 2, 4, EPI_LN | EPI_BIAS);
    gemm!(gemm_128w8_lnbiasrelu, 2, 4, EPI_LN | EPI_BIAS | EPI_RELU);
    gemm!(gemm_128w8_lnbiasgelu, 2, 4, EPI_LN | EPI_BIAS | EPI_GELU);
    gemm!(gemm_128w8_biasresstats, 2, 4, EPI_BIAS | EPI_RES | EPI_STATS);
}

// needed for binary crate; never ran.
fn main() {}
