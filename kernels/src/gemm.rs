//! The tensor-core GEMM's building blocks, adapted from an earlier GEMM microbenchmark (f16
//! only). The kernels themselves (`gemm_{shape}_{epilogue}`) are generated in `main.rs` by
//! `gemm!`.
//!
//! `c = a · wᵀ`: `a` is `[m, k]`, `w` `[n, k]`, f16, f32 accumulation. A block computes a
//! BM × 64 tile with 2·WR warps (WR warp rows × 2 warp columns), each warp a 16·MI × 32 piece,
//! so BM = 16·MI·WR and the block has 64·WR threads. K is walked in steps of BK = 32 through
//! swizzled shared tiles, double-buffered: the next step's global loads are in flight (in
//! registers) while the current step computes. Shapes: 32 × 64 (MI 1, WR 2), 64 × 64 (2, 2),
//! 128 × 64 on 4 warps (4, 2) and on 8 warps (2, 4).
//!
//! Requirements (checked by the host): n is a multiple of 64, k of 32 (and of 32·splits for
//! split-K to be balanced), all rows 16-byte aligned. Rows of `a` past m are read as zeros and
//! never stored.

use core::ptr::addr_of_mut;

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::vector::{F32x4, U32x4};
use cuda_device::{SharedArray, device, ptx_asm, thread};

/// K step.
pub const BK: u32 = 32;
/// A 64 × 32 f16 tile (the W tile, and the A tile of the 64-row shapes).
pub const TILE: usize = 64 * 32;
/// A-tile stride per pipeline stage (the largest BM, 128 rows × 32).
pub const ATILE: usize = 128 * 32;

/// Element offset of (row, 16-byte chunk) in a swizzled [rows][32] tile: chunk c of row r is
/// stored at chunk c ^ ((r / 2) % 4), so the 8 rows an `ldmatrix` reads hit 8 different 16-byte
/// bank groups.
#[inline(always)]
pub fn swz(row: u32, chunk: u32) -> u32 {
    row * BK + ((chunk ^ ((row >> 1) & 3)) << 3)
}

/// The shared-memory address of `p` (which must point into shared memory).
#[inline(always)]
pub fn saddr(p: *const u16) -> u32 {
    // SAFETY: callers pass pointers into the GEMM's shared tiles.
    unsafe { cvta_generic_to_shared_u32(p as *const u8) }
}

/// Four 8 × 8 b16 matrices from shared memory, in the mma fragment layout: lane l supplies the
/// address of row l % 8 of matrix l / 8.
#[inline(always)]
pub fn ldsm_x4(addr: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // SAFETY: `addr` is a 16-byte aligned shared address of a row inside the tile; all 32 lanes
    // execute this (the GEMM has no divergence before it).
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") addr);
    }
    [a, b, c, d]
}

/// c += a · b for one 16 × 8 × 16 step: two m16n8k8 on Turing, which has no k16 shape.
#[inline(always)]
pub fn mma(c: &mut [f32; 4], a: &[u32; 4], b0: u32, b1: u32) {
    // SAFETY: register-only; all 32 lanes execute it.
    unsafe {
        #[cfg(feature = "sm_80")]
        ptx_asm!("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
            inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3],
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3], in("r") b0, in("r") b1);
        #[cfg(not(feature = "sm_80"))]
        {
            ptx_asm!("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
                inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3],
                in("r") a[0], in("r") a[1], in("r") b0);
            ptx_asm!("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
                inout("+f") c[0], inout("+f") c[1], inout("+f") c[2], inout("+f") c[3],
                in("r") a[2], in("r") a[3], in("r") b1);
        }
    }
}

/// Two f32 as f16x2 (`lo` in the low half).
#[inline(always)]
pub fn pack(lo: f32, hi: f32) -> u32 {
    let r: u32;
    // SAFETY: register-only.
    unsafe {
        ptx_asm!("{ .reg .f16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h}; }",
            out("=r") r, in("f") lo, in("f") hi);
    }
    r
}

/// f16x2 as two f32 (low half first).
#[inline(always)]
pub fn unpack(v: u32) -> (f32, f32) {
    let (lo, hi): (f32, f32);
    // SAFETY: register-only.
    unsafe {
        ptx_asm!("{ .reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }",
            out("=f") lo, out("=f") hi, in("r") v);
    }
    (lo, hi)
}

#[inline(always)]
fn ex2(x: f32) -> f32 {
    let r: f32;
    // SAFETY: register-only.
    unsafe { ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x) };
    r
}

#[inline(always)]
fn rsqrt(x: f32) -> f32 {
    let r: f32;
    // SAFETY: register-only.
    unsafe { ptx_asm!("rsqrt.approx.f32 %0, %1;", out("=f") r, in("f") x) };
    r
}

/// Butterfly shuffle across the warp.
///
/// # Safety
/// All 32 lanes must call it.
#[inline(always)]
pub unsafe fn shfl_xor(x: f32, mask: u32) -> f32 {
    let r: f32;
    // SAFETY: the caller guarantees the whole warp is here.
    unsafe {
        ptx_asm!("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;", out("=f") r, in("f") x, in("r") mask)
    };
    r
}

/// `*p += x` in global memory, atomically, without a return value.
///
/// # Safety
/// `p` must be a valid, 4-byte aligned global f32.
#[inline(always)]
pub unsafe fn red_add(p: *mut f32, x: f32) {
    // SAFETY: the caller guarantees `p`.
    unsafe { ptx_asm!("red.global.add.f32 [%0], %1;", in("l") p, in("f") x, clobber("memory")) };
}

/// Exact (erf) GELU via Abramowitz & Stegun 7.1.26, as in julia-rs.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let mut p = 0.5 * 1.061_405_4;
    p = p * t + 0.5 * -1.453_152;
    p = p * t + 0.5 * 1.421_413_7;
    p = p * t + 0.5 * -0.284_496_74;
    p = p * t + 0.5 * 0.254_829_6;
    let q = p * t * ex2(-x * x * (0.5 * core::f32::consts::LOG2_E));
    x * if x >= 0.0 { 1.0 - q } else { q }
}

/// LayerNorm (mean, rstd) of row r from its Σx, Σx² (`stats[2r]`, `stats[2r + 1]`) over k values.
///
/// # Safety
/// `stats` must hold at least 2·(r + 1) values.
#[inline(always)]
pub unsafe fn ln_stats(stats: *const f32, r: u32, k: u32, eps: f32) -> (f32, f32) {
    // SAFETY: the caller guarantees the length.
    let (s, q) = unsafe { (*stats.add(2 * r as usize), *stats.add(2 * r as usize + 1)) };
    let mean = s / k as f32;
    let var = q / k as f32 - mean * mean;
    (mean, rsqrt(if var > 0.0 { var } else { 0.0 } + eps))
}

/// One BK-deep step of this warp's 16·MI × 32 piece, from the current shared tiles.
#[device]
pub fn compute<const MI: usize>(acc: &mut [[[f32; 4]; 4]; MI], a_s: *const u16, w_s: *const u16) {
    let tid = thread::threadIdx_x();
    let (lane, warp) = (tid & 31, tid >> 5);
    let (wm, wn) = ((warp >> 1) * 16 * MI as u32, (warp & 1) * 32);
    #[unroll]
    for ks in 0..2u32 {
        let nrow = (lane & 7) + ((lane >> 4) << 3);
        let kc = ks * 2 + ((lane >> 3) & 1);
        // SAFETY: rows wn + nrow (+ 16) < 64 and chunks kc < 4: inside the W tile.
        let b = unsafe {
            [
                ldsm_x4(saddr(w_s.add(swz(wn + nrow, kc) as usize))),
                ldsm_x4(saddr(w_s.add(swz(wn + 16 + nrow, kc) as usize))),
            ]
        };
        #[unroll]
        for i in 0..MI {
            // SAFETY: row wm + 16i + lane % 16 < BM, chunk < 4: inside the A tile.
            let a = ldsm_x4(saddr(unsafe {
                a_s.add(swz(wm + i as u32 * 16 + (lane & 15), ks * 2 + (lane >> 4)) as usize)
            }));
            #[unroll]
            for j in 0..4usize {
                mma(&mut acc[i][j], &a, b[j >> 1][(j & 1) * 2], b[j >> 1][(j & 1) * 2 + 1]);
            }
        }
    }
}

/// Thread t's A chunks t + 64·WR·h (h < MI) into v[h], and its W chunks (4/WR of them) into
/// v[4..]. The A tile has BM·4 16-byte chunks, the W tile 256. Rows of A past m stay zero.
#[device]
pub fn load_regs<const MI: usize, const WR: u32>(
    a: *const u16,
    w: *const u16,
    m: u32,
    k: u32,
    k0: u32,
) -> [U32x4; 6] {
    let mut v = [U32x4([0; 4]); 6];
    let (tid, nt, bm) = (thread::threadIdx_x(), 64 * WR, 16 * MI as u32 * WR);
    #[unroll]
    for h in 0..MI {
        let ch = tid + h as u32 * nt;
        let (row, col) = (ch >> 2, k0 + (ch & 3) * 8);
        let ar = thread::blockIdx_y() * bm + row;
        if ar < m {
            // SAFETY: row ar < m, col + 8 <= k; 16-byte aligned (k and col are multiples of 8,
            // and the host passes 16-byte aligned buffers).
            v[h] = unsafe { *(a.add(ar as usize * k as usize + col as usize) as *const U32x4) };
        }
    }
    #[unroll]
    for h in 0..2usize {
        if (h as u32) < 4 / WR {
            let ch = tid + h as u32 * nt;
            let (row, col) = (ch >> 2, k0 + (ch & 3) * 8);
            let wr = thread::blockIdx_x() * 64 + row;
            // SAFETY: as for A, with wr < n because n is a multiple of 64.
            v[4 + h] = unsafe { *(w.add(wr as usize * k as usize + col as usize) as *const U32x4) };
        }
    }
    v
}

/// The chunks `load_regs` loaded, into the swizzled shared tiles.
#[device]
pub fn store_regs<const MI: usize, const WR: u32>(v: &[U32x4; 6], a_s: *mut u16, w_s: *mut u16) {
    let (tid, nt) = (thread::threadIdx_x(), 64 * WR);
    #[unroll]
    for h in 0..MI {
        let ch = tid + h as u32 * nt;
        // SAFETY: chunk ch < BM·4 of the A tile; each thread writes its own chunks.
        unsafe { *(a_s.add(swz(ch >> 2, ch & 3) as usize) as *mut U32x4) = v[h] };
    }
    #[unroll]
    for h in 0..2usize {
        if (h as u32) < 4 / WR {
            let ch = tid + h as u32 * nt;
            // SAFETY: chunk ch < 256 of the W tile; each thread writes its own chunks.
            unsafe { *(w_s.add(swz(ch >> 2, ch & 3) as usize) as *mut U32x4) = v[4 + h] };
        }
    }
}

/// This block's K-step range [kt0, kt1) under split-K (gridDim.z splits).
#[inline(always)]
pub fn k_range(k: u32) -> (u32, u32) {
    let (nk, z, splits) = (k / BK, thread::blockIdx_z(), thread::gridDim_z());
    (z * nk / splits, (z + 1) * nk / splits)
}

/// The register-prefetch main loop. Not #[device]: LLVM must inline it, or the accumulator goes
/// through local memory.
#[inline(always)]
pub fn mainloop<const MI: usize, const WR: u32>(
    acc: &mut [[[f32; 4]; 4]; MI],
    a: *const u16,
    w: *const u16,
    m: u32,
    k: u32,
) {
    static mut AS: SharedArray<u16, { 2 * ATILE }, 128> = SharedArray::UNINIT;
    static mut WS: SharedArray<u16, { 2 * TILE }, 128> = SharedArray::UNINIT;
    let (a_s, w_s) = (addr_of_mut!(AS) as *mut u16, addr_of_mut!(WS) as *mut u16);
    let (kt0, kt1) = k_range(k);
    let mut v = load_regs::<MI, WR>(a, w, m, k, kt0 * BK);
    store_regs::<MI, WR>(&v, a_s, w_s);
    thread::sync_threads();
    let mut kt = kt0;
    while kt < kt1 {
        if kt + 1 < kt1 {
            v = load_regs::<MI, WR>(a, w, m, k, (kt + 1) * BK);
        }
        let cur = ((kt - kt0) & 1) as usize;
        // SAFETY: stage `cur` of the two-stage buffers.
        compute::<MI>(acc, unsafe { a_s.add(cur * ATILE) }, unsafe { w_s.add(cur * TILE) });
        if kt + 1 < kt1 {
            let nxt = 1 - cur;
            // The other stage: nobody reads it this step (the barrier below ends the step).
            store_regs::<MI, WR>(&v, unsafe { a_s.add(nxt * ATILE) }, unsafe {
                w_s.add(nxt * TILE)
            });
        }
        thread::sync_threads();
        kt += 1;
    }
}

/// Split-K reduction. Every block writes its f32 partial tile to `ws`; the last block of a tile to
/// arrive (counted in `counters`, which it resets to 0) sums all partials in split order, so the
/// result doesn't depend on arrival order. Returns whether this block holds the full sums and
/// runs the epilogue.
///
/// # Safety
/// With more than one split, `ws` must hold BM·64 floats per (tile, split) and `counters` one
/// zeroed u32 per tile.
#[device]
pub unsafe fn split_reduce<const MI: usize, const WR: u32>(
    acc: &mut [[[f32; 4]; 4]; MI],
    ws: *mut f32,
    counters: *mut u32,
) -> bool {
    let splits = thread::gridDim_z();
    if splits == 1 {
        return true;
    }
    static mut LAST: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
    let last = addr_of_mut!(LAST) as *mut u32;
    let (tid, nt) = (thread::threadIdx_x(), (64 * WR) as usize);
    let tile = thread::blockIdx_y() * thread::gridDim_x() + thread::blockIdx_x();
    // Partial of split z: [4·MI fragments][64·WR threads] of F32x4 (BM × 64 floats), so the
    // stores coalesce.
    let part = |z: u32| unsafe {
        ws.add((tile * splits + z) as usize * 1024 * MI * WR as usize) as *mut F32x4
    };
    let mine = part(thread::blockIdx_z());
    #[unroll]
    for i in 0..MI {
        #[unroll]
        for j in 0..4usize {
            unsafe { *mine.add((i * 4 + j) * nt + tid as usize) = F32x4(acc[i][j]) };
        }
    }
    unsafe { ptx_asm!("membar.gl;", clobber("memory")) };
    thread::sync_threads();
    if tid == 0 {
        let old: u32;
        unsafe {
            ptx_asm!("atom.global.add.u32 %0, [%1], 1;", out("=r") old, in("l") counters.add(tile as usize), clobber("memory"));
            let is_last = old == splits - 1;
            if is_last {
                *counters.add(tile as usize) = 0;
            }
            *last = is_last as u32;
        }
    }
    thread::sync_threads();
    if unsafe { *last } == 0 {
        return false;
    }
    unsafe { ptx_asm!("membar.gl;", clobber("memory")) };
    *acc = [[[0f32; 4]; 4]; MI];
    let mut z = 0;
    while z < splits {
        let p = part(z) as *const f32;
        #[unroll]
        for i in 0..MI {
            #[unroll]
            for j in 0..4usize {
                let (x, y, u, v): (f32, f32, f32, f32);
                // .cg: from L2, not a stale L1 line (other blocks wrote these).
                unsafe {
                    ptx_asm!("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];",
                        out("=f") x, out("=f") y, out("=f") u, out("=f") v,
                        in("l") p.add(((i * 4 + j) * nt + tid as usize) * 4));
                }
                let c = &mut acc[i][j];
                c[0] += x;
                c[1] += y;
                c[2] += u;
                c[3] += v;
            }
        }
        z += 1;
    }
    true
}
