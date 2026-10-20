//! Attention's building blocks (the kernel itself is `attention` in `main.rs`): a FlashAttention-2
//! style loop on tensor cores. A block takes 64 queries of one head, 4 warps of 16; K and V go
//! through swizzled shared tiles 64 keys at a time; S = Q·Kᵀ uses the GEMM's fragment code, the
//! softmax is online (running max and sum per row, in registers), and P·V reuses S's
//! accumulators as the A operand and loads V with `ldmatrix.trans`.
//!
//! Layout: `qkv` is `[n, 3·D]` f16, rows `[q | k | v]`, each 6 heads of 64; q and k permuted for
//! RoPE (the GEMM's order), which the dot products don't see. `o` is `[n, D]`.

use cuda_device::vector::U32x4;
use cuda_device::{device, ptx_asm, thread};

use crate::gemm::saddr;

/// Model width (`[q | k | v]` thirds of a qkv row).
pub const D: usize = 384;
/// Dimensions per head.
pub const HD: usize = 64;
/// Queries per block, and keys per step.
pub const BQ: usize = 64;

/// Element offset of (row, 16-byte chunk) in a swizzled [rows][64] f16 tile: 128-byte rows would
/// put the 8 rows an `ldmatrix` reads in the same banks, so chunk c of row r is stored at chunk
/// c ^ (r % 8).
#[inline(always)]
pub fn swz64(row: u32, chunk: u32) -> u32 {
    row * HD as u32 + ((chunk ^ (row & 7)) << 3)
}

/// `ldmatrix` with `.trans`: lane l's address is row l % 8 of matrix l / 8, and each thread
/// receives elements of the transposed matrices (for V as the col-major B operand).
#[inline(always)]
pub fn ldsm_x4_trans(addr: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // SAFETY: `addr` is a 16-byte aligned shared address of a row inside the tile; all 32 lanes
    // execute this.
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") addr);
    }
    [a, b, c, d]
}

/// Thread t's 4 chunks (16 bytes each; chunk c = t + 128i is row c / 8, chunk c % 8) of one
/// head's 64 rows starting at `row0` of the third `third` (0 q, 1 k, 2 v), into registers; rows
/// past n are zero. `store_tile` puts them in a swizzled shared tile; apart, the loads can be in
/// flight while the block computes on the previous tile.
#[device]
pub fn load_tile_regs(qkv: *const u16, n: u32, row0: u32, head: u32, third: u32) -> [U32x4; 4] {
    let tid = thread::threadIdx_x();
    let mut v = [U32x4([0; 4]); 4];
    #[unroll]
    for i in 0..4u32 {
        let c = tid + 128 * i;
        let (row, ch) = (c >> 3, c & 7);
        let r = row0 + row;
        if r < n {
            let col = third * D as u32 + head * HD as u32 + ch * 8;
            // SAFETY: r < n and col + 8 <= 3·D: inside qkv; 16-byte aligned (rows of 3·D f16,
            // col a multiple of 8, and the host passes a 16-byte aligned buffer).
            v[i as usize] =
                unsafe { *(qkv.add(r as usize * 3 * D + col as usize) as *const U32x4) };
        }
    }
    v
}

/// `load_tile_regs`'s chunks into the swizzled shared tile `tile` ([64][64] f16).
#[device]
pub fn store_tile(tile: *mut u16, v: &[U32x4; 4]) {
    let tid = thread::threadIdx_x();
    #[unroll]
    for i in 0..4u32 {
        let c = tid + 128 * i;
        let (row, ch) = (c >> 3, c & 7);
        // SAFETY: row < 64, ch < 8: inside the tile; each thread writes its own chunks.
        unsafe { *(tile.add(swz64(row, ch) as usize) as *mut U32x4) = v[i as usize] };
    }
}

/// The shared address of `p`'s element `off`.
#[inline(always)]
pub fn at(p: *const u16, off: u32) -> u32 {
    // SAFETY: callers pass offsets inside the tile.
    saddr(unsafe { p.add(off as usize) })
}

/// f32 values per thread in a key-range partial: the 32 unnormalized output values, then the
/// running max and this lane's part of the running sum for its two rows.
pub const PART: usize = 36;

/// The merge of a (64 queries, head) tile's key-range partials (`attention_task` with several
/// chunks). Every chunk's task writes its `acc`, `mx`, `sum` to its slot of `ws`; the last of the
/// `chunks` tasks of tile `tile` to arrive (counted in `counters`, which it resets) reads every
/// slot back in chunk order and rescales them to the common max:
/// acc = Σ_z acc_z · 2^(mx_z − mx), sum likewise. Returns whether this task holds the merged
/// values (and stores the output).
///
/// # Safety
/// `ws` must hold PART·128 floats per (tile, chunk) and `counters` one zeroed u32 per tile;
/// `last` is one u32 of shared memory; every thread of the block calls it.
#[device]
#[allow(clippy::too_many_arguments)]
pub unsafe fn split_merge(
    acc: &mut [[f32; 4]; 8],
    sum: &mut [f32; 2],
    mx: &mut [f32; 2],
    ws: *mut f32,
    counters: *mut u32,
    tile: u32,
    chunk: u32,
    chunks: u32,
    last: *mut u32,
) -> bool {
    let tid = thread::threadIdx_x() as usize;
    // Slot of chunk z: [PART values][128 threads], so the stores coalesce.
    let slot = |z: u32| unsafe { ws.add((tile * chunks + z) as usize * PART * 128 + tid) };
    let mine = slot(chunk);
    #[unroll]
    for jt in 0..8usize {
        #[unroll]
        for e in 0..4usize {
            // SAFETY: inside this (tile, chunk)'s slot (the caller's size); only this thread
            // writes its column of it.
            unsafe { *mine.add((jt * 4 + e) * 128) = acc[jt][e] };
        }
    }
    #[unroll]
    for h in 0..2usize {
        // SAFETY: as above.
        unsafe {
            *mine.add((32 + h) * 128) = mx[h];
            *mine.add((34 + h) * 128) = sum[h];
        }
    }
    // SAFETY: the partial is written before the arrival is counted (fence), and the counter
    // is read by thread 0 only; as gemm::split_reduce.
    unsafe { ptx_asm!("membar.gl;", clobber("memory")) };
    thread::sync_threads();
    if tid == 0 {
        let old: u32;
        // SAFETY: counters has one u32 per tile (the caller's size); last is shared.
        unsafe {
            ptx_asm!("atom.global.add.u32 %0, [%1], 1;", out("=r") old, in("l") counters.add(tile as usize), clobber("memory"));
            let is_last = old == chunks - 1;
            if is_last {
                *counters.add(tile as usize) = 0;
            }
            *last = is_last as u32;
        }
    }
    thread::sync_threads();
    // SAFETY: last is shared, written by thread 0 before the barrier above.
    if unsafe { *last } == 0 {
        return false;
    }
    // SAFETY: as above: the other chunks' partials are complete before their arrivals.
    unsafe { ptx_asm!("membar.gl;", clobber("memory")) };
    // .cg: from L2, not a stale L1 line (other blocks wrote these).
    let ld = |p: *const f32| {
        let v: f32;
        // SAFETY: callers pass addresses inside a slot.
        unsafe { ptx_asm!("ld.global.cg.f32 %0, [%1];", out("=f") v, in("l") p) };
        v
    };
    let mut m = [f32::NEG_INFINITY; 2];
    let mut z = 0;
    while z < chunks {
        let p = slot(z);
        #[unroll]
        for h in 0..2usize {
            // SAFETY: inside slot z.
            let v = ld(unsafe { p.add((32 + h) * 128) });
            m[h] = if v > m[h] { v } else { m[h] };
        }
        z += 1;
    }
    *acc = [[0f32; 4]; 8];
    *sum = [0f32; 2];
    let mut z = 0;
    while z < chunks {
        let p = slot(z);
        #[unroll]
        for h in 0..2usize {
            // A row no chunk has a key for (only rows past n) keeps -inf and zeros.
            let base = if m[h] == f32::NEG_INFINITY { 0.0 } else { m[h] };
            // SAFETY: inside slot z.
            let f = ex2(ld(unsafe { p.add((32 + h) * 128) }) - base);
            // SAFETY: inside slot z.
            sum[h] += ld(unsafe { p.add((34 + h) * 128) }) * f;
            #[unroll]
            for jt in 0..8usize {
                #[unroll]
                for e in 0..2usize {
                    // SAFETY: inside slot z.
                    acc[jt][2 * h + e] += ld(unsafe { p.add((jt * 4 + 2 * h + e) * 128) }) * f;
                }
            }
        }
        z += 1;
    }
    *mx = m;
    true
}

#[inline(always)]
fn ex2(x: f32) -> f32 {
    let r: f32;
    // SAFETY: register-only.
    unsafe { ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") r, in("f") x) };
    r
}
