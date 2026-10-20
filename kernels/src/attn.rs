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

/// One head's 64 rows starting at `row0` of the third `third` (0 q, 1 k, 2 v) into a swizzled
/// shared tile, 4 chunks per thread (128 threads); rows past n are zero.
#[device]
pub fn load_tile(qkv: *const u16, tile: *mut u16, n: u32, row0: u32, head: u32, third: u32) {
    let tid = thread::threadIdx_x();
    #[unroll]
    for i in 0..4u32 {
        let c = tid + 128 * i;
        let (row, ch) = (c >> 3, c & 7);
        let r = row0 + row;
        let mut v = U32x4([0; 4]);
        if r < n {
            let col = third * D as u32 + head * HD as u32 + ch * 8;
            // SAFETY: r < n and col + 8 <= 3·D: inside qkv; 16-byte aligned (rows of 3·D f16,
            // col a multiple of 8, and the host passes a 16-byte aligned buffer).
            v = unsafe { *(qkv.add(r as usize * 3 * D + col as usize) as *const U32x4) };
        }
        // SAFETY: row < 64, ch < 8: inside the tile; each thread writes its own chunks.
        unsafe { *(tile.add(swz64(row, ch) as usize) as *mut U32x4) = v };
    }
}

/// The shared address of `p`'s element `off`.
#[inline(always)]
pub fn at(p: *const u16, off: u32) -> u32 {
    // SAFETY: callers pass offsets inside the tile.
    saddr(unsafe { p.add(off as usize) })
}
