//! Timestamps for profiling the layer kernel (`--time`): thread 0 of each block writes, at
//! each phase's start and end, the GPU-wide `%globaltimer` (ns, comparable across SMs, but
//! updated only every ~µs on some GPUs) and its SM's `%clock64` (cycles, exact, but per SM), and
//! the SM it runs on.
//!
//! A block's record is PROF_WORDS u64 at `prof + blockIdx.x · PROF_WORDS`: words 0..5 the
//! `%smid` during phase p, words 8 + 2s and 9 + 2s the (globaltimer, clock64) of stamp s:
//! s = 2p is phase p's start (after the barrier before it), s = 2p + 1 its end (after the
//! block's last task, before the barrier after it).

use cuda_device::{ptx_asm, thread};

/// u64 words per block.
pub const PROF_WORDS: usize = 32;

#[inline(always)]
fn write(prof: *mut u64, slot: u32) {
    let (gt, ck): (u64, u64);
    // SAFETY: register-only reads; the memory clobber keeps them in place relative to the
    // phase's loads and stores.
    unsafe {
        ptx_asm!("mov.u64 %0, %%globaltimer;", out("=l") gt, clobber("memory"));
        ptx_asm!("mov.u64 %0, %%clock64;", out("=l") ck, clobber("memory"));
    }
    let rec = thread::blockIdx_x() as usize * PROF_WORDS;
    // SAFETY: the caller guarantees PROF_WORDS words per block; slot < 10, so words
    // 8 + 2·slot + 1 < PROF_WORDS. Only thread 0 of this block writes its record.
    unsafe {
        *prof.add(rec + 8 + 2 * slot as usize) = gt;
        *prof.add(rec + 9 + 2 * slot as usize) = ck;
    }
}

/// Stamp `slot` (and the SM, as phase `phase`'s) by thread 0, if `prof` isn't null. Every thread
/// may call it; no barrier.
///
/// # Safety
/// `prof` is null or holds PROF_WORDS u64 per block of the grid; phase < 5, slot < 10.
#[inline(always)]
pub unsafe fn stamp(prof: *mut u64, phase: u32, slot: u32) {
    if !prof.is_null() && thread::threadIdx_x() == 0 {
        write(prof, slot);
        let rec = thread::blockIdx_x() as usize * PROF_WORDS;
        // SAFETY: phase < 5 < PROF_WORDS, inside this block's record (the caller's size).
        unsafe { *prof.add(rec + phase as usize) = thread::smid() as u64 };
    }
}

/// Phase `phase`'s end stamp, once every thread of the block is done with it (a block barrier,
/// only when profiling).
///
/// # Safety
/// As `stamp`; and every thread of the block must call it (it may contain `bar.sync`).
#[inline(always)]
pub unsafe fn stamp_end(prof: *mut u64, phase: u32) {
    if !prof.is_null() {
        thread::sync_threads();
        if thread::threadIdx_x() == 0 {
            write(prof, 2 * phase + 1);
        }
    }
}
