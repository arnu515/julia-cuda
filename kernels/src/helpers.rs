use cuda_device::warp::shuffle_xor_f32_sync;

/// The sum of `v` over a warp's 32 lanes. Returned to every lane.
///
/// # Safety
///
/// All 32 lanes of the warp must call this function. There must be
/// no warp divergence, otherwise the warp can hang.
///
#[inline(always)]
pub unsafe fn warp_sum(mut v: f32) -> f32 {
    let mut offset = 16;
    while offset > 0 {
        v += shuffle_xor_f32_sync(0xFF_FF_FF_FF, v, offset);
        offset /= 2;
    }
    v
}
