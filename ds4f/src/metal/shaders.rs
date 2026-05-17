//! Metal shader source embedding.
//!
//! The C version concatenates a preamble with all 19 .metal source files
//! from the `metal/` directory. We reproduce the exact same concatenation
//! order so the runtime-compiled Metal library is identical.

use objc2_foundation::NSString;

pub const METAL_PREAMBLE: &str = r###"#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define M_PI_F 3.14159265358979323846f

// Reads one byte per stride to warm model-backed pages without copying the
// model. This is outside inference and exists only to reduce first-use stalls.
kernel void kernel_touch_u8_stride(
        device const uchar    *src        [[buffer(0)]],
        device uchar          *dst        [[buffer(1)]],
        constant ulong        &stride     [[buffer(2)]],
        constant ulong        &bytes      [[buffer(3)]],
        constant ulong        &dst_offset [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    ulong off = (ulong)gid * stride;
    if (off >= bytes) return;
    dst[dst_offset + (ulong)gid] = src[off];
}

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

"###;

// Include the auto-generated source constants from build.rs.
include!(concat!(env!("OUT_DIR"), "/metal_sources.rs"));

/// Concatenate the preamble and all 19 Metal source files in the exact order
/// used by ds4_metal.m.
pub fn full_source() -> objc2::rc::Retained<NSString> {
    let mut s = String::with_capacity(256 * 1024);
    s.push_str(METAL_PREAMBLE);
    for src in METAL_SOURCE_SLICE {
        s.push_str(src);
        s.push('\n');
    }
    NSString::from_str(&s)
}
