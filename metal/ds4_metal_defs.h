// Metal kernel definitions extracted from ds4.h
// This allows the .metal files to compile independently without C build context

#ifndef DS4_METAL_DEFS_H
#define DS4_METAL_DEFS_H

// Quantization block sizes
#define QK8_0 32
#define QK_K 256

// Quantization types
typedef struct {
    float d;
    int8_t qs[32];
} block_q8_0;

typedef struct {
    uint8_t d;
    int8_t qs[32];
} block_q4_k;

typedef struct {
    uint8_t d[2];
    int8_t qs[64];
} block_q2_k;

typedef struct {
    uint16_t d;
    int8_t qs[32];
} block_iq2_xxs;

// Loop unrolling macro for Metal
#define FOR_UNROLL(type, iter, count) \
    _Pragma("clang loop unroll_count(8)") \
    for (type iter = 0; iter < count; ++iter)

// SIMD width for Metal compute
#define N_SIMDWIDTH 32

// Matvec tuning parameters
#define N_R0_Q8_0 2
#define N_R0_Q4_K 8
#define N_R0_Q2_K 4

#endif // DS4_METAL_DEFS_H
