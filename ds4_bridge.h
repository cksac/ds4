// Bridge: exposes ds4_engine internals for the Rust session.
// Included at the end of ds4.c, called via FFI from Rust.

#ifndef DS4_BRIDGE_H
#define DS4_BRIDGE_H

#include <stdint.h>

// ds4_engine is already typedef'd in ds4.h

// Model info
const void *ds4_bridge_model_map(const struct ds4_engine *e);
uint64_t    ds4_bridge_model_size(const struct ds4_engine *e);
uint64_t    ds4_bridge_tensor_data_offset(const struct ds4_engine *e);
uint64_t    ds4_bridge_tensor_data_size(const struct ds4_engine *e);

// Model geometry
int ds4_bridge_n_layer(const struct ds4_engine *e);
int ds4_bridge_n_embd(const struct ds4_engine *e);
int ds4_bridge_n_hc(const struct ds4_engine *e);
int ds4_bridge_n_head(const struct ds4_engine *e);
int ds4_bridge_head_dim(const struct ds4_engine *e);
int ds4_bridge_n_rot(const struct ds4_engine *e);
int ds4_bridge_n_vocab(const struct ds4_engine *e);

// Per-layer weight offsets (returns 0 on success, -1 if il out of range)
int ds4_bridge_layer_weights(const struct ds4_engine *e, int il,
    uint64_t *attn_norm,
    uint64_t *attn_q_a, uint64_t *attn_q_b, uint64_t *attn_kv,
    uint64_t *attn_out_a, uint64_t *attn_out_b,
    uint64_t *hc_attn_fn, uint64_t *hc_attn_scale, uint64_t *hc_attn_base,
    uint64_t *hc_ffn_fn, uint64_t *hc_ffn_scale, uint64_t *hc_ffn_base,
    uint64_t *ffn_norm,
    uint64_t *ffn_gate_shexp, uint64_t *ffn_up_shexp, uint64_t *ffn_down_shexp,
    uint64_t *ffn_gate_exps, uint64_t *ffn_up_exps, uint64_t *ffn_down_exps,
    int *ffn_gate_type, int *ffn_down_type,
    uint64_t *ffn_gate_expert_bytes, uint64_t *ffn_gate_row_bytes,
    uint64_t *ffn_down_expert_bytes, uint64_t *ffn_down_row_bytes,
    int *ffn_expert_in_dim, int *ffn_expert_mid_dim, int *ffn_expert_out_dim,
    int *compress_ratio,
    uint64_t *compress_ape, int *compress_ape_type,
    uint64_t *compress_norm, int *compress_norm_type,
    float *rope_freq_base, float *rope_freq_scale,
    uint64_t *router_bias, uint64_t *router_hash, int *router_hash_rows,
    int *has_bias, int *hash_mode,
    uint64_t *sink_offset,
    uint64_t *output_norm_offset, uint64_t *output_weight_offset);

#endif
