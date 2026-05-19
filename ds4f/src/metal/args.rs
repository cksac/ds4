//! Kernel argument structs.
//!
//! Each kernel family has a small `#[repr(C)]` struct passed to
//! `setBytes:length:atIndex:0` on the compute encoder. These must match
//! the corresponding Metal shader struct layouts exactly (same field types,
//! same order, same alignment).

/// Args for `kernel_rms_norm_fuse_impl` (norm.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RmsNormArgs {
    pub ne00: i32,
    pub ne00_t: i32,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub eps: f32,
    pub nef1: [i32; 3],
    pub nef2: [i32; 3],
    pub nef3: [i32; 3],
    pub nbf1: [u64; 3],
    pub nbf2: [u64; 3],
    pub nbf3: [u64; 3],
}

impl RmsNormArgs {
    pub fn new(n: u32, rows: u32, eps: f32) -> Self {
        let row_bytes = n as u64 * std::mem::size_of::<f32>() as u64;
        let total_bytes = row_bytes * rows as u64;
        Self {
            ne00: n as i32,
            ne00_t: (n / 4) as i32,
            nb1: row_bytes,
            nb2: total_bytes,
            nb3: total_bytes,
            eps,
            nef1: [rows as i32, 1, 1],
            nef2: [1, 1, 1],
            nef3: [1, 1, 1],
            nbf1: [row_bytes, row_bytes, row_bytes],
            nbf2: [total_bytes, row_bytes, row_bytes],
            nbf3: [total_bytes, row_bytes, row_bytes],
        }
    }

    pub fn new_3d(n0: u32, n1: u32, n2: u32, eps: f32) -> Self {
        let row_bytes = n0 as u64 * std::mem::size_of::<f32>() as u64;
        let plane_bytes = row_bytes * n1 as u64;
        Self {
            ne00: n0 as i32,
            ne00_t: (n0 / 4) as i32,
            nb1: row_bytes,
            nb2: plane_bytes,
            nb3: plane_bytes * n2 as u64,
            eps,
            nef1: [n1 as i32, 1, 1],
            nef2: [n2 as i32, 1, 1],
            nef3: [1, 1, 1],
            nbf1: [row_bytes, row_bytes, row_bytes],
            nbf2: [plane_bytes, row_bytes, row_bytes],
            nbf3: [plane_bytes * n2 as u64, row_bytes, row_bytes],
        }
    }
}

/// Args for `kernel_dsv4_qkv_rms_norm_f32_4` (norm.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QkvRmsNormArgs {
    pub q_n: i32,
    pub q_n4: i32,
    pub kv_n: i32,
    pub kv_n4: i32,
    pub q_row_stride: u64,
    pub kv_row_stride: u64,
    pub eps: f32,
}

/// Args for `kernel_mul_mv_*` (dense.metal).
/// Must match `ds4_metal_args_mul_mv` exactly (112 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MulMvArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nr0: i32,
    pub r2: i16,
    pub r3: i16,
}

/// Args for `kernel_mul_mv_ext_*` (dense.metal) — small-batch matvec.
/// Must match `ds4_metal_args_mul_mv_ext` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MulMvExtArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub r2: i16,
    pub r3: i16,
}

/// Args for `kernel_mul_mm_*` (dense.metal) — tiled matmul for prefill.
/// Must match `ds4_metal_args_mul_mm` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MulMmArgs {
    pub ne00: i32,
    pub ne02: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub r2: i16,
    pub r3: i16,
}

/// Args for `kernel_mul_mv_id_*` (moe.metal) — MoE matvec with expert ids.
/// Must match `ds4_metal_args_mul_mv_id` exactly (120 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MulMvIdArgs {
    pub nei0: i32,
    pub nei1: i32,
    pub nbi1: u64,
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nb1: u64,
    pub nr0: i32,
}

/// Args for `kernel_mul_mm_id_map0` (moe.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MulMmIdMap0Args {
    pub ne20: i32,
    pub ne0: i32,
    pub n_experts: i32,
    pub n_tokens: i32,
}

/// Args for MoE SwiGLU weight activation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MoeSwigluWeightArgs {
    pub n: u32,
    pub clamp: f32,
    pub weight_scale: f32,
}

/// Args for `kernel_dsv4_hc_split_sinkhorn` (dsv4_hc.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HcSplitSinkhornArgs {
    pub n_hc: i32,
    pub sinkhorn_iters: i32,
    pub n_rows: i64,
    pub mix_hc: i64,
    pub nb01: u64,
    pub nb1: u64,
    pub eps: f32,
}

/// Args for `kernel_dsv4_hc_split_weighted_sum` (dsv4_hc.metal).
/// Must match `ds4_metal_args_dsv4_hc_split_weighted_sum` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HcSplitWeightedSumArgs {
    pub n_embd: i64,
    pub n_hc: i32,
    pub sinkhorn_iters: i32,
    pub n_rows: i64,
    pub mix_hc: i64,
    pub nb_mix1: u64,
    pub nb_split1: u64,
    pub nb_x0: u64,
    pub nb_x1: u64,
    pub nb_x2: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub eps: f32,
}

/// Args for `kernel_dsv4_hc_split_weighted_sum_norm4` (dsv4_hc.metal).
/// Must match `ds4_metal_args_dsv4_hc_split_weighted_sum_norm` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HcSplitWeightedSumNormArgs {
    pub n_embd: i64,
    pub n_hc: i32,
    pub sinkhorn_iters: i32,
    pub n_rows: i64,
    pub mix_hc: i64,
    pub nb_mix1: u64,
    pub nb_split1: u64,
    pub nb_x0: u64,
    pub nb_x1: u64,
    pub nb_x2: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb_norm1: u64,
    pub eps: f32,
    pub norm_eps: f32,
}

/// Args for `kernel_dsv4_hc_weighted_sum` (dsv4_hc.metal).
/// Must match `ds4_metal_args_dsv4_hc_weighted_sum` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HcWeightedSumArgs {
    pub n_embd: i64,
    pub n_hc: i64,
    pub n_tokens: i64,
    pub nb_x0: u64,
    pub nb_x1: u64,
    pub nb_x2: u64,
    pub nb_w0: u64,
    pub nb_w1: u64,
    pub nb0: u64,
    pub nb1: u64,
}

/// Args for `kernel_dsv4_hc_expand` / `kernel_dsv4_hc_expand4` (dsv4_hc.metal).
/// Must match `ds4_metal_args_dsv4_hc_expand` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HcExpandArgs {
    pub n_embd: i64,
    pub n_hc: i64,
    pub n_tokens: i64,
    pub nb_block0: u64,
    pub nb_block1: u64,
    pub nb_add0: u64,
    pub nb_add1: u64,
    pub nb_res0: u64,
    pub nb_res1: u64,
    pub nb_res2: u64,
    pub nb_post0: u64,
    pub nb_post1: u64,
    pub nb_comb0: u64,
    pub nb_comb1: u64,
    pub nb_comb2: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub has_add: i32,
}

/// Args for `kernel_dsv4_rope_tail_f32` (dsv4_rope.metal).
/// Must match `ds4_metal_args_dsv4_rope_tail` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RopeTailArgs {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub n_dims: i32,
    pub mode: i32,
    pub n_ctx_orig: i32,
    pub inverse: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub src2: i32,
}

/// Args for `kernel_dsv4_fp8_kv_quantize_f32` (dsv4_kv.metal).
/// Must match `ds4_metal_args_dsv4_fp8_kv_quantize` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Fp8KvQuantizeArgs {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub n_rot: i32,
}

/// Args for `kernel_dsv4_kv_fp8_store_f32` (dsv4_kv.metal).
/// Mirrors `ds4_metal_args_dsv4_kv_fp8_store` in dsv4_kv.metal.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvFp8StoreArgs {
    pub head_dim: i32,
    pub n_rot: i32,
    pub raw_row: i32,
}

/// Args for `kernel_dsv4_ratio4_shift_f32` (dsv4_kv.metal).
/// Must match `ds4_metal_args_dsv4_ratio4_shift` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Ratio4ShiftArgs {
    pub width: u32,
}

/// Args for `kernel_swiglu_f32` (glu.metal).
/// Must match `ds4_metal_args_glu` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwigluArgs {
    pub ne00: i32,
    pub nb01: u64,
    pub ne10: i32,
    pub nb11: u64,
    pub ne0: i32,
    pub nb1: u64,
    pub i00: i32,
    pub i10: i32,
    pub alpha: f32,
    pub limit: f32,
}

/// Args for `kernel_unary_impl` (unary.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UnaryArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub slope: f32,
    pub scale: f32,
    pub bias: f32,
    pub val: f32,
    pub min: f32,
    pub max: f32,
}

/// Args for `kernel_bin_fuse_impl` (bin.metal).
/// Must match `ds4_metal_args_bin` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BinArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub offs: u64,
    pub o1: [u64; 8],
}

/// Args for `kernel_cpy_t_t` (cpy.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpyArgs {
    pub nk0: i64,
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub ne3: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

/// Args for `kernel_get_rows_f` (get_rows.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GetRowsArgs {
    pub ne00: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub ne0: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

/// Args for `kernel_set_rows_f` (set_rows.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SetRowsArgs {
    pub ne00: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub ne0: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

/// Args for `kernel_repeat_f32` (repeat.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RepeatArgs {
    pub ne00: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

/// Args for `kernel_soft_max` / `kernel_soft_max_4` (softmax.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SoftMaxArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub scale: f32,
    pub max_bias: f32,
    pub mask_nrows: i32,
    pub n_rows_log2: i32,
}

/// Args for `kernel_argsort_f32_i32_desc` (argsort.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ArgsortArgs {
    pub ne00: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub ne0: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub order: i32,
    pub n_passes: i32,
}

/// Args for `kernel_argsort_merge_f32_i32_desc` (argsort.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ArgsortMergeArgs {
    pub ne00: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub ne0: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub order: i32,
    pub run_len: i32,
}

/// Args for `kernel_sum_rows_impl` (sum_rows.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SumRowsArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub mean: i32,
}

/// Args for `kernel_concat` (concat.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConcatArgs {
    pub ne00: i32,
    pub ne01: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

/// Args for `kernel_flash_attn_ext` (flash_attn.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnExtArgs {
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub ns10: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ns20: i32,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: i32,
    pub logit_softcap: f32,
}

/// Args for `kernel_flash_attn_ext_vec` (flash_attn.metal).
/// Must match `ds4_metal_args_flash_attn_ext_vec` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnExtVecArgs {
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub ns10: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ns20: i32,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: i32,
    pub logit_softcap: f32,
}

/// Args for `kernel_flash_attn_ext_vec_reduce` (flash_attn.metal).
/// Must match `ds4_metal_args_flash_attn_ext_vec_reduce` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnExtVecReduceArgs {
    pub nrows: i32,
}

/// Args for `kernel_flash_attn_ext_pad` (flash_attn.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnExtPadArgs {
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
}

/// Args for `kernel_flash_attn_ext_blk` (flash_attn.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnExtBlkArgs {
    pub ne01: i32,
    pub ne30: i32,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
}

/// Args for `kernel_dsv4_softmax_pool` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_softmax_pool` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxPoolArgs {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub nb0: u64,
    pub nb1: u64,
}

/// Args for `kernel_dsv4_compressor_store_one` (dsv4_kv.metal).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompressorStoreOneArgs {
    pub head_dim: i32,
    pub ratio: i32,
    pub pos: i32,
    pub ape_type: i32,
}

/// Args for `kernel_dsv4_topk_mask` / `kernel_dsv4_topk_mask_scatter` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_topk_mask` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TopkMaskArgs {
    pub ne00: i64,
    pub ne01: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub nb0: u64,
    pub nb1: u64,
}

/// Args for `kernel_dsv4_indexer_weighted_sum` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_indexer_weighted_sum` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexerWeightedSumArgs {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub ne10: i64,
    pub ne11: i64,
    pub nb10: u64,
    pub nb11: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub scale: f32,
}

/// Args for `kernel_dsv4_indexer_scores_fused` / `kernel_dsv4_indexer_score_one_direct` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_indexer_scores_fused` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexerScoresFusedArgs {
    pub n_comp: u32,
    pub n_tokens: u32,
    pub n_head: u32,
    pub head_dim: u32,
    pub pos0: u32,
    pub ratio: u32,
    pub q_token_stride: u64,
    pub q_head_stride: u64,
    pub weights_token_stride: u64,
    pub index_row_stride: u64,
    pub score_token_stride: u64,
    pub scale: f32,
}

/// Args for `kernel_dsv4_directional_steering_project_f32` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_directional_steering_project` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectionalSteeringArgs {
    pub width: u32,
    pub rows: u32,
    pub layer: u32,
    pub n_threads: u32,
    pub scale: f32,
}

/// Args for `kernel_dsv4_indexed_mixed_attention_heads8` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_indexed_attention` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IndexedAttentionArgs {
    pub n_tokens: u32,
    pub n_head: u32,
    pub n_raw: u32,
    pub raw_cap: u32,
    pub raw_start: u32,
    pub n_comp: u32,
    pub top_k: u32,
    pub pos0: u32,
    pub window: u32,
    pub ratio: u32,
    pub q_token_stride: u64,
    pub q_head_stride: u64,
    pub raw_row_stride: u64,
    pub comp_row_stride: u64,
    pub topk_token_stride: u64,
    pub dst_token_stride: u64,
    pub dst_head_stride: u64,
    pub scale: f32,
}

/// Args for `kernel_dsv4_router_finalize_one` (dsv4_misc.metal).
/// Must match `ds4_metal_args_dsv4_router_select_one` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RouterSelectOneArgs {
    pub has_bias: u32,
    pub hash_mode: u32,
    pub use_token_buffer: u32,
    pub token: u32,
    pub hash_rows: u32,
}
