use anyhow::Result;
use crate::metal::kernels::{
    attention::attention_decode_heads,
    dense::{matmul_f16, matmul_q8_0, shared_gate_up_swiglu_q8_0},
    hc::{hc_expand_add_split, hc_split_weighted_sum_norm, hc_weighted_sum},
    misc::add,
    moe::{router_select, routed_moe_one},
    norm::{head_rms_norm, rms_norm_plain, rms_norm_weight},
    rope::{kv_fp8_store_raw, rope_tail},
};
use crate::metal::session::*;

const RMS_EPS: f32 = 1e-6;

pub fn decode_token(session: &mut Session, _token: i32) -> Result<&[f32]> {
    let pos = session.pos as u32;
    let n_embd = session.n_embd;
    let n_hc = session.n_hc;
    let n_head = session.n_head;
    let head_dim = session.head_dim;
    let n_rot = session.n_rot;
    let n_vocab = session.n_vocab;
    let hc_dim = n_hc as u64 * n_embd as u64;
    let hc_dim_bytes = hc_dim * 4;

    let map = session.weights.map;
    let size = session.weights.size;
    let model_views = &session.model_views;
    let cache = &mut session.pipeline_cache;
    let batch = &mut session.batch;
    let g = &mut session.graph;

    g.decode.cur_hc.fill_f32(0.0, hc_dim_bytes / 4).map_err(|e| anyhow::anyhow!("fill_f32: {}", e))?;

    batch.begin()?;

    for il in 0..session.n_layer as usize {
        let layer = &session.layers[il];
        let raw_cache = &g.layers[il].raw_cache;
        let mix_hc = (2 * n_hc + n_hc * n_hc) as u64;

        // HC pre (attention side)
        rms_norm_plain(cache, batch, &g.decode.flat_hc, &g.decode.cur_hc, (n_embd * n_hc) as u32, RMS_EPS)?;

        matmul_f16(cache, batch, model_views, map, size, layer.hc_attn_fn,
            hc_dim, mix_hc, &g.decode.flat_hc, 1, &g.decode.hc_mix)?;

        hc_split_weighted_sum_norm(cache, batch, model_views, map, size,
            layer.hc_attn_scale, layer.hc_attn_base, layer.attn_norm,
            &g.decode.attn_cur, &g.decode.attn_norm, &g.decode.hc_split,
            &g.decode.hc_mix, &g.decode.cur_hc,
            n_embd, n_hc, 20, RMS_EPS, RMS_EPS)?;

        // Q projection
        matmul_q8_0(cache, batch, model_views, map, size, layer.attn_q_a,
            n_embd as u64, 256, &g.decode.attn_norm, 1, &g.decode.qr)?;

        matmul_q8_0(cache, batch, model_views, map, size, layer.attn_q_b,
            256, n_head as u64 * head_dim as u64, &g.decode.qr, 1, &g.decode.q)?;

        head_rms_norm(cache, batch, &g.decode.q, 1, n_head, head_dim, RMS_EPS)?;

        rope_tail(cache, batch, &g.decode.q, 1, n_head, head_dim, n_rot,
            pos, 0, false, layer.rope_freq_base, layer.rope_freq_scale, 0.0, 1.0, 0.0, 0.0)?;

        // KV projection + store
        matmul_q8_0(cache, batch, model_views, map, size, layer.attn_kv,
            n_embd as u64, head_dim as u64, &g.decode.attn_norm, 1, &g.decode.kv_raw)?;

        rope_tail(cache, batch, &g.decode.kv_raw, 1, 1, head_dim, n_rot,
            pos, 0, false, layer.rope_freq_base, layer.rope_freq_scale, 0.0, 1.0, 0.0, 0.0)?;

        let raw_row = pos % g.raw_cap;
        kv_fp8_store_raw(cache, batch, &g.decode.kv_raw, raw_cache,
            g.raw_cap, raw_row, head_dim, n_rot)?;

        // Flash attention
        let n_filled = if pos > 0 { std::cmp::min(pos, g.raw_cap) } else { 0 };
        if n_filled > 0 {
            let raw_start = if n_filled < g.raw_cap { 0 } else { raw_row.wrapping_add(1) % g.raw_cap };
            attention_decode_heads(cache, batch, model_views, map, size, session.sink_offset,
                &g.work.heads, &g.decode.q, raw_cache,
                n_filled, g.raw_cap, raw_start,
                None, 0, None, false, n_head, head_dim,
                &g.flash_attn_mask, &g.flash_attn_kv,
                &g.flash_attn_pad, &g.flash_attn_tmp,
                &*session.device)?;
        }

        // Post-attention: SKIPPED (no output proj yet)

        // FFN HC pre
        rms_norm_plain(cache, batch, &g.decode.flat_hc, &g.decode.cur_hc, (n_embd * n_hc) as u32, RMS_EPS)?;

        matmul_f16(cache, batch, model_views, map, size, layer.hc_ffn_fn,
            hc_dim, mix_hc, &g.decode.flat_hc, 1, &g.decode.hc_mix)?;

        hc_split_weighted_sum_norm(cache, batch, model_views, map, size,
            layer.hc_ffn_scale, layer.hc_ffn_base, layer.ffn_norm,
            &g.work.ffn_cur, &g.work.ffn_norm, &g.decode.hc_split,
            &g.decode.hc_mix, &g.decode.cur_hc,
            n_embd, n_hc, 20, RMS_EPS, RMS_EPS)?;

        // Router
        matmul_f16(cache, batch, model_views, map, size, layer.router_bias,
            n_embd as u64, 256, &g.work.ffn_norm, 1, &g.work.router_logits)?;

        router_select(cache, batch, model_views, map, size,
            layer.router_bias, layer.router_hash, layer.router_hash_rows,
            256, 6, layer.has_bias, layer.hash_mode,
            pos, &g.work.router_logits,
            &g.work.router_selected, &g.work.router_weights, &g.work.router_probs)?;

        // Routed MoE
        routed_moe_one(cache, batch, model_views, map, size,
            layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps,
            layer.ffn_gate_type, layer.ffn_down_type,
            layer.ffn_gate_expert_bytes, layer.ffn_gate_row_bytes,
            layer.ffn_down_expert_bytes, layer.ffn_down_row_bytes,
            layer.ffn_expert_in_dim, layer.ffn_expert_mid_dim, n_embd,
            &g.work.ffn_norm, &g.work.router_selected, &g.work.router_weights,
            6, 10.0,
            &g.work.routed_out, &g.work.routed_gate, &g.work.routed_up, &g.work.routed_mid)?;

        // Shared expert
        shared_gate_up_swiglu_q8_0(cache, batch, model_views, map, size,
            layer.ffn_gate_shexp, layer.ffn_up_shexp,
            n_embd as u64, 2048,
            &g.work.ffn_norm, &g.work.shared_gate, &g.work.shared_up, &g.work.shared_mid)?;

        matmul_q8_0(cache, batch, model_views, map, size, layer.ffn_down_shexp,
            2048, n_embd as u64, &g.work.shared_mid, 1, &g.work.shared_out)?;

        add(cache, batch, &g.work.ffn_out, &g.work.shared_out, &g.work.routed_out, n_embd)?;

        hc_expand_add_split(cache, batch,
            &g.work.after_ffn_hc, &g.work.ffn_out, &g.work.routed_out,
            &g.decode.cur_hc, &g.decode.hc_split, n_embd, n_hc)?;

        // Swap HC for next layer
        std::mem::swap(&mut g.decode.cur_hc, &mut g.work.after_ffn_hc);
    }

    // Output head
    hc_weighted_sum(cache, batch,
        &g.work.output_embd, &g.decode.cur_hc, &g.work.output_weights,
        n_embd, n_hc)?;

    rms_norm_weight(cache, batch, model_views, map, size,
        session.output_norm_offset, &g.work.output_norm, &g.work.output_embd, n_embd, RMS_EPS)?;

    matmul_q8_0(cache, batch, model_views, map, size,
        session.output_weight_offset,
        n_embd as u64, n_vocab as u64, &g.work.output_norm, 1, &g.work.logits)?;

    batch.end()?;

    session.pos += 1;
    for i in 0..n_vocab as usize { session.logits[i] = 0.0; }
    unsafe {
        let src = g.work.logits.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, session.logits.as_mut_ptr(), n_vocab as usize);
    }
    Ok(&session.logits)
}
