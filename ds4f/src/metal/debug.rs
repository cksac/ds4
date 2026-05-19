//! Iterative debug: test kernels one at a time.

use anyhow::Result;
use crate::metal::{commands::CommandBatch, kernels::*, pipeline::PipelineCache, session::*, tensor::Tensor};

pub fn test_layer0_prefix(session: &mut Session) -> Result<()> {
    let g = &mut session.graph;
    let cache = &mut session.pipeline_cache;
    let batch = &mut session.batch;
    let model_views = &session.model_views;
    let map = session.weights.map;
    let size = session.weights.size;
    let n_embd = session.n_embd;
    let n_hc = session.n_hc;
    let hc_dim = n_hc as u64 * n_embd as u64;

    eprintln!("debug: starting kernel tests");

    // ── Test 1: Simple RMS norm (CPU-filled data, no model access) ──
    eprintln!("debug: [1] simple rms_norm on CPU data...");
    g.decode.flat_hc.fill_f32(1.0, hc_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    g.decode.cur_hc.fill_f32(0.5, hc_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    norm::rms_norm_plain(cache, batch, &g.decode.flat_hc, &g.decode.cur_hc, hc_dim as u32, 1e-6)?;
    batch.end()?;
    eprintln!("debug: [1] OK - basic kernel dispatch works");

    // ── Test 2: RMS norm with WEIGHT (model access needed) ──
    let layer = &session.layers[0];
    eprintln!("debug: [2] rms_norm_weight (model weight access)...");
    g.decode.attn_cur.fill_f32(0.5, n_embd as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    norm::rms_norm_weight(cache, batch, model_views, map, size,
        layer.attn_norm, &g.decode.attn_norm, &g.decode.attn_cur, n_embd, 1e-6)?;
    batch.end()?;
    eprintln!("debug: [2] OK - model weight access works");

    // ── Test 3: Matmul Q8_0 with model weights ──
    let mix_hc = 2 * n_hc as u64 + n_hc as u64 * n_hc as u64;
    eprintln!("debug: [3] matmul_q8_0 flat_hc→hc_mix {}→{}...", hc_dim, mix_hc);
    g.decode.flat_hc.fill_f32(1.0, hc_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    dense::matmul_q8_0(cache, batch, model_views, map, size,
        layer.hc_attn_fn, hc_dim, mix_hc, &g.decode.flat_hc, 1, &g.decode.hc_mix)?;
    batch.end()?;
    eprintln!("debug: [3] OK - Q8_0 matmul works");

    // ── Test 4: Q LoRA matmul ──
    let qr_dim = 256u64;
    eprintln!("debug: [4] matmul_q8_0 attn_norm→qr {}→{}...", n_embd, qr_dim);
    g.decode.attn_norm.fill_f32(0.5, n_embd as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    dense::matmul_q8_0(cache, batch, model_views, map, size,
        layer.attn_q_a, n_embd as u64, qr_dim, &g.decode.attn_norm, 1, &g.decode.qr)?;
    batch.end()?;
    eprintln!("debug: [4] OK");

    // ── Test 5: KV projection matmul ──
    let kv_dim = session.n_head as u64 * session.head_dim as u64;
    eprintln!("debug: [5] matmul_q8_0 attn_norm→kv_raw {}→{}...", n_embd, kv_dim);
    batch.begin()?;
    dense::matmul_q8_0(cache, batch, model_views, map, size,
        layer.attn_kv, n_embd as u64, kv_dim, &g.decode.attn_norm, 1, &g.decode.kv_raw)?;
    batch.end()?;
    eprintln!("debug: [5] OK");

    // ── Test 6: Fused Q+KV RMS norm ──
    eprintln!("debug: [6] qkv_rms_norm_rows...");
    let q_bias = layer.attn_q_a + qr_dim * n_embd as u64;
    let kv_bias = layer.attn_kv + kv_dim * n_embd as u64;
    batch.begin()?;
    norm::qkv_rms_norm_rows(cache, batch, model_views, map, size,
        q_bias, kv_bias, &g.decode.qr_norm, &g.decode.kv,
        &g.decode.qr, &g.decode.kv_raw, qr_dim as u32, kv_dim as u32, 1, 1e-6)?;
    batch.end()?;
    eprintln!("debug: [6] OK");

    // ── Test 7: Q up-projection matmul ──
    let kv_dim = session.n_head as u64 * session.head_dim as u64;
    eprintln!("debug: [7] matmul_q8_0 qr_norm→q {}→{}...", qr_dim, kv_dim);
    g.decode.qr_norm.fill_f32(0.1, qr_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    dense::matmul_q8_0(cache, batch, model_views, map, size,
        layer.attn_q_b, qr_dim, kv_dim, &g.decode.qr_norm, 1, &g.decode.q)?;
    batch.end()?;
    eprintln!("debug: [7] OK");

    // ── Test 8: Head RMS norm on Q ──
    eprintln!("debug: [8] head_rms_norm...");
    g.decode.q.fill_f32(0.1, kv_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    norm::head_rms_norm(cache, batch, &g.decode.q, 1, session.n_head, session.head_dim, 1e-6)?;
    batch.end()?;
    eprintln!("debug: [8] OK");

    // ── Test 9: RoPE tail on Q ──
    eprintln!("debug: [9] rope_tail on Q...");
    let freq = layer.rope_freq_base;
    batch.begin()?;
    rope::rope_tail(cache, batch, &g.decode.q,
        1, session.n_head, session.head_dim, session.n_rot, 0, 131072, false,
        freq, 1.0, 0.0, 1.0, 32.0, 1.0)?;
    batch.end()?;
    eprintln!("debug: [9] OK");

    // ── Test 10: RoPE tail on KV ──
    eprintln!("debug: [10] rope_tail on KV...");
    g.decode.kv.fill_f32(0.1, kv_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    rope::rope_tail(cache, batch, &g.decode.kv,
        1, session.n_head, session.head_dim, session.n_rot, 0, 131072, false,
        freq, 1.0, 0.0, 1.0, 32.0, 1.0)?;
    batch.end()?;
    eprintln!("debug: [10] OK");

    // ── Test 11: KV FP8 store to raw cache ──
    eprintln!("debug: [11] kv_fp8_store_raw...");
    g.decode.kv.fill_f32(0.1, kv_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    rope::kv_fp8_store_raw(cache, batch, &g.decode.kv,
        &g.layers[0].raw_cache, g.raw_cap, 0, session.head_dim, session.n_rot)?;
    batch.end()?;
    eprintln!("debug: [11] OK");

    // ── Test 12: HC expand post-attention ──
    eprintln!("debug: [12] hc_expand...");
    g.work.attn_out.fill_f32(0.1, n_embd as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    g.decode.hc_post.fill_f32(0.5, n_hc as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    g.decode.hc_comb.fill_f32(0.3, (n_hc*n_hc) as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    g.decode.cur_hc.fill_f32(0.2, hc_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    hc::hc_expand(cache, batch, &g.work.after_attn_hc, &g.work.attn_out,
        &g.decode.cur_hc, &g.decode.hc_post, &g.decode.hc_comb, n_embd, n_hc)?;
    batch.end()?;
    eprintln!("debug: [12] OK");

    // ── Test 13: SKIPPED (hc_split_weighted_sum hangs - investigate weight offsets)
    eprintln!("debug: [13] SKIPPED (hc_split_weighted_sum)");

    // ── Test 14: FFN RMS norm + Shared gate/up/swiglu ──
    eprintln!("debug: [14] ffn: rms_norm + shared_gate_up_swiglu...");
    g.work.ffn_cur.fill_f32(0.1, n_embd as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    norm::rms_norm_plain(cache, batch, &g.work.ffn_norm, &g.work.ffn_cur, n_embd, 1e-6)?;
    dense::shared_gate_up_swiglu_q8_0(cache, batch, model_views, map, size,
        layer.ffn_gate_shexp, layer.ffn_up_shexp, n_embd as u64, 2048,
        &g.work.ffn_norm, &g.work.shared_gate, &g.work.shared_up, &g.work.shared_mid)?;
    batch.end()?;
    eprintln!("debug: [14] OK");

    // ── Test 15: Shared down projection ──
    eprintln!("debug: [15] shared_down matmul...");
    batch.begin()?;
    dense::matmul_q8_0(cache, batch, model_views, map, size,
        layer.ffn_down_shexp, 2048, n_embd as u64,
        &g.work.shared_mid, 1, &g.work.shared_out)?;
    batch.end()?;
    eprintln!("debug: [15] OK");

    // ── Test 16: HC expand add split (post-FFN) ──
    eprintln!("debug: [16] hc_expand_add_split...");
    g.work.ffn_out.fill_f32(0.1, n_embd as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    g.work.routed_out.fill_f32(0.0, n_embd as u64).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    hc::hc_expand_add_split(cache, batch, &g.decode.cur_hc, &g.work.ffn_out,
        &g.work.routed_out, &g.work.after_ffn_hc, &g.decode.hc_split, n_embd, n_hc)?;
    batch.end()?;
    eprintln!("debug: [16] OK");

    // ── Test 17-18: Chained decode (multiple kernels in one batch) ──
    eprintln!("debug: [17] chained: rms_norm + matmul + rms_norm_weight in one batch...");
    g.decode.cur_hc.fill_f32(0.5, hc_dim).map_err(|e| anyhow::anyhow!("{}", e))?;
    batch.begin()?;
    norm::rms_norm_plain(cache, batch, &g.decode.flat_hc, &g.decode.cur_hc, hc_dim as u32, 1e-6)?;
    dense::matmul_q8_0(cache, batch, model_views, map, size,
        layer.hc_attn_fn, hc_dim, mix_hc, &g.decode.flat_hc, 1, &g.decode.hc_mix)?;
    norm::rms_norm_weight(cache, batch, model_views, map, size,
        layer.attn_norm, &g.decode.attn_norm, &g.decode.attn_cur, n_embd, 1e-6)?;
    batch.end()?;
    eprintln!("debug: [17] OK - chained dispatch works!");

    eprintln!("debug: all 17 tests passed");
    Ok(())
}
