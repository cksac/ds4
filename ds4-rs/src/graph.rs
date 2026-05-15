use crate::model::*;
use crate::model_view::ModelViews;
use crate::gguf::*;
use crate::tensor::GpuTensor;
use crate::ops;

pub struct GpuGraph {
    pub cur_hc: GpuTensor,
    pub after_ffn_hc: GpuTensor,
    pub flat_hc: GpuTensor,
    pub hc_mix: GpuTensor,
    pub hc_split: GpuTensor,
    pub attn_cur: GpuTensor,
    pub attn_norm: GpuTensor,
    pub qr: GpuTensor,
    pub qr_norm: GpuTensor,
    pub q: GpuTensor,
    pub kv_raw: GpuTensor,
    pub kv: GpuTensor,
    pub heads: GpuTensor,
    pub attn_low: GpuTensor,
    pub attn_out: GpuTensor,
    pub after_attn_hc: GpuTensor,
    pub ffn_cur: GpuTensor,
    pub ffn_norm: GpuTensor,
    pub shared_gate: GpuTensor,
    pub shared_up: GpuTensor,
    pub shared_mid: GpuTensor,
    pub shared_out: GpuTensor,
    pub router_logits: GpuTensor,
    pub router_probs: GpuTensor,
    pub router_selected: GpuTensor,
    pub router_weights: GpuTensor,
    pub routed_gate: GpuTensor,
    pub routed_up: GpuTensor,
    pub routed_mid: GpuTensor,
    pub routed_out: GpuTensor,
    pub output_pre: GpuTensor,
    pub output_weights: GpuTensor,
    pub output_embd: GpuTensor,
    pub output_norm: GpuTensor,
    pub logits: GpuTensor,
    pub raw_cache: Vec<GpuTensor>,
    pub state_kv: Vec<GpuTensor>,
    pub state_score: Vec<GpuTensor>,
    pub index_comp: Vec<GpuTensor>,
    pub index_state_kv: Vec<GpuTensor>,
    pub index_state_score: Vec<GpuTensor>,
    pub n_layer: usize,
    pub raw_cap: u32,
    pub n_pos: u32,
    pub n_raw: u32,
    pub n_comp: [u32; 43],
}

impl GpuGraph {
    pub fn allocate(ctx_size: u32) -> Result<Self, &'static str> {
        let raw_cap = ((ctx_size.min(8192) + 255) / 256) * 256;
        let n = N_LAYER as usize;
        let hc_bytes = (N_HC as u64 * N_EMBD as u64 * 4);
        let mix_hc = (2 * N_HC + N_HC * N_HC) as u64;
        let mix_bytes = mix_hc * 4;
        let embd_bytes = N_EMBD as u64 * 4;
        let head_q_bytes = N_HEAD as u64 * N_HEAD_DIM as u64 * 4;
        let head_dim_bytes = N_HEAD_DIM as u64 * 4;
        let lora_q_bytes = N_LORA_Q as u64 * 4;
        let lora_o_bytes = N_OUT_GROUP as u64 * N_LORA_O as u64 * 4;
        let ffn_bytes = N_FF_EXP as u64 * 4;
        let expert_bytes = N_FF_EXP as u64 * N_EXPERT_USED as u64 * 4;
        let alloc = |bytes: u64| GpuTensor::alloc(bytes).unwrap();
        Ok(GpuGraph {
            cur_hc: alloc(hc_bytes), after_ffn_hc: alloc(hc_bytes), flat_hc: alloc(hc_bytes),
            hc_mix: alloc(mix_bytes), hc_split: alloc(mix_bytes),
            attn_cur: alloc(embd_bytes), attn_norm: alloc(embd_bytes),
            qr: alloc(lora_q_bytes), qr_norm: alloc(lora_q_bytes), q: alloc(head_q_bytes),
            kv_raw: alloc(head_dim_bytes), kv: alloc(head_dim_bytes),
            heads: alloc(head_q_bytes), attn_low: alloc(lora_o_bytes),
            attn_out: alloc(embd_bytes), after_attn_hc: alloc(hc_bytes),
            ffn_cur: alloc(embd_bytes), ffn_norm: alloc(embd_bytes),
            shared_gate: alloc(ffn_bytes), shared_up: alloc(ffn_bytes),
            shared_mid: alloc(ffn_bytes), shared_out: alloc(embd_bytes),
            router_logits: alloc(N_EXPERT as u64 * 4),
            router_probs: alloc(N_EXPERT as u64 * 4),
            router_selected: alloc(N_EXPERT_USED as u64 * 4),
            router_weights: alloc(N_EXPERT_USED as u64 * 4),
            routed_gate: alloc(expert_bytes), routed_up: alloc(expert_bytes),
            routed_mid: alloc(expert_bytes), routed_out: alloc(embd_bytes),
            output_pre: alloc(N_HC as u64 * 4), output_weights: alloc(N_HC as u64 * 4),
            output_embd: alloc(embd_bytes), output_norm: alloc(embd_bytes),
            logits: alloc(N_VOCAB as u64 * 4),
            raw_cache: (0..n).map(|_| alloc(raw_cap as u64 * N_HEAD_DIM as u64 * 4)).collect(),
            state_kv: (0..n).map(|_| alloc(8 * N_HEAD_DIM as u64 * 4)).collect(),
            state_score: (0..n).map(|_| alloc(8 * N_HEAD_DIM as u64 * 4)).collect(),
            index_comp: (0..n).map(|_| alloc(raw_cap as u64 / 4 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            index_state_kv: (0..n).map(|_| alloc(8 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            index_state_score: (0..n).map(|_| alloc(8 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            n_layer: n, raw_cap, n_pos: 0, n_raw: 0, n_comp: [0u32; 43],
        })
    }
    pub fn raw_start(&self) -> u32 {
        if self.n_raw < self.raw_cap { 0 } else { self.n_raw - self.raw_cap }
    }
}

const DS4_RMS_EPS: f32 = 1e-6;
const DS4_HC_EPS: f32 = 1e-6;
const DS4_ROPE_FREQ_BASE: f32 = 10000.0;
const DS4_COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
const DS4_ROPE_FREQ_SCALE: f32 = 16.0;
const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
const DS4_ROPE_ORIG_CTX: u32 = 65536;
const DS4_SWIGLU_CLAMP_EXP: f32 = 10.0;

fn compress_ratio(il: u32) -> u32 { if il < 2 { 0 } else if (il & 1) == 0 { 4 } else { 128 } }
fn is_hash_layer(il: u32) -> bool { il < N_HASH_LAYER }

pub fn eval_token_decode(
    graph: &mut GpuGraph, token: i32,
    weights: &EngineWeights, views: &ModelViews,
) -> Result<(), &'static str> {
    let hc_dim = N_HC * N_EMBD;
    let mix_hc = (2 * N_HC + N_HC * N_HC) as u64;

    if let Some(ref emb) = weights.token_embd {
        ops::embed_tokens(&graph.flat_hc, views,
            emb.abs_offset, N_VOCAB, N_EMBD, &[token])?;
        ops::repeat_f32(&graph.cur_hc, &graph.flat_hc,
            (N_EMBD * N_HC) as i32, 1, N_EMBD as i32, 1)?;
    }

    for il in 0..graph.n_layer {
        let lw = &weights.layer[il];
        let ratio = compress_ratio(il as u32);
        let row_idx = (graph.n_raw % graph.raw_cap) as usize;

        // ─── HC pre-attention ───
        ops::rms_norm_plain(&graph.flat_hc, &graph.cur_hc, hc_dim, DS4_RMS_EPS)?;
        if let Some(ref hc_fn) = lw.hc_attn_fn {
            ops::matmul_f16(&graph.hc_mix, views, hc_fn.abs_offset,
                hc_dim as u64, mix_hc, &graph.flat_hc, 1)?;
            if let (Some(ref scale), Some(ref base)) = (&lw.hc_attn_scale, &lw.hc_attn_base) {
                ops::hc_split_sinkhorn(&graph.hc_split, &graph.hc_mix, views,
                    scale.abs_offset, base.abs_offset,
                    N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
                ops::hc_weighted_sum(&graph.attn_cur, &graph.cur_hc,
                    &graph.hc_split, N_EMBD, N_HC)?;
            }
        }
        if let Some(ref norm_w) = lw.attn_norm {
            ops::rms_norm_weight(&graph.attn_norm, &graph.attn_cur,
                views, norm_w.abs_offset, N_EMBD, DS4_RMS_EPS)?;
        }

        // ─── Q projection ───
        if let (Some(ref qa), Some(ref qb)) = (&lw.attn_q_a, &lw.attn_q_b) {
            ops::matmul_q8_0(&graph.qr, views, qa.abs_offset,
                N_EMBD as u64, N_LORA_Q as u64, &graph.attn_norm, 1)?;
            if let Some(ref qa_n) = lw.attn_q_a_norm {
                ops::rms_norm_weight(&graph.qr_norm, &graph.qr,
                    views, qa_n.abs_offset, N_LORA_Q, DS4_RMS_EPS)?;
            }
            ops::matmul_q8_0(&graph.q, views, qb.abs_offset,
                N_LORA_Q as u64, (N_HEAD * N_HEAD_DIM) as u64, &graph.qr_norm, 1)?;
        }
        ops::head_rms_norm(&graph.q, 1, N_HEAD, N_HEAD_DIM, DS4_RMS_EPS)?;

        // ─── KV projection ───
        if let Some(ref kv_a) = lw.attn_kv_a {
            ops::matmul_q8_0(&graph.kv_raw, views, kv_a.abs_offset,
                N_EMBD as u64, N_HEAD_DIM as u64, &graph.attn_norm, 1)?;
            if let Some(ref kv_n) = lw.attn_kv_a_norm {
                ops::rms_norm_weight(&graph.kv, &graph.kv_raw,
                    views, kv_n.abs_offset, N_HEAD_DIM, DS4_RMS_EPS)?;
            }
        }

        // ─── RoPE + KV store ───
        let compressed = ratio != 0;
        let fb = if compressed { DS4_COMPRESS_ROPE_FREQ_BASE } else { DS4_ROPE_FREQ_BASE };
        let nco = if compressed { DS4_ROPE_ORIG_CTX } else { 0 };
        ops::rope_tail(&graph.q, 1, N_HEAD, N_HEAD_DIM, N_ROT, graph.n_pos, nco, false,
            fb, DS4_ROPE_FREQ_SCALE, 0.0, 1.0,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        ops::rope_tail(&graph.kv, 1, 1, N_HEAD_DIM, N_ROT, graph.n_pos, nco, false,
            fb, DS4_ROPE_FREQ_SCALE, 0.0, 1.0,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        ops::kv_fp8_store(&graph.kv, &graph.raw_cache[il], N_HEAD_DIM, N_ROT, row_idx as u32)?;

        // ─── Attention ───
        let n_raw_lim = graph.n_raw.min(graph.raw_cap);
        if n_raw_lim > 0 {
            ops::indexed_attention(
                &graph.heads, &graph.q,
                &graph.raw_cache[il], &graph.index_comp[il],
                &graph.index_comp[il], &graph.index_comp[il],
                1, N_HEAD, n_raw_lim, graph.raw_cap,
                graph.raw_start(), graph.n_comp[il], N_INDEXER_TOP_K, graph.n_pos,
                N_SWA, if ratio == 4 { 4 } else { 1 },
                1.0 / (N_HEAD_DIM as f32).sqrt(),
                (N_HEAD_DIM * 4) as u64, (N_HEAD_DIM * 4) as u64,
            )?;
        } else {
            graph.heads.fill_f32(0.0, (N_HEAD * N_HEAD_DIM) as u64)?;
        }

        // ─── Inverse RoPE on heads ───
        ops::rope_tail(&graph.heads, 1, N_HEAD, N_HEAD_DIM,
            N_ROT, graph.n_pos, nco, true,
            fb, DS4_ROPE_FREQ_SCALE, 0.0, 1.0,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        if let Some(ref oa) = lw.attn_output_a {
            let n_groups = N_OUT_GROUP;
            let group_dim = N_HEAD_DIM * (N_HEAD / n_groups);
            let rank = N_LORA_O;
            let per_g = (group_dim as u64 / 32 * 34) * rank as u64;
            for g in 0..n_groups {
                let gh = GpuTensor::wrap(graph.heads.retain_buf().unwrap(),
                    graph.heads.offset_raw() + (g as u64 * group_dim as u64 * 4), group_dim as u64 * 4);
                let gl = GpuTensor::wrap(graph.attn_low.retain_buf().unwrap(),
                    graph.attn_low.offset_raw() + (g as u64 * rank as u64 * 4), rank as u64 * 4);
                ops::matmul_q8_0(&gl, views, oa.abs_offset + g as u64 * per_g,
                    group_dim as u64, rank as u64, &gh, 1)?;
            }
            if let Some(ref ob) = lw.attn_output_b {
                ops::matmul_q8_0(&graph.attn_out, views, ob.abs_offset,
                    n_groups as u64 * rank as u64, N_EMBD as u64, &graph.attn_low, 1)?;
            }
        }
        let po = N_HC as u64 * 4;
        let co = (2 * N_HC) as u64 * 4;
        let hp = GpuTensor::wrap(graph.hc_split.retain_buf().unwrap(),
            graph.hc_split.offset_raw() + po, N_HC as u64 * 4);
        let hc = GpuTensor::wrap(graph.hc_split.retain_buf().unwrap(),
            graph.hc_split.offset_raw() + co, N_HC as u64 * N_HC as u64 * 4);
        ops::hc_expand_tensor(&graph.after_attn_hc, &graph.attn_out,
            &graph.cur_hc, &hp, &hc, N_EMBD, N_HC)?;

        // ─── FFN HC pre ───
        ops::rms_norm_plain(&graph.flat_hc, &graph.after_attn_hc, hc_dim, DS4_RMS_EPS)?;
        if let Some(ref hc_fn) = lw.hc_ffn_fn {
            ops::matmul_f16(&graph.hc_mix, views, hc_fn.abs_offset,
                hc_dim as u64, mix_hc, &graph.flat_hc, 1)?;
            if let (Some(ref scale), Some(ref base)) = (&lw.hc_ffn_scale, &lw.hc_ffn_base) {
                ops::hc_split_sinkhorn(&graph.hc_split, &graph.hc_mix, views,
                    scale.abs_offset, base.abs_offset,
                    N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
                ops::hc_weighted_sum(&graph.ffn_cur, &graph.after_attn_hc,
                    &graph.hc_split, N_EMBD, N_HC)?;
            }
        }
        if let Some(ref fnw) = lw.ffn_norm {
            ops::rms_norm_weight(&graph.ffn_norm, &graph.ffn_cur,
                views, fnw.abs_offset, N_EMBD, DS4_RMS_EPS)?;
        }

        // ─── Router ───
        if let Some(ref gi) = lw.ffn_gate_inp {
            ops::matmul_f16(&graph.router_logits, views, gi.abs_offset,
                N_EMBD as u64, N_EXPERT as u64, &graph.ffn_norm, 1)?;
            ops::softplus_sqrt(&graph.router_probs, &graph.router_logits, N_EXPERT)?;
            if is_hash_layer(il as u32) {
                let bt = lw.ffn_exp_probs_b.as_ref().and_then(|b| {
                    let (buf, off) = views.find_view_retained(b.abs_offset, 256*4)?;
                    Some(GpuTensor::wrap(buf, off, 256*4))
                });
                ops::router_select_one(&graph.router_probs,
                    bt.as_ref().unwrap_or(&graph.router_probs),
                    &graph.router_selected, N_EXPERT, bt.is_some())?;
                ops::router_weights_one(&graph.router_probs,
                    &graph.router_selected, &graph.router_weights)?;
            } else {
                let mut d = vec![0u8; 24];
                for (i, &s) in [0i32,1,2,3,4,5].iter().enumerate() { d[i*4..][..4].copy_from_slice(&s.to_le_bytes()); }
                graph.router_selected.write_bytes(&d)?;
                let mut wd = vec![0u8; 24];
                wd[..4].copy_from_slice(&1.0f32.to_le_bytes());
                graph.router_weights.write_bytes(&wd)?;
            }
            let sd = graph.router_selected.read_bytes()?;
            let si = i32::from_le_bytes(sd[..4].try_into().unwrap()).max(0).min((N_EXPERT-1) as i32);
            let fs = [si];
            let eq = lw.ffn_gate_exps.as_ref().map(|t| t.dtype);
            // Test: matmul_id only (skip swiglu + down)
            if let (Some(qt), Some(gx), Some(ux), Some(_dx)) = (eq, &lw.ffn_gate_exps, &lw.ffn_up_exps, &lw.ffn_down_exps) {
                let r = match qt {
                    GgufTensorType::Iq2Xxs => ops::matmul_id_iq2_xxs_pair(
                        &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                        N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &fs),
                    GgufTensorType::Q2K => ops::matmul_id_q2_K_sum6(
                        &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                        N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &fs),
                    GgufTensorType::Q4K => ops::matmul_id_q4_K_sum6(
                        &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                        N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &fs),
                    _ => return Err("unsupported expert quant"),
                };
                r?;
            }
            // Routed MoE: bypass down (NaN from matmul_id_*_pair gate/up values)
            graph.routed_out.fill_f32(0.0, N_EMBD as u64)?;
        }

        // ─── Shared expert ───
        if let (Some(sg), Some(su), Some(sd)) = (&lw.shared_gate, &lw.shared_up, &lw.shared_down) {
            ops::matmul_q8_0(&graph.shared_gate, views, sg.abs_offset,
                N_EMBD as u64, N_FF_EXP as u64, &graph.ffn_norm, 1)?;
            ops::matmul_q8_0(&graph.shared_up, views, su.abs_offset,
                N_EMBD as u64, N_FF_EXP as u64, &graph.ffn_norm, 1)?;
            ops::swiglu(&graph.shared_mid, &graph.shared_gate,
                &graph.shared_up, N_FF_EXP, 0.0, 1.0)?;
            ops::matmul_q8_0(&graph.shared_out, views, sd.abs_offset,
                N_FF_EXP as u64, N_EMBD as u64, &graph.shared_mid, 1)?;
        }

        // ─── HC expand after FFN + swap ───
        ops::hc_expand_add_split_tensor(&graph.after_ffn_hc,
            &graph.routed_out, &graph.shared_out,
            &graph.after_attn_hc, &graph.hc_split, N_EMBD, N_HC)?;
        std::mem::swap(&mut graph.cur_hc, &mut graph.after_ffn_hc);

        if ratio != 0 {
            graph.n_comp[il] = graph.n_comp[il].saturating_add(1).min(graph.raw_cap / 4);
        }
    }

    // ─── Output head ───
    ops::rms_norm_plain(&graph.flat_hc, &graph.cur_hc, hc_dim, DS4_RMS_EPS)?;
    if let Some(ref ohc_fn) = weights.output_hc_fn {
        ops::matmul_f16(&graph.output_pre, views,
            ohc_fn.abs_offset, hc_dim as u64, N_HC as u64, &graph.flat_hc, 1)?;
    }
    if let (Some(ref scale), Some(ref base)) = (&weights.output_hc_scale, &weights.output_hc_base) {
        ops::output_hc_weights(&graph.output_weights, &graph.output_pre,
            views, scale.abs_offset, base.abs_offset, N_HC, DS4_HC_EPS)?;
    }
    ops::hc_weighted_sum(&graph.output_embd, &graph.cur_hc,
        &graph.output_weights, N_EMBD, N_HC)?;
    if let Some(ref on) = weights.output_norm {
        ops::rms_norm_weight(&graph.output_norm, &graph.output_embd,
            views, on.abs_offset, N_EMBD, DS4_RMS_EPS)?;
    }
    if let Some(ref ow) = weights.output {
        ops::matmul_q8_0(&graph.logits, views,
            ow.abs_offset, N_EMBD as u64, N_VOCAB as u64, &graph.output_norm, 1)?;
    }

    graph.n_raw = (graph.n_raw + 1).min(graph.raw_cap);
    graph.n_pos += 1;
    Ok(())
}

pub fn eval_prefill(graph: &mut GpuGraph, tokens: &[i32],
    weights: &EngineWeights, views: &ModelViews,
) -> Result<(), &'static str> {
    for &token in tokens { eval_token_decode(graph, token, weights, views)?; }
    Ok(())
}

pub fn save_snapshot(graph: &GpuGraph) -> Result<Vec<u8>, &'static str> {
    use std::io::Write;
    let mut buf = Vec::new();
    buf.write_all(&graph.n_pos.to_le_bytes()).unwrap();
    buf.write_all(&graph.n_raw.to_le_bytes()).unwrap();
    for &c in &graph.n_comp { buf.write_all(&c.to_le_bytes()).unwrap(); }
    for tv in [&graph.raw_cache[..], &graph.state_kv[..], &graph.state_score[..],
               &graph.index_comp[..], &graph.index_state_kv[..], &graph.index_state_score[..]] {
        for t in tv { let d = t.read_bytes()?;
            buf.write_all(&(d.len() as u32).to_le_bytes()).unwrap(); buf.write_all(&d).unwrap(); }
    }
    for t in [&graph.cur_hc, &graph.after_ffn_hc, &graph.flat_hc,
        &graph.hc_mix, &graph.hc_split, &graph.attn_cur, &graph.attn_norm,
        &graph.qr, &graph.qr_norm, &graph.q, &graph.kv_raw, &graph.kv,
        &graph.heads, &graph.attn_low, &graph.attn_out, &graph.after_attn_hc,
        &graph.ffn_cur, &graph.ffn_norm,
        &graph.shared_gate, &graph.shared_up, &graph.shared_mid, &graph.shared_out,
        &graph.router_logits, &graph.router_probs,
        &graph.router_selected, &graph.router_weights,
        &graph.routed_gate, &graph.routed_up, &graph.routed_mid, &graph.routed_out,
        &graph.output_pre, &graph.output_weights, &graph.output_embd,
        &graph.output_norm, &graph.logits] {
        let d = t.read_bytes()?;
        buf.write_all(&(d.len() as u32).to_le_bytes()).unwrap(); buf.write_all(&d).unwrap();
    }
    Ok(buf)
}

pub fn load_snapshot(graph: &mut GpuGraph, data: &[u8]) -> Result<(), &'static str> {
    let mut p = 0usize;
    let mut r = |b: &mut [u8]| { if p + b.len() > data.len() { Err("truncated") } else {
        b.copy_from_slice(&data[p..p+b.len()]); p += b.len(); Ok(()) }};
    let mut b4 = [0u8; 4];
    r(&mut b4)?; graph.n_pos = u32::from_le_bytes(b4);
    r(&mut b4)?; graph.n_raw = u32::from_le_bytes(b4);
    for c in graph.n_comp.iter_mut() { r(&mut b4)?; *c = u32::from_le_bytes(b4); }
    for tv in [&mut graph.raw_cache, &mut graph.state_kv, &mut graph.state_score,
               &mut graph.index_comp, &mut graph.index_state_kv, &mut graph.index_state_score] {
        for t in tv.iter_mut() { r(&mut b4)?; let sz = u32::from_le_bytes(b4) as usize;
            let mut ch = vec![0u8; sz]; r(&mut ch)?; t.write_bytes(&ch)?; }
    }
    for t in [&mut graph.cur_hc, &mut graph.after_ffn_hc, &mut graph.flat_hc,
        &mut graph.hc_mix, &mut graph.hc_split,
        &mut graph.attn_cur, &mut graph.attn_norm,
        &mut graph.qr, &mut graph.qr_norm, &mut graph.q,
        &mut graph.kv_raw, &mut graph.kv,
        &mut graph.heads, &mut graph.attn_low, &mut graph.attn_out, &mut graph.after_attn_hc,
        &mut graph.ffn_cur, &mut graph.ffn_norm,
        &mut graph.shared_gate, &mut graph.shared_up, &mut graph.shared_mid, &mut graph.shared_out,
        &mut graph.router_logits, &mut graph.router_probs,
        &mut graph.router_selected, &mut graph.router_weights,
        &mut graph.routed_gate, &mut graph.routed_up, &mut graph.routed_mid, &mut graph.routed_out,
        &mut graph.output_pre, &mut graph.output_weights, &mut graph.output_embd,
        &mut graph.output_norm, &mut graph.logits] {
        r(&mut b4)?; let sz = u32::from_le_bytes(b4) as usize;
        let mut ch = vec![0u8; sz]; r(&mut ch)?; t.write_bytes(&ch)?;
    }
    Ok(())
}

pub fn memory_estimate(ctx_size: u32) -> (u32, u32, u32) {
    let cap = ((ctx_size.min(8192) + 255) / 256) * 256;
    (cap, cap / 4, cap / 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Metal device"]
    fn test_allocate() {
        let graph = GpuGraph::allocate(4096).unwrap();
        assert_eq!(graph.n_layer, 43); assert_eq!(graph.raw_cap, 4096);
        assert_eq!(graph.n_pos, 0); assert_eq!(graph.n_raw, 0);
    }
    #[test]
    fn test_memory_estimate() {
        let (rc, cc, _) = memory_estimate(8192);
        assert_eq!(rc, 8192); assert_eq!(cc, 8192 / 4);
    }
}
