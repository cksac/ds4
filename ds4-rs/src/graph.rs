use crate::model::*;
use crate::model_view::ModelViews;
use crate::gguf::*;
use crate::tensor::GpuTensor;
use crate::ops;

pub struct GpuGraph {
    pub hc_mix: Vec<GpuTensor>,
    pub hc_split: Vec<GpuTensor>,
    pub cur: Vec<GpuTensor>,
    pub q: Vec<GpuTensor>,
    pub kv: Vec<GpuTensor>,
    pub raw_cache: Vec<GpuTensor>,
    pub state_kv: Vec<GpuTensor>,
    pub state_score: Vec<GpuTensor>,
    pub index_comp: Vec<GpuTensor>,
    pub index_state_kv: Vec<GpuTensor>,
    pub index_state_score: Vec<GpuTensor>,
    pub q_after_norm: Vec<GpuTensor>,
    pub heads: Vec<GpuTensor>,
    pub attn_out: Vec<GpuTensor>,
    pub attn_low: Vec<GpuTensor>,
    pub shared_mid: Vec<GpuTensor>,
    pub routed_out: Vec<GpuTensor>,
    // Router/MoE scratch per layer
    pub router_logits: Vec<GpuTensor>,
    pub router_probs: Vec<GpuTensor>,
    pub router_selected: Vec<GpuTensor>,
    pub router_weights: Vec<GpuTensor>,
    pub expert_gate: Vec<GpuTensor>,
    pub expert_up: Vec<GpuTensor>,
    pub expert_mid: Vec<GpuTensor>,
    // Logits
    pub logits: GpuTensor,
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
        let alloc = |bytes: u64| GpuTensor::alloc(bytes).unwrap();
        Ok(GpuGraph {
            hc_mix: (0..n).map(|_| alloc(4 * N_HC as u64 * N_EMBD as u64)).collect(),
            hc_split: (0..n).map(|_| alloc(4 * 5 * 4 * 4)).collect(),
            cur: (0..n).map(|_| alloc(N_EMBD as u64 * 4)).collect(),
            q: (0..n).map(|_| alloc(N_HEAD as u64 * N_HEAD_DIM as u64 * 4)).collect(),
            kv: (0..n).map(|_| alloc(N_HEAD_DIM as u64 * 4)).collect(),
            raw_cache: (0..n).map(|_| alloc(raw_cap as u64 * N_HEAD_DIM as u64 * 4)).collect(),
            state_kv: (0..n).map(|_| alloc(8 * N_HEAD_DIM as u64 * 4)).collect(),
            state_score: (0..n).map(|_| alloc(8 * N_HEAD_DIM as u64 * 4)).collect(),
            index_comp: (0..n).map(|_| alloc(raw_cap as u64 / 4 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            index_state_kv: (0..n).map(|_| alloc(8 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            index_state_score: (0..n).map(|_| alloc(8 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            q_after_norm: (0..n).map(|_| alloc(N_LORA_Q as u64 * 4)).collect(),
            heads: (0..n).map(|_| alloc(N_HEAD as u64 * N_HEAD_DIM as u64 * 4)).collect(),
            attn_out: (0..n).map(|_| alloc(N_EMBD as u64 * 4)).collect(),
            attn_low: (0..n).map(|_| alloc(N_OUT_GROUP as u64 * N_LORA_O as u64 * 4)).collect(),
            shared_mid: (0..n).map(|_| alloc(N_FF_EXP as u64 * 4)).collect(),
            routed_out: (0..n).map(|_| alloc(N_FF_EXP as u64 * 4)).collect(),
            router_logits: (0..n).map(|_| alloc(N_EXPERT as u64 * 4)).collect(),
            router_probs: (0..n).map(|_| alloc(N_EXPERT as u64 * 4)).collect(),
            router_selected: (0..n).map(|_| alloc(N_EXPERT_USED as u64 * 4)).collect(),
            router_weights: (0..n).map(|_| alloc(N_EXPERT_USED as u64 * 4)).collect(),
            expert_gate: (0..n).map(|_| alloc(N_FF_EXP as u64 * 4)).collect(),
            expert_up: (0..n).map(|_| alloc(N_FF_EXP as u64 * 4)).collect(),
            expert_mid: (0..n).map(|_| alloc(N_FF_EXP as u64 * 4)).collect(),
            logits: alloc(N_VOCAB as u64 * 4),
            n_layer: n,
            raw_cap,
            n_pos: 0,
            n_raw: 0,
            n_comp: [0u32; 43],
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
#[allow(dead_code)]
const DS4_EXPERT_WEIGHT_SCALE: f32 = 1.5;

fn compress_ratio(il: u32) -> u32 {
    if il < 2 { 0 } else if (il & 1) == 0 { 4 } else { 128 }
}

fn is_hash_layer(il: u32) -> bool {
    il < N_HASH_LAYER
}

fn model_buf(model_map: &[u8], offset: u64, bytes: u64) -> metal::Buffer {
    let device = crate::bridge::with_device(|d| d.clone()).unwrap();
    let data = &model_map[offset as usize..][..bytes as usize];
    device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void, bytes,
        metal::MTLResourceOptions::StorageModeShared)
}

pub fn eval_token_decode(
    graph: &mut GpuGraph, token: i32,
    weights: &EngineWeights, _views: &ModelViews,
    model_map: &[u8], _model_size: u64,
) -> Result<(), &'static str> {
    let pos = graph.n_pos;
    let n = graph.n_layer;

    // Embed token → repeat across HC channels
    if let Some(ref emb) = weights.token_embd {
        ops::embed_tokens(&graph.hc_mix[0], model_map, _model_size,
            emb.abs_offset, N_VOCAB, N_EMBD, &[token])?;
        // Repeat across 4 HC channels and all layers
        for il in 0..n {
            ops::repeat_f32(&graph.hc_mix[il], &graph.hc_mix[il],
                (N_EMBD * N_HC) as i32, 1,
                N_EMBD as i32, 1)?;
        }
    }

    for il in 0..n {
        let lw = &weights.layer[il];
        let ratio = compress_ratio(il as u32);
        let is_indexed = ratio == 4;
        let row_idx = (graph.n_pos as usize) % graph.raw_cap as usize;

        // ─── HC pre-attention ───
        if lw.hc_attn_fn.is_some() {
            ops::hc_split_sinkhorn(&graph.hc_split[il], &graph.hc_mix[il],
                model_map, _model_size, 0, 0,
                N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
            ops::hc_weighted_sum(&graph.cur[il], &graph.hc_mix[il],
                &graph.hc_split[il], N_EMBD, N_HC)?;
        }

        // ─── RMS norm before attention ───
        ops::rms_norm_plain(&graph.cur[il], &graph.cur[il],
            N_EMBD, DS4_RMS_EPS)?;

        // ─── Q projection ───
        if let (Some(qa), Some(qb)) = (&lw.attn_q_a, &lw.attn_q_b) {
            ops::matmul_f16(&graph.q_after_norm[il], model_map, _model_size,
                qa.abs_offset, N_EMBD as u64, N_LORA_Q as u64,
                &graph.cur[il], 1)?;
            ops::rms_norm_plain(&graph.q_after_norm[il], &graph.q_after_norm[il],
                N_LORA_Q, DS4_RMS_EPS)?;
            ops::matmul_f16(&graph.q[il], model_map, _model_size,
                qb.abs_offset, N_LORA_Q as u64,
                (N_HEAD * N_HEAD_DIM) as u64,
                &graph.q_after_norm[il], 1)?;
        }

        // ─── KV projection ───
        if let (Some(kva), Some(kvb)) = (&lw.attn_kv_a, &lw.attn_kv_b) {
            ops::matmul_f16(&graph.kv[il], model_map, _model_size,
                kva.abs_offset, N_EMBD as u64, N_LORA_Q as u64,
                &graph.cur[il], 1)?;
            ops::matmul_f16(&graph.kv[il], model_map, _model_size,
                kvb.abs_offset, N_LORA_Q as u64, N_HEAD_DIM as u64,
                &graph.kv[il], 1)?;
        }

        // ─── RoPE ───
        let rope_base = if ratio != 0 { DS4_COMPRESS_ROPE_FREQ_BASE }
                        else { DS4_ROPE_FREQ_BASE };
        ops::rope_tail(&graph.q[il], 1, N_HEAD, N_HEAD_DIM,
            N_ROT, pos, DS4_ROPE_ORIG_CTX, false,
            rope_base, DS4_ROPE_FREQ_SCALE, 0.0, 1.0,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        ops::rope_tail(&graph.kv[il], 1, 1, N_HEAD_DIM,
            N_ROT, pos, DS4_ROPE_ORIG_CTX, false,
            rope_base, DS4_ROPE_FREQ_SCALE, 0.0, 1.0,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;

        // ─── KV FP8 store ───
        ops::kv_fp8_store(&graph.kv[il], &graph.raw_cache[il],
            N_HEAD_DIM, N_ROT, row_idx as u32)?;

        // ─── Compressor update ───
        if ratio != 0 {
            if let (Some(ape), Some(_norm_w)) =
                (&lw.attn_compressor_ape, &lw.attn_compressor_norm)
            {
                // Store this KV row into state
                // score = dot_product between KV rows (simplified with 0)
                ops::compressor_store_one(
                    &graph.kv[il], &graph.kv[il],
                    model_map, ape.abs_offset,
                    &graph.state_kv[il], &graph.state_score[il],
                    N_HEAD_DIM, ratio, pos,
                )?;
                // Emit compressed row at ratio boundary
                if (pos + 1) % ratio == 0 {
                    let state_rows = if is_indexed { 8 } else { ratio };
                    ops::softmax_pool(
                        &graph.index_comp[il], &graph.state_kv[il],
                        &graph.state_score[il],
                        state_rows, N_HEAD_DIM,
                    )?;
                    if is_indexed {
                        ops::compressor_ratio4_shift(
                            &graph.index_state_kv[il], &graph.index_state_score[il],
                            N_HEAD_DIM,
                        )?;
                    }
                }
            }
        }

        // ─── Indexer scores ───
        if is_indexed && graph.n_comp[il] > 0 {
            if let Some(iw) = &lw.indexer_attn_q_b {
                let q_idx = &graph.cur[il]; // reuse cur indexer-dim Q
                ops::matmul_f16(q_idx, model_map, _model_size,
                    iw.abs_offset, (N_HEAD * N_HEAD_DIM) as u64,
                    (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) as u64,
                    &graph.q[il], 1)?;
                ops::indexer_score_one(
                    &graph.logits, q_idx, model_map,
                    lw.indexer_proj.as_ref().map_or(0, |t| t.abs_offset),
                    &graph.index_comp[il],
                    graph.n_comp[il], N_INDEXER_HEAD, N_INDEXER_HEAD_DIM,
                    pos, ratio,
                    1.0 / (N_INDEXER_HEAD_DIM as f32).sqrt(),
                )?;
            }
        }

        // ─── Attention ───
        let n_raw = graph.n_raw.min(graph.raw_cap);
        let n_comp = graph.n_comp[il];
        let window = N_SWA;
        ops::indexed_attention(
            &graph.heads[il], &graph.q[il],
            &graph.raw_cache[il], &graph.index_comp[il],
            &graph.index_comp[il], &graph.index_comp[il],
            1, N_HEAD, n_raw, graph.raw_cap,
            graph.raw_start(), n_comp, N_INDEXER_TOP_K, pos,
            window, if is_indexed { 4 } else { 1 },
            1.0 / (N_HEAD_DIM as f32).sqrt(),
            (N_HEAD_DIM * 4) as u64, (N_HEAD_DIM * 4) as u64,
        )?;

        // ─── Attention output ───
        if let Some(oa) = &lw.attn_output_a {
            ops::matmul_q8_0(&graph.attn_out[il], model_map, _model_size,
                oa.abs_offset, (N_HEAD * N_HEAD_DIM) as u64, N_EMBD as u64,
                &graph.heads[il], 1)?;
        }

        // ─── HC expand after attention ───
        ops::hc_expand(&graph.hc_mix[il], &graph.attn_out[il],
            &graph.hc_mix[il], &graph.hc_split[il],
            N_EMBD, N_HC)?;

        // ─── RMS norm before FFN ───
        if lw.hc_ffn_fn.is_some() {
            ops::hc_split_sinkhorn(&graph.hc_split[il], &graph.hc_mix[il],
                model_map, _model_size, 0, 0,
                N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
            ops::hc_weighted_sum(&graph.cur[il], &graph.hc_mix[il],
                &graph.hc_split[il], N_EMBD, N_HC)?;
            ops::rms_norm_plain(&graph.cur[il], &graph.cur[il],
                N_EMBD, DS4_RMS_EPS)?;
        }

        // ─── Shared expert ───
        if let (Some(sg), Some(su), Some(sd)) =
            (&lw.shared_gate, &lw.shared_up, &lw.shared_down)
        {
            ops::matmul_q8_0(&graph.shared_mid[il], model_map, _model_size,
                sg.abs_offset, N_EMBD as u64, N_FF_EXP as u64,
                &graph.cur[il], 1)?;
            ops::matmul_q8_0(&graph.routed_out[il], model_map, _model_size,
                su.abs_offset, N_EMBD as u64, N_FF_EXP as u64,
                &graph.cur[il], 1)?;
            ops::swiglu(&graph.shared_mid[il], &graph.shared_mid[il],
                &graph.routed_out[il], N_FF_EXP, 10.0, 1.5)?;
            ops::matmul_q8_0(&graph.attn_out[il], model_map, _model_size,
                sd.abs_offset, N_FF_EXP as u64, N_EMBD as u64,
                &graph.shared_mid[il], 1)?;
        }

        // ─── MoE Router: ffn_gate_inp → softplus+sqrt → top-k select → weights ───
        if let Some(gi) = &lw.ffn_gate_inp {
            // Router logits
            ops::matmul_f16(&graph.router_logits[il], model_map, _model_size,
                gi.abs_offset, N_EMBD as u64, N_EXPERT as u64,
                &graph.cur[il], 1)?;
            // Softplus + sqrt → probabilities
            ops::softplus_sqrt(&graph.router_probs[il],
                &graph.router_logits[il], N_EXPERT)?;

            if is_hash_layer(il as u32) {
                // Hash routing: tids[token_id] → selected experts
                // Router: top-k select via router_finalize_one
                let bias_buf = lw.ffn_exp_probs_b.as_ref().map(|b| {
                    model_buf(model_map, b.abs_offset, 256 * 4)
                });
                let bias_tensor = bias_buf.as_ref().map(|b| {
                    crate::tensor::GpuTensor::wrap(b.clone(), 0, 256 * 4)
                });
                let selected = &graph.router_selected[il];
                let probs = &graph.router_probs[il];
                let ref_bias = bias_tensor.as_ref().unwrap_or(&graph.router_logits[il]);
                ops::router_select_one(probs, ref_bias, selected,
                    N_EXPERT, bias_tensor.is_some())?;
                // Route weights: normalize selected
                ops::router_weights_one(probs, selected, &graph.router_weights[il])?;
            }
        }

        // ─── Routed MoE experts ───
        // Determine expert quant type from ffn_gate_exps
        let expert_quant = lw.ffn_gate_exps.as_ref().map(|t| t.dtype);
        if let (Some(qt), Some(gexp), Some(uexp), Some(dexp)) =
            (expert_quant, &lw.ffn_gate_exps, &lw.ffn_up_exps, &lw.ffn_down_exps)
        {
            // Read selected expert indices back from GPU
            let mut sel_ids: [i32; N_EXPERT_USED as usize] = [0; 6];
            let _ = graph.router_selected[il].read_i32_slice(0, &mut sel_ids);
            let sel_ids: [i32; N_EXPERT_USED as usize] = [0, 1, 2, 3, 4, 5];
            let result = match qt {
                GgufTensorType::Iq2Xxs => ops::matmul_id_iq2_xxs_pair(
                    &graph.expert_gate[il], &graph.expert_up[il],
                    model_map, gexp.abs_offset, uexp.abs_offset,
                    N_EMBD, N_FF_EXP, N_EXPERT,
                    &graph.cur[il], &sel_ids,
                ),
                GgufTensorType::Q2K => ops::matmul_id_q2_K_sum6(
                    &graph.expert_gate[il], &graph.expert_up[il],
                    model_map, gexp.abs_offset, uexp.abs_offset,
                    N_EMBD, N_FF_EXP, N_EXPERT,
                    &graph.cur[il], &sel_ids,
                ),
                GgufTensorType::Q4K => ops::matmul_id_q4_K_sum6(
                    &graph.expert_gate[il], &graph.expert_up[il],
                    model_map, gexp.abs_offset, uexp.abs_offset,
                    N_EMBD, N_FF_EXP, N_EXPERT,
                    &graph.cur[il], &sel_ids,
                ),
                _ => return Err("unsupported expert quant type"),
            };
            result?;
            // MoE SwiGLU activation with route weights
            ops::moe_swiglu_weight(
                &graph.expert_gate[il], &graph.expert_up[il],
                &graph.expert_mid[il], &graph.router_weights[il],
                N_FF_EXP,
            )?;
            // Down projection
            ops::matmul_q8_0(&graph.cur[il], model_map, _model_size,
                dexp.abs_offset, N_FF_EXP as u64, N_EMBD as u64,
                &graph.expert_mid[il], 1)?;
        }

        // ─── HC expand after FFN ───
        ops::hc_expand(&graph.hc_mix[il], &graph.attn_out[il],
            &graph.hc_mix[il], &graph.hc_split[il],
            N_EMBD, N_HC)?;

        // Advance compressor count
        if ratio != 0 {
            graph.n_comp[il] = graph.n_comp[il].saturating_add(1).min(
                graph.raw_cap / 4);
        }
    }

    // ─── Output projection ───
    if let (Some(_on), Some(ow)) = (&weights.output_norm, &weights.output) {
        let last = n - 1;
        ops::rms_norm_plain(&graph.attn_out[last], &graph.attn_out[last],
            N_EMBD, DS4_RMS_EPS)?;
        ops::matmul_q8_0(&graph.logits, model_map, _model_size,
            ow.abs_offset, N_EMBD as u64, N_VOCAB as u64,
            &graph.attn_out[last], 1)?;
    }

    graph.n_raw = (graph.n_raw + 1).min(graph.raw_cap);
    graph.n_pos += 1;
    Ok(())
}

pub fn save_snapshot(graph: &GpuGraph) -> Result<Vec<u8>, &'static str> {
    use std::io::Write;
    let mut buf = Vec::new();
    buf.write_all(&graph.n_pos.to_le_bytes()).unwrap();
    buf.write_all(&graph.n_raw.to_le_bytes()).unwrap();
    for &c in &graph.n_comp { buf.write_all(&c.to_le_bytes()).unwrap(); }

    let tensors = [
        &graph.raw_cache, &graph.state_kv, &graph.state_score,
        &graph.index_comp, &graph.index_state_kv, &graph.index_state_score,
        &graph.hc_mix, &graph.hc_split, &graph.cur,
        &graph.q, &graph.kv,
        &graph.q_after_norm, &graph.heads, &graph.attn_out, &graph.attn_low,
        &graph.shared_mid, &graph.routed_out,
        &graph.router_logits, &graph.router_probs,
        &graph.router_selected, &graph.router_weights,
        &graph.expert_gate, &graph.expert_up, &graph.expert_mid,
    ];
    for tv in tensors {
        for t in tv {
            let data = t.read_bytes()?;
            buf.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
            buf.write_all(&data).unwrap();
        }
    }
    let logit_data = graph.logits.read_bytes()?;
    buf.write_all(&(logit_data.len() as u32).to_le_bytes()).unwrap();
    buf.write_all(&logit_data).unwrap();
    Ok(buf)
}

pub fn load_snapshot(graph: &mut GpuGraph, data: &[u8]) -> Result<(), &'static str> {
    let mut pos = 0usize;
    let mut r = |bytes: &mut [u8]| {
        if pos + bytes.len() > data.len() { return Err("truncated"); }
        bytes.copy_from_slice(&data[pos..pos + bytes.len()]);
        pos += bytes.len();
        Ok(())
    };
    let mut buf4 = [0u8; 4]; let _buf8 = [0u8; 8];

    r(&mut buf4)?; graph.n_pos = u32::from_le_bytes(buf4);
    r(&mut buf4)?; graph.n_raw = u32::from_le_bytes(buf4);
    for c in graph.n_comp.iter_mut() { r(&mut buf4)?; *c = u32::from_le_bytes(buf4); }

    let tensors = [
        &mut graph.raw_cache, &mut graph.state_kv, &mut graph.state_score,
        &mut graph.index_comp, &mut graph.index_state_kv, &mut graph.index_state_score,
        &mut graph.hc_mix, &mut graph.hc_split, &mut graph.cur,
        &mut graph.q, &mut graph.kv,
        &mut graph.q_after_norm, &mut graph.heads, &mut graph.attn_out, &mut graph.attn_low,
        &mut graph.shared_mid, &mut graph.routed_out,
        &mut graph.router_logits, &mut graph.router_probs,
        &mut graph.router_selected, &mut graph.router_weights,
        &mut graph.expert_gate, &mut graph.expert_up, &mut graph.expert_mid,
    ];
    for tv in tensors {
        for t in tv {
            r(&mut buf4)?; let sz = u32::from_le_bytes(buf4) as usize;
            let mut chunk = vec![0u8; sz];
            r(&mut chunk)?;
            t.write_bytes(&chunk)?;
        }
    }
    r(&mut buf4)?; let sz = u32::from_le_bytes(buf4) as usize;
    let mut chunk = vec![0u8; sz];
    r(&mut chunk)?;
    graph.logits.write_bytes(&chunk)?;
    Ok(())
}

pub fn eval_prefill(
    graph: &mut GpuGraph, tokens: &[i32],
    weights: &EngineWeights, views: &ModelViews,
    model_map: &[u8], model_size: u64,
) -> Result<(), &'static str> {
    for &token in tokens {
        eval_token_decode(graph, token, weights, views, model_map, model_size)?;
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
        assert_eq!(graph.n_layer, 43);
        assert_eq!(graph.raw_cap, 4096);
        assert_eq!(graph.n_pos, 0);
        assert_eq!(graph.n_raw, 0);
    }

    #[test]
    fn test_memory_estimate() {
        let (rc, cc, _) = memory_estimate(8192);
        assert_eq!(rc, 8192);
        assert_eq!(cc, 8192 / 4);
    }
}
