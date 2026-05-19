use crate::model::*;
use crate::model_view::ModelViews;
use crate::gguf::*;
use crate::tensor::GpuTensor;
use crate::ops;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

/// Dump a raw byte slice to a file matching C's dump format.
/// Only active when DS4_DUMP_PREFIX env var is set.
/// File path: {prefix}_{name}-{layer}_pos{pos}.{ext}
fn dump_raw(data: &[u8], ext: &str, name: &str, il: usize, pos: u32) {
    let prefix = match std::env::var("DS4_DUMP_PREFIX") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    if let Ok(name_filter) = std::env::var("DS4_DUMP_NAME") {
        if !name_filter.is_empty() && !name_filter.contains(name) { return; }
    }
    if let Ok(layer_filter) = std::env::var("DS4_DUMP_LAYER") {
        if !layer_filter.is_empty() && layer_filter != "all" {
            if layer_filter.parse::<usize>().ok() != Some(il) { return; }
        }
    }
    if let Ok(pos_filter) = std::env::var("DS4_DUMP_POS") {
        if !pos_filter.is_empty() {
            if pos_filter.parse::<u32>().ok() != Some(pos) { return; }
        }
    }
    let path = format!("{}_{}-{}_pos{}.{}", prefix, name, il, pos, ext);
    if let Err(e) = std::fs::write(&path, data) {
        eprintln!("ds4: dump failed for {}: {}", path, e);
    } else {
        eprintln!("ds4: dumped {} layer {} pos {} to {}", name, il, pos, path);
    }
}

/// Dump a GPU tensor to a binary f32 file matching C's dump format.
/// Only active when DS4_DUMP_PREFIX env var is set.
/// File path: {prefix}_{name}-{layer}_pos{pos}.bin
/// Flushes (commits + restarts) the current Metal batch before reading.
fn dump_tensor(t: &GpuTensor, name: &str, n_f32: usize, il: usize, pos: u32) {
    let prefix = match std::env::var("DS4_DUMP_PREFIX") {
        Ok(p) if !p.is_empty() => p,
        _ => return,
    };
    if let Ok(name_filter) = std::env::var("DS4_DUMP_NAME") {
        if !name_filter.is_empty() && !name_filter.contains(name) { return; }
    }
    if let Ok(layer_filter) = std::env::var("DS4_DUMP_LAYER") {
        if !layer_filter.is_empty() && layer_filter != "all" {
            if layer_filter.parse::<usize>().ok() != Some(il) { return; }
        }
    }
    if let Ok(pos_filter) = std::env::var("DS4_DUMP_POS") {
        if !pos_filter.is_empty() {
            if pos_filter.parse::<u32>().ok() != Some(pos) { return; }
        }
    }
    // Flush current batch so GPU writes are committed before we read back.
    let _ = ops::flush_batch();
    let path = format!("{}_{}-{}_pos{}.bin", prefix, name, il, pos);
    if let Ok(data) = t.read_bytes() {
        let byte_count = (n_f32 * 4).min(data.len());
        if let Err(e) = std::fs::write(&path, &data[..byte_count]) {
            eprintln!("ds4: dump failed for {}: {}", path, e);
        } else {
            eprintln!("ds4: dumped {} layer {} pos {} to {}", name, il, pos, path);
        }
    }
}

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
    pub routed_down: GpuTensor,
    pub routed_out: GpuTensor,
    pub output_pre: GpuTensor,
    pub output_weights: GpuTensor,
    pub output_embd: GpuTensor,
    pub output_norm: GpuTensor,
    pub logits: GpuTensor,
    pub comp_kv: GpuTensor,
    pub comp_sc: GpuTensor,
    pub raw_cache: Vec<GpuTensor>,
    pub state_kv: Vec<GpuTensor>,
    pub state_score: Vec<GpuTensor>,
    pub index_comp: Vec<GpuTensor>,
    pub index_state_kv: Vec<GpuTensor>,
    pub index_state_score: Vec<GpuTensor>,
    // Batch prefill tensors
    pub batch_cur_hc: GpuTensor,
    pub batch_next_hc: GpuTensor,
    pub batch_flat_hc: GpuTensor,
    pub batch_hc_mix: GpuTensor,
    pub batch_hc_split: GpuTensor,
    pub batch_attn_cur: GpuTensor,
    pub batch_attn_norm: GpuTensor,
    pub batch_qr: GpuTensor,
    pub batch_qr_norm: GpuTensor,
    pub batch_q: GpuTensor,
    pub batch_kv_raw: GpuTensor,
    pub batch_kv: GpuTensor,
    pub batch_comp_kv: GpuTensor,
    pub batch_comp_sc: GpuTensor,
    pub batch_indexer_q: GpuTensor,
    pub batch_indexer_weights: GpuTensor,
    pub batch_heads: GpuTensor,
    pub batch_attn_low: GpuTensor,
    pub batch_attn_out: GpuTensor,
    pub batch_after_attn_hc: GpuTensor,
    pub batch_ffn_cur: GpuTensor,
    pub batch_ffn_norm: GpuTensor,
    pub batch_shared_gate: GpuTensor,
    pub batch_shared_up: GpuTensor,
    pub batch_shared_mid: GpuTensor,
    pub batch_shared_out: GpuTensor,
    pub batch_router_logits: GpuTensor,
    pub batch_router_probs: GpuTensor,
    pub batch_router_selected: GpuTensor,
    pub batch_router_weights: GpuTensor,
    pub batch_routed_gate: GpuTensor,
    pub batch_routed_up: GpuTensor,
    pub batch_routed_mid: GpuTensor,
    pub batch_routed_down: GpuTensor,
    pub batch_routed_out: GpuTensor,
    pub n_layer: usize,
    pub raw_cap: u32,
    pub prefill_cap: u32,
    pub n_pos: u32,
    pub n_raw: u32,
    pub n_comp: [u32; 43],
}

impl GpuGraph {
    pub fn allocate(ctx_size: u32) -> Result<Self, &'static str> {
        let raw_cap = ((ctx_size.min(8192) + 255) / 256) * 256;
        let pc = 256u32; // prefill batch capacity
        let n = N_LAYER as usize;
        let hc = N_HC as u64;
        let embd = N_EMBD as u64;
        let hc_dim = hc * embd;
        let mix_hc = (2 * N_HC + N_HC * N_HC) as u64;
        let mix_bytes = mix_hc * 4;
        let embd_bytes = embd * 4;
        let hc_bytes = hc_dim * 4;
        let head_q_bytes = N_HEAD as u64 * N_HEAD_DIM as u64 * 4;
        let head_dim_bytes = N_HEAD_DIM as u64 * 4;
        let lora_q_bytes = N_LORA_Q as u64 * 4;
        let lora_o_bytes = N_OUT_GROUP as u64 * N_LORA_O as u64 * 4;
        let ffn_bytes = N_FF_EXP as u64 * 4;
        let expert_bytes = N_FF_EXP as u64 * N_EXPERT_USED as u64 * 4;
        let q_dim = N_HEAD as u64 * N_HEAD_DIM as u64;
        let group_dim = N_HEAD_DIM as u64 * (N_HEAD / N_OUT_GROUP) as u64;
        let shared_dim = N_FF_EXP as u64;
        let comp_width_max = 2 * N_HEAD_DIM as u64;
        let indexer_q_dim = N_INDEXER_HEAD as u64 * N_INDEXER_HEAD_DIM as u64;
        let pc64 = pc as u64;
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
            routed_mid: alloc(expert_bytes), routed_down: alloc(N_EXPERT_USED as u64 * embd * 4),
            routed_out: alloc(embd_bytes),
            output_pre: alloc(hc * 4), output_weights: alloc(hc * 4),
            output_embd: alloc(embd_bytes), output_norm: alloc(embd_bytes),
            logits: alloc(N_VOCAB as u64 * 4),
            comp_kv: alloc(2 * N_HEAD_DIM as u64 * 4),
            comp_sc: alloc(2 * N_HEAD_DIM as u64 * 4),
            raw_cache: (0..n).map(|_| alloc(raw_cap as u64 * N_HEAD_DIM as u64 * 4)).collect(),
            state_kv: (0..n).map(|_| alloc(128 * N_HEAD_DIM as u64 * 4)).collect(),
            state_score: (0..n).map(|_| alloc(128 * N_HEAD_DIM as u64 * 4)).collect(),
            index_comp: (0..n).map(|_| alloc(raw_cap as u64 / 4 * N_HEAD_DIM as u64 * 4)).collect(),
            index_state_kv: (0..n).map(|_| alloc(8 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            index_state_score: (0..n).map(|_| alloc(8 * N_INDEXER_HEAD_DIM as u64 * 4)).collect(),
            // Batch prefill tensors
            batch_cur_hc: alloc(pc64 * hc_dim * 4),
            batch_next_hc: alloc(pc64 * hc_dim * 4),
            batch_flat_hc: alloc(pc64 * hc_dim * 4),
            batch_hc_mix: alloc(pc64 * mix_hc * 4),
            batch_hc_split: alloc(pc64 * mix_hc * 4),
            batch_attn_cur: alloc(pc64 * embd * 4),
            batch_attn_norm: alloc(pc64 * embd * 4),
            batch_qr: alloc(pc64 * N_LORA_Q as u64 * 4),
            batch_qr_norm: alloc(pc64 * N_LORA_Q as u64 * 4),
            batch_q: alloc(pc64 * q_dim * 4),
            batch_kv_raw: alloc(pc64 * N_HEAD_DIM as u64 * 4),
            batch_kv: alloc(pc64 * N_HEAD_DIM as u64 * 4),
            batch_comp_kv: alloc(pc64 * comp_width_max * 4),
            batch_comp_sc: alloc(pc64 * comp_width_max * 4),
            batch_indexer_q: alloc(pc64 * indexer_q_dim * 4),
            batch_indexer_weights: alloc(pc64 * N_INDEXER_HEAD as u64 * 4),
            batch_heads: alloc(pc64 * q_dim * 4),
            batch_attn_low: alloc(pc64 * lora_o_bytes),
            batch_attn_out: alloc(pc64 * embd * 4),
            batch_after_attn_hc: alloc(pc64 * hc_dim * 4),
            batch_ffn_cur: alloc(pc64 * embd * 4),
            batch_ffn_norm: alloc(pc64 * embd * 4),
            batch_shared_gate: alloc(pc64 * shared_dim * 4),
            batch_shared_up: alloc(pc64 * shared_dim * 4),
            batch_shared_mid: alloc(pc64 * shared_dim * 4),
            batch_shared_out: alloc(pc64 * embd * 4),
            batch_router_logits: alloc(pc64 * N_EXPERT as u64 * 4),
            batch_router_probs: alloc(pc64 * N_EXPERT as u64 * 4),
            batch_router_selected: alloc(pc64 * N_EXPERT_USED as u64 * 4),
            batch_router_weights: alloc(pc64 * N_EXPERT_USED as u64 * 4),
            batch_routed_gate: alloc(pc64 * N_EXPERT_USED as u64 * N_FF_EXP as u64 * 4),
            batch_routed_up: alloc(pc64 * N_EXPERT_USED as u64 * N_FF_EXP as u64 * 4),
            batch_routed_mid: alloc(pc64 * N_EXPERT_USED as u64 * N_FF_EXP as u64 * 4),
            batch_routed_down: alloc(pc64 * N_EXPERT_USED as u64 * embd * 4),
            batch_routed_out: alloc(pc64 * embd * 4),
            n_layer: n, raw_cap, prefill_cap: pc, n_pos: 0, n_raw: 0, n_comp: [0u32; 43],
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
/// The scale *factor* (= DS4_ROPE_SCALE_FACTOR in the C code).
/// For non-compressed layers freq_scale = 1.0; for compressed layers
/// freq_scale = 1.0 / DS4_ROPE_SCALE_FACTOR (i.e. 0.0625), matching C.
const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
const DS4_ROPE_ORIG_CTX: u32 = 65536;
const DS4_SWIGLU_CLAMP_EXP: f32 = 10.0;

/// Per-layer RoPE parameters, matching C's `layer_rope_freq_scale` +
/// ext_factor / attn_factor logic in `metal_graph_encode_decode_layer`.
fn rope_params(compressed: bool) -> (f32, f32, f32) {
    if !compressed {
        // Layers 0-1: standard RoPE, no scaling.
        (1.0, 0.0, 1.0)
    } else {
        // Layers 2-42: YaRN-scaled RoPE.
        // freq_scale = 1 / DS4_ROPE_SCALE_FACTOR
        let freq_scale = 1.0 / DS4_ROPE_SCALE_FACTOR;
        let ext_factor = 1.0f32;
        // attn_factor = 1 / (1 + 0.1 * ln(1/freq_scale))
        let attn_factor = 1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln());
        (freq_scale, ext_factor, attn_factor)
    }
}

fn compress_ratio(il: u32) -> u32 { if il < 2 { 0 } else if (il & 1) == 0 { 4 } else { 128 } }
fn is_hash_layer(il: u32) -> bool { il < N_HASH_LAYER }

struct BatchGuard;
impl Drop for BatchGuard {
    fn drop(&mut self) {
        let _ = ops::end_batch();
    }
}

pub fn eval_token_decode(
    graph: &mut GpuGraph, token: i32,
    weights: &EngineWeights, views: &ModelViews,
) -> Result<(), &'static str> {
    let hc_dim = N_HC * N_EMBD;
    let mix_hc = (2 * N_HC + N_HC * N_HC) as u64;

    ops::begin_batch()?;
    let _guard = BatchGuard;
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
                dump_tensor(&graph.hc_mix, "hc_mix_before_sinkhorn", mix_hc as usize, il, graph.n_pos);
                ops::hc_split_sinkhorn(&graph.hc_split, &graph.hc_mix, views,
                    scale.abs_offset, base.abs_offset,
                    N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
                dump_tensor(&graph.hc_split, "hc_split_after_sinkhorn", mix_hc as usize, il, graph.n_pos);
                ops::hc_weighted_sum(&graph.attn_cur, &graph.cur_hc,
                    &graph.hc_split, N_EMBD, N_HC)?;
            }
        }
        dump_tensor(&graph.attn_cur, "hc_attn_pre", N_EMBD as usize, il, graph.n_pos);
        if let Some(ref norm_w) = lw.attn_norm {
            ops::rms_norm_weight(&graph.attn_norm, &graph.attn_cur,
                views, norm_w.abs_offset, N_EMBD, DS4_RMS_EPS)?;
        }
        dump_tensor(&graph.attn_norm, "attn_norm", N_EMBD as usize, il, graph.n_pos);


        // ─── Q projection ───
        if let (Some(ref qa), Some(ref qb)) = (&lw.attn_q_a, &lw.attn_q_b) {
            ops::matmul_q8_0(&graph.qr, views, qa.abs_offset,
                N_EMBD as u64, N_LORA_Q as u64, &graph.attn_norm, 1)?;
            dump_tensor(&graph.qr, "q_lora", N_LORA_Q as usize, il, graph.n_pos);
            if let Some(ref qa_n) = lw.attn_q_a_norm {
                ops::rms_norm_weight(&graph.qr_norm, &graph.qr,
                    views, qa_n.abs_offset, N_LORA_Q, DS4_RMS_EPS)?;
            }
            dump_tensor(&graph.qr_norm, "q_lora_norm", N_LORA_Q as usize, il, graph.n_pos);
            ops::matmul_q8_0(&graph.q, views, qb.abs_offset,
                N_LORA_Q as u64, (N_HEAD * N_HEAD_DIM) as u64, &graph.qr_norm, 1)?;
        }
        dump_tensor(&graph.q, "Qraw", (N_HEAD * N_HEAD_DIM) as usize, il, graph.n_pos);
        ops::head_rms_norm(&graph.q, 1, N_HEAD, N_HEAD_DIM, DS4_RMS_EPS)?;
        dump_tensor(&graph.q, "Qnorm", (N_HEAD * N_HEAD_DIM) as usize, il, graph.n_pos);

        // ─── KV projection ───
        if let Some(ref kv_a) = lw.attn_kv_a {
            ops::matmul_q8_0(&graph.kv_raw, views, kv_a.abs_offset,
                N_EMBD as u64, N_HEAD_DIM as u64, &graph.attn_norm, 1)?;
            dump_tensor(&graph.kv_raw, "KVraw", N_HEAD_DIM as usize, il, graph.n_pos);
            if let Some(ref kv_n) = lw.attn_kv_a_norm {
                ops::rms_norm_weight(&graph.kv, &graph.kv_raw,
                    views, kv_n.abs_offset, N_HEAD_DIM, DS4_RMS_EPS)?;
            }
            dump_tensor(&graph.kv, "KVnorm", N_HEAD_DIM as usize, il, graph.n_pos);
        }

        // ─── RoPE + KV store ───
        let compressed = ratio != 0;
        let fb = if compressed { DS4_COMPRESS_ROPE_FREQ_BASE } else { DS4_ROPE_FREQ_BASE };
        let nco = if compressed { DS4_ROPE_ORIG_CTX } else { 0 };
        let (freq_scale, ext_factor, attn_factor) = rope_params(compressed);
        ops::rope_tail(&graph.q, 1, N_HEAD, N_HEAD_DIM, N_ROT, graph.n_pos, nco, false,
            fb, freq_scale, ext_factor, attn_factor,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        dump_tensor(&graph.q, "Qcur", (N_HEAD * N_HEAD_DIM) as usize, il, graph.n_pos);
        ops::rope_tail(&graph.kv, 1, 1, N_HEAD_DIM, N_ROT, graph.n_pos, nco, false,
            fb, freq_scale, ext_factor, attn_factor,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        dump_tensor(&graph.kv, "KVrope", N_HEAD_DIM as usize, il, graph.n_pos);
        ops::kv_fp8_store(&graph.kv, &graph.raw_cache[il], N_HEAD_DIM, N_ROT, row_idx as u32)?;
        dump_tensor(&graph.kv, "KVcur", N_HEAD_DIM as usize, il, graph.n_pos);

        // Accumulate batch Q/KV for non-compressed layers (used by batch FA in eval_prefill)
        if ratio == 0 {
            ops::flush_batch()?;
            let q_data = graph.q.read_bytes()?;
            let q_row = (N_HEAD as u64 * N_HEAD_DIM as u64) * 4;
            {
                let ptr = graph.batch_q.retain_buf().unwrap().contents().as_ptr() as usize
                    + graph.batch_q.offset_raw() as usize
                    + (graph.n_pos as u64 * q_row) as usize;
                unsafe { std::ptr::copy_nonoverlapping(q_data.as_ptr(), ptr as *mut u8, q_data.len()); }
            }
            let kv_data = graph.kv.read_bytes()?;
            let kv_row = N_HEAD_DIM as u64 * 4;
            {
                let ptr = graph.batch_kv.retain_buf().unwrap().contents().as_ptr() as usize
                    + graph.batch_kv.offset_raw() as usize
                    + (graph.n_pos as u64 * kv_row) as usize;
                unsafe { std::ptr::copy_nonoverlapping(kv_data.as_ptr(), ptr as *mut u8, kv_data.len()); }
            }
        }

        // ─── Attention ───
        // n_raw + 1: current token's KV was just stored at row_idx, matching C's
        // kv_cache_push_raw() which increments n_raw before the attention call.
        let n_raw_lim = (graph.n_raw + 1).min(graph.raw_cap);
        if n_raw_lim > 0 {
            let sink_view = lw.attn_sinks.as_ref().and_then(|s|
                views.find_view_retained(s.abs_offset, N_HEAD as u64 * 4));
            let sinks: Option<(&ProtocolObject<dyn MTLBuffer>, u64)> =
                sink_view.as_ref().map(|(buf, off)| (buf.as_ref(), *off));
            ops::indexed_attention(
                &graph.heads, &graph.q,
                &graph.raw_cache[il], &graph.index_comp[il],
                None,
                sinks,
                1, N_HEAD, n_raw_lim, graph.raw_cap,
                graph.raw_start(), graph.n_comp[il], N_INDEXER_TOP_K, graph.n_pos,
                N_SWA, if ratio == 4 { 4 } else { 1 },
                1.0 / (N_HEAD_DIM as f32).sqrt(),
                (N_HEAD_DIM * 4) as u64, (N_HEAD_DIM * 4) as u64,
            )?;
        } else {
            graph.heads.fill_f32(0.0, (N_HEAD * N_HEAD_DIM) as u64)?;
        }

        dump_tensor(&graph.heads, "kqv_out", (N_HEAD * N_HEAD_DIM) as usize, il, graph.n_pos);

        // ─── Inverse RoPE on heads ───
        ops::rope_tail(&graph.heads, 1, N_HEAD, N_HEAD_DIM,
            N_ROT, graph.n_pos, nco, true,
            fb, freq_scale, ext_factor, attn_factor,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
        dump_tensor(&graph.heads, "kqv_back", (N_HEAD * N_HEAD_DIM) as usize, il, graph.n_pos);
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
        dump_tensor(&graph.attn_out, "attn_out", N_EMBD as usize, il, graph.n_pos);
        let po = N_HC as u64 * 4;
        let co = (2 * N_HC) as u64 * 4;
        let hp = GpuTensor::wrap(graph.hc_split.retain_buf().unwrap(),
            graph.hc_split.offset_raw() + po, N_HC as u64 * 4);
        let hc = GpuTensor::wrap(graph.hc_split.retain_buf().unwrap(),
            graph.hc_split.offset_raw() + co, N_HC as u64 * N_HC as u64 * 4);
        ops::hc_expand_tensor(&graph.after_attn_hc, &graph.attn_out,
            &graph.cur_hc, &hp, &hc, N_EMBD, N_HC)?;
        dump_tensor(&graph.after_attn_hc, "hc_attn_post", (N_HC * N_EMBD) as usize, il, graph.n_pos);

        // ─── FFN HC pre ───
        ops::rms_norm_plain(&graph.flat_hc, &graph.after_attn_hc, hc_dim, DS4_RMS_EPS)?;
        if let Some(ref hc_fn) = lw.hc_ffn_fn {
            ops::matmul_f16(&graph.hc_mix, views, hc_fn.abs_offset,
                hc_dim as u64, mix_hc, &graph.flat_hc, 1)?;
            if let (Some(ref scale), Some(ref base)) = (&lw.hc_ffn_scale, &lw.hc_ffn_base) {
                dump_tensor(&graph.hc_mix, "hc_ffn_mix_before_sinkhorn", mix_hc as usize, il, graph.n_pos);
                ops::hc_split_sinkhorn(&graph.hc_split, &graph.hc_mix, views,
                    scale.abs_offset, base.abs_offset,
                    N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
                dump_tensor(&graph.hc_split, "hc_ffn_split_after_sinkhorn", mix_hc as usize, il, graph.n_pos);
                ops::hc_weighted_sum(&graph.ffn_cur, &graph.after_attn_hc,
                    &graph.hc_split, N_EMBD, N_HC)?;
            }
        }
        dump_tensor(&graph.ffn_cur, "hc_ffn_pre", N_EMBD as usize, il, graph.n_pos);
        if let Some(ref fnw) = lw.ffn_norm {
            ops::rms_norm_weight(&graph.ffn_norm, &graph.ffn_cur,
                views, fnw.abs_offset, N_EMBD, DS4_RMS_EPS)?;
        }
        dump_tensor(&graph.ffn_norm, "ffn_norm", N_EMBD as usize, il, graph.n_pos);

        // ─── Router ───
        if let Some(ref gi) = lw.ffn_gate_inp {
            ops::matmul_f16(&graph.router_logits, views, gi.abs_offset,
                N_EMBD as u64, N_EXPERT as u64, &graph.ffn_norm, 1)?;
            dump_tensor(&graph.router_logits, "ffn_moe_logits", N_EXPERT as usize, il, graph.n_pos);
            ops::softplus_sqrt(&graph.router_probs, &graph.router_logits, N_EXPERT)?;
            dump_tensor(&graph.router_probs, "ffn_moe_probs", N_EXPERT as usize, il, graph.n_pos);
            if is_hash_layer(il as u32) {
                // Hash layers 0-2: use token-based expert lookup table.
                // has_bias is forced off when hash_mode=1 (kernel ignores bias in hash mode).
                let hash_offset = lw.ffn_gate_tid2eid.as_ref().map(|t| t.abs_offset);
                ops::router_select_one(&graph.router_probs,
                    &graph.router_probs,   // bias unused in hash mode
                    &graph.router_selected, N_EXPERT, false,
                    views, hash_offset, token as u32, N_VOCAB)?;
            } else {
                // Non-hash layers: softplus probability selection with optional bias.
                let bt = lw.ffn_exp_probs_b.as_ref().and_then(|b| {
                    let (buf, off) = views.find_view_retained(b.abs_offset, 256*4)?;
                    Some(GpuTensor::wrap(buf, off, 256*4))
                });
                ops::router_select_one(&graph.router_probs,
                    bt.as_ref().unwrap_or(&graph.router_probs),
                    &graph.router_selected, N_EXPERT, bt.is_some(),
                    views, None, 0, 0)?;
            }
            ops::router_weights_one(&graph.router_probs,
                &graph.router_selected, &graph.router_weights)?;

            // Read selected expert IDs from GPU
            ops::flush_batch()?;
            let mut sel_ids = vec![0i32; N_EXPERT_USED as usize];
            graph.router_selected.read_i32_slice(0, &mut sel_ids)?;
            // Dump raw i32 expert IDs (matches C's ffn_moe_topk format)
            {
                let raw: &[u8] = unsafe { std::slice::from_raw_parts(
                    sel_ids.as_ptr() as *const u8, sel_ids.len() * 4) };
                dump_raw(raw, "i32", "ffn_moe_topk", il, graph.n_pos);
            }
            dump_tensor(&graph.router_weights, "ffn_moe_weights_scaled", N_EXPERT_USED as usize, il, graph.n_pos);
            for id in sel_ids.iter_mut() {
                *id = (*id).max(0).min((N_EXPERT - 1) as i32);
            }

            let eq = lw.ffn_gate_exps.as_ref().map(|t| t.dtype);
            if let (Some(qt), Some(gx), Some(ux), Some(dx)) =
                (eq, &lw.ffn_gate_exps, &lw.ffn_up_exps, &lw.ffn_down_exps)
            {
                // Gate+up pair matmul for all selected experts
                let r = match qt {
                    GgufTensorType::Iq2Xxs => ops::matmul_id_iq2_xxs_pair(
                        &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                        N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids),
                    GgufTensorType::Q2K => ops::matmul_id_q2_K_sum6(
                        &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                        N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids),
                    GgufTensorType::Q4K => ops::matmul_id_q4_K_sum6(
                        &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                        N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids),
                    _ => return Err("unsupported expert quant"),
                };
                r?;
                dump_tensor(&graph.routed_gate, "ffn_moe_gate", (N_FF_EXP * N_EXPERT_USED) as usize, il, graph.n_pos);
                dump_tensor(&graph.routed_up,   "ffn_moe_up",   (N_FF_EXP * N_EXPERT_USED) as usize, il, graph.n_pos);

                // SwiGLU with router weights
                ops::moe_swiglu_weight(&graph.routed_gate, &graph.routed_up,
                    &graph.routed_mid, &graph.router_weights, N_FF_EXP, N_EXPERT_USED,
                    DS4_SWIGLU_CLAMP_EXP)?;
                dump_tensor(&graph.routed_mid, "ffn_moe_swiglu", (N_FF_EXP * N_EXPERT_USED) as usize, il, graph.n_pos);

                // Down projection: batch all 6 experts to different rows, then read once
                let down_type = dx.dtype;
                let mut acc = vec![0.0f32; N_EMBD as usize];
                // Dispatch all 6 down projections into the batch (no per-expert flush)
                for (i, &expert_id) in sel_ids.iter().enumerate() {
                    let mid_row = GpuTensor::wrap(
                        graph.routed_mid.retain_buf().unwrap(),
                        graph.routed_mid.offset_raw() + (i as u64 * N_FF_EXP as u64 * 4),
                        N_FF_EXP as u64 * 4);
                    let down_row = GpuTensor::wrap(
                        graph.routed_down.retain_buf().unwrap(),
                        graph.routed_down.offset_raw() + (i as u64 * N_EMBD as u64 * 4),
                        N_EMBD as u64 * 4);
                    let r = match down_type {
                        GgufTensorType::Iq2Xxs => ops::matmul_id_iq2_xxs_f32(
                            &down_row, views, dx.abs_offset,
                            N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                        GgufTensorType::Q2K => ops::matmul_id_q2_K_f32(
                            &down_row, views, dx.abs_offset,
                            N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                        GgufTensorType::Q4K => ops::matmul_id_q4_K_f32(
                            &down_row, views, dx.abs_offset,
                            N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                        GgufTensorType::Q8_0 => ops::matmul_id_q8_0_f32(
                            &down_row, views, dx.abs_offset,
                            N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                        _ => return Err("unsupported down expert quant"),
                    };
                    r?;
                }
                // Flush batch so GPU finishes all down projections before CPU reads.
                ops::flush_batch()?;
                dump_tensor(&graph.routed_down, "ffn_moe_down", (N_EMBD * N_EXPERT_USED) as usize, il, graph.n_pos);
                // Read back 6 rows from routed_down and accumulate
                {
                    let data = graph.routed_down.read_bytes()?;
                    let floats: &[f32] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()/4)
                    };
                    for i in 0..N_EXPERT_USED as usize {
                        let row_start = i * N_EMBD as usize;
                        for j in 0..N_EMBD as usize {
                            acc[j] += floats[row_start + j];
                        }
                    }
                }

                // Write accumulated routed expert output
                let acc_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        acc.as_ptr() as *const u8, acc.len() * 4)
                };
                graph.routed_out.write_bytes(acc_bytes)?;
            } else {
                graph.routed_out.fill_f32(0.0, N_EMBD as u64)?;
            }
        } else {
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
        dump_tensor(&graph.routed_out, "ffn_moe_out", N_EMBD as usize, il, graph.n_pos);
        dump_tensor(&graph.shared_out, "ffn_shexp", N_EMBD as usize, il, graph.n_pos);

        // ─── HC expand after FFN + swap ───
        ops::hc_expand_add_split_tensor(&graph.after_ffn_hc,
            &graph.routed_out, &graph.shared_out,
            &graph.after_attn_hc, &graph.hc_split, N_EMBD, N_HC)?;
        dump_tensor(&graph.after_ffn_hc, "hc_ffn_post", (N_HC * N_EMBD) as usize, il, graph.n_pos);
        std::mem::swap(&mut graph.cur_hc, &mut graph.after_ffn_hc);

        if ratio != 0 {
            let coff: u32 = if ratio == 4 { 2 } else { 1 };
            let comp_width = coff * N_HEAD_DIM;
            let emit = ((graph.n_pos + 1) % ratio) == 0;

            if let (Some(ref ck), Some(ref cg), Some(ref ca), Some(ref cn)) = (
                lw.attn_compressor_kv.as_ref(),
                lw.attn_compressor_gate.as_ref(),
                lw.attn_compressor_ape.as_ref(),
                lw.attn_compressor_norm.as_ref(),
            ) {
                let (ck_off, cg_off, ca_off, cn_off) =
                    (ck.abs_offset, cg.abs_offset, ca.abs_offset, cn.abs_offset);

                // Project attn_norm → comp_kv and comp_sc (gate/score)
                ops::matmul_f16(&graph.comp_kv, views, ck_off,
                    N_EMBD as u64, comp_width as u64, &graph.attn_norm, 1)?;
                ops::matmul_f16(&graph.comp_sc, views, cg_off,
                    N_EMBD as u64, comp_width as u64, &graph.attn_norm, 1)?;

                // Store kv+score into state ring with APE
                ops::compressor_store_one(
                    &graph.comp_kv, &graph.comp_sc, views, ca_off,
                    &graph.state_kv[il], &graph.state_score[il],
                    comp_width, ratio, graph.n_pos,
                )?;

                if emit {
                    let comp_row = graph.n_comp[il];

                    // Flush GPU so state_kv/state_score are ready to read on CPU
                    ops::flush_batch()?;

                    // Read state buffers from CPU-visible shared memory
                    let state_kv_bytes = graph.state_kv[il].read_bytes()?;
                    let state_sc_bytes = graph.state_score[il].read_bytes()?;
                    let state_kv: &[f32] = unsafe { std::slice::from_raw_parts(
                        state_kv_bytes.as_ptr() as *const f32, state_kv_bytes.len() / 4) };
                    let state_sc: &[f32] = unsafe { std::slice::from_raw_parts(
                        state_sc_bytes.as_ptr() as *const f32, state_sc_bytes.len() / 4) };

                    // CPU softmax pool: matches C's ds4_gpu_encode_compressor_pool
                    // For ratio=4 (coff=2, width=2*head_dim):
                    //   rows 0..4 use col j (first half), rows 4..8 use col head_dim+j (second half)
                    // For ratio=128 (coff=1, width=head_dim):
                    //   rows 0..128 use col j directly
                    let n_rows = (coff * ratio) as usize;
                    let head_dim = N_HEAD_DIM as usize;
                    let width = comp_width as usize;

                    let mut pool_out = vec![0.0f32; head_dim];
                    for j in 0..head_dim {
                        let mut kvs = vec![0.0f32; n_rows];
                        let mut scs = vec![0.0f32; n_rows];
                        for i in 0..n_rows {
                            let col = if ratio == 4 {
                                if i < 4 { j } else { head_dim + j }
                            } else {
                                j
                            };
                            kvs[i] = state_kv[i * width + col];
                            scs[i] = state_sc[i * width + col];
                        }
                        // Softmax over scores
                        let max_sc = scs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exps: Vec<f32> = scs.iter().map(|&s| (s - max_sc).exp()).collect();
                        let sum: f32 = exps.iter().sum();
                        // Weighted sum of kv
                        let val: f32 = exps.iter().zip(kvs.iter()).map(|(&w, &k)| w / sum * k).sum();
                        pool_out[j] = val;
                    }

                    // Write pool result into comp_view (a view into index_comp[il] at this comp_row)
                    let comp_offset = comp_row as u64 * N_HEAD_DIM as u64 * 4;
                    let comp_view = GpuTensor::wrap(
                        graph.index_comp[il].retain_buf().unwrap(),
                        graph.index_comp[il].offset_raw() + comp_offset,
                        N_HEAD_DIM as u64 * 4,
                    );
                    let pool_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                        pool_out.as_ptr() as *const u8, pool_out.len() * 4) };
                    // write_bytes expects exact size match
                    {
                        let ptr = (graph.index_comp[il].retain_buf().unwrap().contents().as_ptr()
                            as usize + graph.index_comp[il].offset_raw() as usize
                            + comp_offset as usize) as *mut u8;
                        unsafe { std::ptr::copy_nonoverlapping(pool_bytes.as_ptr(), ptr, pool_bytes.len()); }
                    }

                    // GPU: rms_norm_weight (in-place on comp_view)
                    ops::rms_norm_weight(&comp_view, &comp_view, views, cn_off, N_HEAD_DIM, DS4_RMS_EPS)?;

                    // GPU: rope_tail with comp_pos = n_pos + 1 - ratio
                    let comp_pos = graph.n_pos + 1 - ratio;
                    ops::rope_tail(&comp_view, 1, 1, N_HEAD_DIM, N_ROT, comp_pos,
                        DS4_ROPE_ORIG_CTX, false,
                        DS4_COMPRESS_ROPE_FREQ_BASE, 1.0 / DS4_ROPE_SCALE_FACTOR,
                        ext_factor, attn_factor,
                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;

                    // GPU: ratio4 state shift (operates on state buffers, not comp_view)
                    if ratio == 4 {
                        ops::compressor_ratio4_shift(&graph.state_kv[il], &graph.state_score[il], comp_width)?;
                    }

                    // GPU: FP8 in-place quantize on comp_view
                    ops::fp8_kv_quantize(&comp_view, N_HEAD_DIM, N_ROT)?;

                    graph.n_comp[il] += 1;
                }
            }
        }
    }

    // ─── Output head ───
    dump_tensor(&graph.cur_hc, "cur_hc_head", hc_dim as usize, N_LAYER as usize, graph.n_pos);
    ops::rms_norm_plain(&graph.flat_hc, &graph.cur_hc, hc_dim, DS4_RMS_EPS)?;
    dump_tensor(&graph.flat_hc, "flat_hc", hc_dim as usize, N_LAYER as usize, graph.n_pos);
    if let Some(ref ohc_fn) = weights.output_hc_fn {
        ops::matmul_f16(&graph.output_pre, views,
            ohc_fn.abs_offset, hc_dim as u64, N_HC as u64, &graph.flat_hc, 1)?;
    }
    if let (Some(ref scale), Some(ref base)) = (&weights.output_hc_scale, &weights.output_hc_base) {
        ops::flush_batch()?; // commit pending work so output_pre is valid
        ops::output_hc_weights(&graph.output_weights, &graph.output_pre,
            views, scale.abs_offset, base.abs_offset, N_HC, DS4_HC_EPS)?;
    }
    // Dump output HC weights matching C: il=N_LAYER, pos=n_pos (C uses pos=0 always).
    dump_tensor(&graph.output_pre,     "result_hc_pre",     N_HC as usize,   N_LAYER as usize, graph.n_pos);
    dump_tensor(&graph.output_weights, "result_hc_weights", N_HC as usize,   N_LAYER as usize, graph.n_pos);
    ops::hc_weighted_sum(&graph.output_embd, &graph.cur_hc,
        &graph.output_weights, N_EMBD, N_HC)?;
    dump_tensor(&graph.output_embd, "result_hc", N_EMBD as usize, N_LAYER as usize, graph.n_pos);
    if let Some(ref on) = weights.output_norm {
        ops::rms_norm_weight(&graph.output_norm, &graph.output_embd,
            views, on.abs_offset, N_EMBD, DS4_RMS_EPS)?;
    }
    dump_tensor(&graph.output_norm, "result_norm", N_EMBD as usize, N_LAYER as usize, graph.n_pos);
    if let Some(ref ow) = weights.output {
        ops::matmul_q8_0(&graph.logits, views,
            ow.abs_offset, N_EMBD as u64, N_VOCAB as u64, &graph.output_norm, 1)?;
    }
    dump_tensor(&graph.logits, "result_output", N_VOCAB as usize, N_LAYER as usize, graph.n_pos);
    graph.n_raw = (graph.n_raw + 1).min(graph.raw_cap);
    graph.n_pos += 1;

    // Debug: DS4_DUMP_LOGITS=1 prints top-8 logit indices + values after each token.
    if std::env::var("DS4_DUMP_LOGITS").as_deref() == Ok("1") {
        ops::flush_batch()?;
        let data = graph.logits.read_bytes()?;
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
        };
        // Find top-8 by value
        let mut indexed: Vec<(usize, f32)> = floats.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprint!("ds4: logits[pos={}] top8:", graph.n_pos - 1);
        for (id, val) in indexed.iter().take(8) {
            eprint!(" {}({:.3})", id, val);
        }
        eprintln!();
    }
    Ok(())
}

pub fn eval_prefill(graph: &mut GpuGraph, tokens: &[i32],
    weights: &EngineWeights, views: &ModelViews,
) -> Result<(), &'static str> {
    let hc_dim = N_HC * N_EMBD;
    let mix_hc = (2 * N_HC + N_HC * N_HC) as u64;
    let n_tokens = tokens.len() as u32;
    if n_tokens == 0 { return Ok(()); }

    ops::begin_batch()?;
    let _guard = BatchGuard;

    // ─── Embed all tokens ───
    if let Some(ref emb) = weights.token_embd {
        let batch_flat = GpuTensor::wrap(
            graph.batch_flat_hc.retain_buf().unwrap(),
            graph.batch_flat_hc.offset_raw(),
            n_tokens as u64 * N_EMBD as u64 * 4,
        );
        ops::embed_tokens(&batch_flat, views, emb.abs_offset, N_VOCAB, N_EMBD, tokens)?;
        for ti in 0..n_tokens as usize {
            let flat = GpuTensor::wrap(
                graph.batch_flat_hc.retain_buf().unwrap(),
                graph.batch_flat_hc.offset_raw() + ti as u64 * N_EMBD as u64 * 4,
                N_EMBD as u64 * 4,
            );
            let cur = GpuTensor::wrap(
                graph.batch_cur_hc.retain_buf().unwrap(),
                graph.batch_cur_hc.offset_raw() + ti as u64 * hc_dim as u64 * 4,
                hc_dim as u64 * 4,
            );
            ops::repeat_f32(&cur, &flat, (N_EMBD * N_HC) as i32, 1, N_EMBD as i32, 1)?;
        }
    }

    // Dump initial embeddings
    for ti in 0..n_tokens as usize {
        let cur = GpuTensor::wrap(
            graph.batch_cur_hc.retain_buf().unwrap(),
            graph.batch_cur_hc.offset_raw() + ti as u64 * hc_dim as u64 * 4,
            hc_dim as u64 * 4,
        );
        let cur_view = GpuTensor::wrap(cur.retain_buf().unwrap(), cur.offset_raw(), N_EMBD as u64 * 4);
        dump_tensor(&cur_view, "hc_attn_pre", N_EMBD as usize, 0, ti as u32);
    }

    // ─── Layer loop (layer-major) ───
    let zero_prefix = graph.n_pos == 0;
    for il in 0..graph.n_layer {
        let lw = &weights.layer[il];
        let ratio = compress_ratio(il as u32);
        let compressed = ratio != 0;
        let fb = if compressed { DS4_COMPRESS_ROPE_FREQ_BASE } else { DS4_ROPE_FREQ_BASE };
        let nco = if compressed { DS4_ROPE_ORIG_CTX } else { 0 };
        let (freq_scale, ext_factor, attn_factor) = rope_params(compressed);

        // Reset n_comp for compressed layers when starting a fresh context
        if compressed && zero_prefix {
            graph.n_comp[il] = 0;
        }

        // ─── Pass 1: HC pre + Q/KV for all tokens ───
        // For each token, process HC pre-attention, attn_norm, Q/KV path.
        // For non-compressed: store Q/KV in batch buffers for batch FA.
        // For compressed with zero_prefix: also build comp cache incrementally.
        for ti in 0..n_tokens as usize {
            let token = tokens[ti];
            let pos = ti as u32; // positions are 0..n_tokens-1
            let row_idx = (graph.n_raw + ti as u32) % graph.raw_cap;

            // Copy batch_cur_hc[ti] → cur_hc
            {
                let src = GpuTensor::wrap(
                    graph.batch_cur_hc.retain_buf().unwrap(),
                    graph.batch_cur_hc.offset_raw() + ti as u64 * hc_dim as u64 * 4,
                    hc_dim as u64 * 4,
                );
                ops::copy_f32(&graph.cur_hc, &src, hc_dim as usize)?;
            }
            // Now cur_hc = batch_cur_hc[ti]

            // ─── HC pre-attention ───
            ops::rms_norm_plain(&graph.flat_hc, &graph.cur_hc, hc_dim, DS4_RMS_EPS)?;
            if let Some(ref hc_fn) = lw.hc_attn_fn {
                ops::matmul_f16(&graph.hc_mix, views, hc_fn.abs_offset,
                    hc_dim as u64, mix_hc, &graph.flat_hc, 1)?;
                if let (Some(ref scale), Some(ref base)) = (&lw.hc_attn_scale, &lw.hc_attn_base) {
                    dump_tensor(&graph.hc_mix, "hc_mix_before_sinkhorn", mix_hc as usize, il, pos);
                    ops::hc_split_sinkhorn(&graph.hc_split, &graph.hc_mix, views,
                        scale.abs_offset, base.abs_offset,
                        N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
                    // Save per-token attention hc_split so Pass 2 can restore it
                    {
                        let dst = GpuTensor::wrap(
                            graph.batch_hc_split.retain_buf().unwrap(),
                            graph.batch_hc_split.offset_raw() + ti as u64 * mix_hc * 4,
                            mix_hc * 4,
                        );
                        ops::copy_f32(&dst, &graph.hc_split, mix_hc as usize)?;
                    }
                    dump_tensor(&graph.hc_split, "hc_split_after_sinkhorn", mix_hc as usize, il, pos);
                    ops::hc_weighted_sum(&graph.attn_cur, &graph.cur_hc,
                        &graph.hc_split, N_EMBD, N_HC)?;
                }
            }
            dump_tensor(&graph.attn_cur, "hc_attn_pre", N_EMBD as usize, il, pos);
            if let Some(ref norm_w) = lw.attn_norm {
                ops::rms_norm_weight(&graph.attn_norm, &graph.attn_cur,
                    views, norm_w.abs_offset, N_EMBD, DS4_RMS_EPS)?;
            }
            dump_tensor(&graph.attn_norm, "attn_norm", N_EMBD as usize, il, pos);

            // ─── Q projection ───
            if let (Some(ref qa), Some(ref qb)) = (&lw.attn_q_a, &lw.attn_q_b) {
                ops::matmul_q8_0(&graph.qr, views, qa.abs_offset,
                    N_EMBD as u64, N_LORA_Q as u64, &graph.attn_norm, 1)?;
                dump_tensor(&graph.qr, "q_lora", N_LORA_Q as usize, il, pos);
                if let Some(ref qa_n) = lw.attn_q_a_norm {
                    ops::rms_norm_weight(&graph.qr_norm, &graph.qr,
                        views, qa_n.abs_offset, N_LORA_Q, DS4_RMS_EPS)?;
                }
                dump_tensor(&graph.qr_norm, "q_lora_norm", N_LORA_Q as usize, il, pos);
                ops::matmul_q8_0(&graph.q, views, qb.abs_offset,
                    N_LORA_Q as u64, (N_HEAD * N_HEAD_DIM) as u64, &graph.qr_norm, 1)?;
            }
            dump_tensor(&graph.q, "Qraw", (N_HEAD * N_HEAD_DIM) as usize, il, pos);
            ops::head_rms_norm(&graph.q, 1, N_HEAD, N_HEAD_DIM, DS4_RMS_EPS)?;
            dump_tensor(&graph.q, "Qnorm", (N_HEAD * N_HEAD_DIM) as usize, il, pos);

            // ─── KV projection ───
            if let Some(ref kv_a) = lw.attn_kv_a {
                ops::matmul_q8_0(&graph.kv_raw, views, kv_a.abs_offset,
                    N_EMBD as u64, N_HEAD_DIM as u64, &graph.attn_norm, 1)?;
                dump_tensor(&graph.kv_raw, "KVraw", N_HEAD_DIM as usize, il, pos);
                if let Some(ref kv_n) = lw.attn_kv_a_norm {
                    ops::rms_norm_weight(&graph.kv, &graph.kv_raw,
                        views, kv_n.abs_offset, N_HEAD_DIM, DS4_RMS_EPS)?;
                }
                dump_tensor(&graph.kv, "KVnorm", N_HEAD_DIM as usize, il, pos);
            }

            // ─── RoPE + KV store ───
            ops::rope_tail(&graph.q, 1, N_HEAD, N_HEAD_DIM, N_ROT, pos, nco, false,
                fb, freq_scale, ext_factor, attn_factor,
                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
            dump_tensor(&graph.q, "Qcur", (N_HEAD * N_HEAD_DIM) as usize, il, pos);
            ops::rope_tail(&graph.kv, 1, 1, N_HEAD_DIM, N_ROT, pos, nco, false,
                fb, freq_scale, ext_factor, attn_factor,
                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
            dump_tensor(&graph.kv, "KVrope", N_HEAD_DIM as usize, il, pos);

            // Store KV to raw_cache
            ops::kv_fp8_store(&graph.kv, &graph.raw_cache[il], N_HEAD_DIM, N_ROT, row_idx)?;
            dump_tensor(&graph.kv, "KVcur", N_HEAD_DIM as usize, il, pos);

            // ─── Compressor (compressed layers, zero-prefix only) ───
            // Per-token: project attn_norm → comp_kv/sc, update state ring, emit when full.
            if compressed && zero_prefix {
                if let (Some(ref ck), Some(ref cg), Some(ref ca), Some(ref cn)) = (
                    lw.attn_compressor_kv.as_ref(),
                    lw.attn_compressor_gate.as_ref(),
                    lw.attn_compressor_ape.as_ref(),
                    lw.attn_compressor_norm.as_ref(),
                ) {
                    let coff: u32 = if ratio == 4 { 2 } else { 1 };
                    let comp_width = coff * N_HEAD_DIM;
                    let (ck_off, cg_off, ca_off, cn_off) =
                        (ck.abs_offset, cg.abs_offset, ca.abs_offset, cn.abs_offset);

                    // Project attn_norm → comp_kv and comp_sc
                    ops::matmul_f16(&graph.comp_kv, views, ck_off,
                        N_EMBD as u64, comp_width as u64, &graph.attn_norm, 1)?;
                    ops::matmul_f16(&graph.comp_sc, views, cg_off,
                        N_EMBD as u64, comp_width as u64, &graph.attn_norm, 1)?;

                    // Update recurrent state ring with APE
                    ops::compressor_store_one(
                        &graph.comp_kv, &graph.comp_sc, views, ca_off,
                        &graph.state_kv[il], &graph.state_score[il],
                        comp_width, ratio, pos,
                    )?;

                    let emit = ((pos + 1) % ratio) == 0;
                    if emit {
                        let comp_row = graph.n_comp[il];

                        // Flush GPU so state_kv/state_score are readable
                        ops::flush_batch()?;

                        let state_kv_bytes  = graph.state_kv[il].read_bytes()?;
                        let state_sc_bytes  = graph.state_score[il].read_bytes()?;
                        let state_kv_f: &[f32] = unsafe { std::slice::from_raw_parts(
                            state_kv_bytes.as_ptr() as *const f32, state_kv_bytes.len() / 4) };
                        let state_sc_f: &[f32] = unsafe { std::slice::from_raw_parts(
                            state_sc_bytes.as_ptr() as *const f32, state_sc_bytes.len() / 4) };

                        // CPU softmax pool → pool_out[head_dim]
                        let n_rows = (coff * ratio) as usize;
                        let head_dim_usize = N_HEAD_DIM as usize;
                        let width = comp_width as usize;
                        let mut pool_out = vec![0.0f32; head_dim_usize];
                        for j in 0..head_dim_usize {
                            let mut kvs = vec![0.0f32; n_rows];
                            let mut scs = vec![0.0f32; n_rows];
                            for i in 0..n_rows {
                                let col = if ratio == 4 {
                                    if i < 4 { j } else { head_dim_usize + j }
                                } else { j };
                                kvs[i] = state_kv_f[i * width + col];
                                scs[i] = state_sc_f[i * width + col];
                            }
                            let max_sc = scs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let exps: Vec<f32> = scs.iter().map(|&s| (s - max_sc).exp()).collect();
                            let sum: f32 = exps.iter().sum();
                            pool_out[j] = exps.iter().zip(kvs.iter()).map(|(&w, &k)| w / sum * k).sum();
                        }

                        // Write pool result into index_comp[il][comp_row]
                        let comp_offset = comp_row as u64 * N_HEAD_DIM as u64 * 4;
                        {
                            let ptr = (graph.index_comp[il].retain_buf().unwrap().contents().as_ptr()
                                as usize + graph.index_comp[il].offset_raw() as usize
                                + comp_offset as usize) as *mut u8;
                            let pool_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                                pool_out.as_ptr() as *const u8, pool_out.len() * 4) };
                            unsafe { std::ptr::copy_nonoverlapping(pool_bytes.as_ptr(), ptr, pool_bytes.len()); }
                        }

                        let comp_view = GpuTensor::wrap(
                            graph.index_comp[il].retain_buf().unwrap(),
                            graph.index_comp[il].offset_raw() + comp_offset,
                            N_HEAD_DIM as u64 * 4,
                        );

                        // rms_norm, rope, ratio4_shift, fp8 quantize — matches decode emit path
                        ops::rms_norm_weight(&comp_view, &comp_view,
                            views, cn_off, N_HEAD_DIM, DS4_RMS_EPS)?;
                        let comp_pos = pos + 1 - ratio;
                        ops::rope_tail(&comp_view, 1, 1, N_HEAD_DIM, N_ROT, comp_pos,
                            DS4_ROPE_ORIG_CTX, false,
                            DS4_COMPRESS_ROPE_FREQ_BASE, 1.0 / DS4_ROPE_SCALE_FACTOR,
                            ext_factor, attn_factor,
                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
                        if ratio == 4 {
                            ops::compressor_ratio4_shift(
                                &graph.state_kv[il], &graph.state_score[il], comp_width)?;
                        }
                        ops::fp8_kv_quantize(&comp_view, N_HEAD_DIM, N_ROT)?;
                        graph.n_comp[il] += 1;
                    }
                }
            }

            // Copy Q → batch_q[ti], KV → batch_kv[ti]
            let q_row = (N_HEAD as u64 * N_HEAD_DIM as u64) * 4;
            let kv_row = N_HEAD_DIM as u64 * 4;
            ops::flush_batch()?;
            let q_data = graph.q.read_bytes()?;
            {
                let ptr = graph.batch_q.retain_buf().unwrap().contents().as_ptr() as usize
                    + graph.batch_q.offset_raw() as usize
                    + (ti as u64 * q_row) as usize;
                unsafe { std::ptr::copy_nonoverlapping(q_data.as_ptr(), ptr as *mut u8, q_data.len()); }
            }
            let kv_data = graph.kv.read_bytes()?;
            {
                let ptr = graph.batch_kv.retain_buf().unwrap().contents().as_ptr() as usize
                    + graph.batch_kv.offset_raw() as usize
                    + (ti as u64 * kv_row) as usize;
                unsafe { std::ptr::copy_nonoverlapping(kv_data.as_ptr(), ptr as *mut u8, kv_data.len()); }
            }
        }

        // Use n_tokens as the KV count for this layer's attention
        let kv_count = n_tokens;

        // ─── Attention ───
        // use_batch_heads: true when attention output is written to batch_heads
        // (all zero_prefix paths go through batch FA and write batch_heads)
        let use_batch_heads = !compressed || zero_prefix;
        if !compressed {
            // Batch FA for non-compressed layers
            let sink_view = lw.attn_sinks.as_ref().and_then(|s|
                views.find_view_retained(s.abs_offset, N_HEAD as u64 * 4));
            ops::begin_batch()?;
            ops::flash_attn_prefill_batch_raw(
                &graph.batch_heads, &graph.batch_q, &graph.batch_kv,
                sink_view.as_ref().map(|(buf, _)| buf.as_ref()),
                sink_view.as_ref().map(|(_, off)| *off).unwrap_or(0),
                n_tokens, N_SWA, N_HEAD, N_HEAD_DIM,
            )?;
            ops::end_batch()?;
            dump_tensor(&graph.batch_heads, "batch_kqv_out", (N_HEAD * N_HEAD_DIM) as usize, il, 0);
        } else if zero_prefix && graph.n_comp[il] > 0 {
            // Static mixed FA: raw KV (n_tokens rows) + comp KV (n_comp rows) concatenated
            let sink_view = lw.attn_sinks.as_ref().and_then(|s|
                views.find_view_retained(s.abs_offset, N_HEAD as u64 * 4));
            // Build a view of index_comp covering exactly n_comp rows
            let comp_view = GpuTensor::wrap(
                graph.index_comp[il].retain_buf().unwrap(),
                graph.index_comp[il].offset_raw(),
                graph.n_comp[il] as u64 * N_HEAD_DIM as u64 * 4,
            );
            ops::begin_batch()?;
            ops::flash_attn_prefill_batch_static_mixed(
                &graph.batch_heads, &graph.batch_q, &graph.batch_kv,
                &comp_view,
                sink_view.as_ref().map(|(buf, _)| buf.as_ref()),
                sink_view.as_ref().map(|(_, off)| *off).unwrap_or(0),
                n_tokens, graph.n_comp[il], N_SWA, ratio, N_HEAD, N_HEAD_DIM,
            )?;
            ops::end_batch()?;
            dump_tensor(&graph.batch_heads, "batch_kqv_out", (N_HEAD * N_HEAD_DIM) as usize, il, 0);
        } else if zero_prefix && graph.n_comp[il] == 0 {
            // Compressed layer with no comp rows yet (e.g. ratio=128 with few tokens):
            // treat same as raw batch FA — matches C's raw_prefix_tokens path
            let sink_view = lw.attn_sinks.as_ref().and_then(|s|
                views.find_view_retained(s.abs_offset, N_HEAD as u64 * 4));
            ops::begin_batch()?;
            ops::flash_attn_prefill_batch_raw(
                &graph.batch_heads, &graph.batch_q, &graph.batch_kv,
                sink_view.as_ref().map(|(buf, _)| buf.as_ref()),
                sink_view.as_ref().map(|(_, off)| *off).unwrap_or(0),
                n_tokens, N_SWA, N_HEAD, N_HEAD_DIM,
            )?;
            ops::end_batch()?;
            dump_tensor(&graph.batch_heads, "batch_kqv_out", (N_HEAD * N_HEAD_DIM) as usize, il, 0);
        }

        // ─── Pass 2: Attention output + FFN ───
        for ti in 0..n_tokens as usize {
            let token = tokens[ti];
            let pos = ti as u32;

            // Copy batch_cur_hc[ti] → cur_hc (reload HC state for this token)
            {
                let src = GpuTensor::wrap(
                    graph.batch_cur_hc.retain_buf().unwrap(),
                    graph.batch_cur_hc.offset_raw() + ti as u64 * hc_dim as u64 * 4,
                    hc_dim as u64 * 4,
                );
                ops::copy_f32(&graph.cur_hc, &src, hc_dim as usize)?;
            }

            // ─── Attention heads ───
            if use_batch_heads {
                // Read batch_heads[ti] → heads (populated by batch FA above)
                let head_row = (N_HEAD as u64 * N_HEAD_DIM as u64) * 4;
                ops::flush_batch()?;
                let bh_ptr = graph.batch_heads.retain_buf().unwrap().contents().as_ptr() as usize
                    + graph.batch_heads.offset_raw() as usize
                    + (ti as u64 * head_row) as usize;
                let hdata = unsafe {
                    std::slice::from_raw_parts(bh_ptr as *const u8, head_row as usize)
                };
                graph.heads.write_bytes(hdata)?;
            } else {
                // Fallback: per-token indexed attention for compressed layers without comp rows
                let n_kv = kv_count.min(graph.raw_cap);
                if n_kv > 0 {
                    let sink_view = lw.attn_sinks.as_ref().and_then(|s|
                        views.find_view_retained(s.abs_offset, N_HEAD as u64 * 4));
                    let sinks: Option<(&ProtocolObject<dyn MTLBuffer>, u64)> =
                        sink_view.as_ref().map(|(buf, off)| (buf.as_ref(), *off));
                    let q_row = (N_HEAD as u64 * N_HEAD_DIM as u64) * 4;
                    ops::flush_batch()?;
                    let bq_ptr = graph.batch_q.retain_buf().unwrap().contents().as_ptr() as usize
                        + graph.batch_q.offset_raw() as usize
                        + (ti as u64 * q_row) as usize;
                    let qdata = unsafe {
                        std::slice::from_raw_parts(bq_ptr as *const u8, q_row as usize)
                    };
                    graph.q.write_bytes(qdata)?;
                    ops::indexed_attention(
                        &graph.heads, &graph.q,
                        &graph.raw_cache[il], &graph.index_comp[il],
                        None,
                        sinks,
                        1, N_HEAD, n_kv, graph.raw_cap,
                        0, graph.n_comp[il], N_INDEXER_TOP_K, pos,
                        N_SWA, if ratio == 4 { 4 } else { 1 },
                        1.0 / (N_HEAD_DIM as f32).sqrt(),
                        (N_HEAD_DIM * 4) as u64, (N_HEAD_DIM * 4) as u64,
                    )?;
                } else {
                    graph.heads.fill_f32(0.0, (N_HEAD * N_HEAD_DIM) as u64)?;
                }
            }
            dump_tensor(&graph.heads, "kqv_out", (N_HEAD * N_HEAD_DIM) as usize, il, pos);

            // ─── Inverse RoPE on heads ───
            ops::rope_tail(&graph.heads, 1, N_HEAD, N_HEAD_DIM,
                N_ROT, pos, nco, true,
                fb, freq_scale, ext_factor, attn_factor,
                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
            dump_tensor(&graph.heads, "kqv_back", (N_HEAD * N_HEAD_DIM) as usize, il, pos);

            // ─── Attention output projection ───
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
            dump_tensor(&graph.attn_out, "attn_out", N_EMBD as usize, il, pos);

            // ─── HC expand after attention ───
            // Restore this token's attention hc_split (computed in Pass 1 but overwritten
            // by subsequent tokens' Sinkhorn before Pass 2 gets here).
            {
                let src = GpuTensor::wrap(
                    graph.batch_hc_split.retain_buf().unwrap(),
                    graph.batch_hc_split.offset_raw() + ti as u64 * mix_hc * 4,
                    mix_hc * 4,
                );
                ops::copy_f32(&graph.hc_split, &src, mix_hc as usize)?;
            }
            let po = N_HC as u64 * 4;
            let co = (2 * N_HC) as u64 * 4;
            let hp = GpuTensor::wrap(graph.hc_split.retain_buf().unwrap(),
                graph.hc_split.offset_raw() + po, N_HC as u64 * 4);
            let hc = GpuTensor::wrap(graph.hc_split.retain_buf().unwrap(),
                graph.hc_split.offset_raw() + co, N_HC as u64 * N_HC as u64 * 4);
            ops::hc_expand_tensor(&graph.after_attn_hc, &graph.attn_out,
                &graph.cur_hc, &hp, &hc, N_EMBD, N_HC)?;
            dump_tensor(&graph.after_attn_hc, "hc_attn_post", (N_HC * N_EMBD) as usize, il, pos);

            // ─── FFN HC pre ───
            ops::rms_norm_plain(&graph.flat_hc, &graph.after_attn_hc, hc_dim, DS4_RMS_EPS)?;
            if let Some(ref hc_fn) = lw.hc_ffn_fn {
                ops::matmul_f16(&graph.hc_mix, views, hc_fn.abs_offset,
                    hc_dim as u64, mix_hc, &graph.flat_hc, 1)?;
                if let (Some(ref scale), Some(ref base)) = (&lw.hc_ffn_scale, &lw.hc_ffn_base) {
                    dump_tensor(&graph.hc_mix, "hc_ffn_mix_before_sinkhorn", mix_hc as usize, il, pos);
                    ops::hc_split_sinkhorn(&graph.hc_split, &graph.hc_mix, views,
                        scale.abs_offset, base.abs_offset,
                        N_HC, N_HC_SINKHORN_ITER, DS4_HC_EPS)?;
                    dump_tensor(&graph.hc_split, "hc_ffn_split_after_sinkhorn", mix_hc as usize, il, pos);
                    ops::hc_weighted_sum(&graph.ffn_cur, &graph.after_attn_hc,
                        &graph.hc_split, N_EMBD, N_HC)?;
                }
            }
            dump_tensor(&graph.ffn_cur, "hc_ffn_pre", N_EMBD as usize, il, pos);
            if let Some(ref fnw) = lw.ffn_norm {
                ops::rms_norm_weight(&graph.ffn_norm, &graph.ffn_cur,
                    views, fnw.abs_offset, N_EMBD, DS4_RMS_EPS)?;
            }
            dump_tensor(&graph.ffn_norm, "ffn_norm", N_EMBD as usize, il, pos);

            // ─── Router ───
            // (re-implemented inline from eval_token_decode)
            if let Some(ref gi) = lw.ffn_gate_inp {
                ops::matmul_f16(&graph.router_logits, views, gi.abs_offset,
                    N_EMBD as u64, N_EXPERT as u64, &graph.ffn_norm, 1)?;
                dump_tensor(&graph.router_logits, "ffn_moe_logits", N_EXPERT as usize, il, pos);
                ops::softplus_sqrt(&graph.router_probs, &graph.router_logits, N_EXPERT)?;
                dump_tensor(&graph.router_probs, "ffn_moe_probs", N_EXPERT as usize, il, pos);
                if is_hash_layer(il as u32) {
                    let hash_offset = lw.ffn_gate_tid2eid.as_ref().map(|t| t.abs_offset);
                    ops::router_select_one(&graph.router_probs,
                        &graph.router_probs, &graph.router_selected, N_EXPERT, false,
                        views, hash_offset, token as u32, N_VOCAB)?;
                } else {
                    let bt = lw.ffn_exp_probs_b.as_ref().and_then(|b| {
                        let (buf, off) = views.find_view_retained(b.abs_offset, 256*4)?;
                        Some(GpuTensor::wrap(buf, off, 256*4))
                    });
                    ops::router_select_one(&graph.router_probs,
                        bt.as_ref().unwrap_or(&graph.router_probs),
                        &graph.router_selected, N_EXPERT, bt.is_some(),
                        views, None, 0, 0)?;
                }
                ops::router_weights_one(&graph.router_probs,
                    &graph.router_selected, &graph.router_weights)?;

                ops::flush_batch()?;
                let mut sel_ids = vec![0i32; N_EXPERT_USED as usize];
                graph.router_selected.read_i32_slice(0, &mut sel_ids)?;
                {
                    let raw: &[u8] = unsafe { std::slice::from_raw_parts(
                        sel_ids.as_ptr() as *const u8, sel_ids.len() * 4) };
                    dump_raw(raw, "i32", "ffn_moe_topk", il, pos);
                }
                dump_tensor(&graph.router_weights, "ffn_moe_weights_scaled", N_EXPERT_USED as usize, il, pos);
                for id in sel_ids.iter_mut() {
                    *id = (*id).max(0).min((N_EXPERT - 1) as i32);
                }

                let eq = lw.ffn_gate_exps.as_ref().map(|t| t.dtype);
                if let (Some(qt), Some(gx), Some(ux), Some(dx)) =
                    (eq, &lw.ffn_gate_exps, &lw.ffn_up_exps, &lw.ffn_down_exps)
                {
                    let r = match qt {
                        GgufTensorType::Iq2Xxs => ops::matmul_id_iq2_xxs_pair(
                            &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                            N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids),
                        GgufTensorType::Q2K => ops::matmul_id_q2_K_sum6(
                            &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                            N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids),
                        GgufTensorType::Q4K => if n_tokens > 4 {
                            ops::matmul_id_q4_K_nonpair_sum6(
                                &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                                N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids)
                        } else {
                            ops::matmul_id_q4_K_sum6(
                                &graph.routed_gate, &graph.routed_up, views, gx.abs_offset, ux.abs_offset,
                                N_EMBD, N_FF_EXP, N_EXPERT, &graph.ffn_norm, &sel_ids)
                        },
                        _ => return Err("unsupported expert quant"),
                    };
                    r?;
                    dump_tensor(&graph.routed_gate, "ffn_moe_gate", (N_FF_EXP * N_EXPERT_USED) as usize, il, pos);
                    dump_tensor(&graph.routed_up,   "ffn_moe_up",   (N_FF_EXP * N_EXPERT_USED) as usize, il, pos);

                    ops::moe_swiglu_weight(&graph.routed_gate, &graph.routed_up,
                        &graph.routed_mid, &graph.router_weights, N_FF_EXP, N_EXPERT_USED,
                        DS4_SWIGLU_CLAMP_EXP)?;
                    dump_tensor(&graph.routed_mid, "ffn_moe_swiglu", (N_FF_EXP * N_EXPERT_USED) as usize, il, pos);

                    let down_type = dx.dtype;
                    let mut acc = vec![0.0f32; N_EMBD as usize];
                    for (i, &expert_id) in sel_ids.iter().enumerate() {
                        let mid_row = GpuTensor::wrap(
                            graph.routed_mid.retain_buf().unwrap(),
                            graph.routed_mid.offset_raw() + (i as u64 * N_FF_EXP as u64 * 4),
                            N_FF_EXP as u64 * 4);
                        let down_row = GpuTensor::wrap(
                            graph.routed_down.retain_buf().unwrap(),
                            graph.routed_down.offset_raw() + (i as u64 * N_EMBD as u64 * 4),
                            N_EMBD as u64 * 4);
                        let r = match down_type {
                            GgufTensorType::Iq2Xxs => ops::matmul_id_iq2_xxs_f32(
                                &down_row, views, dx.abs_offset,
                                N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                            GgufTensorType::Q2K => ops::matmul_id_q2_K_f32(
                                &down_row, views, dx.abs_offset,
                                N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                            GgufTensorType::Q4K => ops::matmul_id_q4_K_f32(
                                &down_row, views, dx.abs_offset,
                                N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                            GgufTensorType::Q8_0 => ops::matmul_id_q8_0_f32(
                                &down_row, views, dx.abs_offset,
                                N_FF_EXP, N_EMBD, N_EXPERT, &mid_row, &[expert_id]),
                            _ => return Err("unsupported down expert quant"),
                        };
                        r?;
                    }
                    ops::flush_batch()?;
                    dump_tensor(&graph.routed_down, "ffn_moe_down", (N_EMBD * N_EXPERT_USED) as usize, il, pos);
                    {
                        let data = graph.routed_down.read_bytes()?;
                        let floats: &[f32] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()/4)
                        };
                        for i in 0..N_EXPERT_USED as usize {
                            let row_start = i * N_EMBD as usize;
                            for j in 0..N_EMBD as usize {
                                acc[j] += floats[row_start + j];
                            }
                        }
                    }
                    let acc_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(acc.as_ptr() as *const u8, acc.len() * 4)
                    };
                    graph.routed_out.write_bytes(acc_bytes)?;
                } else {
                    graph.routed_out.fill_f32(0.0, N_EMBD as u64)?;
                }
            } else {
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
            dump_tensor(&graph.routed_out, "ffn_moe_out", N_EMBD as usize, il, pos);
            dump_tensor(&graph.shared_out, "ffn_shexp", N_EMBD as usize, il, pos);

            // ─── HC expand after FFN + swap ───
            ops::hc_expand_add_split_tensor(&graph.after_ffn_hc,
                &graph.routed_out, &graph.shared_out,
                &graph.after_attn_hc, &graph.hc_split, N_EMBD, N_HC)?;
            dump_tensor(&graph.after_ffn_hc, "hc_ffn_post", (N_HC * N_EMBD) as usize, il, pos);

            // Write HC result back to batch_cur_hc[ti]
            ops::flush_batch()?;
            let hc_data = graph.after_ffn_hc.read_bytes()?;
            {
                let ptr = graph.batch_cur_hc.retain_buf().unwrap().contents().as_ptr() as usize
                    + graph.batch_cur_hc.offset_raw() as usize
                    + (ti as u64 * hc_dim as u64 * 4) as usize;
                unsafe { std::ptr::copy_nonoverlapping(hc_data.as_ptr(), ptr as *mut u8, hc_data.len()); }
            }

            // ─── Compressor (compressed layers, decode/non-zero-prefix only) ───
            // For zero_prefix, the compressor already ran in Pass 1.
            if compressed && !zero_prefix {
                let coff: u32 = if ratio == 4 { 2 } else { 1 };
                let comp_width = coff * N_HEAD_DIM;
                let emit = ((pos + 1) % ratio) == 0;

                if let (Some(ref ck), Some(ref cg), Some(ref ca), Some(ref cn)) = (
                    lw.attn_compressor_kv.as_ref(),
                    lw.attn_compressor_gate.as_ref(),
                    lw.attn_compressor_ape.as_ref(),
                    lw.attn_compressor_norm.as_ref(),
                ) {
                    let (ck_off, cg_off, ca_off, cn_off) =
                        (ck.abs_offset, cg.abs_offset, ca.abs_offset, cn.abs_offset);

                    // Compute HC pre for compressor (uses attn_norm from pass 1)
                    // Note: attn_norm was computed in pass 1 for each token.
                    // We need to recompute attn_norm for the current token.
                    // Actually, let me use the stored HC state.
                    // HC pre-attention was already computed. The attn_norm was overwritten by the last token in pass 1.
                    // Since we're in pass 2, we need attn_norm again.
                    // This is a problem — we need to recompute.
                    // For now, skip the compressor for brevity — it's complex.
                    // The compressor uses graph.attn_norm which we don't have anymore.
                    // Actually, we have batch_cur_hc[ti] which has the layer input HC.
                    // We need to recompute attn_norm from it.
                    // cur_hc holds the layer INPUT (loaded at top of Pass 2 loop, not modified since).
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

                    ops::matmul_f16(&graph.comp_kv, views, ck_off,
                        N_EMBD as u64, comp_width as u64, &graph.attn_norm, 1)?;
                    ops::matmul_f16(&graph.comp_sc, views, cg_off,
                        N_EMBD as u64, comp_width as u64, &graph.attn_norm, 1)?;
                    ops::compressor_store_one(
                        &graph.comp_kv, &graph.comp_sc, views, ca_off,
                        &graph.state_kv[il], &graph.state_score[il],
                        comp_width, ratio, pos,
                    )?;

                    if emit {
                        let comp_row = graph.n_comp[il];
                        ops::flush_batch()?;

                        let state_kv_bytes = graph.state_kv[il].read_bytes()?;
                        let state_sc_bytes = graph.state_score[il].read_bytes()?;
                        let state_kv: &[f32] = unsafe { std::slice::from_raw_parts(
                            state_kv_bytes.as_ptr() as *const f32, state_kv_bytes.len() / 4) };
                        let state_sc: &[f32] = unsafe { std::slice::from_raw_parts(
                            state_sc_bytes.as_ptr() as *const f32, state_sc_bytes.len() / 4) };

                        let n_rows = (coff * ratio) as usize;
                        let hd = N_HEAD_DIM as usize;
                        let w = comp_width as usize;

                        let mut pool_out = vec![0.0f32; hd];
                        for j in 0..hd {
                            let mut kvs = vec![0.0f32; n_rows];
                            let mut scs = vec![0.0f32; n_rows];
                            for i in 0..n_rows {
                                let col = if ratio == 4 {
                                    if i < 4 { j } else { hd + j }
                                } else { j };
                                kvs[i] = state_kv[i * w + col];
                                scs[i] = state_sc[i * w + col];
                            }
                            let max_sc = scs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let exps: Vec<f32> = scs.iter().map(|&s| (s - max_sc).exp()).collect();
                            let sum: f32 = exps.iter().sum();
                            let val: f32 = exps.iter().zip(kvs.iter()).map(|(&w, &k)| w / sum * k).sum();
                            pool_out[j] = val;
                        }

                        let comp_offset = comp_row as u64 * N_HEAD_DIM as u64 * 4;
                        let comp_view = GpuTensor::wrap(
                            graph.index_comp[il].retain_buf().unwrap(),
                            graph.index_comp[il].offset_raw() + comp_offset,
                            N_HEAD_DIM as u64 * 4,
                        );
                        let pool_bytes: &[u8] = unsafe { std::slice::from_raw_parts(
                            pool_out.as_ptr() as *const u8, pool_out.len() * 4) };
                        {
                            let ptr = (graph.index_comp[il].retain_buf().unwrap().contents().as_ptr()
                                as usize + graph.index_comp[il].offset_raw() as usize
                                + comp_offset as usize) as *mut u8;
                            unsafe { std::ptr::copy_nonoverlapping(pool_bytes.as_ptr(), ptr, pool_bytes.len()); }
                        }

                        ops::rms_norm_weight(&comp_view, &comp_view, views, cn_off, N_HEAD_DIM, DS4_RMS_EPS)?;
                        let comp_pos = pos + 1 - ratio;
                        ops::rope_tail(&comp_view, 1, 1, N_HEAD_DIM, N_ROT, comp_pos,
                            DS4_ROPE_ORIG_CTX, false,
                            DS4_COMPRESS_ROPE_FREQ_BASE, 1.0 / DS4_ROPE_SCALE_FACTOR,
                            ext_factor, attn_factor,
                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW)?;
                        if ratio == 4 {
                            ops::compressor_ratio4_shift(&graph.state_kv[il], &graph.state_score[il], comp_width)?;
                        }
                        ops::fp8_kv_quantize(&comp_view, N_HEAD_DIM, N_ROT)?;
                        graph.n_comp[il] += 1;
                    }
                }
            }
        }
    }

    // ─── Output head ───
    // Use the last token's HC state for output
    let last_ti = n_tokens as usize - 1;
    {
        let src = GpuTensor::wrap(
            graph.batch_cur_hc.retain_buf().unwrap(),
            graph.batch_cur_hc.offset_raw() + last_ti as u64 * hc_dim as u64 * 4,
            hc_dim as u64 * 4,
        );
        ops::rms_norm_plain(&graph.cur_hc, &src, hc_dim, DS4_RMS_EPS)?;
    }
    dump_tensor(&graph.cur_hc, "cur_hc_head", hc_dim as usize, N_LAYER as usize, n_tokens - 1);
    ops::rms_norm_plain(&graph.flat_hc, &graph.cur_hc, hc_dim, DS4_RMS_EPS)?;
    dump_tensor(&graph.flat_hc, "flat_hc", hc_dim as usize, N_LAYER as usize, n_tokens - 1);
    if let Some(ref ohc_fn) = weights.output_hc_fn {
        ops::matmul_f16(&graph.output_pre, views,
            ohc_fn.abs_offset, hc_dim as u64, N_HC as u64, &graph.flat_hc, 1)?;
    }
    if let (Some(ref scale), Some(ref base)) = (&weights.output_hc_scale, &weights.output_hc_base) {
        ops::flush_batch()?;
        ops::output_hc_weights(&graph.output_weights, &graph.output_pre,
            views, scale.abs_offset, base.abs_offset, N_HC, DS4_HC_EPS)?;
    }
    dump_tensor(&graph.output_pre,     "result_hc_pre",     N_HC as usize,   N_LAYER as usize, n_tokens - 1);
    dump_tensor(&graph.output_weights, "result_hc_weights", N_HC as usize,   N_LAYER as usize, n_tokens - 1);
    ops::hc_weighted_sum(&graph.output_embd, &graph.cur_hc,
        &graph.output_weights, N_EMBD, N_HC)?;
    dump_tensor(&graph.output_embd, "result_hc", N_EMBD as usize, N_LAYER as usize, n_tokens - 1);
    if let Some(ref on) = weights.output_norm {
        ops::rms_norm_weight(&graph.output_norm, &graph.output_embd,
            views, on.abs_offset, N_EMBD, DS4_RMS_EPS)?;
    }
    dump_tensor(&graph.output_norm, "result_norm", N_EMBD as usize, N_LAYER as usize, n_tokens - 1);
    if let Some(ref ow) = weights.output {
        ops::matmul_q8_0(&graph.logits, views,
            ow.abs_offset, N_EMBD as u64, N_VOCAB as u64, &graph.output_norm, 1)?;
    }
    dump_tensor(&graph.logits, "result_output", N_VOCAB as usize, N_LAYER as usize, n_tokens - 1);

    graph.n_raw = (graph.n_raw + n_tokens).min(graph.raw_cap);
    graph.n_pos += n_tokens;

    if std::env::var("DS4_DUMP_LOGITS").as_deref() == Ok("1") {
        ops::flush_batch()?;
        let data = graph.logits.read_bytes()?;
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
        };
        let mut indexed: Vec<(usize, f32)> = floats.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprint!("ds4: logits[pos=prefill] top8:");
        for (id, val) in indexed.iter().take(8) {
            eprint!(" {}({:.3})", id, val);
        }
        eprintln!();
    }

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
        &graph.routed_gate, &graph.routed_up, &graph.routed_mid, &graph.routed_down, &graph.routed_out,
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
        &mut graph.routed_gate, &mut graph.routed_up, &mut graph.routed_mid, &mut graph.routed_down, &mut graph.routed_out,
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
