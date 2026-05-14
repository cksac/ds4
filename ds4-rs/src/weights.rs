//! Weight binding: maps GGUF tensor names to model weight structures.

use crate::gguf::*;
use crate::model::*;
use crate::model::MtpWeights;

/// Look up a tensor by formatted name with layer substitution.
fn tensor_by_namef<'a>(model: &'a GgufModel, fmt: &str, layer: u32) -> Option<&'a GgufTensor> {
    let name = if fmt.contains("%u") {
        let parts: Vec<&str> = fmt.split("%u").collect();
        if parts.len() == 2 {
            format!("{}{}{}", parts[0], layer, parts[1])
        } else {
            fmt.to_string()
        }
    } else {
        fmt.to_string()
    };
    model.find_tensor(&name)
}

/// Bind MTP (Multi-Token Prediction) weights from model.
pub fn mtp_weights_bind(model: &GgufModel) -> MtpWeights {
    let mut w = MtpWeights::new();
    w.hc_head_base = model.find_tensor("mtp.0.hc_head_base.weight").cloned();
    w.hc_head_fn = model.find_tensor("mtp.0.hc_head_fn.weight").cloned();
    w.hc_head_scale = model.find_tensor("mtp.0.hc_head_scale.weight").cloned();
    w.e_proj = model.find_tensor("mtp.0.e_proj.weight").cloned();
    w.h_proj = model.find_tensor("mtp.0.h_proj.weight").cloned();
    w.enorm = model.find_tensor("mtp.0.enorm.weight").cloned();
    w.hnorm = model.find_tensor("mtp.0.hnorm.weight").cloned();
    w.norm = model.find_tensor("mtp.0.norm.weight").cloned();

    // Single transformer layer with standard weight names
    let mut l = LayerWeights::new();
    l.hc_attn_fn = model.find_tensor("mtp.0.hc_attn_fn.weight").cloned();
    l.hc_attn_scale = model.find_tensor("mtp.0.hc_attn_scale.weight").cloned();
    l.hc_attn_base = model.find_tensor("mtp.0.hc_attn_base.weight").cloned();
    l.attn_norm = model.find_tensor("mtp.0.attn_norm.weight").cloned();
    l.attn_q_a = model.find_tensor("mtp.0.attn_q_a.weight").cloned();
    l.attn_q_a_norm = model.find_tensor("mtp.0.attn_q_a_norm.weight").cloned();
    l.attn_q_b = model.find_tensor("mtp.0.attn_q_b.weight").cloned();
    l.attn_kv_a = model.find_tensor("mtp.0.attn_kv.weight").cloned();
    l.attn_kv_a_norm = model.find_tensor("mtp.0.attn_kv_a_norm.weight").cloned();
    l.attn_sinks = model.find_tensor("mtp.0.attn_sinks.weight").cloned();
    l.attn_output_a = model.find_tensor("mtp.0.attn_output_a.weight").cloned();
    l.attn_output_b = model.find_tensor("mtp.0.attn_output_b.weight").cloned();
    l.hc_ffn_fn = model.find_tensor("mtp.0.hc_ffn_fn.weight").cloned();
    l.hc_ffn_scale = model.find_tensor("mtp.0.hc_ffn_scale.weight").cloned();
    l.hc_ffn_base = model.find_tensor("mtp.0.hc_ffn_base.weight").cloned();
    l.ffn_norm = model.find_tensor("mtp.0.ffn_norm.weight").cloned();
    l.ffn_gate_inp = model.find_tensor("mtp.0.ffn_gate_inp.weight").cloned();
    l.ffn_gate_exps = model.find_tensor("mtp.0.ffn_gate_exps.weight").cloned();
    l.ffn_up_exps = model.find_tensor("mtp.0.ffn_up_exps.weight").cloned();
    l.ffn_down_exps = model.find_tensor("mtp.0.ffn_down_exps.weight").cloned();
    l.shared_gate = model.find_tensor("mtp.0.ffn_gate_shexp.weight").cloned();
    l.shared_up = model.find_tensor("mtp.0.ffn_up_shexp.weight").cloned();
    l.shared_down = model.find_tensor("mtp.0.ffn_down_shexp.weight").cloned();
    l.ffn_exp_probs_b = model.find_tensor("mtp.0.exp_probs_b.bias").cloned();
    w.layer = Some(Box::new(l));
    w
}

/// Bind all weights for the main model.
pub fn weights_bind(model: &GgufModel) -> EngineWeights {
    let mut ew = EngineWeights::new();
    let n_layer = N_LAYER;
    let compress_ratio = 4u32; // DS4 Flash uses ratio 4 for all layers
    let compress_layer_status: Vec<bool> = (0..n_layer).map(|il| {
        // Layers have compress_ratio=4; in production read from metadata
        let token_ratio = vec![4i32; n_layer as usize];
        token_ratio[il as usize] != 0
    }).collect();

    // ─── Global (non-block) tensors ───
    if let Some(t) = model.find_tensor("token_embd.weight") {
        ew.token_embd = Some(t.clone());
    }
    if let Some(t) = model.find_tensor("output_hc_base.weight") {
        ew.output_hc_base = Some(t.clone());
    }
    if let Some(t) = model.find_tensor("output_hc_fn.weight") {
        ew.output_hc_fn = Some(t.clone());
    }
    if let Some(t) = model.find_tensor("output_hc_scale.weight") {
        ew.output_hc_scale = Some(t.clone());
    }
    if let Some(t) = model.find_tensor("output_norm.weight") {
        ew.output_norm = Some(t.clone());
    }
    if let Some(t) = model.find_tensor("output.weight") {
        ew.output = Some(t.clone());
    }

    // ─── Per-layer tensor bindings ───
    for il in 0..n_layer {
        let l = &mut ew.layer[il as usize];

        // Required for every layer
        l.hc_attn_fn = tensor_by_namef(model, "blk.%u.hc_attn_fn.weight", il).map(|t| t.clone());
        l.hc_attn_scale = tensor_by_namef(model, "blk.%u.hc_attn_scale.weight", il).map(|t| t.clone());
        l.hc_attn_base = tensor_by_namef(model, "blk.%u.hc_attn_base.weight", il).map(|t| t.clone());
        l.attn_norm = tensor_by_namef(model, "blk.%u.attn_norm.weight", il).map(|t| t.clone());
        l.attn_q_a = tensor_by_namef(model, "blk.%u.attn_q_a.weight", il).map(|t| t.clone());
        l.attn_q_a_norm = tensor_by_namef(model, "blk.%u.attn_q_a_norm.weight", il).map(|t| t.clone());
        l.attn_q_b = tensor_by_namef(model, "blk.%u.attn_q_b.weight", il).map(|t| t.clone());
        l.attn_kv_a = tensor_by_namef(model, "blk.%u.attn_kv.weight", il).map(|t| t.clone());
        l.attn_kv_a_norm = tensor_by_namef(model, "blk.%u.attn_kv_a_norm.weight", il).map(|t| t.clone());
        l.attn_sinks = tensor_by_namef(model, "blk.%u.attn_sinks.weight", il).map(|t| t.clone());
        l.attn_output_a = tensor_by_namef(model, "blk.%u.attn_output_a.weight", il).map(|t| t.clone());
        l.attn_output_b = tensor_by_namef(model, "blk.%u.attn_output_b.weight", il).map(|t| t.clone());

        // Compressor tensors (ratio != 0)
        if compress_layer_status[il as usize] {
            l.attn_compressor_ape = tensor_by_namef(model, "blk.%u.attn_compressor_ape.weight", il).map(|t| t.clone());
            l.attn_compressor_kv = tensor_by_namef(model, "blk.%u.attn_compressor_kv.weight", il).map(|t| t.clone());
            l.attn_compressor_gate = tensor_by_namef(model, "blk.%u.attn_compressor_gate.weight", il).map(|t| t.clone());
            l.attn_compressor_norm = tensor_by_namef(model, "blk.%u.attn_compressor_norm.weight", il).map(|t| t.clone());
        }

        // Indexer tensors (ratio == 4)
        if compress_ratio == 4 {
            l.indexer_attn_q_b = tensor_by_namef(model, "blk.%u.indexer.attn_q_b.weight", il).map(|t| t.clone());
            l.indexer_proj = tensor_by_namef(model, "blk.%u.indexer.proj.weight", il).map(|t| t.clone());
            l.indexer_comp_ape = tensor_by_namef(model, "blk.%u.indexer_compressor_ape.weight", il).map(|t| t.clone());
            l.indexer_comp_kv = tensor_by_namef(model, "blk.%u.indexer_compressor_kv.weight", il).map(|t| t.clone());
            l.indexer_comp_gate = tensor_by_namef(model, "blk.%u.indexer_compressor_gate.weight", il).map(|t| t.clone());
            l.indexer_comp_norm = tensor_by_namef(model, "blk.%u.indexer_compressor_norm.weight", il).map(|t| t.clone());
        }

        // HC FFN (required for every layer)
        l.hc_ffn_fn = tensor_by_namef(model, "blk.%u.hc_ffn_fn.weight", il).map(|t| t.clone());
        l.hc_ffn_scale = tensor_by_namef(model, "blk.%u.hc_ffn_scale.weight", il).map(|t| t.clone());
        l.hc_ffn_base = tensor_by_namef(model, "blk.%u.hc_ffn_base.weight", il).map(|t| t.clone());

        // FFN core
        l.ffn_norm = tensor_by_namef(model, "blk.%u.ffn_norm.weight", il).map(|t| t.clone());
        l.ffn_gate_inp = tensor_by_namef(model, "blk.%u.ffn_gate_inp.weight", il).map(|t| t.clone());
        l.ffn_gate_exps = tensor_by_namef(model, "blk.%u.ffn_gate_exps.weight", il).map(|t| t.clone());
        l.ffn_up_exps = tensor_by_namef(model, "blk.%u.ffn_up_exps.weight", il).map(|t| t.clone());
        l.ffn_down_exps = tensor_by_namef(model, "blk.%u.ffn_down_exps.weight", il).map(|t| t.clone());

        // Shared experts
        l.shared_gate = tensor_by_namef(model, "blk.%u.ffn_gate_shexp.weight", il).map(|t| t.clone());
        l.shared_up = tensor_by_namef(model, "blk.%u.ffn_up_shexp.weight", il).map(|t| t.clone());
        l.shared_down = tensor_by_namef(model, "blk.%u.ffn_down_shexp.weight", il).map(|t| t.clone());

        // Hash routing (only layers il < N_HASH_LAYER)
        if (il as u32) < N_HASH_LAYER {
            l.ffn_gate_tid2eid = tensor_by_namef(model, "blk.%u.ffn_gate_tid2eid.weight", il).map(|t| t.clone());
        }

        // Optional: exp_probs_b.bias (not .weight)
        l.ffn_exp_probs_b = tensor_by_namef(model, "blk.%u.exp_probs_b.bias", il).map(|t| t.clone());
    }

    ew
}

/// Validate all expected tensor shapes match the fixed model constants.
pub fn validate_shapes(_model: &GgufModel, weights: &EngineWeights) -> Result<(), String> {
    let mut others: Vec<String> = Vec::new();
    for il in 0..N_LAYER as usize {
        let l = &weights.layer[il];
        // Check a few key tensors for shape
        if let Some(ref t) = l.attn_q_a {
            if t.ndim != 2 || t.dim[0] as u32 != N_LORA_Q || t.dim[1] as u32 != N_EMBD {
                others.push(format!("layer {}.attn_q_a bad shape: {}x{} != {}x{}",
                    il, t.dim[0], t.dim[1], N_LORA_Q, N_EMBD));
            }
        }
        if let Some(ref t) = l.attn_q_b {
            if t.ndim != 2 || t.dim[0] as u32 != N_HEAD * N_HEAD_DIM || t.dim[1] as u32 != N_LORA_Q {
                others.push(format!("layer {}.attn_q_b bad shape", il));
            }
        }
        if let Some(ref t) = l.attn_kv_a {
            if t.ndim != 2 || t.dim[1] as u32 != N_EMBD {
                others.push(format!("layer {}.attn_kv_a bad shape", il));
            }
        }
        if let Some(ref t) = l.ffn_gate_inp {
            if t.ndim != 2 || t.dim[0] as u32 != N_EMBD {
                others.push(format!("layer {}.ffn_gate_inp bad shape", il));
            }
        }
    }
    if others.is_empty() { Ok(()) } else { Err(others.join("; ")) }
}
