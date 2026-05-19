//! Native Rust Metal inference path.

use std::ffi::{c_int, CString};
use anyhow::Result;
use crate::metal::{device::MetalDevice, session::{LayerWeights, Session}, decode};
use crate::ffi;

pub struct NativeSession {
    pub device: MetalDevice,
    pub session: Session,
}

impl NativeSession {
    pub fn init_from_engine(
        engine: *const ffi::ds4_engine,
        model_map: *const std::ffi::c_void, model_size: u64,
        tensor_data_offset: u64, tensor_data_size: u64,
        ctx_size: i32,
        n_layer: u32, n_embd: u32, n_hc: u32,
        n_head: u32, head_dim: u32, n_rot: u32, n_vocab: u32,
    ) -> Result<Self> {
        eprintln!("ds4: initializing native Rust Metal session...");
        let device = MetalDevice::init()?;

        let mut session = Session::new(
            device.device.clone(),
            device.queue.clone(),
            device.library.clone(),
            model_map, model_size,
            ctx_size,
            n_layer, n_embd, n_hc, n_head, head_dim, n_rot, n_vocab,
        )?;

        session.set_model_map(model_map, model_size, tensor_data_offset, tensor_data_size)?;

        // Populate per-layer weight offsets from C engine bridge
        for il in 0..n_layer as i32 {
            let mut lw = LayerWeights {
                layer: il as u32,
                attn_norm: 0, attn_q_a: 0, attn_q_b: 0, attn_kv: 0,
                attn_out_a: 0, attn_out_b: 0, attn_out_group_dim: 32768,
                hc_attn_fn: 0, hc_attn_scale: 0, hc_attn_base: 0,
                hc_ffn_fn: 0, hc_ffn_scale: 0, hc_ffn_base: 0,
                ffn_norm: 0,
                ffn_gate_shexp: 0, ffn_up_shexp: 0, ffn_down_shexp: 0,
                ffn_gate_exps: 0, ffn_up_exps: 0, ffn_down_exps: 0,
                ffn_gate_type: 0, ffn_down_type: 0,
                ffn_gate_expert_bytes: 0, ffn_gate_row_bytes: 0,
                ffn_down_expert_bytes: 0, ffn_down_row_bytes: 0,
                ffn_expert_in_dim: 0, ffn_expert_mid_dim: 0, ffn_expert_out_dim: 0,
                compress_ratio: 0,
                compress_ape: 0, compress_ape_type: 0,
                compress_norm: 0, compress_norm_type: 0,
                rope_freq_base: 0.0, rope_freq_scale: 0.0,
                router_bias: 0, router_hash: 0, router_hash_rows: 0,
                has_bias: false, hash_mode: false,
            };
            unsafe {
                let (mut ffn_gate_type, mut ffn_down_type) = (0i32, 0i32);
                let (mut ffn_expert_in_dim, mut ffn_expert_mid_dim, mut ffn_expert_out_dim) = (0i32, 0i32, 0i32);
                let (mut compress_ratio, mut compress_ape_type, mut compress_norm_type) = (0i32, 0i32, 0i32);
                let (mut router_hash_rows, mut has_bias, mut hash_mode) = (0i32, 0i32, 0i32);
                let (mut sink_offset, mut output_norm_offset, mut output_weight_offset) = (0u64, 0u64, 0u64);
                ffi::ds4_bridge_layer_weights(
                    engine, il,
                    &mut lw.attn_norm,
                    &mut lw.attn_q_a, &mut lw.attn_q_b, &mut lw.attn_kv,
                    &mut lw.attn_out_a, &mut lw.attn_out_b,
                    &mut lw.hc_attn_fn, &mut lw.hc_attn_scale, &mut lw.hc_attn_base,
                    &mut lw.hc_ffn_fn, &mut lw.hc_ffn_scale, &mut lw.hc_ffn_base,
                    &mut lw.ffn_norm,
                    &mut lw.ffn_gate_shexp, &mut lw.ffn_up_shexp, &mut lw.ffn_down_shexp,
                    &mut lw.ffn_gate_exps, &mut lw.ffn_up_exps, &mut lw.ffn_down_exps,
                    &mut ffn_gate_type, &mut ffn_down_type,
                    &mut lw.ffn_gate_expert_bytes, &mut lw.ffn_gate_row_bytes,
                    &mut lw.ffn_down_expert_bytes, &mut lw.ffn_down_row_bytes,
                    &mut ffn_expert_in_dim, &mut ffn_expert_mid_dim, &mut ffn_expert_out_dim,
                    &mut compress_ratio,
                    &mut lw.compress_ape, &mut compress_ape_type,
                    &mut lw.compress_norm, &mut compress_norm_type,
                    &mut lw.rope_freq_base, &mut lw.rope_freq_scale,
                    &mut lw.router_bias, &mut lw.router_hash, &mut router_hash_rows,
                    &mut has_bias, &mut hash_mode,
                    &mut sink_offset,
                    &mut output_norm_offset, &mut output_weight_offset,
                );
                lw.ffn_gate_type = ffn_gate_type as u32;
                lw.ffn_down_type = ffn_down_type as u32;
                lw.ffn_expert_in_dim = ffn_expert_in_dim as u32;
                lw.ffn_expert_mid_dim = ffn_expert_mid_dim as u32;
                lw.ffn_expert_out_dim = ffn_expert_out_dim as u32;
                lw.compress_ratio = compress_ratio as u32;
                lw.compress_ape_type = compress_ape_type as u32;
                lw.compress_norm_type = compress_norm_type as u32;
                lw.router_hash_rows = router_hash_rows as u32;
                lw.has_bias = has_bias != 0;
                lw.hash_mode = hash_mode != 0;
                session.sink_offset = sink_offset;
                session.output_norm_offset = output_norm_offset;
                session.output_weight_offset = output_weight_offset;
            }
            session.layers.push(lw);
        }

        // Print first layer weight offsets for verification
        if !session.layers.is_empty() {
            let l0 = &session.layers[0];
            eprintln!("ds4: layer0 weights: attn_norm={:#x} attn_q_a={:#x} attn_kv={:#x} hc_attn_fn={:#x}",
                l0.attn_norm, l0.attn_q_a, l0.attn_kv, l0.hc_attn_fn);
            eprintln!("ds4: layer0 hc_ffn_scale={:#x} hc_ffn_base={:#x} hc_ffn_fn={:#x}",
                l0.hc_ffn_scale, l0.hc_ffn_base, l0.hc_ffn_fn);
            eprintln!("ds4: layer0: ffn_gate_shexp={:#x} ffn_down_shexp={:#x} gate_type={} down_type={}",
                l0.ffn_gate_shexp, l0.ffn_down_shexp, l0.ffn_gate_type, l0.ffn_down_type);
        }
        eprintln!("ds4: Rust Metal session ready ({} layers, {} tensors)",
            session.layers.len(), session.graph.layers.len() * 7 + 50);

        Ok(Self { device, session })
    }

    pub fn decode(&mut self, token: i32) -> Result<&[f32]> {
        decode::decode_token(&mut self.session, token)
    }

    pub fn argmax(logits: &[f32]) -> i32 {
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .map(|(i, _)| i as i32).unwrap_or(0)
    }

    pub fn sample(logits: &[f32], temperature: f32, _top_p: f32, min_p: f32, rng: &mut u64) -> i32 {
        let n = logits.len();
        if temperature < 0.001 { return Self::argmax(logits); }
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_threshold = max_logit + (min_p as f64).ln() as f32;
        let mut probs: Vec<f32> = logits.iter()
            .map(|&l| if l >= min_threshold { ((l - max_logit) / temperature).exp() } else { 0.0 })
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 { return Self::argmax(logits); }
        for p in &mut probs { *p /= sum; }
        let r = random_f32(rng);
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum { return i as i32; }
        }
        n as i32 - 1
    }
}

fn random_f32(rng: &mut u64) -> f32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    (*rng as f32) / (u64::MAX as f32)
}
