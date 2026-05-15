use crate::model::*;
use crate::graph::{GpuGraph, memory_estimate, eval_token_decode, eval_prefill};
use crate::gguf::{GgufModel, N_VOCAB, N_LAYER};
use crate::model_view::ModelViews;
use crate::weights;
use crate::tokenizer;
use std::sync::{Arc, Mutex};
use memmap2::Mmap;
use objc2_metal::MTLBuffer;

pub static SESSION: std::sync::OnceLock<Mutex<SessionState>> = std::sync::OnceLock::new();

pub struct SessionState {
    pub graph: Option<GpuGraph>,
    pub logits: Vec<f32>,
    pub checkpoint: Vec<i32>,
    pub n_pos: u32,
    pub prefill_cap: u32, pub raw_cap: u32, pub comp_cap: u32,
    pub model_views: Option<ModelViews>,
    pub weights: Option<EngineWeights>,
    pub vocab: Option<Vocab>,
    pub model_map: Option<Arc<Mmap>>,
    pub model_size: u64,
    pub stop_tokens: Vec<i32>,
}

impl SessionState {
    pub fn new(ctx_size: u32) -> Self {
        let (rc, cc, _) = memory_estimate(ctx_size);
        SessionState {
            graph: None, logits: vec![0.0; N_VOCAB as usize],
            checkpoint: Vec::new(), n_pos: 0,
            prefill_cap: 2048, raw_cap: rc, comp_cap: cc,
            model_views: None, weights: None, vocab: None,
            model_map: None, model_size: 0,
            stop_tokens: vec![2], // default EOS
        }
    }

    pub fn from_model(model: &GgufModel) -> Result<Self, &'static str> {
        let ctx_size = 65536u32;
        let (rc, cc, _) = memory_estimate(ctx_size);
        let w = weights::weights_bind(model);
        let views = ModelViews::new(model)?;
        Ok(SessionState {
            graph: Some(GpuGraph::allocate(ctx_size)?),
            logits: vec![0.0; N_VOCAB as usize],
            checkpoint: Vec::new(), n_pos: 0,
            prefill_cap: 2048, raw_cap: rc, comp_cap: cc,
            model_views: Some(views), weights: Some(w),
            vocab: Some(tokenizer::vocab_load(model)),
            model_map: Some(model.map.clone()), // Arc::clone, cheap
            model_size: model.size,
            stop_tokens: vec![2],
        })
    }

    pub fn common_prefix(&self, prompt: &[i32]) -> usize {
        let ck = &self.checkpoint;
        let n = ck.len().min(prompt.len());
        for i in 0..n { if ck[i] != prompt[i] { return i; } }
        n
    }

    pub fn sync(&mut self, prompt: &[i32]) -> Result<(), &'static str> {
        let common = self.common_prefix(prompt);
        let new_tokens = &prompt[common..];
        if new_tokens.is_empty() { return Ok(()); }

        if common > 0 && common == self.checkpoint.len() {
            if let Some(ref mut graph) = self.graph {
                graph.n_pos = common as u32;
                graph.n_raw = common as u32;
            }
            self.checkpoint.truncate(common);
        } else {
            if let (Some(ref w), Some(ref v)) =
                (&self.weights, &self.model_views)
            {
                let mut g = GpuGraph::allocate(65536)?;
                let short = &prompt[..common.min(prompt.len())];
                eval_prefill(&mut g, short, w, v)?;
                self.graph = Some(g);
            }
            self.checkpoint.clear();
            self.checkpoint.extend_from_slice(&prompt[..common]);
        }
        self.prefill(new_tokens)
    }

    pub fn prefill(&mut self, tokens: &[i32]) -> Result<(), &'static str> {
        if let (Some(ref mut graph), Some(ref weights), Some(ref views)) =
            (&mut self.graph, &self.weights, &self.model_views)
        {
            eval_prefill(graph, tokens, weights, views)?;
            self.n_pos = graph.n_pos;
        } else {
            self.n_pos += tokens.len() as u32;
        }
        self.checkpoint.extend_from_slice(tokens);
        Ok(())
    }

    pub fn decode(&mut self, token: i32) -> Result<(), &'static str> {
        if let (Some(ref mut graph), Some(ref weights), Some(ref views)) =
            (&mut self.graph, &self.weights, &self.model_views)
        {
            eval_token_decode(graph, token, weights, views)?;
        }
        self.checkpoint.push(token);
        self.n_pos += 1;
        Ok(())
    }

    pub fn get_logits(&self) -> &[f32] {
        if let Some(ref graph) = self.graph {
            if let Some(buf) = graph.logits.buf_ref() {
                let ptr = buf.contents().as_ptr() as *const f32;
                let n = N_VOCAB as usize;
                return unsafe { std::slice::from_raw_parts(ptr, n) };
            }
        }
        &self.logits
    }

    fn rand_u32() -> u32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let mut state = t.as_nanos() as u32;
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        state
    }

    pub fn sample(&self, temperature: f32, top_k: usize) -> i32 {
        let logits = self.get_logits();
        if temperature < 0.01 {
            let mut best = 0i32;
            let mut best_score = std::f32::MIN;
            for (i, &s) in logits.iter().enumerate() {
                if s > best_score { best_score = s; best = i as i32; }
            }
            return best;
        }
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let mut candidates: Vec<(i32, f32)> = scaled.iter().copied()
            .enumerate().map(|(i, s)| (i as i32, s)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(candidates.len());
        candidates.truncate(k);
        let max_logit = candidates.iter().map(|(_, s)| *s).fold(std::f32::MIN, f32::max);
        let mut sum = 0.0f32;
        let probs: Vec<(i32, f32)> = candidates.iter().map(|&(id, s)| {
            let p = (s - max_logit).exp(); sum += p; (id, p)
        }).collect();
        let r = (Self::rand_u32() as f32 / u32::MAX as f32) * sum;
        let mut cum = 0.0;
        for (id, p) in &probs { cum += p; if cum >= r { return *id; } }
        probs.last().map(|(id, _)| *id).unwrap_or(0)
    }

    pub fn argmax(&self) -> (i32, f32) {
        let logits = self.get_logits();
        let mut best = 0i32;
        let mut best_score = std::f32::MIN;
        for (i, &s) in logits.iter().enumerate() {
            if s > best_score { best_score = s; best = i as i32; }
        }
        (best, best_score)
    }

    pub fn top_logprobs(&self, k: usize) -> Vec<(i32, f32)> {
        let logits = self.get_logits();
        let mut candidates: Vec<(i32, f32)> = logits.iter().copied()
            .enumerate().map(|(i, s)| (i as i32, s)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    pub fn is_stop_token(&self, token: i32) -> bool {
        self.stop_tokens.contains(&token) || token == self.vocab.as_ref().map_or(2, |v| v.eos_id)
    }

    pub fn generate_with_stops(&mut self, prompt: &[i32], max_new: usize,
        stop_strs: &[String], temperature: f32, top_k: usize,
    ) -> Vec<i32> {
        if let Err(e) = self.prefill(prompt) {
            eprintln!("prefill error: {}", e);
            return vec![];
        }
        let mut out = Vec::new();
        let mut text_buf = String::new();
        for _ in 0..max_new {
            let token = self.sample(temperature, top_k);
            if self.is_stop_token(token) { break; }
            out.push(token);

            // Check string-based stop sequences
            if !stop_strs.is_empty() {
                if let Some(ref vocab) = self.vocab {
                    text_buf.push_str(&crate::tokenizer::token_decode(vocab, &[token]));
                    for stop in stop_strs {
                        if text_buf.contains(stop.as_str()) {
                            // Trim the stop sequence from output
                            if let Some(pos) = text_buf.find(stop.as_str()) {
                                text_buf.truncate(pos);
                            }
                            // Remove the stop-triggering tokens
                            while !out.is_empty() {
                                let tk = out.pop().unwrap();
                                let tk_str = crate::tokenizer::token_decode(vocab, &[tk]);
                                if tk_str.contains(stop.as_str()) {
                                    break;
                                }
                            }
                            return out;
                        }
                    }
                }
            }

            if let Err(e) = self.decode(token) {
                eprintln!("decode error: {}", e);
                break;
            }
        }
        out
    }

    pub fn generate(&mut self, prompt: &[i32], max_new: usize) -> Vec<i32> {
        self.generate_with_stops(prompt, max_new, &[], 0.0, 1)
    }

    pub fn generate_stream<F: FnMut(i32)>(&mut self, prompt: &[i32], max_new: usize, mut emit: F) {
        if let Err(e) = self.prefill(prompt) {
            eprintln!("prefill error: {}", e);
            return;
        }
        for _ in 0..max_new {
            let (token, _) = self.argmax();
            if self.is_stop_token(token) { break; }
            emit(token);
            if let Err(e) = self.decode(token) {
                eprintln!("decode error: {}", e);
                break;
            }
        }
    }

    pub fn save_snapshot(&self) -> Result<Vec<u8>, &'static str> {
        if let Some(ref graph) = self.graph {
            crate::graph::save_snapshot(graph)
        } else {
            Err("no graph")
        }
    }

    pub fn load_snapshot(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if let Some(ref mut graph) = self.graph {
            crate::graph::load_snapshot(graph, data)?;
            self.n_pos = graph.n_pos;
            Ok(())
        } else {
            Err("no graph")
        }
    }

    pub fn mtp_draft(&mut self, n_draft: usize) -> Vec<i32> {
        // MTP speculative decoding: generates draft tokens from the MTP head.
        // The MTP head is a small transformer that predicts the next N tokens.
        // This is a simplified implementation: without a loaded MTP model,
        // we just return argmax tokens from the main model.
        let mut drafts = Vec::new();
        if self.weights.as_ref().map_or(true, |w| w.token_embd.is_none()) {
            return drafts;
        }

        for _ in 0..n_draft {
            let (token, _) = self.argmax();
            if self.is_stop_token(token) { break; }
            drafts.push(token);
            if let Err(e) = self.decode(token) {
                eprintln!("mtp draft decode error: {}", e);
                break;
            }
        }
        drafts
    }

    pub fn generate_speculative(&mut self, prompt: &[i32], max_new: usize,
        n_draft: usize, temperature: f32, top_k: usize,
    ) -> Vec<i32> {
        if let Err(e) = self.prefill(prompt) {
            eprintln!("prefill error: {}", e);
            return vec![];
        }
        let mut out = Vec::new();
        let mut buffer = Vec::new();

        while out.len() < max_new {
            // Generate draft tokens
            buffer.clear();
            let n = n_draft.min(max_new - out.len());
            for _ in 0..n {
                let tok = self.sample(temperature, top_k);
                if self.is_stop_token(tok) { buffer.push(tok); break; }
                buffer.push(tok);
            }

            if buffer.is_empty() { break; }

            // Verify drafts: check that each draft matches the model's argmax
            // after having seen all previous drafts
            let mut accepted = 0usize;
            for &draft in &buffer {
                if self.is_stop_token(draft) {
                    out.push(draft);
                    accepted = buffer.len();
                    break;
                }
                let (expected, _) = self.argmax();
                if draft == expected {
                    // Accept draft
                    out.push(draft);
                    accepted += 1;
                    if let Err(e) = self.decode(draft) {
                        eprintln!("verify decode error: {}", e);
                        break;
                    }
                } else {
                    // Reject: use the model's own token instead
                    out.push(expected);
                    if let Err(e) = self.decode(expected) {
                        eprintln!("verify decode error: {}", e);
                    }
                    break;
                }
            }
            if accepted == buffer.len() && !buffer.is_empty() {
                // All accepted: generate one more token to maintain spec window
                let tok = self.sample(temperature, top_k);
                if !self.is_stop_token(tok) {
                    out.push(tok);
                    if let Err(e) = self.decode(tok) {
                        eprintln!("extra decode error: {}", e);
                        break;
                    }
                }
            }
            if out.last().map_or(false, |&t| self.is_stop_token(t)) { break; }
        }
        out.truncate(max_new);
        out
    }

    pub fn reset_cache(&mut self) {
        if let Some(ref mut graph) = self.graph {
            graph.n_pos = 0;
            graph.n_raw = 0;
            graph.n_comp = [0u32; N_LAYER as usize];
        }
        self.checkpoint.clear();
        self.n_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let s = SessionState::new(65536);
        assert!(s.graph.is_none());
        assert_eq!(s.checkpoint.len(), 0);
        assert_eq!(s.stop_tokens, vec![2]);
    }

    #[test]
    fn test_common_prefix() {
        let mut s = SessionState::new(65536);
        s.checkpoint = vec![1, 2, 3, 4, 5];
        assert_eq!(s.common_prefix(&[1, 2, 3]), 3);
        assert_eq!(s.common_prefix(&[1, 2, 3, 4, 5, 6]), 5);
        assert_eq!(s.common_prefix(&[0, 1, 2]), 0);
        assert_eq!(s.common_prefix(&[]), 0);
    }

    #[test]
    fn test_stop_token() {
        let mut s = SessionState::new(65536);
        s.stop_tokens = vec![2, 13];
        assert!(s.is_stop_token(2));
        assert!(s.is_stop_token(13));
        assert!(!s.is_stop_token(0));
        assert!(!s.is_stop_token(42));
    }

    #[test]
    fn test_argmax_zeros() {
        let s = SessionState::new(65536);
        let (token, _) = s.argmax();
        assert_eq!(token, 0);
    }
}
