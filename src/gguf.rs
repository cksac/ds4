use anyhow::{bail, Context, Result};
use libc::{mmap, munmap, MAP_FAILED, MAP_PRIVATE, MAP_SHARED, PROT_READ};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::ptr;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
const GGUF_MAX_DIMS: u32 = 8;
const GGUF_MIN_FILE_SIZE: u64 = 32;
const DS4_N_LAYER: u32 = 43;
const DS4_N_EMBD: u32 = 4096;
const DS4_N_VOCAB: u32 = 129280;
const DS4_N_HEAD: u32 = 64;
const DS4_N_HEAD_KV: u32 = 1;
const DS4_N_HEAD_DIM: u32 = 512;
const DS4_N_VALUE_DIM: u32 = 512;
const DS4_N_ROT: u32 = 64;
const DS4_N_OUT_GROUP: u32 = 8;
const DS4_N_LORA_Q: u32 = 1024;
const DS4_N_LORA_O: u32 = 1024;
const DS4_N_EXPERT: u32 = 256;
const DS4_N_EXPERT_USED: u32 = 6;
const DS4_N_EXPERT_SHARED: u32 = 1;
const DS4_N_FF_EXP: u32 = 2048;
const DS4_N_HASH_LAYER: u32 = 3;
const DS4_N_SWA: u32 = 128;
const DS4_N_INDEXER_HEAD: u32 = 64;
const DS4_N_INDEXER_HEAD_DIM: u32 = 128;
const DS4_N_INDEXER_TOP_K: u32 = 512;
const DS4_N_HC: u32 = 4;
const DS4_N_HC_SINKHORN_ITER: u32 = 20;
const DS4_RMS_EPS: f32 = 1.0e-6;
const DS4_HC_EPS: f32 = 1.0e-6;
const DS4_EXPERT_WEIGHT_SCALE: f32 = 1.5;
const DS4_SWIGLU_CLAMP_EXP: f32 = 10.0;
const DS4_ROPE_FREQ_BASE: f32 = 10000.0;
const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
const DS4_COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
const DS4_ROPE_ORIG_CTX: u64 = 65536;

const GGUF_VALUE_UINT8: u32 = 0;
const GGUF_VALUE_INT8: u32 = 1;
const GGUF_VALUE_UINT16: u32 = 2;
const GGUF_VALUE_INT16: u32 = 3;
const GGUF_VALUE_UINT32: u32 = 4;
const GGUF_VALUE_INT32: u32 = 5;
const GGUF_VALUE_FLOAT32: u32 = 6;
const GGUF_VALUE_BOOL: u32 = 7;
const GGUF_VALUE_STRING: u32 = 8;
const GGUF_VALUE_ARRAY: u32 = 9;
const GGUF_VALUE_UINT64: u32 = 10;
const GGUF_VALUE_INT64: u32 = 11;
const GGUF_VALUE_FLOAT64: u32 = 12;

const DS4_TENSOR_F32: u32 = 0;
const DS4_TENSOR_F16: u32 = 1;
const DS4_TENSOR_Q8_0: u32 = 8;
const DS4_TENSOR_Q2_K: u32 = 10;
const DS4_TENSOR_Q4_K: u32 = 12;
const DS4_TENSOR_IQ2_XXS: u32 = 16;
const DS4_TENSOR_I32: u32 = 26;

const FULLWIDTH_BAR_UTF8: [u8; 3] = [0xef, 0xbd, 0x9c];

#[derive(Debug, Clone)]
pub(crate) struct TokenizerMetadata {
    tokens: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, i32>,
    merge_rank: HashMap<Vec<u8>, i32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ModelSummary {
    name: Option<String>,
    architecture: Option<String>,
    version: u32,
    n_kv: u64,
    n_tensors: u64,
    file_size: u64,
    layers: Option<u32>,
    train_context: Option<u64>,
    attention_heads: Option<u32>,
    attention_kv_heads: Option<u32>,
    attention_head_dim: Option<u32>,
    attention_swa: Option<u32>,
    indexer_heads: Option<u32>,
    indexer_head_dim: Option<u32>,
    indexer_top_k: Option<u32>,
    expert_count: Option<u32>,
    expert_used_count: Option<u32>,
    expert_group_count: Option<u32>,
    expert_group_used_count: Option<u32>,
    tensor_bytes: u64,
    params: u64,
    tensor_types: BTreeMap<u32, TensorTypeSummary>,
}

#[derive(Debug)]
pub(crate) struct GgufMap {
    #[allow(dead_code)]
    file: File,
    map: *mut libc::c_void,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
struct TensorTypeSummary {
    count: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct TensorDirectoryEntry {
    rel_offset: u64,
    bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct TensorTypeInfo {
    name: &'static str,
    block_elems: u32,
    block_bytes: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct TensorDirectory {
    tensors: HashMap<String, TensorDescriptor>,
    tensor_data_pos: u64,
}

impl TensorDirectory {
    pub(crate) fn tensor_data_pos(&self) -> u64 {
        self.tensor_data_pos
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct TensorDescriptor {
    pub ndim: u32,
    pub dims: Vec<u64>,
    pub tensor_type: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct BoundTensor {
    pub name: String,
    pub descriptor: TensorDescriptor,
}

impl BoundTensor {
    fn expect_layout(&self, type_id: u32, dims: &[u64]) -> Result<()> {
        if self.descriptor.tensor_type != type_id {
            bail!(
                "ds4: tensor {} has type {}, expected {}",
                self.name,
                tensor_type_name(self.descriptor.tensor_type),
                tensor_type_name(type_id)
            )
        }
        if self.descriptor.ndim as usize != dims.len() {
            bail!(
                "ds4: tensor {} has {} dimensions, expected {}",
                self.name,
                self.descriptor.ndim,
                dims.len()
            )
        }
        for (index, want) in dims.iter().copied().enumerate() {
            let got = self.descriptor.dims[index];
            if got != want {
                bail!(
                    "ds4: tensor {} has dim[{}]={}, expected {}",
                    self.name,
                    index,
                    got,
                    want
                )
            }
        }
        Ok(())
    }

    fn expect_plain_layout(&self, dims: &[u64]) -> Result<()> {
        if self.descriptor.tensor_type != DS4_TENSOR_F16 && self.descriptor.tensor_type != DS4_TENSOR_F32 {
            bail!(
                "ds4: tensor {} has type {}, expected F16 or F32",
                self.name,
                tensor_type_name(self.descriptor.tensor_type)
            )
        }
        self.expect_layout(self.descriptor.tensor_type, dims)
    }

    fn expect_routed_expert(&self, dims: &[u64]) -> Result<()> {
        if !tensor_is_routed_expert_type(self.descriptor.tensor_type) {
            bail!(
                "ds4: tensor {} has type {} ({}), expected a routed expert quant type",
                self.name,
                self.descriptor.tensor_type,
                tensor_type_name(self.descriptor.tensor_type)
            )
        }
        if self.descriptor.ndim as usize != dims.len() {
            bail!(
                "ds4: tensor {} has {} dimensions, expected {}",
                self.name,
                self.descriptor.ndim,
                dims.len()
            )
        }
        for (index, want) in dims.iter().copied().enumerate() {
            let got = self.descriptor.dims[index];
            if got != want {
                bail!(
                    "ds4: tensor {} has dim[{}]={}, expected {}",
                    self.name,
                    index,
                    got,
                    want
                )
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn data<'a>(&self, gguf_bytes: &'a [u8]) -> Result<&'a [u8]> {
        let start = usize::try_from(self.descriptor.abs_offset)
            .context("tensor offset does not fit in usize")?;
        let len = usize::try_from(self.descriptor.bytes)
            .context("tensor length does not fit in usize")?;
        let end = start
            .checked_add(len)
            .with_context(|| format!("tensor {} range overflow", self.name))?;
        gguf_bytes
            .get(start..end)
            .with_context(|| format!("tensor {} points outside mapped GGUF", self.name))
    }

    pub(crate) fn read_f16_row(&self, gguf_bytes: &[u8], row_index: u64) -> Result<Vec<f32>> {
        if self.descriptor.tensor_type != DS4_TENSOR_F16 || self.descriptor.ndim != 2 {
            bail!("ds4: tensor {} is not a 2D F16 tensor", self.name)
        }

        let width = self.descriptor.dims[0];
        let rows = self.descriptor.dims[1];
        if row_index >= rows {
            bail!(
                "ds4: row {} is outside tensor {} with {} rows",
                row_index,
                self.name,
                rows
            )
        }

        let data = self.data(gguf_bytes)?;
        let row_bytes = usize::try_from(width)
            .context("tensor width does not fit in usize")?
            .checked_mul(2)
            .context("tensor row byte count overflow")?;
        let start = usize::try_from(row_index)
            .context("tensor row index does not fit in usize")?
            .checked_mul(row_bytes)
            .context("tensor row offset overflow")?;
        let end = start
            .checked_add(row_bytes)
            .context("tensor row range overflow")?;
        let row = data
            .get(start..end)
            .with_context(|| format!("tensor {} row {} points outside mapped data", self.name, row_index))?;

        let mut out = Vec::with_capacity(usize::try_from(width).unwrap_or_default());
        for chunk in row.chunks_exact(2) {
            out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
        }
        Ok(out)
    }

    pub(crate) fn read_f32_values(&self, gguf_bytes: &[u8]) -> Result<Vec<f32>> {
        if self.descriptor.tensor_type != DS4_TENSOR_F32 {
            bail!("ds4: tensor {} is not an F32 tensor", self.name)
        }

        let data = self.data(gguf_bytes)?;
        if data.len() % 4 != 0 {
            bail!("ds4: tensor {} has a non-F32 byte length", self.name)
        }

        let mut out = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Ds4LayerTensorBindings {
    pub compress_ratio: u32,
    pub hc_attn_fn: BoundTensor,
    pub hc_attn_scale: BoundTensor,
    pub hc_attn_base: BoundTensor,
    pub attn_norm: BoundTensor,
    pub attn_q_a: BoundTensor,
    pub attn_q_a_norm: BoundTensor,
    pub attn_q_b: BoundTensor,
    pub attn_kv: BoundTensor,
    pub attn_kv_a_norm: BoundTensor,
    pub attn_sinks: BoundTensor,
    pub attn_output_a: BoundTensor,
    pub attn_output_b: BoundTensor,
    pub attn_compressor_ape: Option<BoundTensor>,
    pub attn_compressor_kv: Option<BoundTensor>,
    pub attn_compressor_gate: Option<BoundTensor>,
    pub attn_compressor_norm: Option<BoundTensor>,
    pub indexer_attn_q_b: Option<BoundTensor>,
    pub indexer_proj: Option<BoundTensor>,
    pub indexer_compressor_ape: Option<BoundTensor>,
    pub indexer_compressor_kv: Option<BoundTensor>,
    pub indexer_compressor_gate: Option<BoundTensor>,
    pub indexer_compressor_norm: Option<BoundTensor>,
    pub hc_ffn_fn: BoundTensor,
    pub hc_ffn_scale: BoundTensor,
    pub hc_ffn_base: BoundTensor,
    pub ffn_norm: BoundTensor,
    pub ffn_gate_tid2eid: Option<BoundTensor>,
    pub ffn_gate_inp: BoundTensor,
    pub ffn_exp_probs_b: Option<BoundTensor>,
    pub ffn_gate_exps: BoundTensor,
    pub ffn_up_exps: BoundTensor,
    pub ffn_down_exps: BoundTensor,
    pub ffn_gate_shexp: BoundTensor,
    pub ffn_up_shexp: BoundTensor,
    pub ffn_down_shexp: BoundTensor,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Ds4TensorBindings {
    pub token_embd: BoundTensor,
    pub output_hc_base: BoundTensor,
    pub output_hc_fn: BoundTensor,
    pub output_hc_scale: BoundTensor,
    pub output_norm: BoundTensor,
    pub output: BoundTensor,
    pub layers: Vec<Ds4LayerTensorBindings>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct Ds4MtpTensorBindings {
    pub e_proj: BoundTensor,
    pub h_proj: BoundTensor,
    pub enorm: BoundTensor,
    pub hnorm: BoundTensor,
    pub norm: BoundTensor,
    pub hc_head_base: BoundTensor,
    pub hc_head_fn: BoundTensor,
    pub hc_head_scale: BoundTensor,
    pub block: Ds4LayerTensorBindings,
}

#[derive(Debug, Default, Clone)]
struct ValidationMetadata {
    block_count: Option<u32>,
    embedding_length: Option<u32>,
    vocab_size: Option<u32>,
    attention_head_count: Option<u32>,
    attention_head_count_kv: Option<u32>,
    attention_key_length: Option<u32>,
    attention_value_length: Option<u32>,
    rope_dimension_count: Option<u32>,
    attention_q_lora_rank: Option<u32>,
    attention_output_lora_rank: Option<u32>,
    attention_output_group_count: Option<u32>,
    expert_count: Option<u32>,
    expert_used_count: Option<u32>,
    expert_feed_forward_length: Option<u32>,
    expert_shared_count: Option<u32>,
    hash_layer_count: Option<u32>,
    expert_group_count: Option<u32>,
    expert_group_used_count: Option<u32>,
    attention_sliding_window: Option<u32>,
    attention_indexer_head_count: Option<u32>,
    attention_indexer_key_length: Option<u32>,
    attention_indexer_top_k: Option<u32>,
    hyper_connection_count: Option<u32>,
    hyper_connection_sinkhorn_iterations: Option<u32>,
    rope_scaling_original_context_length: Option<u64>,
    rope_freq_base: Option<f32>,
    rope_scaling_factor: Option<f32>,
    rope_scaling_yarn_beta_fast: Option<f32>,
    rope_scaling_yarn_beta_slow: Option<f32>,
    attention_compress_rope_freq_base: Option<f32>,
    expert_weights_scale: Option<f32>,
    attention_layer_norm_rms_epsilon: Option<f32>,
    hyper_connection_epsilon: Option<f32>,
    expert_weights_norm: Option<bool>,
    compress_ratios: Option<Vec<u32>>,
    swiglu_clamp_exp: Option<Vec<f32>>,
}

impl TokenizerMetadata {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = std::io::Cursor::new(bytes);
        Self::from_reader(&mut reader).context("failed to load tokenizer metadata from mapped GGUF")
    }

    pub(crate) fn token_bytes(&self, token: i32) -> Vec<u8> {
        let Ok(index) = usize::try_from(token) else {
            return Vec::new();
        };
        let Some(raw) = self.tokens.get(index) else {
            return Vec::new();
        };

        decode_token_bytes(raw)
    }

    pub(crate) fn lookup_required(&self, text: &str) -> Result<i32> {
        self.token_to_id
            .get(text.as_bytes())
            .copied()
            .with_context(|| format!("required tokenizer token is missing: {text}"))
    }

    pub(crate) fn tokenize_text(&self, text: &str) -> Vec<i32> {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut pos = 0usize;
        let mut out = Vec::new();

        while pos < len {
            let start = pos;
            let c = bytes[pos];

            if ascii_digit(c) {
                let mut digits = 0;
                while pos < len && ascii_digit(bytes[pos]) && digits < 3 {
                    pos += 1;
                    digits += 1;
                }
            } else if joyai_cjk_at(bytes, pos) {
                loop {
                    pos = next_utf8_char(bytes, pos);
                    if pos >= len || !joyai_cjk_at(bytes, pos) {
                        break;
                    }
                }
            } else if joyai_ascii_punct_symbol(c)
                && pos + 1 < len
                && ascii_alpha(bytes[pos + 1])
            {
                pos += 1;
                while pos < len && ascii_alpha(bytes[pos]) {
                    pos += 1;
                }
            } else if joyai_letter_like_at(bytes, pos) {
                pos = joyai_consume_letters(bytes, pos);
            } else if !ascii_newline(c)
                && !joyai_ascii_punct_symbol(c)
                && pos + 1 < len
                && joyai_letter_like_at(bytes, pos + 1)
            {
                pos += 1;
                pos = joyai_consume_letters(bytes, pos);
            } else if c == b' ' && pos + 1 < len && joyai_ascii_punct_symbol(bytes[pos + 1]) {
                pos += 1;
                while pos < len && joyai_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < len && ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if joyai_ascii_punct_symbol(c) {
                while pos < len && joyai_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < len && ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if ascii_space(c) {
                let mut p = pos;
                let mut last_newline_end = None;
                while p < len && ascii_space(bytes[p]) {
                    let space = bytes[p];
                    p += 1;
                    if ascii_newline(space) {
                        last_newline_end = Some(p);
                    }
                }

                if let Some(newline_end) = last_newline_end {
                    pos = newline_end;
                } else if p < len
                    && p > pos + 1
                    && (joyai_letter_like_at(bytes, p) || joyai_ascii_punct_symbol(bytes[p]))
                {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(bytes, pos);
            }

            if pos == start {
                pos = next_utf8_char(bytes, pos);
            }
            self.bpe_emit_piece(&bytes[start..pos], &mut out);
        }

        out
    }

    fn from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            bail!("model is not a GGUF file")
        }

        let version = read_u32(reader)?;
        if version != GGUF_VERSION {
            bail!("only GGUF v3 is supported")
        }

        let _n_tensors = read_u64(reader)?;
        let n_kv = read_u64(reader)?;
        let mut tokens = None;
        let mut merges = None;

        for _ in 0..n_kv {
            let key = read_string_bytes(reader)?;
            let value_type = read_u32(reader)?;

            if key.as_slice() == b"tokenizer.ggml.tokens" {
                tokens = Some(read_string_array(reader, value_type)?);
            } else if key.as_slice() == b"tokenizer.ggml.merges" {
                merges = Some(read_string_array(reader, value_type)?);
            } else {
                skip_value(reader, value_type, 0)?;
            }

            if tokens.is_some() && merges.is_some() {
                break;
            }
        }

        let tokens = tokens.context("GGUF tokenizer token table is missing or invalid")?;
        let merges = merges.context("GGUF tokenizer merge table is missing or invalid")?;

        if tokens.len() > i32::MAX as usize {
            bail!("GGUF tokenizer token table is too large")
        }

        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(index, token)| (token.clone(), index as i32))
            .collect();
        let merge_rank = merges
            .into_iter()
            .enumerate()
            .map(|(index, merge)| (merge, index as i32))
            .collect();

        Ok(Self {
            tokens,
            token_to_id,
            merge_rank,
        })
    }

    fn bpe_emit_piece(&self, raw_piece: &[u8], out: &mut Vec<i32>) {
        if raw_piece.is_empty() {
            return;
        }

        let encoded = byte_encode(raw_piece);
        let mut symbols = Vec::new();
        let mut offset = 0usize;
        while offset < encoded.len() {
            let width = utf8_len_from_first_byte(encoded[offset]);
            let end = encoded.len().min(offset + width);
            symbols.push(encoded[offset..end].to_vec());
            offset = end;
        }

        loop {
            let mut best_index = None;
            let mut best_rank = i32::MAX;

            for index in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self.bpe_rank(&symbols[index], &symbols[index + 1]) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(index);
                    }
                }
            }

            let Some(best_index) = best_index else {
                break;
            };

            let mut merged = Vec::with_capacity(symbols[best_index].len() + symbols[best_index + 1].len());
            merged.extend_from_slice(&symbols[best_index]);
            merged.extend_from_slice(&symbols[best_index + 1]);
            symbols[best_index] = merged;
            symbols.remove(best_index + 1);
        }

        for symbol in symbols {
            if let Some(token) = self.token_to_id.get(symbol.as_slice()) {
                out.push(*token);
            } else {
                for byte in symbol {
                    if let Some(token) = self.token_to_id.get(&[byte][..]) {
                        out.push(*token);
                    }
                }
            }
        }
    }

    fn bpe_rank(&self, left: &[u8], right: &[u8]) -> Option<i32> {
        let mut key = Vec::with_capacity(left.len() + 1 + right.len());
        key.extend_from_slice(left);
        key.push(b' ');
        key.extend_from_slice(right);
        self.merge_rank.get(key.as_slice()).copied()
    }
}

impl ModelSummary {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = std::io::Cursor::new(bytes);
        Self::from_reader(&mut reader, bytes.len() as u64)
            .context("failed to load GGUF summary from mapped bytes")
    }

    pub(crate) fn print(&self) {
        println!("model: {}", self.name.as_deref().unwrap_or(""));
        println!("arch:  {}", self.architecture.as_deref().unwrap_or(""));
        println!(
            "gguf:  v{}, {} metadata keys, {} tensors",
            self.version, self.n_kv, self.n_tensors
        );

        if self.layers.unwrap_or(0) != 0 {
            println!("layers: {}", self.layers.unwrap_or(0));
        }
        if self.train_context.unwrap_or(0) != 0 {
            println!("train context: {}", self.train_context.unwrap_or(0));
        }

        let n_head = self.attention_heads.unwrap_or(0);
        let n_head_kv = self.attention_kv_heads.unwrap_or(0);
        let head_dim = self.attention_head_dim.unwrap_or(0);
        let n_swa = self.attention_swa.unwrap_or(0);
        if n_head != 0 || n_head_kv != 0 || head_dim != 0 || n_swa != 0 {
            println!(
                "attention: heads={} kv_heads={} head_dim={} swa={}",
                n_head, n_head_kv, head_dim, n_swa
            );
        }

        let indexer_heads = self.indexer_heads.unwrap_or(0);
        let indexer_head_dim = self.indexer_head_dim.unwrap_or(0);
        let indexer_top_k = self.indexer_top_k.unwrap_or(0);
        if indexer_heads != 0 || indexer_head_dim != 0 || indexer_top_k != 0 {
            println!(
                "indexer: heads={} head_dim={} top_k={}",
                indexer_heads, indexer_head_dim, indexer_top_k
            );
        }

        let n_expert = self.expert_count.unwrap_or(0);
        let n_expert_used = self.expert_used_count.unwrap_or(0);
        let n_expert_groups = self.expert_group_count.unwrap_or(0);
        let n_group_used = self.expert_group_used_count.unwrap_or(0);
        if n_expert != 0 || n_expert_used != 0 || n_expert_groups != 0 || n_group_used != 0 {
            println!(
                "experts: count={} used={} groups={} groups_used={}",
                n_expert, n_expert_used, n_expert_groups, n_group_used
            );
        }

        println!("file size: {}", format_gib(self.file_size));
        println!("tensor bytes described by GGUF: {}", format_gib(self.tensor_bytes));
        println!("logical parameters: {:.2} B", self.params as f64 / 1_000_000_000.0);
        println!("tensor types:");
        for (type_id, stats) in &self.tensor_types {
            let Some(info) = tensor_type_info(*type_id) else {
                continue;
            };
            println!(
                "  {:<8} {:>5} tensors, {}",
                info.name,
                stats.count,
                format_gib(stats.bytes)
            );
        }
    }

    fn from_reader<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<Self> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            bail!("model is not a GGUF file")
        }

        let version = read_u32(reader)?;
        if version != GGUF_VERSION {
            bail!("only GGUF v3 is supported")
        }

        let n_tensors = read_u64(reader)?;
        let n_kv = read_u64(reader)?;
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        let mut summary = Self {
            name: None,
            architecture: None,
            version,
            n_kv,
            n_tensors,
            file_size,
            layers: None,
            train_context: None,
            attention_heads: None,
            attention_kv_heads: None,
            attention_head_dim: None,
            attention_swa: None,
            indexer_heads: None,
            indexer_head_dim: None,
            indexer_top_k: None,
            expert_count: None,
            expert_used_count: None,
            expert_group_count: None,
            expert_group_used_count: None,
            tensor_bytes: 0,
            params: 0,
            tensor_types: BTreeMap::new(),
        };

        for _ in 0..n_kv {
            let key = read_string_bytes(reader)?;
            let value_type = read_u32(reader)?;

            match key.as_slice() {
                b"general.alignment" if value_type == GGUF_VALUE_UINT32 => {
                    let value = read_u32(reader)?;
                    if value != 0 {
                        alignment = value;
                    }
                }
                b"general.name" if value_type == GGUF_VALUE_STRING => {
                    summary.name = Some(read_string_lossy(reader)?);
                }
                b"general.architecture" if value_type == GGUF_VALUE_STRING => {
                    summary.architecture = Some(read_string_lossy(reader)?);
                }
                b"deepseek4.block_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.layers = Some(read_u32(reader)?);
                }
                b"deepseek4.context_length" if value_type == GGUF_VALUE_UINT64 => {
                    summary.train_context = Some(read_u64(reader)?);
                }
                b"deepseek4.attention.head_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.attention_heads = Some(read_u32(reader)?);
                }
                b"deepseek4.attention.head_count_kv" if value_type == GGUF_VALUE_UINT32 => {
                    summary.attention_kv_heads = Some(read_u32(reader)?);
                }
                b"deepseek4.attention.key_length" if value_type == GGUF_VALUE_UINT32 => {
                    summary.attention_head_dim = Some(read_u32(reader)?);
                }
                b"deepseek4.attention.sliding_window" if value_type == GGUF_VALUE_UINT32 => {
                    summary.attention_swa = Some(read_u32(reader)?);
                }
                b"deepseek4.attention.indexer.head_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.indexer_heads = Some(read_u32(reader)?);
                }
                b"deepseek4.attention.indexer.key_length" if value_type == GGUF_VALUE_UINT32 => {
                    summary.indexer_head_dim = Some(read_u32(reader)?);
                }
                b"deepseek4.attention.indexer.top_k" if value_type == GGUF_VALUE_UINT32 => {
                    summary.indexer_top_k = Some(read_u32(reader)?);
                }
                b"deepseek4.expert_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.expert_count = Some(read_u32(reader)?);
                }
                b"deepseek4.expert_used_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.expert_used_count = Some(read_u32(reader)?);
                }
                b"deepseek4.expert_group_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.expert_group_count = Some(read_u32(reader)?);
                }
                b"deepseek4.expert_group_used_count" if value_type == GGUF_VALUE_UINT32 => {
                    summary.expert_group_used_count = Some(read_u32(reader)?);
                }
                _ => skip_value(reader, value_type, 0)?,
            }
        }

        parse_tensor_directory(reader, &mut summary, alignment)?;
        Ok(summary)
    }
}

pub(crate) fn validate_model_config_bytes(bytes: &[u8]) -> Result<()> {
    ValidationMetadata::from_bytes(bytes)?.validate()
}

impl TensorDirectory {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = std::io::Cursor::new(bytes);
        Self::from_reader(&mut reader, bytes.len() as u64)
            .context("failed to load GGUF tensor directory from mapped bytes")
    }

    fn from_reader<R: Read + Seek>(reader: &mut R, file_size: u64) -> Result<Self> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            bail!("model is not a GGUF file")
        }

        let version = read_u32(reader)?;
        if version != GGUF_VERSION {
            bail!("only GGUF v3 is supported")
        }

        let n_tensors = read_u64(reader)?;
        let n_kv = read_u64(reader)?;
        let mut alignment = GGUF_DEFAULT_ALIGNMENT;

        for _ in 0..n_kv {
            let key = read_string_bytes(reader)?;
            let value_type = read_u32(reader)?;
            if key.as_slice() == b"general.alignment" && value_type == GGUF_VALUE_UINT32 {
                let value = read_u32(reader)?;
                if value != 0 {
                    alignment = value;
                }
            } else {
                skip_value(reader, value_type, 0)?;
            }
        }

        let mut pending = Vec::new();
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name_bytes = read_string_bytes(reader)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();
            let ndim = read_u32(reader)?;
            if ndim == 0 || ndim > GGUF_MAX_DIMS {
                bail!("tensor has an unsupported number of dimensions")
            }

            let mut dims = Vec::with_capacity(ndim as usize);
            let mut elements = 1u64;
            for _ in 0..ndim {
                let dim = read_u64(reader)?;
                if dim != 0 && elements > u64::MAX / dim {
                    bail!("tensor element count overflow")
                }
                elements *= dim;
                dims.push(dim);
            }

            let tensor_type = read_u32(reader)?;
            let rel_offset = read_u64(reader)?;
            let bytes = match tensor_nbytes(tensor_type, elements) {
                Some(bytes) => bytes,
                None => {
                    eprintln!(
                        "ds4: warning: tensor {} has unsupported GGUF type {}",
                        name,
                        tensor_type
                    );
                    0
                }
            };

            if tensors.contains_key(&name) {
                bail!("duplicate tensor name in GGUF: {name}")
            }
            tensors.insert(
                name,
                TensorDescriptor {
                    ndim,
                    dims,
                    tensor_type,
                    rel_offset,
                    abs_offset: 0,
                    elements,
                    bytes,
                },
            );
            pending.push(rel_offset);
        }

        let tensor_data_pos = align_up(reader.stream_position()?, alignment as u64);
        for rel_offset in pending {
            if rel_offset > u64::MAX - tensor_data_pos {
                bail!("tensor offset overflow")
            }
        }

        for descriptor in tensors.values_mut() {
            descriptor.abs_offset = tensor_data_pos + descriptor.rel_offset;
            if descriptor.bytes != 0
                && (descriptor.abs_offset > file_size
                    || descriptor.bytes > file_size - descriptor.abs_offset)
            {
                bail!("tensor points outside GGUF file")
            }
        }

        Ok(Self {
            tensors,
            tensor_data_pos,
        })
    }

    pub(crate) fn tensor(&self, name: &str) -> Option<BoundTensor> {
        self.tensors.get(name).cloned().map(|descriptor| BoundTensor {
            name: name.to_owned(),
            descriptor,
        })
    }

    pub(crate) fn require_tensor(&self, name: &str) -> Result<BoundTensor> {
        self.tensor(name)
            .with_context(|| format!("ds4: required tensor is missing: {name}"))
    }

    pub(crate) fn bind_ds4_tensors(&self) -> Result<Ds4TensorBindings> {
        let mut layers = Vec::with_capacity(DS4_N_LAYER as usize);
        for layer in 0..DS4_N_LAYER {
            let prefix = format!("blk.{layer}");
            layers.push(bind_layer_tensors(
                self,
                prefix.as_str(),
                layer_compress_ratio(layer),
                false,
                layer < DS4_N_HASH_LAYER,
            )?);
        }

        let bindings = Ds4TensorBindings {
            token_embd: self.require_tensor("token_embd.weight")?,
            output_hc_base: self.require_tensor("output_hc_base.weight")?,
            output_hc_fn: self.require_tensor("output_hc_fn.weight")?,
            output_hc_scale: self.require_tensor("output_hc_scale.weight")?,
            output_norm: self.require_tensor("output_norm.weight")?,
            output: self.require_tensor("output.weight")?,
            layers,
        };
        validate_ds4_tensor_bindings(&bindings)?;
        Ok(bindings)
    }

    pub(crate) fn bind_ds4_mtp_tensors(&self) -> Result<Ds4MtpTensorBindings> {
        let bindings = Ds4MtpTensorBindings {
            hc_head_base: self.require_tensor("mtp.0.hc_head_base.weight")?,
            hc_head_fn: self.require_tensor("mtp.0.hc_head_fn.weight")?,
            hc_head_scale: self.require_tensor("mtp.0.hc_head_scale.weight")?,
            e_proj: self.require_tensor("mtp.0.e_proj.weight")?,
            h_proj: self.require_tensor("mtp.0.h_proj.weight")?,
            enorm: self.require_tensor("mtp.0.enorm.weight")?,
            hnorm: self.require_tensor("mtp.0.hnorm.weight")?,
            norm: self.require_tensor("mtp.0.norm.weight")?,
            block: bind_layer_tensors(self, "mtp.0", 0, true, false)?,
        };
        validate_ds4_mtp_tensor_bindings(&bindings)?;
        Ok(bindings)
    }
}

fn bind_layer_tensors(
    directory: &TensorDirectory,
    prefix: &str,
    compress_ratio: u32,
    require_exp_probs_bias: bool,
    include_hash_tid2eid: bool,
) -> Result<Ds4LayerTensorBindings> {
    let exp_probs_name = format!("{prefix}.exp_probs_b.bias");

    Ok(Ds4LayerTensorBindings {
        compress_ratio,
        hc_attn_fn: directory.require_tensor(&format!("{prefix}.hc_attn_fn.weight"))?,
        hc_attn_scale: directory.require_tensor(&format!("{prefix}.hc_attn_scale.weight"))?,
        hc_attn_base: directory.require_tensor(&format!("{prefix}.hc_attn_base.weight"))?,
        attn_norm: directory.require_tensor(&format!("{prefix}.attn_norm.weight"))?,
        attn_q_a: directory.require_tensor(&format!("{prefix}.attn_q_a.weight"))?,
        attn_q_a_norm: directory.require_tensor(&format!("{prefix}.attn_q_a_norm.weight"))?,
        attn_q_b: directory.require_tensor(&format!("{prefix}.attn_q_b.weight"))?,
        attn_kv: directory.require_tensor(&format!("{prefix}.attn_kv.weight"))?,
        attn_kv_a_norm: directory.require_tensor(&format!("{prefix}.attn_kv_a_norm.weight"))?,
        attn_sinks: directory.require_tensor(&format!("{prefix}.attn_sinks.weight"))?,
        attn_output_a: directory.require_tensor(&format!("{prefix}.attn_output_a.weight"))?,
        attn_output_b: directory.require_tensor(&format!("{prefix}.attn_output_b.weight"))?,
        attn_compressor_ape: if compress_ratio != 0 {
            Some(directory.require_tensor(&format!("{prefix}.attn_compressor_ape.weight"))?)
        } else {
            None
        },
        attn_compressor_kv: if compress_ratio != 0 {
            Some(directory.require_tensor(&format!("{prefix}.attn_compressor_kv.weight"))?)
        } else {
            None
        },
        attn_compressor_gate: if compress_ratio != 0 {
            Some(directory.require_tensor(&format!("{prefix}.attn_compressor_gate.weight"))?)
        } else {
            None
        },
        attn_compressor_norm: if compress_ratio != 0 {
            Some(directory.require_tensor(&format!("{prefix}.attn_compressor_norm.weight"))?)
        } else {
            None
        },
        indexer_attn_q_b: if compress_ratio == 4 {
            Some(directory.require_tensor(&format!("{prefix}.indexer.attn_q_b.weight"))?)
        } else {
            None
        },
        indexer_proj: if compress_ratio == 4 {
            Some(directory.require_tensor(&format!("{prefix}.indexer.proj.weight"))?)
        } else {
            None
        },
        indexer_compressor_ape: if compress_ratio == 4 {
            Some(directory.require_tensor(&format!("{prefix}.indexer_compressor_ape.weight"))?)
        } else {
            None
        },
        indexer_compressor_kv: if compress_ratio == 4 {
            Some(directory.require_tensor(&format!("{prefix}.indexer_compressor_kv.weight"))?)
        } else {
            None
        },
        indexer_compressor_gate: if compress_ratio == 4 {
            Some(directory.require_tensor(&format!("{prefix}.indexer_compressor_gate.weight"))?)
        } else {
            None
        },
        indexer_compressor_norm: if compress_ratio == 4 {
            Some(directory.require_tensor(&format!("{prefix}.indexer_compressor_norm.weight"))?)
        } else {
            None
        },
        hc_ffn_fn: directory.require_tensor(&format!("{prefix}.hc_ffn_fn.weight"))?,
        hc_ffn_scale: directory.require_tensor(&format!("{prefix}.hc_ffn_scale.weight"))?,
        hc_ffn_base: directory.require_tensor(&format!("{prefix}.hc_ffn_base.weight"))?,
        ffn_norm: directory.require_tensor(&format!("{prefix}.ffn_norm.weight"))?,
        ffn_gate_tid2eid: if include_hash_tid2eid {
            Some(directory.require_tensor(&format!("{prefix}.ffn_gate_tid2eid.weight"))?)
        } else {
            None
        },
        ffn_gate_inp: directory.require_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?,
        ffn_exp_probs_b: if require_exp_probs_bias {
            Some(directory.require_tensor(exp_probs_name.as_str())?)
        } else {
            directory.tensor(exp_probs_name.as_str())
        },
        ffn_gate_exps: directory.require_tensor(&format!("{prefix}.ffn_gate_exps.weight"))?,
        ffn_up_exps: directory.require_tensor(&format!("{prefix}.ffn_up_exps.weight"))?,
        ffn_down_exps: directory.require_tensor(&format!("{prefix}.ffn_down_exps.weight"))?,
        ffn_gate_shexp: directory.require_tensor(&format!("{prefix}.ffn_gate_shexp.weight"))?,
        ffn_up_shexp: directory.require_tensor(&format!("{prefix}.ffn_up_shexp.weight"))?,
        ffn_down_shexp: directory.require_tensor(&format!("{prefix}.ffn_down_shexp.weight"))?,
    })
}

fn validate_ds4_tensor_bindings(bindings: &Ds4TensorBindings) -> Result<()> {
    let hc_dim = u64::from(DS4_N_EMBD) * u64::from(DS4_N_HC);
    let hc_mix_dim = 2 * u64::from(DS4_N_HC) + u64::from(DS4_N_HC) * u64::from(DS4_N_HC);
    let q_dim = u64::from(DS4_N_HEAD) * u64::from(DS4_N_HEAD_DIM);
    let out_low_dim = u64::from(DS4_N_OUT_GROUP) * u64::from(DS4_N_LORA_O);
    let attn_output_a_dim = u64::from(DS4_N_HEAD_DIM) * (u64::from(DS4_N_HEAD) / u64::from(DS4_N_OUT_GROUP));

    bindings
        .token_embd
        .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_VOCAB)])?;
    bindings
        .output_hc_base
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HC)])?;
    bindings
        .output_hc_fn
        .expect_layout(DS4_TENSOR_F16, &[hc_dim, u64::from(DS4_N_HC)])?;
    bindings.output_hc_scale.expect_layout(DS4_TENSOR_F32, &[1])?;
    bindings
        .output_norm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
    bindings
        .output
        .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_VOCAB)])?;

    if bindings.layers.len() != DS4_N_LAYER as usize {
        bail!(
            "ds4: expected {} bound layers, got {}",
            DS4_N_LAYER,
            bindings.layers.len()
        )
    }

    for (layer_index, layer) in bindings.layers.iter().enumerate() {
        let expected_ratio = layer_compress_ratio(layer_index as u32);
        if layer.compress_ratio != expected_ratio {
            bail!(
                "ds4: layer {} bound with compression ratio {}, expected {}",
                layer_index,
                layer.compress_ratio,
                expected_ratio
            )
        }

        layer
            .hc_attn_fn
            .expect_layout(DS4_TENSOR_F16, &[hc_dim, hc_mix_dim])?;
        layer.hc_attn_scale.expect_layout(DS4_TENSOR_F32, &[3])?;
        layer
            .hc_attn_base
            .expect_layout(DS4_TENSOR_F32, &[hc_mix_dim])?;
        layer
            .attn_norm
            .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
        layer
            .attn_q_a
            .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_LORA_Q)])?;
        layer
            .attn_q_a_norm
            .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_LORA_Q)])?;
        layer
            .attn_q_b
            .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_LORA_Q), q_dim])?;
        layer
            .attn_kv
            .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_HEAD_DIM)])?;
        layer
            .attn_kv_a_norm
            .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HEAD_DIM)])?;
        layer
            .attn_sinks
            .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HEAD)])?;
        layer
            .attn_output_a
            .expect_layout(DS4_TENSOR_Q8_0, &[attn_output_a_dim, out_low_dim])?;
        layer
            .attn_output_b
            .expect_layout(DS4_TENSOR_Q8_0, &[out_low_dim, u64::from(DS4_N_EMBD)])?;

        if expected_ratio != 0 {
            let coeff = if expected_ratio == 4 { 2u64 } else { 1u64 };
            let comp_width = coeff * u64::from(DS4_N_HEAD_DIM);
            require_bound_tensor(
                &layer.attn_compressor_ape,
                "attention compressor APE",
            )?
            .expect_layout(DS4_TENSOR_F16, &[comp_width, u64::from(expected_ratio)])?;
            require_bound_tensor(
                &layer.attn_compressor_kv,
                "attention compressor KV",
            )?
            .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), comp_width])?;
            require_bound_tensor(
                &layer.attn_compressor_gate,
                "attention compressor gate",
            )?
            .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), comp_width])?;
            require_bound_tensor(
                &layer.attn_compressor_norm,
                "attention compressor norm",
            )?
            .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HEAD_DIM)])?;
        }

        if expected_ratio == 4 {
            let index_q_dim = u64::from(DS4_N_INDEXER_HEAD) * u64::from(DS4_N_INDEXER_HEAD_DIM);
            let index_width = 2 * u64::from(DS4_N_INDEXER_HEAD_DIM);
            require_bound_tensor(&layer.indexer_attn_q_b, "indexer attention q_b")?
                .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_LORA_Q), index_q_dim])?;
            require_bound_tensor(&layer.indexer_proj, "indexer projection")?
                .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_INDEXER_HEAD)])?;
            require_bound_tensor(&layer.indexer_compressor_ape, "indexer compressor APE")?
                .expect_layout(DS4_TENSOR_F16, &[index_width, u64::from(expected_ratio)])?;
            require_bound_tensor(&layer.indexer_compressor_kv, "indexer compressor KV")?
                .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), index_width])?;
            require_bound_tensor(&layer.indexer_compressor_gate, "indexer compressor gate")?
                .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), index_width])?;
            require_bound_tensor(&layer.indexer_compressor_norm, "indexer compressor norm")?
                .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_INDEXER_HEAD_DIM)])?;
        }

        layer
            .hc_ffn_fn
            .expect_layout(DS4_TENSOR_F16, &[hc_dim, hc_mix_dim])?;
        layer.hc_ffn_scale.expect_layout(DS4_TENSOR_F32, &[3])?;
        layer.hc_ffn_base.expect_layout(DS4_TENSOR_F32, &[hc_mix_dim])?;
        layer
            .ffn_norm
            .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
        layer
            .ffn_gate_inp
            .expect_layout(DS4_TENSOR_F16, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EXPERT)])?;
        if let Some(exp_probs_b) = &layer.ffn_exp_probs_b {
            exp_probs_b.expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EXPERT)])?;
        }
        layer.ffn_gate_exps.expect_routed_expert(&[
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EXPERT),
        ])?;
        layer.ffn_up_exps.expect_routed_expert(&[
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EXPERT),
        ])?;
        layer.ffn_down_exps.expect_routed_expert(&[
            u64::from(DS4_N_FF_EXP),
            u64::from(DS4_N_EMBD),
            u64::from(DS4_N_EXPERT),
        ])?;
        if layer.ffn_gate_exps.descriptor.tensor_type != layer.ffn_up_exps.descriptor.tensor_type {
            bail!(
                "ds4: routed gate/up experts use different quant types in layer {}",
                layer_index
            )
        }
        layer.ffn_gate_shexp.expect_layout(
            DS4_TENSOR_Q8_0,
            &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
        )?;
        layer.ffn_up_shexp.expect_layout(
            DS4_TENSOR_Q8_0,
            &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
        )?;
        layer.ffn_down_shexp.expect_layout(
            DS4_TENSOR_Q8_0,
            &[u64::from(DS4_N_FF_EXP), u64::from(DS4_N_EMBD)],
        )?;

        if layer_index < DS4_N_HASH_LAYER as usize {
            require_bound_tensor(&layer.ffn_gate_tid2eid, "hash-layer token-to-expert map")?
                .expect_layout(
                    DS4_TENSOR_I32,
                    &[u64::from(DS4_N_EXPERT_USED), u64::from(DS4_N_VOCAB)],
                )?;
        }
    }

    Ok(())
}

fn validate_ds4_mtp_tensor_bindings(bindings: &Ds4MtpTensorBindings) -> Result<()> {
    let hc_dim = u64::from(DS4_N_EMBD) * u64::from(DS4_N_HC);
    let hc_mix_dim = 2 * u64::from(DS4_N_HC) + u64::from(DS4_N_HC) * u64::from(DS4_N_HC);
    let q_dim = u64::from(DS4_N_HEAD) * u64::from(DS4_N_HEAD_DIM);
    let out_low_dim = u64::from(DS4_N_OUT_GROUP) * u64::from(DS4_N_LORA_O);
    let attn_output_a_dim = u64::from(DS4_N_HEAD_DIM) * (u64::from(DS4_N_HEAD) / u64::from(DS4_N_OUT_GROUP));
    let layer = &bindings.block;

    bindings
        .hc_head_base
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HC)])?;
    bindings
        .hc_head_fn
        .expect_plain_layout(&[hc_dim, u64::from(DS4_N_HC)])?;
    bindings.hc_head_scale.expect_layout(DS4_TENSOR_F32, &[1])?;
    bindings
        .e_proj
        .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EMBD)])?;
    bindings
        .h_proj
        .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_EMBD)])?;
    bindings
        .enorm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
    bindings
        .hnorm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
    bindings
        .norm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;

    layer.hc_attn_fn.expect_plain_layout(&[hc_dim, hc_mix_dim])?;
    layer.hc_attn_scale.expect_layout(DS4_TENSOR_F32, &[3])?;
    layer.hc_attn_base.expect_layout(DS4_TENSOR_F32, &[hc_mix_dim])?;
    layer
        .attn_norm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
    layer
        .attn_q_a
        .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_LORA_Q)])?;
    layer
        .attn_q_a_norm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_LORA_Q)])?;
    layer
        .attn_q_b
        .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_LORA_Q), q_dim])?;
    layer
        .attn_kv
        .expect_layout(DS4_TENSOR_Q8_0, &[u64::from(DS4_N_EMBD), u64::from(DS4_N_HEAD_DIM)])?;
    layer
        .attn_kv_a_norm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HEAD_DIM)])?;
    layer
        .attn_sinks
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_HEAD)])?;
    layer
        .attn_output_a
        .expect_layout(DS4_TENSOR_Q8_0, &[attn_output_a_dim, out_low_dim])?;
    layer
        .attn_output_b
        .expect_layout(DS4_TENSOR_Q8_0, &[out_low_dim, u64::from(DS4_N_EMBD)])?;

    layer.hc_ffn_fn.expect_plain_layout(&[hc_dim, hc_mix_dim])?;
    layer.hc_ffn_scale.expect_layout(DS4_TENSOR_F32, &[3])?;
    layer.hc_ffn_base.expect_layout(DS4_TENSOR_F32, &[hc_mix_dim])?;
    layer
        .ffn_norm
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EMBD)])?;
    layer
        .ffn_gate_inp
        .expect_plain_layout(&[u64::from(DS4_N_EMBD), u64::from(DS4_N_EXPERT)])?;
    require_bound_tensor(&layer.ffn_exp_probs_b, "MTP expert-probability bias")?
        .expect_layout(DS4_TENSOR_F32, &[u64::from(DS4_N_EXPERT)])?;
    layer.ffn_gate_exps.expect_routed_expert(&[
        u64::from(DS4_N_EMBD),
        u64::from(DS4_N_FF_EXP),
        u64::from(DS4_N_EXPERT),
    ])?;
    layer.ffn_up_exps.expect_routed_expert(&[
        u64::from(DS4_N_EMBD),
        u64::from(DS4_N_FF_EXP),
        u64::from(DS4_N_EXPERT),
    ])?;
    layer.ffn_down_exps.expect_routed_expert(&[
        u64::from(DS4_N_FF_EXP),
        u64::from(DS4_N_EMBD),
        u64::from(DS4_N_EXPERT),
    ])?;
    if layer.ffn_gate_exps.descriptor.tensor_type != layer.ffn_up_exps.descriptor.tensor_type {
        bail!("ds4: MTP routed gate/up experts use different quant types")
    }
    layer.ffn_gate_shexp.expect_layout(
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
    )?;
    layer.ffn_up_shexp.expect_layout(
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_EMBD), u64::from(DS4_N_FF_EXP)],
    )?;
    layer.ffn_down_shexp.expect_layout(
        DS4_TENSOR_Q8_0,
        &[u64::from(DS4_N_FF_EXP), u64::from(DS4_N_EMBD)],
    )?;

    Ok(())
}

fn require_bound_tensor<'a>(tensor: &'a Option<BoundTensor>, label: &str) -> Result<&'a BoundTensor> {
    tensor
        .as_ref()
        .with_context(|| format!("ds4: required tensor binding is missing: {label}"))
}

fn tensor_is_routed_expert_type(type_id: u32) -> bool {
    matches!(type_id, DS4_TENSOR_IQ2_XXS | DS4_TENSOR_Q2_K | DS4_TENSOR_Q4_K)
}

impl ValidationMetadata {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = std::io::Cursor::new(bytes);
        Self::from_reader(&mut reader).context("failed to validate mapped GGUF metadata")
    }

    fn from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            bail!("model is not a GGUF file")
        }

        let version = read_u32(reader)?;
        if version != GGUF_VERSION {
            bail!("only GGUF v3 is supported")
        }

        let _n_tensors = read_u64(reader)?;
        let n_kv = read_u64(reader)?;
        let mut metadata = Self::default();

        for _ in 0..n_kv {
            let key = read_string_bytes(reader)?;
            let value_type = read_u32(reader)?;

            match key.as_slice() {
                b"deepseek4.block_count" => {
                    metadata.block_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.embedding_length" => {
                    metadata.embedding_length = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.vocab_size" => {
                    metadata.vocab_size = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.head_count" => {
                    metadata.attention_head_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.head_count_kv" => {
                    metadata.attention_head_count_kv = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.key_length" => {
                    metadata.attention_key_length = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.value_length" => {
                    metadata.attention_value_length = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.rope.dimension_count" => {
                    metadata.rope_dimension_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.q_lora_rank" => {
                    metadata.attention_q_lora_rank = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.output_lora_rank" => {
                    metadata.attention_output_lora_rank = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.output_group_count" => {
                    metadata.attention_output_group_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.expert_count" => {
                    metadata.expert_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.expert_used_count" => {
                    metadata.expert_used_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.expert_feed_forward_length" => {
                    metadata.expert_feed_forward_length = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.expert_shared_count" => {
                    metadata.expert_shared_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.hash_layer_count" => {
                    metadata.hash_layer_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.expert_group_count" => {
                    metadata.expert_group_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.expert_group_used_count" => {
                    metadata.expert_group_used_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.sliding_window" => {
                    metadata.attention_sliding_window = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.indexer.head_count" => {
                    metadata.attention_indexer_head_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.indexer.key_length" => {
                    metadata.attention_indexer_key_length = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.attention.indexer.top_k" => {
                    metadata.attention_indexer_top_k = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.hyper_connection.count" => {
                    metadata.hyper_connection_count = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.hyper_connection.sinkhorn_iterations" => {
                    metadata.hyper_connection_sinkhorn_iterations = read_optional_u32(reader, value_type)?;
                }
                b"deepseek4.rope.scaling.original_context_length" => {
                    metadata.rope_scaling_original_context_length = read_optional_u64(reader, value_type)?;
                }
                b"deepseek4.rope.freq_base" => {
                    metadata.rope_freq_base = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.rope.scaling.factor" => {
                    metadata.rope_scaling_factor = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.rope.scaling.yarn_beta_fast" => {
                    metadata.rope_scaling_yarn_beta_fast = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.rope.scaling.yarn_beta_slow" => {
                    metadata.rope_scaling_yarn_beta_slow = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.attention.compress_rope_freq_base" => {
                    metadata.attention_compress_rope_freq_base = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.expert_weights_scale" => {
                    metadata.expert_weights_scale = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.attention.layer_norm_rms_epsilon" => {
                    metadata.attention_layer_norm_rms_epsilon = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.hyper_connection.epsilon" => {
                    metadata.hyper_connection_epsilon = read_optional_f32(reader, value_type)?;
                }
                b"deepseek4.expert_weights_norm" => {
                    metadata.expert_weights_norm = read_optional_bool(reader, value_type)?;
                }
                b"deepseek4.attention.compress_ratios" => {
                    metadata.compress_ratios = read_optional_u32_array(reader, value_type)?;
                }
                b"deepseek4.swiglu_clamp_exp" => {
                    metadata.swiglu_clamp_exp = read_optional_f32_array(reader, value_type)?;
                }
                _ => skip_value(reader, value_type, 0)?,
            }
        }

        Ok(metadata)
    }

    fn validate(&self) -> Result<()> {
        expect_u32(
            "embedding_length",
            required_u32(self.embedding_length, "deepseek4.embedding_length")?,
            DS4_N_EMBD,
        )?;
        expect_u32(
            "vocab_size",
            required_u32(self.vocab_size, "deepseek4.vocab_size")?,
            DS4_N_VOCAB,
        )?;
        expect_u32(
            "attention.head_count",
            required_u32(self.attention_head_count, "deepseek4.attention.head_count")?,
            DS4_N_HEAD,
        )?;
        expect_u32(
            "attention.key_length",
            required_u32(self.attention_key_length, "deepseek4.attention.key_length")?,
            DS4_N_HEAD_DIM,
        )?;
        expect_u32(
            "attention.head_count_kv",
            required_u32(self.attention_head_count_kv, "deepseek4.attention.head_count_kv")?,
            DS4_N_HEAD_KV,
        )?;
        expect_u32(
            "attention.value_length",
            required_u32(self.attention_value_length, "deepseek4.attention.value_length")?,
            DS4_N_VALUE_DIM,
        )?;
        expect_u32(
            "rope.dimension_count",
            required_u32(self.rope_dimension_count, "deepseek4.rope.dimension_count")?,
            DS4_N_ROT,
        )?;
        expect_u32(
            "attention.output_group_count",
            required_u32(
                self.attention_output_group_count,
                "deepseek4.attention.output_group_count",
            )?,
            DS4_N_OUT_GROUP,
        )?;
        expect_u32(
            "attention.q_lora_rank",
            required_u32(self.attention_q_lora_rank, "deepseek4.attention.q_lora_rank")?,
            DS4_N_LORA_Q,
        )?;
        expect_u32(
            "attention.output_lora_rank",
            required_u32(
                self.attention_output_lora_rank,
                "deepseek4.attention.output_lora_rank",
            )?,
            DS4_N_LORA_O,
        )?;
        expect_u32(
            "expert_count",
            required_u32(self.expert_count, "deepseek4.expert_count")?,
            DS4_N_EXPERT,
        )?;
        expect_u32(
            "expert_used_count",
            required_u32(self.expert_used_count, "deepseek4.expert_used_count")?,
            DS4_N_EXPERT_USED,
        )?;
        expect_u32(
            "expert_feed_forward_length",
            required_u32(
                self.expert_feed_forward_length,
                "deepseek4.expert_feed_forward_length",
            )?,
            DS4_N_FF_EXP,
        )?;
        expect_u32(
            "expert_shared_count",
            required_u32(self.expert_shared_count, "deepseek4.expert_shared_count")?,
            DS4_N_EXPERT_SHARED,
        )?;
        expect_u32(
            "hash_layer_count",
            required_u32(self.hash_layer_count, "deepseek4.hash_layer_count")?,
            DS4_N_HASH_LAYER,
        )?;
        expect_u32(
            "expert_group_count",
            self.expert_group_count.unwrap_or(0),
            0,
        )?;
        expect_u32(
            "expert_group_used_count",
            self.expert_group_used_count.unwrap_or(0),
            0,
        )?;
        expect_u32(
            "attention.sliding_window",
            required_u32(
                self.attention_sliding_window,
                "deepseek4.attention.sliding_window",
            )?,
            DS4_N_SWA,
        )?;
        expect_u32(
            "attention.indexer.head_count",
            required_u32(
                self.attention_indexer_head_count,
                "deepseek4.attention.indexer.head_count",
            )?,
            DS4_N_INDEXER_HEAD,
        )?;
        expect_u32(
            "attention.indexer.key_length",
            required_u32(
                self.attention_indexer_key_length,
                "deepseek4.attention.indexer.key_length",
            )?,
            DS4_N_INDEXER_HEAD_DIM,
        )?;
        expect_u32(
            "attention.indexer.top_k",
            required_u32(
                self.attention_indexer_top_k,
                "deepseek4.attention.indexer.top_k",
            )?,
            DS4_N_INDEXER_TOP_K,
        )?;
        expect_u32(
            "hyper_connection.count",
            required_u32(self.hyper_connection_count, "deepseek4.hyper_connection.count")?,
            DS4_N_HC,
        )?;
        expect_u32(
            "hyper_connection.sinkhorn_iterations",
            required_u32(
                self.hyper_connection_sinkhorn_iterations,
                "deepseek4.hyper_connection.sinkhorn_iterations",
            )?,
            DS4_N_HC_SINKHORN_ITER,
        )?;
        expect_u32(
            "block_count",
            required_u32(self.block_count, "deepseek4.block_count")?,
            DS4_N_LAYER,
        )?;

        validate_compress_ratios(self.compress_ratios.as_deref())?;
        validate_swiglu_clamp(self.swiglu_clamp_exp.as_deref())?;

        let rope_orig_ctx = required_u64(
            self.rope_scaling_original_context_length,
            "deepseek4.rope.scaling.original_context_length",
        )?;
        if rope_orig_ctx != DS4_ROPE_ORIG_CTX {
            bail!(
                "ds4: expected rope.scaling.original_context_length={} for DeepSeek4 Flash, got {}",
                DS4_ROPE_ORIG_CTX,
                rope_orig_ctx
            )
        }

        expect_f32(
            "rope.freq_base",
            required_f32(self.rope_freq_base, "deepseek4.rope.freq_base")?,
            DS4_ROPE_FREQ_BASE,
        )?;
        expect_f32(
            "rope.scaling.factor",
            required_f32(self.rope_scaling_factor, "deepseek4.rope.scaling.factor")?,
            DS4_ROPE_SCALE_FACTOR,
        )?;
        expect_f32(
            "rope.scaling.yarn_beta_fast",
            required_f32(
                self.rope_scaling_yarn_beta_fast,
                "deepseek4.rope.scaling.yarn_beta_fast",
            )?,
            DS4_ROPE_YARN_BETA_FAST,
        )?;
        expect_f32(
            "rope.scaling.yarn_beta_slow",
            required_f32(
                self.rope_scaling_yarn_beta_slow,
                "deepseek4.rope.scaling.yarn_beta_slow",
            )?,
            DS4_ROPE_YARN_BETA_SLOW,
        )?;
        expect_f32(
            "attention.compress_rope_freq_base",
            required_f32(
                self.attention_compress_rope_freq_base,
                "deepseek4.attention.compress_rope_freq_base",
            )?,
            DS4_COMPRESS_ROPE_FREQ_BASE,
        )?;
        expect_f32(
            "expert_weights_scale",
            required_f32(self.expert_weights_scale, "deepseek4.expert_weights_scale")?,
            DS4_EXPERT_WEIGHT_SCALE,
        )?;
        expect_f32(
            "attention.layer_norm_rms_epsilon",
            required_f32(
                self.attention_layer_norm_rms_epsilon,
                "deepseek4.attention.layer_norm_rms_epsilon",
            )?,
            DS4_RMS_EPS,
        )?;
        expect_f32(
            "hyper_connection.epsilon",
            required_f32(
                self.hyper_connection_epsilon,
                "deepseek4.hyper_connection.epsilon",
            )?,
            DS4_HC_EPS,
        )?;
        expect_bool(
            "expert_weights_norm",
            required_bool(self.expert_weights_norm, "deepseek4.expert_weights_norm")?,
            true,
        )?;

        Ok(())
    }
}

impl GgufMap {
    pub(crate) fn open(path: &str, shared: bool) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open GGUF model at {path}"))?;
        let file_size = file.metadata()?.len();
        if file_size < GGUF_MIN_FILE_SIZE {
            bail!("model file is too small to be GGUF")
        }

        let len = usize::try_from(file_size).context("GGUF file is too large to mmap")?;
        let flags = if shared { MAP_SHARED } else { MAP_PRIVATE };
        let map = unsafe { mmap(ptr::null_mut(), len, PROT_READ, flags, file.as_raw_fd(), 0) };
        if map == MAP_FAILED {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to mmap GGUF model at {path}"));
        }

        Ok(Self { file, map, len })
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.map.cast::<u8>(), self.len) }
    }

    pub(crate) fn as_ptr(&self) -> *const libc::c_void {
        self.map.cast_const()
    }

    pub(crate) fn len_u64(&self) -> u64 {
        self.len as u64
    }
}

impl Drop for GgufMap {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.map, self.len);
        }
    }
}

fn validate_compress_ratios(values: Option<&[u32]>) -> Result<()> {
    let Some(values) = values else {
        bail!("ds4: required int32/uint32 array metadata key is missing: deepseek4.attention.compress_ratios")
    };
    if values.len() < DS4_N_LAYER as usize {
        bail!("deepseek4.attention.compress_ratios is shorter than the layer count")
    }

    for layer in 0..DS4_N_LAYER as usize {
        let expected = layer_compress_ratio(layer as u32);
        let got = values[layer];
        if got != expected {
            bail!(
                "ds4: unexpected DeepSeek4 compression ratio at layer {}: got {}, expected {}",
                layer,
                got,
                expected
            )
        }
    }

    Ok(())
}

fn validate_swiglu_clamp(values: Option<&[f32]>) -> Result<()> {
    let Some(values) = values else {
        bail!("ds4: required float array metadata key is missing: deepseek4.swiglu_clamp_exp")
    };
    if values.len() < DS4_N_LAYER as usize {
        bail!("deepseek4.swiglu_clamp_exp is shorter than the layer count")
    }

    for value in values.iter().take(DS4_N_LAYER as usize) {
        expect_f32("swiglu_clamp_exp", *value, DS4_SWIGLU_CLAMP_EXP)?;
    }
    Ok(())
}

fn required_u32(value: Option<u32>, key: &str) -> Result<u32> {
    match value {
        Some(value) => Ok(value),
        None => bail!("ds4: required metadata key is missing: {key}"),
    }
}

fn required_u64(value: Option<u64>, key: &str) -> Result<u64> {
    match value {
        Some(value) => Ok(value),
        None => bail!("ds4: required metadata key is missing: {key}"),
    }
}

fn required_f32(value: Option<f32>, key: &str) -> Result<f32> {
    match value {
        Some(value) => Ok(value),
        None => bail!("ds4: required metadata key is missing: {key}"),
    }
}

fn required_bool(value: Option<bool>, key: &str) -> Result<bool> {
    match value {
        Some(value) => Ok(value),
        None => bail!("ds4: required metadata key is missing: {key}"),
    }
}

fn expect_u32(name: &str, got: u32, expected: u32) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        bail!("ds4: expected {}={} for DeepSeek4 Flash, got {}", name, expected, got)
    }
}

fn expect_f32(name: &str, got: f32, expected: f32) -> Result<()> {
    let scale = expected.abs().max(1.0);
    if (got - expected).abs() <= scale * 1.0e-6 {
        Ok(())
    } else {
        bail!(
            "ds4: expected {}={:.9} for DeepSeek4 Flash, got {:.9}",
            name,
            expected,
            got
        )
    }
}

fn expect_bool(name: &str, got: bool, expected: bool) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        bail!(
            "ds4: expected {}={} for DeepSeek4 Flash, got {}",
            name,
            if expected { "true" } else { "false" },
            if got { "true" } else { "false" }
        )
    }
}

fn layer_compress_ratio(layer: u32) -> u32 {
    if layer < 2 {
        0
    } else if layer & 1 == 0 {
        4
    } else {
        128
    }
}

fn parse_tensor_directory<R: Read + Seek>(reader: &mut R, summary: &mut ModelSummary, alignment: u32) -> Result<()> {
    let mut entries = Vec::new();
    for _ in 0..summary.n_tensors {
        let name = read_string_bytes(reader)?;
        let ndim = read_u32(reader)?;
        if ndim == 0 || ndim > GGUF_MAX_DIMS {
            bail!("tensor has an unsupported number of dimensions")
        }

        let mut elements = 1u64;
        for _ in 0..ndim {
            let dim = read_u64(reader)?;
            if dim != 0 && elements > u64::MAX / dim {
                bail!("tensor element count overflow")
            }
            elements *= dim;
        }

        let tensor_type = read_u32(reader)?;
        let rel_offset = read_u64(reader)?;
        let bytes = match tensor_nbytes(tensor_type, elements) {
            Some(bytes) => bytes,
            None => {
                eprintln!(
                    "ds4: warning: tensor {} has unsupported GGUF type {}",
                    String::from_utf8_lossy(&name),
                    tensor_type
                );
                0
            }
        };

        summary.tensor_bytes = summary
            .tensor_bytes
            .checked_add(bytes)
            .context("tensor byte count overflow")?;
        summary.params = summary
            .params
            .checked_add(elements)
            .context("tensor element count overflow")?;

        if tensor_type_info(tensor_type).is_some() {
            let stats = summary
                .tensor_types
                .entry(tensor_type)
                .or_insert(TensorTypeSummary { count: 0, bytes: 0 });
            stats.count += 1;
            stats.bytes += bytes;
        }

        entries.push(TensorDirectoryEntry { rel_offset, bytes });
    }

    let tensor_data_pos = align_up(reader.stream_position()?, alignment as u64);
    for entry in entries {
        if entry.rel_offset > u64::MAX - tensor_data_pos {
            bail!("tensor offset overflow")
        }
        let abs_offset = tensor_data_pos + entry.rel_offset;
        if entry.bytes != 0
            && (abs_offset > summary.file_size || entry.bytes > summary.file_size - abs_offset)
        {
            bail!("tensor points outside GGUF file")
        }
    }

    Ok(())
}

fn tensor_type_info(type_id: u32) -> Option<TensorTypeInfo> {
    match type_id {
        0 => Some(TensorTypeInfo { name: "f32", block_elems: 1, block_bytes: 4 }),
        1 => Some(TensorTypeInfo { name: "f16", block_elems: 1, block_bytes: 2 }),
        2 => Some(TensorTypeInfo { name: "q4_0", block_elems: 32, block_bytes: 18 }),
        3 => Some(TensorTypeInfo { name: "q4_1", block_elems: 32, block_bytes: 20 }),
        6 => Some(TensorTypeInfo { name: "q5_0", block_elems: 32, block_bytes: 22 }),
        7 => Some(TensorTypeInfo { name: "q5_1", block_elems: 32, block_bytes: 24 }),
        8 => Some(TensorTypeInfo { name: "q8_0", block_elems: 32, block_bytes: 34 }),
        9 => Some(TensorTypeInfo { name: "q8_1", block_elems: 32, block_bytes: 40 }),
        10 => Some(TensorTypeInfo { name: "q2_k", block_elems: 256, block_bytes: 84 }),
        11 => Some(TensorTypeInfo { name: "q3_k", block_elems: 256, block_bytes: 110 }),
        12 => Some(TensorTypeInfo { name: "q4_k", block_elems: 256, block_bytes: 144 }),
        13 => Some(TensorTypeInfo { name: "q5_k", block_elems: 256, block_bytes: 176 }),
        14 => Some(TensorTypeInfo { name: "q6_k", block_elems: 256, block_bytes: 210 }),
        15 => Some(TensorTypeInfo { name: "q8_k", block_elems: 256, block_bytes: 292 }),
        16 => Some(TensorTypeInfo { name: "iq2_xxs", block_elems: 256, block_bytes: 66 }),
        17 => Some(TensorTypeInfo { name: "iq2_xs", block_elems: 256, block_bytes: 74 }),
        18 => Some(TensorTypeInfo { name: "iq3_xxs", block_elems: 256, block_bytes: 98 }),
        19 => Some(TensorTypeInfo { name: "iq1_s", block_elems: 256, block_bytes: 110 }),
        20 => Some(TensorTypeInfo { name: "iq4_nl", block_elems: 256, block_bytes: 50 }),
        21 => Some(TensorTypeInfo { name: "iq3_s", block_elems: 256, block_bytes: 110 }),
        22 => Some(TensorTypeInfo { name: "iq2_s", block_elems: 256, block_bytes: 82 }),
        23 => Some(TensorTypeInfo { name: "iq4_xs", block_elems: 256, block_bytes: 136 }),
        24 => Some(TensorTypeInfo { name: "i8", block_elems: 1, block_bytes: 1 }),
        25 => Some(TensorTypeInfo { name: "i16", block_elems: 1, block_bytes: 2 }),
        26 => Some(TensorTypeInfo { name: "i32", block_elems: 1, block_bytes: 4 }),
        27 => Some(TensorTypeInfo { name: "i64", block_elems: 1, block_bytes: 8 }),
        28 => Some(TensorTypeInfo { name: "f64", block_elems: 1, block_bytes: 8 }),
        29 => Some(TensorTypeInfo { name: "iq1_m", block_elems: 256, block_bytes: 56 }),
        30 => Some(TensorTypeInfo { name: "bf16", block_elems: 1, block_bytes: 2 }),
        _ => None,
    }
}

fn tensor_type_name(type_id: u32) -> &'static str {
    tensor_type_info(type_id)
        .map(|info| info.name)
        .unwrap_or("unknown")
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h & 0x8000)) << 16;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mut mant = u32::from(h & 0x03ff);

    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut exp = 1i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
    };

    f32::from_bits(bits)
}

fn tensor_nbytes(type_id: u32, elements: u64) -> Option<u64> {
    let info = tensor_type_info(type_id)?;
    if info.block_elems == 0 {
        return None;
    }
    let blocks = elements.div_ceil(info.block_elems as u64);
    blocks.checked_mul(info.block_bytes as u64)
}

fn format_gib(bytes: u64) -> String {
    let gib = 1024.0 * 1024.0 * 1024.0;
    format!("{:.2} GiB", bytes as f64 / gib)
}

fn byte_encode(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() * 4);
    for byte in raw {
        utf8_put(&mut out, gpt2_byte_to_codepoint(*byte));
    }
    out
}

fn decode_token_bytes(raw: &[u8]) -> Vec<u8> {
    if token_is_literal_special(raw) {
        return raw.to_vec();
    }

    let mut out = Vec::with_capacity(raw.len());
    let mut pos = 0usize;
    while pos < raw.len() {
        let cp = utf8_decode_one(raw, &mut pos);
        let byte = gpt2_codepoint_to_byte(cp);
        if byte >= 0 {
            out.push(byte as u8);
        }
    }
    out
}

fn token_is_literal_special(raw: &[u8]) -> bool {
    raw.windows(FULLWIDTH_BAR_UTF8.len())
        .any(|window| window == FULLWIDTH_BAR_UTF8)
}

fn utf8_decode_one(raw: &[u8], pos: &mut usize) -> u32 {
    let c = raw[*pos];
    if c < 0x80 || *pos + 1 >= raw.len() {
        *pos += 1;
        return c as u32;
    }
    if (c & 0xe0) == 0xc0 && *pos + 1 < raw.len() {
        let cp = (((c & 0x1f) as u32) << 6) | ((raw[*pos + 1] & 0x3f) as u32);
        *pos += 2;
        return cp;
    }
    if (c & 0xf0) == 0xe0 && *pos + 2 < raw.len() {
        let cp = (((c & 0x0f) as u32) << 12)
            | (((raw[*pos + 1] & 0x3f) as u32) << 6)
            | ((raw[*pos + 2] & 0x3f) as u32);
        *pos += 3;
        return cp;
    }
    if (c & 0xf8) == 0xf0 && *pos + 3 < raw.len() {
        let cp = (((c & 0x07) as u32) << 18)
            | (((raw[*pos + 1] & 0x3f) as u32) << 12)
            | (((raw[*pos + 2] & 0x3f) as u32) << 6)
            | ((raw[*pos + 3] & 0x3f) as u32);
        *pos += 4;
        return cp;
    }
    *pos += 1;
    c as u32
}

fn utf8_put(out: &mut Vec<u8>, cp: u32) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push((0xc0 | (cp >> 6)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else if cp <= 0xffff {
        out.push((0xe0 | (cp >> 12)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (cp >> 18)) as u8);
        out.push((0x80 | ((cp >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((cp >> 6) & 0x3f)) as u8);
        out.push((0x80 | (cp & 0x3f)) as u8);
    }
}

fn gpt2_byte_to_codepoint(byte: u8) -> u32 {
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
        return byte as u32;
    }

    let mut n = 0u32;
    for candidate in 0u32..256 {
        if (33..=126).contains(&candidate) || (161..=172).contains(&candidate) || candidate >= 174 {
            continue;
        }
        if candidate == byte as u32 {
            return 256 + n;
        }
        n += 1;
    }
    byte as u32
}

fn utf8_len_from_first_byte(byte: u8) -> usize {
    if byte < 0x80 {
        1
    } else if (byte & 0xe0) == 0xc0 {
        2
    } else if (byte & 0xf0) == 0xe0 {
        3
    } else if (byte & 0xf8) == 0xf0 {
        4
    } else {
        1
    }
}

fn next_utf8_char(raw: &[u8], pos: usize) -> usize {
    let mut width = utf8_len_from_first_byte(raw[pos]);
    if pos + width > raw.len() {
        width = 1;
    }
    pos + width
}

fn ascii_alpha(byte: u8) -> bool {
    byte.is_ascii_alphabetic()
}

fn ascii_digit(byte: u8) -> bool {
    byte.is_ascii_digit()
}

fn ascii_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn ascii_newline(byte: u8) -> bool {
    matches!(byte, b'\n' | b'\r')
}

fn joyai_ascii_punct_symbol(byte: u8) -> bool {
    (b'!'..=b'/').contains(&byte)
        || (b':'..=b'@').contains(&byte)
        || (b'['..=b'`').contains(&byte)
        || (b'{'..=b'~').contains(&byte)
}

fn utf8_is_cjk_hira_kata(cp: u32) -> bool {
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

fn utf8_peek_one(raw: &[u8], pos: usize) -> (u32, usize) {
    let c0 = raw[pos];
    let mut width = utf8_len_from_first_byte(c0);
    if pos + width > raw.len() {
        width = 1;
    }
    let next = pos + width;

    if width == 1 || pos + 1 >= raw.len() {
        return (c0 as u32, next);
    }
    if width == 2 {
        return (
            (((c0 & 0x1f) as u32) << 6) | ((raw[pos + 1] & 0x3f) as u32),
            next,
        );
    }
    if width == 3 && pos + 2 < raw.len() {
        return (
            (((c0 & 0x0f) as u32) << 12)
                | (((raw[pos + 1] & 0x3f) as u32) << 6)
                | ((raw[pos + 2] & 0x3f) as u32),
            next,
        );
    }
    (
        (((c0 & 0x07) as u32) << 18)
            | (((raw[pos + 1] & 0x3f) as u32) << 12)
            | (((raw[pos + 2] & 0x3f) as u32) << 6)
            | ((raw[pos + 3] & 0x3f) as u32),
        next,
    )
}

fn joyai_letter_like_at(raw: &[u8], pos: usize) -> bool {
    let byte = raw[pos];
    if byte < 128 {
        ascii_alpha(byte)
    } else {
        true
    }
}

fn joyai_consume_letters(raw: &[u8], mut pos: usize) -> usize {
    while pos < raw.len() && joyai_letter_like_at(raw, pos) {
        pos = next_utf8_char(raw, pos);
    }
    pos
}

fn joyai_cjk_at(raw: &[u8], pos: usize) -> bool {
    if raw[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_peek_one(raw, pos);
    utf8_is_cjk_hira_kata(cp)
}

fn gpt2_codepoint_to_byte(cp: u32) -> i32 {
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return cp as i32;
    }

    let mut n = 0u32;
    for byte in 0u32..256 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
            continue;
        }
        if cp == 256 + n {
            return byte as i32;
        }
        n += 1;
    }
    -1
}

fn read_string_array<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Vec<Vec<u8>>> {
    if value_type != GGUF_VALUE_ARRAY {
        bail!("GGUF tokenizer token table is missing or invalid")
    }

    let item_type = read_u32(reader)?;
    let len = read_u64(reader)?;
    if item_type != GGUF_VALUE_STRING {
        bail!("GGUF tokenizer token table is missing or invalid")
    }

    let len = usize::try_from(len).context("GGUF tokenizer token table is too large")?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(read_string_bytes(reader)?);
    }
    Ok(out)
}

fn scalar_value_size(value_type: u32) -> u64 {
    match value_type {
        GGUF_VALUE_UINT8 | GGUF_VALUE_INT8 | GGUF_VALUE_BOOL => 1,
        GGUF_VALUE_UINT16 | GGUF_VALUE_INT16 => 2,
        GGUF_VALUE_UINT32 | GGUF_VALUE_INT32 | GGUF_VALUE_FLOAT32 => 4,
        GGUF_VALUE_UINT64 | GGUF_VALUE_INT64 | GGUF_VALUE_FLOAT64 => 8,
        _ => 0,
    }
}

fn skip_value<R: Read + Seek>(reader: &mut R, value_type: u32, depth: usize) -> Result<()> {
    if depth > 8 {
        bail!("metadata array nesting is too deep")
    }

    let scalar_size = scalar_value_size(value_type);
    if scalar_size != 0 {
        return skip_bytes(reader, scalar_size);
    }

    if value_type == GGUF_VALUE_STRING {
        let _ = read_string_bytes(reader)?;
        return Ok(());
    }

    if value_type == GGUF_VALUE_ARRAY {
        let item_type = read_u32(reader)?;
        let len = read_u64(reader)?;

        let item_size = scalar_value_size(item_type);
        if item_size != 0 {
            let bytes = len
                .checked_mul(item_size)
                .context("metadata array is too large")?;
            return skip_bytes(reader, bytes);
        }

        for _ in 0..len {
            skip_value(reader, item_type, depth + 1)?;
        }
        return Ok(());
    }

    bail!("unknown GGUF metadata type")
}

fn skip_bytes<R: Seek>(reader: &mut R, n: u64) -> Result<()> {
    let offset = i64::try_from(n).context("metadata entry is too large to skip")?;
    reader.seek(SeekFrom::Current(offset))?;
    Ok(())
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string_bytes<R: Read>(reader: &mut R) -> Result<Vec<u8>> {
    let len = read_u64(reader)?;
    let len = usize::try_from(len).context("GGUF string is too large")?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_string_lossy<R: Read>(reader: &mut R) -> Result<String> {
    let bytes = read_string_bytes(reader)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn read_optional_u32<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Option<u32>> {
    if value_type == GGUF_VALUE_UINT32 {
        Ok(Some(read_u32(reader)?))
    } else {
        skip_value(reader, value_type, 0)?;
        Ok(None)
    }
}

fn read_optional_u64<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Option<u64>> {
    match value_type {
        GGUF_VALUE_UINT64 => Ok(Some(read_u64(reader)?)),
        GGUF_VALUE_UINT32 => Ok(Some(read_u32(reader)? as u64)),
        _ => {
            skip_value(reader, value_type, 0)?;
            Ok(None)
        }
    }
}

fn read_optional_f32<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Option<f32>> {
    match value_type {
        GGUF_VALUE_FLOAT32 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(Some(f32::from_le_bytes(buf)))
        }
        GGUF_VALUE_FLOAT64 => {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(Some(f64::from_le_bytes(buf) as f32))
        }
        GGUF_VALUE_UINT32 => Ok(Some(read_u32(reader)? as f32)),
        GGUF_VALUE_INT32 => {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(Some(i32::from_le_bytes(buf) as f32))
        }
        _ => {
            skip_value(reader, value_type, 0)?;
            Ok(None)
        }
    }
}

fn read_optional_bool<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Option<bool>> {
    if value_type == GGUF_VALUE_BOOL {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(Some(buf[0] != 0))
    } else {
        skip_value(reader, value_type, 0)?;
        Ok(None)
    }
}

fn read_optional_u32_array<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Option<Vec<u32>>> {
    if value_type != GGUF_VALUE_ARRAY {
        skip_value(reader, value_type, 0)?;
        return Ok(None);
    }

    let item_type = read_u32(reader)?;
    let len = usize::try_from(read_u64(reader)?).context("metadata array is too large")?;
    match item_type {
        GGUF_VALUE_UINT32 => {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_u32(reader)?);
            }
            Ok(Some(values))
        }
        GGUF_VALUE_INT32 => {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let value = i32::from_le_bytes(buf);
                if value < 0 {
                    bail!("metadata array contains a negative value")
                }
                values.push(value as u32);
            }
            Ok(Some(values))
        }
        _ => {
            skip_array_items(reader, item_type, len)?;
            Ok(None)
        }
    }
}

fn read_optional_f32_array<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<Option<Vec<f32>>> {
    if value_type != GGUF_VALUE_ARRAY {
        skip_value(reader, value_type, 0)?;
        return Ok(None);
    }

    let item_type = read_u32(reader)?;
    let len = usize::try_from(read_u64(reader)?).context("metadata array is too large")?;
    match item_type {
        GGUF_VALUE_FLOAT32 => {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                values.push(f32::from_le_bytes(buf));
            }
            Ok(Some(values))
        }
        GGUF_VALUE_FLOAT64 => {
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                values.push(f64::from_le_bytes(buf) as f32);
            }
            Ok(Some(values))
        }
        _ => {
            skip_array_items(reader, item_type, len)?;
            Ok(None)
        }
    }
}

fn skip_array_items<R: Read + Seek>(reader: &mut R, item_type: u32, len: usize) -> Result<()> {
    let item_size = scalar_value_size(item_type);
    if item_size != 0 {
        skip_bytes(reader, (len as u64) * item_size)?;
        return Ok(());
    }
    for _ in 0..len {
        skip_value(reader, item_type, 1)?;
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn loads_tokenizer_tokens_from_minimal_gguf() {
        let bytes = build_test_gguf(&[
            b"<think>".as_slice(),
            b"abc".as_slice(),
            "<｜User｜>".as_bytes(),
        ], &[]);
        let mut cursor = Cursor::new(bytes);

        let tokenizer = TokenizerMetadata::from_reader(&mut cursor).unwrap();

        assert_eq!(tokenizer.lookup_required("<think>").unwrap(), 0);
        assert_eq!(tokenizer.lookup_required("<｜User｜>").unwrap(), 2);
    }

    #[test]
    fn decodes_token_bytes_like_c_helper() {
        let bytes = build_test_gguf(&[
            &[0xc4, 0x80],
            b"abc".as_slice(),
            "<｜Assistant｜>".as_bytes(),
        ], &[]);
        let mut cursor = Cursor::new(bytes);

        let tokenizer = TokenizerMetadata::from_reader(&mut cursor).unwrap();

        assert_eq!(tokenizer.token_bytes(0), vec![0]);
        assert_eq!(tokenizer.token_bytes(1), b"abc".to_vec());
        assert_eq!(tokenizer.token_bytes(2), "<｜Assistant｜>".as_bytes().to_vec());
    }

    #[test]
    fn tokenizes_ascii_with_bpe_merge() {
        let bytes = build_test_gguf(&[b"a".as_slice(), b"b".as_slice(), b"ab".as_slice()], &[b"a b".as_slice()]);
        let mut cursor = Cursor::new(bytes);

        let tokenizer = TokenizerMetadata::from_reader(&mut cursor).unwrap();

        assert_eq!(tokenizer.tokenize_text("ab"), vec![2]);
    }

    #[test]
    fn parses_model_summary_from_minimal_gguf() {
        let bytes = build_summary_test_gguf();
        let file_size = bytes.len() as u64;
        let mut cursor = Cursor::new(bytes);

        let summary = ModelSummary::from_reader(&mut cursor, file_size).unwrap();

        assert_eq!(summary.name.as_deref(), Some("mini-ds4"));
        assert_eq!(summary.architecture.as_deref(), Some("deepseek4"));
        assert_eq!(summary.layers, Some(43));
        assert_eq!(summary.train_context, Some(32768));
        assert_eq!(summary.tensor_bytes, 24);
        assert_eq!(summary.params, 6);
        let f32_stats = summary.tensor_types.get(&0).unwrap();
        assert_eq!(f32_stats.count, 1);
        assert_eq!(f32_stats.bytes, 24);
    }

    #[test]
    fn validates_deepseek4_metadata_from_minimal_gguf() {
        validate_model_config_bytes(&build_validation_test_gguf()).unwrap();
    }

    #[test]
    fn loads_tensor_directory_from_minimal_gguf() {
        let bytes = build_summary_test_gguf();
        let file_size = bytes.len() as u64;
        let mut cursor = Cursor::new(bytes);

        let directory = TensorDirectory::from_reader(&mut cursor, file_size).unwrap();

        assert_eq!(directory.tensor_data_pos % 32, 0);
        let tensor = directory.tensors.get("tensor.w").unwrap();
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.dims, vec![2, 3]);
        assert_eq!(tensor.tensor_type, 0);
        assert_eq!(tensor.rel_offset, 0);
        assert_eq!(tensor.abs_offset, directory.tensor_data_pos);
        assert_eq!(tensor.elements, 6);
        assert_eq!(tensor.bytes, 24);
    }

    #[test]
    fn bound_tensor_reads_gguf_payload_slice() {
        let bytes = build_summary_test_gguf();
        let file_size = bytes.len() as u64;
        let mut cursor = Cursor::new(bytes.clone());

        let directory = TensorDirectory::from_reader(&mut cursor, file_size).unwrap();
        let tensor = directory.require_tensor("tensor.w").unwrap();

        assert_eq!(tensor.data(&bytes).unwrap(), &[0u8; 24]);
    }

    #[test]
    fn bound_tensor_reads_f16_row() {
        let bytes = build_f16_tensor_test_gguf();
        let file_size = bytes.len() as u64;
        let mut cursor = Cursor::new(bytes.clone());

        let directory = TensorDirectory::from_reader(&mut cursor, file_size).unwrap();
        let tensor = directory.require_tensor("token_embd.weight").unwrap();

        assert_eq!(tensor.read_f16_row(&bytes, 0).unwrap(), vec![1.0, -2.0]);
        assert_eq!(tensor.read_f16_row(&bytes, 1).unwrap(), vec![0.5, 4.0]);
        assert_eq!(tensor.read_f16_row(&bytes, 2).unwrap(), vec![0.0, -0.25]);
    }

    #[test]
    fn bound_tensor_reads_f32_values() {
        let bytes = build_summary_test_gguf();
        let file_size = bytes.len() as u64;
        let mut cursor = Cursor::new(bytes.clone());

        let directory = TensorDirectory::from_reader(&mut cursor, file_size).unwrap();
        let tensor = directory.require_tensor("tensor.w").unwrap();

        assert_eq!(tensor.read_f32_values(&bytes).unwrap(), vec![0.0; 6]);
    }

    fn build_test_gguf(tokens: &[&[u8]], merges: &[&[u8]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 3);

        push_string(&mut bytes, b"general.alignment");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 32);

        push_string(&mut bytes, b"tokenizer.ggml.tokens");
        push_u32(&mut bytes, 9);
        push_u32(&mut bytes, 8);
        push_u64(&mut bytes, tokens.len() as u64);
        for token in tokens {
            push_string(&mut bytes, token);
        }

        push_string(&mut bytes, b"tokenizer.ggml.merges");
        push_u32(&mut bytes, 9);
        push_u32(&mut bytes, 8);
        push_u64(&mut bytes, merges.len() as u64);
        for merge in merges {
            push_string(&mut bytes, merge);
        }

        bytes
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &[u8]) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value);
    }

    fn build_summary_test_gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 5);

        push_string(&mut bytes, b"general.alignment");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 32);

        push_string(&mut bytes, b"general.name");
        push_u32(&mut bytes, 8);
        push_string(&mut bytes, b"mini-ds4");

        push_string(&mut bytes, b"general.architecture");
        push_u32(&mut bytes, 8);
        push_string(&mut bytes, b"deepseek4");

        push_string(&mut bytes, b"deepseek4.block_count");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 43);

        push_string(&mut bytes, b"deepseek4.context_length");
        push_u32(&mut bytes, 10);
        push_u64(&mut bytes, 32768);

        push_string(&mut bytes, b"tensor.w");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 2);
        push_u64(&mut bytes, 3);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 0);

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0u8; 24]);
        bytes
    }

    fn build_validation_test_gguf() -> Vec<u8> {
        let mut entries = Vec::new();
        entries.push(metadata_u32(b"deepseek4.block_count", DS4_N_LAYER));
        entries.push(metadata_u32(b"deepseek4.embedding_length", DS4_N_EMBD));
        entries.push(metadata_u32(b"deepseek4.vocab_size", DS4_N_VOCAB));
        entries.push(metadata_u32(b"deepseek4.attention.head_count", DS4_N_HEAD));
        entries.push(metadata_u32(b"deepseek4.attention.head_count_kv", DS4_N_HEAD_KV));
        entries.push(metadata_u32(b"deepseek4.attention.key_length", DS4_N_HEAD_DIM));
        entries.push(metadata_u32(b"deepseek4.attention.value_length", DS4_N_VALUE_DIM));
        entries.push(metadata_u32(b"deepseek4.rope.dimension_count", DS4_N_ROT));
        entries.push(metadata_u32(b"deepseek4.attention.q_lora_rank", DS4_N_LORA_Q));
        entries.push(metadata_u32(b"deepseek4.attention.output_lora_rank", DS4_N_LORA_O));
        entries.push(metadata_u32(b"deepseek4.attention.output_group_count", DS4_N_OUT_GROUP));
        entries.push(metadata_u32(b"deepseek4.expert_count", DS4_N_EXPERT));
        entries.push(metadata_u32(b"deepseek4.expert_used_count", DS4_N_EXPERT_USED));
        entries.push(metadata_u32(b"deepseek4.expert_feed_forward_length", DS4_N_FF_EXP));
        entries.push(metadata_u32(b"deepseek4.expert_shared_count", DS4_N_EXPERT_SHARED));
        entries.push(metadata_u32(b"deepseek4.hash_layer_count", DS4_N_HASH_LAYER));
        entries.push(metadata_u32(b"deepseek4.expert_group_count", 0));
        entries.push(metadata_u32(b"deepseek4.expert_group_used_count", 0));
        entries.push(metadata_u32(b"deepseek4.attention.sliding_window", DS4_N_SWA));
        entries.push(metadata_u32(b"deepseek4.attention.indexer.head_count", DS4_N_INDEXER_HEAD));
        entries.push(metadata_u32(b"deepseek4.attention.indexer.key_length", DS4_N_INDEXER_HEAD_DIM));
        entries.push(metadata_u32(b"deepseek4.attention.indexer.top_k", DS4_N_INDEXER_TOP_K));
        entries.push(metadata_u32(b"deepseek4.hyper_connection.count", DS4_N_HC));
        entries.push(metadata_u32(
            b"deepseek4.hyper_connection.sinkhorn_iterations",
            DS4_N_HC_SINKHORN_ITER,
        ));
        entries.push(metadata_u64(
            b"deepseek4.rope.scaling.original_context_length",
            DS4_ROPE_ORIG_CTX,
        ));
        entries.push(metadata_f32(b"deepseek4.rope.freq_base", DS4_ROPE_FREQ_BASE));
        entries.push(metadata_f32(b"deepseek4.rope.scaling.factor", DS4_ROPE_SCALE_FACTOR));
        entries.push(metadata_f32(
            b"deepseek4.rope.scaling.yarn_beta_fast",
            DS4_ROPE_YARN_BETA_FAST,
        ));
        entries.push(metadata_f32(
            b"deepseek4.rope.scaling.yarn_beta_slow",
            DS4_ROPE_YARN_BETA_SLOW,
        ));
        entries.push(metadata_f32(
            b"deepseek4.attention.compress_rope_freq_base",
            DS4_COMPRESS_ROPE_FREQ_BASE,
        ));
        entries.push(metadata_f32(
            b"deepseek4.expert_weights_scale",
            DS4_EXPERT_WEIGHT_SCALE,
        ));
        entries.push(metadata_f32(
            b"deepseek4.attention.layer_norm_rms_epsilon",
            DS4_RMS_EPS,
        ));
        entries.push(metadata_f32(
            b"deepseek4.hyper_connection.epsilon",
            DS4_HC_EPS,
        ));
        entries.push(metadata_bool(b"deepseek4.expert_weights_norm", true));
        entries.push(metadata_u32_array(
            b"deepseek4.attention.compress_ratios",
            &(0..DS4_N_LAYER).map(layer_compress_ratio).collect::<Vec<_>>(),
        ));
        entries.push(metadata_f32_array(
            b"deepseek4.swiglu_clamp_exp",
            &vec![DS4_SWIGLU_CLAMP_EXP; DS4_N_LAYER as usize],
        ));

        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, entries.len() as u64);
        for entry in entries {
            bytes.extend_from_slice(&entry);
        }
        bytes
    }

    fn build_f16_tensor_test_gguf() -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x4655_4747);
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 1);

        push_string(&mut bytes, b"general.alignment");
        push_u32(&mut bytes, 4);
        push_u32(&mut bytes, 32);

        push_string(&mut bytes, b"token_embd.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 2);
        push_u64(&mut bytes, 3);
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 0);

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }

        for bits in [0x3c00u16, 0xc000, 0x3800, 0x4400, 0x0000, 0xb400] {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }

        bytes
    }

    fn metadata_u32(key: &[u8], value: u32) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, key);
        push_u32(&mut entry, 4);
        push_u32(&mut entry, value);
        entry
    }

    fn metadata_u64(key: &[u8], value: u64) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, key);
        push_u32(&mut entry, 10);
        push_u64(&mut entry, value);
        entry
    }

    fn metadata_f32(key: &[u8], value: f32) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, key);
        push_u32(&mut entry, 6);
        entry.extend_from_slice(&value.to_le_bytes());
        entry
    }

    fn metadata_bool(key: &[u8], value: bool) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, key);
        push_u32(&mut entry, 7);
        entry.push(u8::from(value));
        entry
    }

    fn metadata_u32_array(key: &[u8], values: &[u32]) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, key);
        push_u32(&mut entry, 9);
        push_u32(&mut entry, 4);
        push_u64(&mut entry, values.len() as u64);
        for value in values {
            push_u32(&mut entry, *value);
        }
        entry
    }

    fn metadata_f32_array(key: &[u8], values: &[f32]) -> Vec<u8> {
        let mut entry = Vec::new();
        push_string(&mut entry, key);
        push_u32(&mut entry, 9);
        push_u32(&mut entry, 6);
        push_u64(&mut entry, values.len() as u64);
        for value in values {
            entry.extend_from_slice(&value.to_le_bytes());
        }
        entry
    }
}