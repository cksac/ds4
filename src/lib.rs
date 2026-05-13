//! DS4 — DeepSeek V4 Flash inference engine (pure Rust + Metal).

pub mod gguf;
#[cfg(target_os = "macos")]
pub mod metal;

// ============================================================================
// Model constants — validated against GGUF metadata at load time.
// ============================================================================

pub const N_LAYER: usize = 43;
pub const N_EMBD: usize = 4096;
pub const N_VOCAB: usize = 129280;
pub const N_HEAD: usize = 64;
pub const N_HEAD_KV: usize = 1;
pub const N_HEAD_DIM: usize = 512;
pub const N_VALUE_DIM: usize = 512;
pub const N_ROT: usize = 64;
pub const N_HC: usize = 4;
pub const N_OUT_GROUP: usize = 8;
pub const N_LORA_Q: usize = 1024;
pub const N_LORA_O: usize = 1024;
pub const N_EXPERT: usize = 256;
pub const N_EXPERT_USED: usize = 6;
pub const N_EXPERT_SHARED: usize = 1;
pub const N_FF_EXP: usize = 2048;
pub const N_HASH_LAYER: usize = 3;
pub const N_SWA: usize = 128;
pub const N_INDEXER_HEAD: usize = 64;
pub const N_INDEXER_HEAD_DIM: usize = 128;
pub const N_INDEXER_TOP_K: usize = 512;
pub const N_HC_SINKHORN_ITER: usize = 20;

// Derived dimensions.
pub const HC_DIM: usize = N_EMBD * N_HC; // 16384
pub const Q_DIM: usize = N_HEAD * N_HEAD_DIM; // 32768
pub const OUT_LOW_DIM: usize = N_OUT_GROUP * N_LORA_O; // 8192

// Float constants.
pub const RMS_EPS: f32 = 1e-6;
pub const HC_EPS: f32 = 1e-6;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const ROPE_FREQ_BASE: f32 = 10000.0;
pub const ROPE_SCALE_FACTOR: f32 = 16.0;
pub const ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const ROPE_YARN_BETA_SLOW: f32 = 1.0;
pub const COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
pub const ROPE_ORIG_CTX: u32 = 65536;

// ============================================================================
// Layer compression schedule.
// ============================================================================

/// Returns the compression ratio for a given layer index.
/// - Layers 0–1: 0 (dense, raw SWA only)
/// - Even layers 2,4,…,42: 4 (indexer + compressed KV)
/// - Odd layers 3,5,…,41: 128 (compressed KV only)
pub const fn layer_ratio(layer: usize) -> u32 {
    if layer < 2 {
        0
    } else if layer % 2 == 0 {
        4
    } else {
        128
    }
}

// ============================================================================
// Tensor type identifiers (matches GGUF spec).
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
}

impl GgmlType {
    /// Block size (number of elements per quantization block).
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => 1,
            Self::Q8_0 | Self::Q8_1 => 32,
            _ => 256,
        }
    }

    /// Size in bytes of one quantization block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 => 1,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 36,
            Self::IQ1_M => 56,
        }
    }

    /// Number of bytes for `n_elements` of this type.
    pub const fn row_bytes(self, n_elements: usize) -> usize {
        let bs = self.block_size();
        let nb = (n_elements + bs - 1) / bs;
        nb * self.block_bytes()
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }
}
