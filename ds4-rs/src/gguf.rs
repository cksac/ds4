use std::path::Path;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::sync::Arc;
use memmap2::Mmap;

pub const N_LAYER: u32 = 43;
pub const N_EMBD: u32 = 4096;
pub const N_VOCAB: u32 = 129280;
pub const N_HEAD: u32 = 64;
#[allow(dead_code)]
pub const N_HEAD_KV: u32 = 1;
pub const N_HEAD_DIM: u32 = 512;
#[allow(dead_code)]
pub const N_VALUE_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
pub const N_OUT_GROUP: u32 = 8;
pub const N_LORA_Q: u32 = 1024;
pub const N_LORA_O: u32 = 1024;
pub const N_EXPERT: u32 = 256;
pub const N_EXPERT_USED: u32 = 6;
#[allow(dead_code)]
pub const N_EXPERT_SHARED: u32 = 1;
pub const N_FF_EXP: u32 = 2048;
pub const N_HASH_LAYER: u32 = 3;
pub const N_SWA: u32 = 128;
pub const N_INDEXER_HEAD: u32 = 64;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const N_INDEXER_TOP_K: u32 = 512;
pub const N_HC: u32 = 4;
pub const N_HC_SINKHORN_ITER: u32 = 20;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Standard GGUF tensor type codes matching llama.cpp/ggml.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
}

impl GgufTensorType {
    pub fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            0 => GgufTensorType::F32,
            1 => GgufTensorType::F16,
            2 => GgufTensorType::Q4_0,
            3 => GgufTensorType::Q4_1,
            6 => GgufTensorType::Q5_0,
            7 => GgufTensorType::Q5_1,
            8 => GgufTensorType::Q8_0,
            9 => GgufTensorType::Q8_1,
            10 => GgufTensorType::Q2K,
            11 => GgufTensorType::Q3K,
            12 => GgufTensorType::Q4K,
            13 => GgufTensorType::Q5K,
            14 => GgufTensorType::Q6K,
            15 => GgufTensorType::Q8K,
            16 => GgufTensorType::Iq2Xxs,
            17 => GgufTensorType::Iq2Xs,
            18 => GgufTensorType::Iq3Xxs,
            19 => GgufTensorType::Iq1S,
            20 => GgufTensorType::Iq4Nl,
            21 => GgufTensorType::Iq3S,
            22 => GgufTensorType::Iq2S,
            23 => GgufTensorType::Iq4Xs,
            24 => GgufTensorType::I8,
            25 => GgufTensorType::I16,
            26 => GgufTensorType::I32,
            27 => GgufTensorType::I64,
            28 => GgufTensorType::F64,
            29 => GgufTensorType::Iq1M,
            30 => GgufTensorType::Bf16,
            _ => return None,
        })
    }

    pub fn type_size(&self) -> (u32, u32) {
        match self {
            GgufTensorType::F32 => (1, 4),
            GgufTensorType::F16 => (1, 2),
            GgufTensorType::Bf16 => (1, 2),
            GgufTensorType::Q4_0 => (32, 18),
            GgufTensorType::Q4_1 => (32, 20),
            GgufTensorType::Q5_0 => (32, 22),
            GgufTensorType::Q5_1 => (32, 24),
            GgufTensorType::Q8_0 => (32, 34),
            GgufTensorType::Q8_1 => (32, 40),
            GgufTensorType::Q2K => (256, 84),
            GgufTensorType::Q3K => (256, 110),
            GgufTensorType::Q4K => (256, 144),
            GgufTensorType::Q5K => (256, 176),
            GgufTensorType::Q6K => (256, 210),
            GgufTensorType::Q8K => (256, 292),
            GgufTensorType::Iq2Xxs => (256, 66),
            GgufTensorType::Iq2Xs => (256, 74),
            GgufTensorType::Iq3Xxs => (256, 98),
            GgufTensorType::Iq1S => (256, 110),
            GgufTensorType::Iq4Nl => (256, 50),
            GgufTensorType::Iq3S => (256, 110),
            GgufTensorType::Iq2S => (256, 82),
            GgufTensorType::Iq4Xs => (256, 136),
            GgufTensorType::I8 => (1, 1),
            GgufTensorType::I16 => (1, 2),
            GgufTensorType::I32 => (1, 4),
            GgufTensorType::I64 => (1, 8),
            GgufTensorType::F64 => (1, 8),
            GgufTensorType::Iq1M => (256, 56),
        }
    }

    pub fn nbytes(&self, elements: u64) -> u64 {
        let (block_elems, block_bytes) = self.type_size();
        let blocks = (elements + block_elems as u64 - 1) / block_elems as u64;
        blocks * block_bytes as u64
    }
}

#[derive(Clone)]
pub struct GgufTensor {
    pub name: String, pub ndim: u32, pub dim: [i64; 4],
    pub dtype: GgufTensorType, pub rel_offset: u64,
    pub abs_offset: u64, pub elements: u64, pub bytes: u64,
}

impl GgufTensor {
    pub fn data_ptr(&self, map: &[u8]) -> *const u8 {
        map.as_ptr().wrapping_add(self.abs_offset as usize)
    }
}

pub struct GgufModel {
    pub map: Arc<Mmap>,
    pub file: File,
    pub fd: i32,
    pub size: u64,
    pub version: u32,
    pub n_kv: u32,
    pub n_tensors: u32,
    pub alignment: u64,
    pub tensor_data_pos: u64,
    pub tensors: Vec<GgufTensor>,
}

impl GgufModel {
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let fd = file.as_raw_fd();

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != GGUF_MAGIC {
            return Err("not a GGUF file".into());
        }

        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 3 {
            return Err("only GGUF v3 is supported".into());
        }

        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let n_tensors = u64::from_le_bytes(buf8);
        file.read_exact(&mut buf8)?;
        let n_kv = u64::from_le_bytes(buf8);

        let mut tensors: Vec<GgufTensor> = Vec::new();

        let alignment = 32u64;

        // Read KV metadata: skip values since we read them later from full map
        for _ in 0..n_kv {
            let mut _key_len_buf = [0u8; 8];
            file.read_exact(&mut _key_len_buf)?;
            let key_len = i64::from_le_bytes(_key_len_buf);
            if key_len > 0 {
                let mut _key_buf = vec![0u8; key_len as usize];
                file.read_exact(&mut _key_buf)?;
            }
            let mut _val_type_buf = [0u8; 4];
            file.read_exact(&mut _val_type_buf)?;
            let val_type = u32::from_le_bytes(_val_type_buf);
            Self::skip_gguf_value(&mut file, val_type)?;
        }

        for _ in 0..n_tensors {
            // read_len
            let mut name_len_buf = [0u8; 8];
            file.read_exact(&mut name_len_buf)?;
            let name_len = i64::from_le_bytes(name_len_buf);
            let mut name = String::new();
            if name_len > 0 {
                let mut name_bytes = vec![0u8; name_len as usize];
                file.read_exact(&mut name_bytes)?;
                name = String::from_utf8(name_bytes)?;
            }

            let mut ndim_buf = [0u8; 4];
            file.read_exact(&mut ndim_buf)?;
            let ndim = u32::from_le_bytes(ndim_buf);

            let mut dim = [0i64; 4];
            for d in 0..ndim as usize {
                let mut d_buf = [0u8; 8];
                file.read_exact(&mut d_buf)?;
                dim[d] = i64::from_le_bytes(d_buf);
            }

            let mut dtype_buf = [0u8; 4];
            file.read_exact(&mut dtype_buf)?;
            let dtype_val = i32::from_le_bytes(dtype_buf);

            let mut off_buf = [0u8; 8];
            file.read_exact(&mut off_buf)?;
            let rel_offset = u64::from_le_bytes(off_buf);

            let dtype = GgufTensorType::from_i32(dtype_val).unwrap_or(GgufTensorType::F32);
            let elements: u64 = dim[..ndim as usize].iter().map(|&d| d as u64).product();
            let bytes = dtype.nbytes(elements);

            tensors.push(GgufTensor {
                name, ndim, dim, dtype, rel_offset,
                abs_offset: 0, elements, bytes,
            });
        }

        let tensor_data_pos = file.stream_position()?;

        // Memory-map the file for lazy weight data access
        let mmap = unsafe { Mmap::map(&file)? };
        let map = Arc::new(mmap);

        // Align tensor data start
        let aligned_pos = ((tensor_data_pos + alignment - 1) / alignment) * alignment;

        for t in &mut tensors {
            t.abs_offset = aligned_pos + t.rel_offset;
        }

        Ok(GgufModel {
            map, file, fd: fd as i32, size, version,
            n_kv: n_kv as u32, n_tensors: n_tensors as u32,
            alignment, tensor_data_pos: aligned_pos,
            tensors,
        })
    }

    fn skip_gguf_value(file: &mut File, val_type: u32) -> Result<(), Box<dyn std::error::Error>> {
        match val_type {
            0 | 1 | 7 => { let mut _b = [0u8; 1]; file.read_exact(&mut _b)?; } // uint8, int8, bool
            2 | 3 => { let mut _b = [0u8; 2]; file.read_exact(&mut _b)?; } // uint16, int16
            4 | 5 | 6 => { let mut _b = [0u8; 4]; file.read_exact(&mut _b)?; } // uint32, int32, float32
            10 | 11 | 12 => { let mut _b = [0u8; 8]; file.read_exact(&mut _b)?; } // uint64, int64, float64
            8 => {
                // string
                let mut _len_buf = [0u8; 8];
                file.read_exact(&mut _len_buf)?;
                let slen = i64::from_le_bytes(_len_buf);
                if slen > 0 {
                    let mut _s = vec![0u8; slen as usize];
                    file.read_exact(&mut _s)?;
                }
            }
            9 => {
                // array
                let mut _item_type_buf = [0u8; 4];
                file.read_exact(&mut _item_type_buf)?;
                let item_type = u32::from_le_bytes(_item_type_buf);
                let mut _len_buf = [0u8; 8];
                file.read_exact(&mut _len_buf)?;
                let arr_len = u64::from_le_bytes(_len_buf);
                for _ in 0..arr_len {
                    Self::skip_gguf_value(file, item_type)?;
                }
            }
            _ => {
                return Err(format!("unknown GGUF metadata type {}", val_type).into());
            }
        }
        Ok(())
    }

    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn find_kv(&self, key: &str) -> Option<Vec<u8>> {
        let mut pos = 0u64;
        let data = &self.map;
        // skip magic, version, n_tensors, n_kv
        pos += 4 + 4 + 8 + 8;

        for _ in 0..self.n_kv {
            let key_len = i64::from_le_bytes(data[pos as usize..pos as usize + 8].try_into().unwrap());
            pos += 8;
            let k = if key_len > 0 {
                let k = String::from_utf8_lossy(&data[pos as usize..pos as usize + key_len as usize]).to_string();
                pos += key_len as u64;
                k
            } else {
                String::new()
            };
            let val_type = u32::from_le_bytes(data[pos as usize..pos as usize + 4].try_into().unwrap());
            pos += 4;

            if k == key {
                // Return raw value bytes
                let val_start = pos;
                let val_end = Self::measure_gguf_value(&data, pos, val_type);
                return Some(data[val_start as usize..val_end as usize].to_vec());
            }

            pos = Self::skip_gguf_value_at(&data, pos, val_type);
        }
        None
    }

    fn measure_gguf_value(data: &[u8], pos: u64, val_type: u32) -> u64 {
        match val_type {
            0 | 1 | 7 => pos + 1,
            2 | 3 => pos + 2,
            4 | 5 | 6 => pos + 4,
            10 | 11 | 12 => pos + 8,
            8 => {
                let slen = i64::from_le_bytes(data[pos as usize..pos as usize + 8].try_into().unwrap_or([0; 8]));
                pos + 8 + if slen > 0 { slen as u64 } else { 0 }
            }
            9 => {
                let item_type = u32::from_le_bytes(data[pos as usize..pos as usize + 4].try_into().unwrap_or([0; 4]));
                let arr_len = u64::from_le_bytes(data[pos as usize + 4..pos as usize + 12].try_into().unwrap_or([0; 8]));
                let mut p = pos + 12;
                for _ in 0..arr_len {
                    p = Self::measure_gguf_value(data, p, item_type);
                }
                p
            }
            _ => pos,
        }
    }

    fn skip_gguf_value_at(data: &[u8], pos: u64, val_type: u32) -> u64 {
        Self::measure_gguf_value(data, pos, val_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(N_LAYER, 43);
        assert_eq!(N_EMBD, 4096);
        assert_eq!(N_VOCAB, 129280);
        assert_eq!(N_HEAD, 64);
        assert_eq!(N_HEAD_DIM, 512);
        assert_eq!(N_EXPERT, 256);
        assert_eq!(N_EXPERT_USED, 6);
        assert_eq!(N_HC, 4);
    }

    #[test]
    fn test_tensor_type_sizes() {
        assert_eq!(GgufTensorType::F32.type_size(), (1, 4));
        assert_eq!(GgufTensorType::F16.type_size(), (1, 2));
        assert_eq!(GgufTensorType::Q8_0.type_size(), (32, 34));
        assert_eq!(GgufTensorType::Q2K.type_size(), (256, 84));
        assert_eq!(GgufTensorType::Q4K.type_size(), (256, 144));
        assert_eq!(GgufTensorType::Iq2Xxs.type_size(), (256, 66));
        assert_eq!(GgufTensorType::Q2K.nbytes(256), 84);
        assert_eq!(GgufTensorType::Q4K.nbytes(256), 144);
        assert_eq!(GgufTensorType::F32.nbytes(100), 400);
    }

    #[test]
    fn test_from_i32() {
        assert_eq!(GgufTensorType::from_i32(0), Some(GgufTensorType::F32));
        assert_eq!(GgufTensorType::from_i32(10), Some(GgufTensorType::Q2K));
        assert_eq!(GgufTensorType::from_i32(12), Some(GgufTensorType::Q4K));
        assert_eq!(GgufTensorType::from_i32(16), Some(GgufTensorType::Iq2Xxs));
    }
}
