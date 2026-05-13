//! GGUF v3 file parser — maps model weights for zero-copy Metal inference.
//!
//! The GGUF format stores metadata (key-value pairs) followed by a tensor
//! directory and the raw tensor data.  We parse the header and directory,
//! then memory-map the data section so Metal can wrap slices as no-copy
//! MTLBuffers.

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// ============================================================================
// GGUF header constants.
// ============================================================================

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3;

// GGUF metadata value types.
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// ============================================================================
// Metadata values.
// ============================================================================

#[derive(Clone, Debug)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrayU8(Vec<u8>),
    ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>),
    ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayBool(Vec<bool>),
    ArrayStr(Vec<String>),
    ArrayU64(Vec<u64>),
    ArrayI64(Vec<i64>),
    ArrayF64(Vec<f64>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str_array(&self) -> Option<&[String]> {
        match self {
            Self::ArrayStr(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<&[f32]> {
        match self {
            Self::ArrayF32(v) => Some(v),
            _ => None,
        }
    }
}

// ============================================================================
// Tensor directory entry.
// ============================================================================

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub tensor_type: crate::GgmlType,
    /// Byte offset of this tensor's data from the start of the data section.
    pub data_offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> u64 {
        let mut n = 1u64;
        for i in 0..self.n_dims as usize {
            n = n.saturating_mul(self.dims[i]);
        }
        n
    }

    /// Total size in bytes.
    pub fn total_bytes(&self) -> u64 {
        let ne = self.n_elements() as usize;
        self.tensor_type.row_bytes(ne) as u64
    }
}

// ============================================================================
// Parsed GGUF file.
// ============================================================================

pub struct GgufFile {
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset of the data section from the start of the file.
    pub data_offset: u64,
    /// Total file size.
    pub file_size: u64,
}

impl GgufFile {
    /// Look up a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get the absolute file offset of a tensor's data.
    pub fn tensor_file_offset(&self, info: &TensorInfo) -> u64 {
        self.data_offset + info.data_offset
    }

    /// Get a required u32 metadata value.
    pub fn require_u32(&self, key: &str) -> Result<u32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| anyhow::anyhow!("missing or invalid metadata: {key}"))
    }

    /// Get a required f32 metadata value.
    pub fn require_f32(&self, key: &str) -> Result<f32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| anyhow::anyhow!("missing or invalid metadata: {key}"))
    }

    /// Get a required string metadata value.
    pub fn require_str(&self, key: &str) -> Result<&str> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing or invalid metadata: {key}"))
    }
}

// ============================================================================
// GGUF parsing.
// ============================================================================

/// Parse a GGUF file from a seekable reader.
pub fn parse<R: Read + Seek>(r: &mut R) -> Result<GgufFile> {
    let file_size = r.seek(SeekFrom::End(0))?;
    r.seek(SeekFrom::Start(0))?;

    // Header: magic, version, tensor_count, metadata_kv_count.
    let magic = read_u32(r)?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (bad magic: {magic:#010x})");
    }
    let version = read_u32(r)?;
    if version != GGUF_VERSION {
        bail!("unsupported GGUF version {version} (expected {GGUF_VERSION})");
    }
    let tensor_count = read_u64(r)? as usize;
    let kv_count = read_u64(r)? as usize;

    // Metadata key-value pairs.
    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = read_string(r)?;
        let value = read_meta_value(r)?;
        metadata.insert(key, value);
    }

    // Tensor directory.
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        if n_dims > 4 {
            bail!("tensor '{name}' has {n_dims} dimensions (max 4)");
        }
        let mut dims = [1u64; 4];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = read_u64(r)?;
        }
        let type_id = read_u32(r)?;
        let tensor_type = crate::GgmlType::from_u32(type_id)
            .ok_or_else(|| anyhow::anyhow!("unknown tensor type {type_id} for '{name}'"))?;
        let data_offset = read_u64(r)?;

        tensors.push(TensorInfo {
            name,
            n_dims,
            dims,
            tensor_type,
            data_offset,
        });
    }

    // The data section starts at the next alignment boundary after the header.
    let header_end = r.stream_position()?;
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;
    let data_offset = (header_end + alignment - 1) & !(alignment - 1);

    Ok(GgufFile {
        metadata,
        tensors,
        data_offset,
        file_size,
    })
}

// ============================================================================
// Binary reading helpers.
// ============================================================================

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    Ok(read_u16(r)? as i16)
}

fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    Ok(read_u32(r)? as i32)
}

fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    Ok(read_u64(r)? as i64)
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    if len > 1 << 24 {
        bail!("string too long: {len} bytes");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_meta_value<R: Read>(r: &mut R) -> Result<MetaValue> {
    let vtype = read_u32(r)?;
    match vtype {
        GGUF_TYPE_UINT8 => Ok(MetaValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(MetaValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(MetaValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(MetaValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(MetaValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(MetaValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(MetaValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(MetaValue::Bool(read_bool(r)?)),
        GGUF_TYPE_STRING => Ok(MetaValue::Str(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(MetaValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(MetaValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(MetaValue::F64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => read_meta_array(r),
        _ => bail!("unknown GGUF metadata type: {vtype}"),
    }
}

fn read_meta_array<R: Read>(r: &mut R) -> Result<MetaValue> {
    let elem_type = read_u32(r)?;
    let count = read_u64(r)? as usize;
    if count > 1 << 24 {
        bail!("array too large: {count} elements");
    }
    match elem_type {
        GGUF_TYPE_UINT8 => {
            let mut v = vec![0u8; count];
            for x in v.iter_mut() { *x = read_u8(r)?; }
            Ok(MetaValue::ArrayU8(v))
        }
        GGUF_TYPE_INT8 => {
            let mut v = vec![0i8; count];
            for x in v.iter_mut() { *x = read_i8(r)?; }
            Ok(MetaValue::ArrayI8(v))
        }
        GGUF_TYPE_UINT16 => {
            let mut v = vec![0u16; count];
            for x in v.iter_mut() { *x = read_u16(r)?; }
            Ok(MetaValue::ArrayU16(v))
        }
        GGUF_TYPE_INT16 => {
            let mut v = vec![0i16; count];
            for x in v.iter_mut() { *x = read_i16(r)?; }
            Ok(MetaValue::ArrayI16(v))
        }
        GGUF_TYPE_UINT32 => {
            let mut v = vec![0u32; count];
            for x in v.iter_mut() { *x = read_u32(r)?; }
            Ok(MetaValue::ArrayU32(v))
        }
        GGUF_TYPE_INT32 => {
            let mut v = vec![0i32; count];
            for x in v.iter_mut() { *x = read_i32(r)?; }
            Ok(MetaValue::ArrayI32(v))
        }
        GGUF_TYPE_FLOAT32 => {
            let mut v = vec![0f32; count];
            for x in v.iter_mut() { *x = read_f32(r)?; }
            Ok(MetaValue::ArrayF32(v))
        }
        GGUF_TYPE_BOOL => {
            let mut v = vec![false; count];
            for x in v.iter_mut() { *x = read_bool(r)?; }
            Ok(MetaValue::ArrayBool(v))
        }
        GGUF_TYPE_STRING => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count { v.push(read_string(r)?); }
            Ok(MetaValue::ArrayStr(v))
        }
        GGUF_TYPE_UINT64 => {
            let mut v = vec![0u64; count];
            for x in v.iter_mut() { *x = read_u64(r)?; }
            Ok(MetaValue::ArrayU64(v))
        }
        GGUF_TYPE_INT64 => {
            let mut v = vec![0i64; count];
            for x in v.iter_mut() { *x = read_i64(r)?; }
            Ok(MetaValue::ArrayI64(v))
        }
        GGUF_TYPE_FLOAT64 => {
            let mut v = vec![0f64; count];
            for x in v.iter_mut() { *x = read_f64(r)?; }
            Ok(MetaValue::ArrayF64(v))
        }
        _ => bail!("unknown GGUF array element type: {elem_type}"),
    }
}
