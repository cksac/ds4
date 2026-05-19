use std::collections::HashMap;
use std::ptr::NonNull;
use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDataType, MTLFunctionConstantValues};

use super::objc_ext;

pub struct PipelineCache {
    device: Retained<AnyObject>,
    library: Retained<AnyObject>,
    pipelines: HashMap<String, Retained<AnyObject>>,
}

impl PipelineCache {
    pub fn new(device: Retained<AnyObject>, library: Retained<AnyObject>) -> Self {
        Self { device, library, pipelines: HashMap::new() }
    }

    /// Get or compile a pipeline. Tries with default function constants
    /// first (nsg=4 at index 600, Short type), then falls back to plain.
    /// This avoids triggering Metal's fatal assertion for functions with
    /// unresolved function constants.
    pub fn put(&mut self, key: &str, ps: Retained<AnyObject>) {
        self.pipelines.insert(key.to_string(), ps);
    }

    pub fn get(&mut self, name: &str) -> Result<Retained<AnyObject>> {
        if let Some(ps) = self.pipelines.get(name) { return Ok(ps.clone()); }

        // Try with default nsg=4 first (avoid Metal assertion for matmul-like kernels)
        if let Ok(ps) = self.compile_with_nsg(name, 4) {
            self.pipelines.insert(name.to_string(), ps.clone());
            return Ok(ps);
        }

        // Try without function constants (for simple kernels like rms_norm, copy, etc.)
        let ns_name = NSString::from_str(name);
        if let Some(fn_obj) = unsafe { objc_ext::library_new_function(&self.library, &ns_name) } {
            if let Ok(ps) = unsafe { objc_ext::device_new_compute_pipeline(&self.device, &fn_obj) } {
                self.pipelines.insert(name.to_string(), ps.clone());
                return Ok(ps);
            }
        }

        // Try other common nsg values
        for &nsg in &[8i16, 2i16] {
            if let Ok(ps) = self.compile_with_nsg(name, nsg) {
                self.pipelines.insert(name.to_string(), ps.clone());
                return Ok(ps);
            }
        }

        anyhow::bail!("pipeline '{}' could not be compiled", name)
    }

    fn compile_with_nsg(&self, name: &str, nsg: i16) -> Result<Retained<AnyObject>> {
        let fcv = unsafe { MTLFunctionConstantValues::new() };
        unsafe { fcv.setConstantValue_type_atIndex(NonNull::from(&nsg).cast(), MTLDataType::Short, 600); }
        let fn_obj = unsafe { objc_ext::library_new_function_with_constants(&self.library, &NSString::from_str(name), &fcv) }
            .map_err(|_| anyhow::anyhow!("fn not found"))?;
        let ps = unsafe { objc_ext::device_new_compute_pipeline(&self.device, &fn_obj) }
            .map_err(|_| anyhow::anyhow!("ps failed"))?;
        Ok(ps)
    }

    pub fn get_with_constants(&mut self, name: &str, constants: &[(usize, usize, i32)]) -> Result<Retained<AnyObject>> {
        let key = Self::make_key(name, constants);
        if let Some(ps) = self.pipelines.get(&key) { return Ok(ps.clone()); }

        let fcv = unsafe { MTLFunctionConstantValues::new() };
        for &(index, data_type_code, value) in constants {
            match data_type_code {
                53 => { let v: bool = value != 0; unsafe { fcv.setConstantValue_type_atIndex(NonNull::from(&v).cast(), MTLDataType::Bool, index); } }
                37 => { let v: i16 = value as i16; unsafe { fcv.setConstantValue_type_atIndex(NonNull::from(&v).cast(), MTLDataType::Short, index); } }
                _ => { let v: i32 = value; unsafe { fcv.setConstantValue_type_atIndex(NonNull::from(&v).cast(), MTLDataType::Int, index); } }
            }
        }

        let fn_obj = unsafe { objc_ext::library_new_function_with_constants(&self.library, &NSString::from_str(name), &fcv) }
            .map_err(|e| anyhow::anyhow!("fn '{}' w/consts: {}", name, e.localizedDescription()))?;
        let ps = unsafe { objc_ext::device_new_compute_pipeline(&self.device, &fn_obj) }
            .map_err(|e| anyhow::anyhow!("ps '{}' w/consts: {}", name, e.localizedDescription()))?;
        self.pipelines.insert(key, ps.clone());
        Ok(ps)
    }

    fn make_key(name: &str, constants: &[(usize, usize, i32)]) -> String {
        let mut key = String::from(name);
        for &(index, _, value) in constants { key.push_str(&format!("_{}={}", index, value)); }
        key
    }
}
