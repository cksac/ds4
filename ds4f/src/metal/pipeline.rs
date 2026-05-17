use std::collections::HashMap;
use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::NSString;

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

    pub fn get(&mut self, name: &str) -> Result<Retained<AnyObject>> {
        if let Some(ps) = self.pipelines.get(name) { return Ok(ps.clone()); }
        let ns_name = NSString::from_str(name);
        let fn_obj = unsafe { objc_ext::library_new_function(&self.library, &ns_name) }
            .with_context(|| format!("Metal function '{}' not found", name))?;
        let ps = unsafe { objc_ext::device_new_compute_pipeline(&self.device, &fn_obj) }
            .map_err(|e| anyhow::anyhow!("Metal pipeline '{}' failed: {}", name, e.localizedDescription()))?;
        self.pipelines.insert(name.to_string(), ps.clone());
        Ok(ps)
    }

    pub fn get_with_constants(
        &mut self, name: &str, constants: &[(usize, usize, i32)],
    ) -> Result<Retained<AnyObject>> {
        let key = Self::make_key(name, constants);
        if let Some(ps) = self.pipelines.get(&key) { return Ok(ps.clone()); }

        let fn_constants = unsafe { objc_ext::fn_constants_new() };
        for &(index, data_type, value) in constants {
            let v: i32 = value;
            unsafe { objc_ext::fn_constants_set_value(&fn_constants, &v as *const i32 as _, data_type, index); }
        }

        let ns_name = NSString::from_str(name);
        let fn_obj = unsafe { objc_ext::library_new_function_with_constants(&self.library, &ns_name, &fn_constants) }
            .map_err(|e| anyhow::anyhow!("Metal function '{}' with constants not found: {}", name, e.localizedDescription()))?;
        let ps = unsafe { objc_ext::device_new_compute_pipeline(&self.device, &fn_obj) }
            .map_err(|e| anyhow::anyhow!("Metal pipeline '{}' with constants failed: {}", name, e.localizedDescription()))?;

        self.pipelines.insert(key, ps.clone());
        Ok(ps)
    }

    fn make_key(name: &str, constants: &[(usize, usize, i32)]) -> String {
        let mut key = String::from(name);
        for &(index, _ty, value) in constants { key.push_str(&format!("_{}={}", index, value)); }
        key
    }
}
