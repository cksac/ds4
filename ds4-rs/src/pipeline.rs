use metal::*;
use std::collections::HashMap;
use std::sync::Mutex;

// ComputePipelineState is !Send in metal 0.20. Wrap for static cache.
struct SendPipeline(ComputePipelineState);
unsafe impl Send for SendPipeline {}

static CACHE: Mutex<Option<HashMap<String, SendPipeline>>> = Mutex::new(None);

pub fn init_cache() {
    *CACHE.lock().unwrap() = Some(HashMap::new());
}

pub fn get_pipeline(name: &str) -> Option<ComputePipelineState> {
    let cache = CACHE.lock().unwrap();
    cache.as_ref()?.get(name).map(|sp| sp.0.clone())
}

pub fn cache_pipeline(name: &str, pipeline: &ComputePipelineState) {
    let mut cache = CACHE.lock().unwrap();
    if let Some(ref mut c) = *cache {
        c.insert(name.to_string(), SendPipeline(pipeline.clone()));
    }
}

pub fn cache_pipeline_owned(name: &str, pipeline: ComputePipelineState) {
    let mut cache = CACHE.lock().unwrap();
    if let Some(ref mut c) = *cache {
        c.insert(name.to_string(), SendPipeline(pipeline));
    }
}
