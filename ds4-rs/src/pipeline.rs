use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;
use std::collections::HashMap;
use std::sync::Mutex;

// MTLComputePipelineState is not Send in objc2-metal. Wrap for the static cache.
struct SendPipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>);
unsafe impl Send for SendPipeline {}

static CACHE: Mutex<Option<HashMap<String, SendPipeline>>> = Mutex::new(None);

pub fn init_cache() {
    *CACHE.lock().unwrap() = Some(HashMap::new());
}

pub fn get_pipeline(name: &str) -> Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    CACHE.lock().unwrap().as_ref()?.get(name).map(|sp| sp.0.clone())
}

pub fn cache_pipeline(name: &str, pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>) {
    if let Some(ref mut c) = *CACHE.lock().unwrap() {
        c.insert(name.to_string(), SendPipeline(pipeline));
    }
}
