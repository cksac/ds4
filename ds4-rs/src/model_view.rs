use crate::gguf::GgufModel;

pub struct ModelViews {
    pub buffers: Vec<()>,
}

impl ModelViews {
    pub fn new(_model: &GgufModel) -> Result<Self, &'static str> {
        Ok(ModelViews { buffers: Vec::new() })
    }
}
