use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;

use super::objc_ext;
use super::shaders;

pub struct MetalDevice {
    pub device: Retained<AnyObject>,
    pub queue: Retained<AnyObject>,
    pub library: Retained<AnyObject>,
}

impl MetalDevice {
    pub fn init() -> Result<Self> {
        let device = unsafe { objc_ext::mtl_create_system_default_device() }
            .context("Metal device not available")?;

        let name = unsafe { objc_ext::device_name(&device) }.to_string();
        eprintln!("ds4: Metal device {}", name);

        let queue = unsafe { objc_ext::device_new_command_queue(&device) }
            .context("failed to create Metal command queue")?;

        let source = shaders::full_source();
        let options = unsafe { objc_ext::compile_options_new() };
        let library = unsafe {
            objc_ext::device_new_library_with_source(&device, &source, &options)
        }
        .map_err(|err| anyhow::anyhow!("Metal shader compilation failed: {}", err.localizedDescription()))?;

        Ok(Self { device, queue, library })
    }
}
