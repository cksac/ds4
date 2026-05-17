use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;

use super::objc_ext;

pub struct CommandBatch {
    queue: Retained<AnyObject>,
    batch_cb: Option<Retained<AnyObject>>,
    batch_enc: Option<Retained<AnyObject>>,
    pending_cbs: Vec<Retained<AnyObject>>,
}

impl CommandBatch {
    pub fn new(queue: Retained<AnyObject>) -> Self {
        Self { queue, batch_cb: None, batch_enc: None, pending_cbs: Vec::new() }
    }

    pub fn begin(&mut self) -> Result<()> {
        if self.batch_cb.is_some() { anyhow::bail!("batch already active"); }
        let cb = unsafe { objc_ext::queue_command_buffer(&self.queue) }
            .context("failed to create command buffer")?;
        self.batch_cb = Some(cb);
        Ok(())
    }

    pub fn command_buffer(&mut self) -> Result<(Retained<AnyObject>, bool)> {
        if let Some(ref cb) = self.batch_cb {
            Ok((cb.clone(), false))
        } else {
            let cb = unsafe { objc_ext::queue_command_buffer(&self.queue) }
                .context("failed to create command buffer")?;
            Ok((cb, true))
        }
    }

    pub fn compute_encoder(&mut self, cb: &AnyObject, is_batch: bool) -> Result<Retained<AnyObject>> {
        if is_batch {
            if self.batch_enc.is_none() {
                self.batch_enc = Some(unsafe { objc_ext::cb_compute_command_encoder(cb) });
            }
            Ok(self.batch_enc.as_ref().unwrap().clone())
        } else {
            Ok(unsafe { objc_ext::cb_compute_command_encoder(cb) })
        }
    }

    pub fn close_batch_encoder(&mut self) {
        if let Some(enc) = self.batch_enc.take() { unsafe { objc_ext::enc_end_encoding(&enc) }; }
    }

    pub fn flush(&mut self) -> Result<()> {
        let cb = self.batch_cb.take().context("no active batch")?;
        self.close_batch_encoder();
        unsafe { objc_ext::cb_commit(&cb) };
        self.pending_cbs.push(cb);
        let new_cb = unsafe { objc_ext::queue_command_buffer(&self.queue) }
            .context("failed to create command buffer after flush")?;
        self.batch_cb = Some(new_cb);
        Ok(())
    }

    pub fn end(&mut self) -> Result<()> {
        let cb = self.batch_cb.take().context("no active batch")?;
        self.close_batch_encoder();
        self.finish_command_buffer(cb, "command batch")
    }

    pub fn synchronize(&mut self) -> Result<()> {
        if self.batch_cb.is_some() { return self.end(); }
        if !self.pending_cbs.is_empty() { self.wait_pending("synchronize")?; }
        let cb = unsafe { objc_ext::queue_command_buffer(&self.queue) }
            .context("failed to create sync command buffer")?;
        self.finish_command_buffer(cb, "synchronize")
    }

    pub fn blit_copy(&mut self, src: &AnyObject, src_off: u64, dst: &AnyObject, dst_off: u64, bytes: u64) -> Result<()> {
        if bytes == 0 { return Ok(()); }
        let cb = self.batch_cb.clone().context("no active batch for blit")?;
        self.close_batch_encoder();
        let blit = unsafe { objc_ext::cb_blit_command_encoder(&cb) };
        unsafe { objc_ext::blit_copy(&blit, src, src_off as usize, dst, dst_off as usize, bytes as usize); }
        unsafe { objc_ext::blit_end_encoding(&blit); }
        Ok(())
    }

    fn finish_command_buffer(&mut self, cb: Retained<AnyObject>, label: &str) -> Result<()> {
        unsafe { objc_ext::cb_commit(&cb) };
        self.wait_pending(label)?;
        self.wait_one(&cb, label)
    }

    fn wait_pending(&mut self, label: &str) -> Result<()> {
        let drained: Vec<_> = self.pending_cbs.drain(..).collect();
        for pending in drained { self.wait_one(&pending, label)?; }
        Ok(())
    }

    fn wait_one(&self, cb: &AnyObject, label: &str) -> Result<()> {
        unsafe { objc_ext::cb_wait_until_completed(cb) };
        let status = unsafe { objc_ext::cb_status(cb) };
        if status == 5 { // MTLCommandBufferStatusError
            let err_msg = unsafe { objc_ext::cb_error(cb) }
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown error".to_string());
            anyhow::bail!("Metal {} failed: {}", label, err_msg);
        }
        Ok(())
    }
}
