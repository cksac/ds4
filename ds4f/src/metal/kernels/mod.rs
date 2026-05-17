//! Kernel dispatch functions.
//!
//! Each sub-module contains thin wrappers around Metal compute kernel
//! dispatches, matching the `ds4_gpu_*` function signatures from ds4_gpu.h.

pub mod attention;
pub mod compressor;
pub mod dense;
pub mod hc;
pub mod indexer;
pub mod misc;
pub mod moe;
pub mod norm;
pub mod rope;
