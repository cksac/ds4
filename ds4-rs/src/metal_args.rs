#[repr(C)]
pub struct RouterSelectOneArgs {
    pub has_bias: u32,
    pub hash_mode: u32,
    pub use_token_buffer: u32,
    pub token: u32,
    pub hash_rows: u32,
}

#[repr(C)]
pub struct Ratio4ShiftArgs {
    pub width: u32,
}
