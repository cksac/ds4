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

#[repr(C)]
pub struct GetRowsArgs {
    pub ne00t: i32,
    pub ne00: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

#[repr(C)]
pub struct BinArgs {
    pub ne00: i32, pub ne01: i32, pub ne02: i32, pub ne03: i32,
    pub nb00: u64, pub nb01: u64, pub nb02: u64, pub nb03: u64,
    pub ne10: i32, pub ne11: i32, pub ne12: i32, pub ne13: i32,
    pub nb10: u64, pub nb11: u64, pub nb12: u64, pub nb13: u64,
    pub ne0:  i32, pub ne1:  i32, pub ne2:  i32, pub ne3:  i32,
    pub nb0:  u64, pub nb1:  u64, pub nb2:  u64, pub nb3:  u64,
    pub offs: u64,
    pub o1: [u64; 8],
}
