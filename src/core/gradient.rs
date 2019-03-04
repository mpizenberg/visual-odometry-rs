//! Helper function to compute gradients

/// Horizontal gradient in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_x(a: u8, b: u8, c: u8, d: u8) -> i16 {
    let a = a as i16;
    let b = b as i16;
    let c = c as i16;
    let d = d as i16;
    (c + d - a - b) / 2
}

/// Vertical gradient in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_y(a: u8, b: u8, c: u8, d: u8) -> i16 {
    let a = a as i16;
    let b = b as i16;
    let c = c as i16;
    let d = d as i16;
    (b - a + d - c) / 2
}

/// Gradient squared norm in a 2x2 pixels block.
///
/// The block is of the form:
///   a c
///   b d
pub fn bloc_squared_norm(a: u8, b: u8, c: u8, d: u8) -> u16 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let d = d as i32;
    let dx = c + d - a - b;
    let dy = b - a + d - c;
    // I have checked that the max value is in u16.
    ((dx * dx + dy * dy) / 4) as u16
}
