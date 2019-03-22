// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Miscellaneous helper functions that didn't fit elsewhere.

use byteorder::{BigEndian, ReadBytesExt};
use nalgebra::{DMatrix, Scalar};
use png::{self, HasParameters};
use std::{self, fs::File, io::Cursor, path::Path};

/// Read a 16 bit gray png image from a file.
pub fn read_png_16bits<P: AsRef<Path>>(
    file_path: P,
) -> Result<(usize, usize, Vec<u16>), png::DecodingError> {
    // Load 16 bits PNG depth image.
    let img_file = File::open(file_path)?;
    let mut decoder = png::Decoder::new(img_file);
    // Use the IDENTITY transformation because by default
    // it will use STRIP_16 which only keep 8 bits.
    // See also SWAP_ENDIAN that might be useful
    //   (but seems not possible to use according to documentation).
    decoder.set(png::Transformations::IDENTITY);
    let (info, mut reader) = decoder.read_info()?;
    let mut buffer = vec![0; info.buffer_size()];
    reader.next_frame(&mut buffer)?;

    // Transform buffer into 16 bits slice.
    // if cfg!(target_endian = "big") ...
    let mut buffer_u16 = vec![0; (info.width * info.height) as usize];
    let mut buffer_cursor = Cursor::new(buffer);
    buffer_cursor.read_u16_into::<BigEndian>(&mut buffer_u16)?;

    // Return u16 buffer.
    Ok((info.width as usize, info.height as usize, buffer_u16))
}

/// Map a function onto a matrix, at positions given by a mask.
/// A default value is used at the other positions.
pub fn zip_mask_map<T, U, F>(mat: &DMatrix<T>, mask: &DMatrix<bool>, default: U, f: F) -> DMatrix<U>
where
    T: Scalar,
    U: Scalar,
    F: Fn(T) -> U,
{
    mat.zip_map(mask, |x, is_true| if is_true { f(x) } else { default })
}

/// Compute the quotient and remainder of x/y both at the same time.
pub fn div_rem<T>(x: T, y: T) -> (T, T)
where
    T: std::ops::Div<Output = T> + std::ops::Rem<Output = T> + Copy,
{
    (x / y, x % y)
}
