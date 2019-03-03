use byteorder::{BigEndian, ReadBytesExt};
use nalgebra::{DMatrix, Scalar};
use png::{self, HasParameters};
use std::{self, fs::File, io::Cursor, path::Path};

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

pub fn zip_mask_map<T, U, F>(mat: &DMatrix<T>, mask: &DMatrix<bool>, default: U, f: F) -> DMatrix<U>
where
    T: Scalar,
    U: Scalar,
    F: Fn(T) -> U,
{
    mat.zip_map(mask, |x, is_true| if is_true { f(x) } else { default })
}

// Compute the quotient and remainder both at the same time.
pub fn div_rem<T>(x: T, y: T) -> (T, T)
where
    T: std::ops::Div<Output = T> + std::ops::Rem<Output = T> + Copy,
{
    (x / y, x % y)
}

// Check that a coordinate is in the bounds of an image of a given size.
pub fn in_image_bounds(pos: (f32, f32), shape: (usize, usize)) -> bool {
    let x = pos.0;
    let y = pos.1;
    let nrows = shape.0;
    let ncols = shape.1;
    0.0 <= x && x < (ncols - 1) as f32 && 0.0 <= y && y < (nrows - 1) as f32
}
