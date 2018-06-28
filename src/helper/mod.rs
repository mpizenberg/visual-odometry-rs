use byteorder::{BigEndian, ReadBytesExt};
use png::{self, HasParameters};
use std::fs::File;
use std::io::Cursor;

pub fn read_png_16bits(file_path: &str) -> Result<(usize, usize, Vec<u16>), png::DecodingError> {
    // Load 16 bits PNG depth image.
    let img_file = File::open(file_path)?;
    let mut decoder = png::Decoder::new(img_file);
    // Use the IDENTITY transformation because by default
    // it will use STRIP_16 which only keep 8 bits.
    decoder.set(png::Transformations::IDENTITY);
    let (info, mut reader) = decoder.read_info()?;
    let mut buffer = vec![0; info.buffer_size()];
    reader.next_frame(&mut buffer)?;

    // Transform buffer into 16 bits slice.
    let mut buffer_u16 = vec![0; (info.width * info.height) as usize];
    let mut buffer_cursor = Cursor::new(buffer);
    buffer_cursor.read_u16_into::<BigEndian>(&mut buffer_u16)?;

    // Return u16 buffer.
    Ok((info.width as usize, info.height as usize, buffer_u16))
}
