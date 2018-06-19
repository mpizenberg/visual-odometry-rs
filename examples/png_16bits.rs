// extern crate byteorder;
extern crate nalgebra as na;
// extern crate png;

mod helper;

// use byteorder::{BigEndian, ReadBytesExt};
use na::DMatrix;
// use png::HasParameters;
// use std::fs::File;
// use std::io::Cursor;

// #[allow(dead_code)]
fn main() {
    // // Load 16 bits PNG depth image.
    // let img_file = File::open("icl-depth/0.png").unwrap();
    // let mut decoder = png::Decoder::new(img_file);
    // // Use the IDENTITY transformation because by default
    // // it will use STRIP_16 which only keep 8 bits.
    // decoder.set(png::Transformations::IDENTITY);
    // let (info, mut reader) = decoder.read_info().unwrap();
    // let mut buffer = vec![0; info.buffer_size()];
    // reader.next_frame(&mut buffer).unwrap();
    //
    // // Display image metadata.
    // println!("info: {:?}", info.width);
    // println!("height: {:?}", info.height);
    // println!("bit depth: {:?}", info.bit_depth);
    // println!("buffer size: {:?}", info.buffer_size());
    //
    // // Transform buffer into 16 bits slice.
    // let mut buffer_u16 = vec![0; (info.width * info.height) as usize];
    // let mut buffer_cursor = Cursor::new(buffer);
    // buffer_cursor
    //     .read_u16_into::<BigEndian>(&mut buffer_u16)
    //     .unwrap();

    // Read 16 bits PNG image.
    let (width, height, buffer_u16) = helper::read_png_16bits("icl-depth/0.png").unwrap();
    // Transform it into a matrix.
    let img_mat: DMatrix<u16> = DMatrix::from_row_slice(height, width, buffer_u16.as_slice());
    let img_slice = img_mat.slice((0, 0), (3, 3));
    println!("Top left: {}", img_slice);
}
