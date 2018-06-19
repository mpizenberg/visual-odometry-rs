extern crate nalgebra as na;

mod helper;

use na::DMatrix;

// #[allow(dead_code)]
fn main() {
    // Read 16 bits PNG image.
    let (width, height, buffer_u16) = helper::read_png_16bits("icl-depth/0.png").unwrap();
    // Transform it into a matrix.
    let img_mat: DMatrix<u16> = DMatrix::from_row_slice(height, width, buffer_u16.as_slice());
    let img_slice = img_mat.slice((0, 0), (3, 3));
    println!("Top left: {}", img_slice);
}
