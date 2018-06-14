extern crate image;
extern crate nalgebra as na;

use image::GrayImage;
use na::DMatrix;
use std::path::Path;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img_path = Path::new("icl-rgb/0.png");
    let img = image::open(&img_path).expect("Cannot open image").to_luma();

    // Save image in a file to visualize it.
    img.save("out/gray.png").unwrap();

    // Create an equivalent matrix.
    let (width, height) = img.dimensions();
    let raw_buffer: Vec<u8> = img.into_raw();
    let img_matrix = DMatrix::from_row_slice(height as usize, width as usize, &raw_buffer);
    let img_matrix_2 = half_resolution(&img_matrix).unwrap();
    let img_matrix_4 = half_resolution(&img_matrix_2).unwrap();
    let img_matrix_8 = half_resolution(&img_matrix_4).unwrap();

    // save half resolution image.
    image_from_matrix(&img_matrix_2)
        .save("out/gray_2.png")
        .unwrap();
    image_from_matrix(&img_matrix_4)
        .save("out/gray_4.png")
        .unwrap();
    image_from_matrix(&img_matrix_8)
        .save("out/gray_8.png")
        .unwrap();
}

fn half_resolution(mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        let half_r_mat = DMatrix::from_fn(half_r, c, |i, j| {
            ((mat[(2 * i, j)] as u16 + mat[(2 * i + 1, j)] as u16) / 2) as u8
        });
        let half_mat = DMatrix::from_fn(half_r, half_c, |i, j| {
            ((half_r_mat[(i, 2 * j)] as u16 + half_r_mat[(i, 2 * j + 1)] as u16) / 2) as u8
        });
        Some(half_mat)
    }
}

fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = image::Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}
