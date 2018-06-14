extern crate image;
extern crate nalgebra as na;

use image::{DynamicImage, GrayImage};
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
    image_from_matrix_1(&img_matrix_2)
        .save("out/gray_2.png")
        .unwrap();
    image_from_matrix_2(&img_matrix_4)
        .save("out/gray_4.png")
        .unwrap();
    image_from_matrix_3(&img_matrix_8)
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

// to_luma() creates a copy of the buffer.
fn image_from_matrix_1(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = DynamicImage::new_luma8(nb_cols as u32, nb_rows as u32).to_luma();
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = image::Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

// I'm not sure what's the memory management for as_luma8().unwrap().to_owned().
fn image_from_matrix_2(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = DynamicImage::new_luma8(nb_cols as u32, nb_rows as u32)
        .as_luma8()
        .unwrap()
        .to_owned();
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = image::Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

// I think that's the only version not copying the buffer but not sure.
fn image_from_matrix_3(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    match DynamicImage::new_luma8(nb_cols as u32, nb_rows as u32) {
        DynamicImage::ImageLuma8(mut gray_img) => {
            for (x, y, pixel) in gray_img.enumerate_pixels_mut() {
                *pixel = image::Luma([mat[(y as usize, x as usize)]]);
            }
            gray_img
        }
        _ => panic!("lol"),
    }
}
