extern crate image;
extern crate nalgebra as na;

use image::GrayImage;
use na::DMatrix;
use std::path::Path;

const NB_PIXELS_THRESH: usize = 200;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img_path = Path::new("icl-rgb/0.png");
    let img = image::open(&img_path).expect("Cannot open image").to_luma();

    // Create an equivalent matrix.
    let (width, height) = img.dimensions();
    let raw_buffer: Vec<u8> = img.into_raw();
    let img_matrix = DMatrix::from_row_slice(height as usize, width as usize, &raw_buffer);
    let mat_hierarchy = hierarchy(NB_PIXELS_THRESH, img_matrix);

    // Save hierarchy of images.
    mat_hierarchy.iter().enumerate().for_each(|(i, mat)| {
        let out_name = &["out/hierarchy_", i.to_string().as_str(), ".png"].concat();
        image_from_matrix(mat).save(out_name).unwrap();
    });
}

fn hierarchy(thresh: usize, mat: DMatrix<u8>) -> Vec<DMatrix<u8>> {
    let mut matrix_hierarchy = Vec::new();
    matrix_hierarchy.push(mat);
    while let Some(half_res) = half_res_with_thresh(thresh, matrix_hierarchy.last().unwrap()) {
        matrix_hierarchy.push(half_res);
    }
    matrix_hierarchy
}

fn half_res_with_thresh(thresh: usize, mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    let (r, c) = mat.shape();
    if r * c > thresh {
        half_res(mat)
    } else {
        None
    }
}

fn half_res(mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
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
