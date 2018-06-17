extern crate image;
extern crate nalgebra as na;
extern crate num_traits;
extern crate num_traits as num;

use image::GrayImage;
use na::{DMatrix, Scalar};
use num::NumCast;
use std::ops::{Add, Div};

const NB_PIXELS_THRESH: usize = 400;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img = image::open("icl-rgb/0.png")
        .expect("Cannot open image")
        .to_luma();

    // Create an equivalent matrix.
    let img_matrix = matrix_from_image(img);

    // Compute pyramid of matrices.
    let pyramid_mat = hierarchy(NB_PIXELS_THRESH, img_matrix);

    // Compute a pyramid of gradients.
    let pyramid_gradients: Vec<DMatrix<u16>> = pyramid_mat.iter().map(gradient_norm).collect();

    // Use the image higher resolution instead of gradient.
    let mut pyramid_g_bis = Vec::new();
    pyramid_g_bis.push(gradient_norm(&pyramid_mat[0]));
    pyramid_mat
        .iter()
        .take(pyramid_mat.len() - 1)
        .for_each(|mat| pyramid_g_bis.push(grad_from_higher_res(&mat)));

    // Save pyramid of gradient norms computed from higher res.
    pyramid_g_bis.iter().enumerate().for_each(|(i, mat)| {
        let out_name = &["out/gradient_level_bis_", i.to_string().as_str(), ".png"].concat();
        image_from_matrix(&mat.map(|x| (x as f32).sqrt() as u8))
            .save(out_name)
            .unwrap();
    });

    // Save pyramid of gradient norms.
    pyramid_gradients.iter().enumerate().for_each(|(i, mat)| {
        let out_name = &["out/gradient_level_", i.to_string().as_str(), ".png"].concat();
        image_from_matrix(&mat.map(|x| (x as f32).sqrt() as u8))
            .save(out_name)
            .unwrap();
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

// Half resolution

fn half_res_with_thresh(thresh: usize, mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    let (r, c) = mat.shape();
    if r * c > thresh {
        half_res_generic::<u8, u16>(mat)
    } else {
        None
    }
}

fn half_res_generic<T, U>(mat: &DMatrix<T>) -> Option<DMatrix<T>>
where
    T: Scalar + NumCast,
    U: Copy + NumCast + Add<U, Output = U> + Div<U, Output = U>,
{
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        let four: U = num::cast(4u8).unwrap();
        let half_mat = DMatrix::<T>::from_fn(half_r, half_c, |i, j| {
            let a: U = num::cast(mat[(2 * i, 2 * j)]).unwrap();
            let b: U = num::cast(mat[(2 * i + 1, 2 * j)]).unwrap();
            let c: U = num::cast(mat[(2 * i, 2 * j + 1)]).unwrap();
            let d: U = num::cast(mat[(2 * i + 1, 2 * j + 1)]).unwrap();
            num::cast((a + b + c + d) / four).unwrap()
        });
        Some(half_mat)
    }
}

// Compute gradients

fn grad_from_higher_res(mat: &DMatrix<u8>) -> DMatrix<u16> {
    let (nb_rows, nb_cols) = mat.shape();
    DMatrix::from_fn(nb_rows / 2, nb_cols / 2, |i, j| {
        // | a  c |
        // | b  d |
        let a = mat[(2 * i, 2 * j)] as i32;
        let b = mat[(2 * i + 1, 2 * j)] as i32;
        let c = mat[(2 * i, 2 * j + 1)] as i32;
        let d = mat[(2 * i + 1, 2 * j + 1)] as i32;
        // max 2 * 255
        let dx = c + d - a - b;
        let dy = b - a + d - c;
        ((dx * dx + dy * dy) / 8) as u16
    })
}

fn gradient_norm(mat: &DMatrix<u8>) -> DMatrix<u16> {
    let (g_x, g_y) = gradients(&mat);
    let g_norm_sqr = g_x.zip_map(&g_y, |gx, gy| {
        (gx as i32 * gx as i32 + gy as i32 * gy as i32) as u16
    });
    g_norm_sqr
}

fn gradients(mat: &DMatrix<u8>) -> (DMatrix<i16>, DMatrix<i16>) {
    (gradient_x(mat), gradient_y(mat))
}

fn gradient_x(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    DMatrix::from_fn(nb_rows - 2, nb_cols - 2, |r, c| {
        mat[(r + 1, c + 2)] as i16 - mat[(r + 1, c)] as i16
    })
}

fn gradient_y(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    DMatrix::from_fn(nb_rows - 2, nb_cols - 2, |r, c| {
        mat[(r + 2, c + 1)] as i16 - mat[(r, c + 1)] as i16
    })
}

// Conversion Image <-> Matrix helpers.

fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = image::Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

fn matrix_from_image(img: GrayImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}
