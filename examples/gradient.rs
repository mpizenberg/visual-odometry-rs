extern crate image;
extern crate nalgebra as na;

use image::GrayImage;
use na::DMatrix;
use std::path::Path;
use std::time::{Duration, Instant};

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img_path = Path::new("icl-rgb/0.png");
    let img = image::open(&img_path).expect("Cannot open image").to_luma();

    // Create an equivalent matrix.
    let img_matrix = matrix_from_image(img);

    // Compute discrete spacial gradient.
    let t_0 = Instant::now();
    for _ in 0..1000 {
        let (g_x, g_y) = gradient_2(&img_matrix);
    }
    let dt = Instant::now().duration_since(t_0);
    println!("{:?}", dt);
    let (g_x, g_y) = gradient_2(&img_matrix);
    let g_norm_sqr = g_x.zip_map(&g_y, |gx, gy| {
        (gx as i32 * gx as i32 + gy as i32 * gy as i32) as u16
    });

    // Save gradient image.
    let g_x_u8 = g_x.map(|x| ((x + 256) / 2) as u8);
    let g_y_u8 = g_y.map(|x| ((x + 256) / 2) as u8);
    let g_norm = g_norm_sqr.map(|x| (x as f32).sqrt() as u8);
    image_from_matrix(&g_x_u8)
        .save("out/gradient_x.png")
        .unwrap();
    image_from_matrix(&g_y_u8)
        .save("out/gradient_y.png")
        .unwrap();
    image_from_matrix(&g_norm)
        .save("out/gradient_norm.png")
        .unwrap();
}

fn gradient_2(mat: &DMatrix<u8>) -> (DMatrix<i16>, DMatrix<i16>) {
    (gradient_x_inner_from_fn(mat), gradient_y_inner_from_fn(mat))
}

fn gradient_x(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    let mut grad_x = DMatrix::zeros(nb_rows, nb_cols);
    if nb_rows >= 2 && nb_cols >= 2 {
        for c in 1..(nb_cols - 1) {
            for r in 1..(nb_rows - 1) {
                grad_x[(r, c)] = mat[(r, c + 1)] as i16 - mat[(r, c - 1)] as i16;
            }
        }
    }
    grad_x
}

fn gradient_x_inner_from_fn(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    DMatrix::from_fn(nb_rows - 2, nb_cols - 2, |r, c| {
        mat[(r + 1, c + 2)] as i16 - mat[(r + 1, c)] as i16
    })
}

fn gradient_x_inner_slice(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    assert!(nb_rows >= 2);
    assert!(nb_cols >= 2);
    mat.slice((1, 2), (nb_rows - 2, nb_cols - 2)).zip_map(
        &mat.slice((1, 0), (nb_rows - 2, nb_cols - 2)),
        |x_2, x_0| x_2 as i16 - x_0 as i16,
    )
}

fn gradient_y(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    let mut grad_y = DMatrix::zeros(nb_rows, nb_cols);
    if nb_rows >= 2 && nb_cols >= 2 {
        for c in 1..(nb_cols - 1) {
            for r in 1..(nb_rows - 1) {
                grad_y[(r, c)] = mat[(r + 1, c)] as i16 - mat[(r - 1, c)] as i16;
            }
        }
    }
    grad_y
}

fn gradient_y_inner_from_fn(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    DMatrix::from_fn(nb_rows - 2, nb_cols - 2, |r, c| {
        mat[(r + 2, c + 1)] as i16 - mat[(r, c + 1)] as i16
    })
}

fn gradient_y_inner_slice(mat: &DMatrix<u8>) -> DMatrix<i16> {
    let (nb_rows, nb_cols) = mat.shape();
    assert!(nb_rows >= 2);
    assert!(nb_cols >= 2);
    mat.slice((2, 1), (nb_rows - 2, nb_cols - 2)).zip_map(
        &mat.slice((0, 1), (nb_rows - 2, nb_cols - 2)),
        |y_2, y_0| y_2 as i16 - y_0 as i16,
    )
}

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
