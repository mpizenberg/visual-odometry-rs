extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;
extern crate num_traits as num;

use cv::interop;

use na::{DMatrix, Scalar};
use num::NumCast;
use std::ops::{Add, Div};
use std::time::Instant;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img = image::open("icl-rgb/0.png")
        .expect("Cannot open image")
        .to_luma();

    // Save image in a file to visualize it.
    img.save("out/gray.png").unwrap();

    // Create an equivalent matrix.
    let (width, height) = img.dimensions();
    let raw_buffer: Vec<u8> = img.into_raw();
    let img_matrix = DMatrix::from_row_slice(height as usize, width as usize, &raw_buffer);
    let img_matrix_2 = half_resolution_from_fn(&img_matrix).unwrap();
    let img_matrix_4 = half_resolution_slice(&img_matrix_2).unwrap();
    let img_matrix_8 = half_res_generic::<u8, u16>(&img_matrix_4).unwrap();

    // Benchmark different versions.
    let t_0 = Instant::now();
    for _ in 0..1000 {
        half_resolution_from_fn(&img_matrix).unwrap();
    }
    let dt = Instant::now().duration_since(t_0);
    println!("{:?}", dt);

    // save half resolution image.
    interop::image_from_matrix(&img_matrix_2)
        .save("out/gray_2.png")
        .unwrap();
    interop::image_from_matrix(&img_matrix_4)
        .save("out/gray_4.png")
        .unwrap();
    interop::image_from_matrix(&img_matrix_8)
        .save("out/gray_8.png")
        .unwrap();
}

// Generic version.
// Use with caution.
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

// This version is more correct since it has only 1 rounding per subpixel
// Slower until this PR get integrated into a new version:
// https://github.com/sebcrozet/nalgebra/pull/355
fn half_resolution_from_fn(mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        Some(DMatrix::from_fn(half_r, half_c, |i, j| {
            let m_00 = mat[(2 * i, 2 * j)] as u16;
            let m_10 = mat[(2 * i + 1, 2 * j)] as u16;
            let m_01 = mat[(2 * i, 2 * j + 1)] as u16;
            let m_11 = mat[(2 * i + 1, 2 * j + 1)] as u16;
            ((m_00 + m_10 + m_01 + m_11) / 4) as u8
        }))
    }
}

// slightly less correct since it does 2 roundings per subpixel.
fn half_resolution_slice(mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    let (r, c) = mat.shape();
    let half_r = r / 2;
    let half_c = c / 2;
    if half_r == 0 || half_c == 0 {
        None
    } else {
        let half_left = mat.columns_with_step(0, half_c, 1);
        let half_right = mat.columns_with_step(1, half_c, 1);
        let half_x = half_right.zip_map(&half_left, |xr, xl| ((xr as u16 + xl as u16) / 2) as u8);
        let half_top = half_x.rows_with_step(0, half_r, 1);
        let half_bottom = half_x.rows_with_step(1, half_r, 1);
        Some(half_bottom.zip_map(&half_top, |yb, yt| ((yb as u16 + yt as u16) / 2) as u8))
    }
}
