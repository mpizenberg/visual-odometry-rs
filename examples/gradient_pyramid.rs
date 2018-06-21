extern crate image;
extern crate nalgebra as na;
extern crate num_traits;
extern crate num_traits as num;

mod interop;
mod multires;

use na::DMatrix;

const NB_PIXELS_THRESH: usize = 400;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img = image::open("icl-rgb/0.png")
        .expect("Cannot open image")
        .to_luma();

    // Create an equivalent matrix.
    let img_matrix = interop::matrix_from_image(img);

    // Compute pyramid of matrices.
    let multires_img = multires::pyramid(
        img_matrix,
        |mat| mat,
        |mat| halve_with_thresh(NB_PIXELS_THRESH, mat),
    );

    // Compute pyramid of gradients (without first level).
    let nb_levels = multires_img.len();
    let multires_gradient_norm: Vec<DMatrix<u16>> = multires_img
        .iter()
        .take(nb_levels - 1)
        .map(half_gradient_norm)
        .collect();

    // Save pyramid of images.
    multires_img.iter().enumerate().for_each(|(i, mat)| {
        let out_name = &["out/img_", i.to_string().as_str(), ".png"].concat();
        interop::image_from_matrix(&mat).save(out_name).unwrap();
    });

    // Save pyramid of gradients.
    multires_gradient_norm
        .iter()
        .map(|mat| mat.map(|x| (x as f32).sqrt() as u8))
        .map(|mat| interop::image_from_matrix(&mat))
        .enumerate()
        .for_each(|(i, img)| {
            let out_name = &["out/gradient_norm_", (i + 1).to_string().as_str(), ".png"].concat();
            img.save(out_name).unwrap();
        });
}

// Half resolution

fn halve_with_thresh(thresh: usize, mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    match mat.shape() {
        (r, c) if r * c > thresh => halve(mat),
        _ => None,
    }
}

fn halve(mat: &DMatrix<u8>) -> Option<DMatrix<u8>> {
    multires::halve(mat, |a, b, c, d| {
        let a = a as u16;
        let b = b as u16;
        let c = c as u16;
        let d = d as u16;
        ((a + b + c + d) / 4) as u8
    })
}

// Gradients

fn half_gradient_norm(mat: &DMatrix<u8>) -> DMatrix<u16> {
    multires::halve(mat, |a, b, c, d| {
        let a = a as i32;
        let b = b as i32;
        let c = c as i32;
        let d = d as i32;
        let dx = c + d - a - b;
        let dy = b - a + d - c;
        ((dx * dx + dy * dy) / 8) as u16
    }).unwrap()
}

fn half_gradients(mat: &DMatrix<u8>) -> DMatrix<(i16, i16, u16)> {
    multires::halve(mat, |a, b, c, d| {
        let a = a as i32;
        let b = b as i32;
        let c = c as i32;
        let d = d as i32;
        let dx = c + d - a - b;
        let dy = b - a + d - c;
        let d2 = ((dx * dx + dy * dy) / 4) as u16;
        (dx as i16 / 2, dy as i16 / 2, d2)
    }).unwrap()
}
