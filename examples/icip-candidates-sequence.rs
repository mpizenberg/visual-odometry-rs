extern crate computer_vision_rs as cv;
extern crate nalgebra as na;

use cv::candidates;
use cv::icl_nuim;
use cv::multires;
use cv::view;

use na::DMatrix;
use std::fs;

const OUT_DIR: &str = "out/icip/candidates_sequence/";

// #[allow(dead_code)]
fn main() {
    // Configuration.
    let nb_levels = 6;
    let diff_threshold = 7;
    let nb_img = 1509;

    // Create output directory.
    fs::create_dir_all(OUT_DIR).unwrap();

    for id_img in 0..nb_img {
        // Load image as grayscale (u8) and depth map (u16).
        let (img, _) = icl_nuim::open_imgs(id_img).unwrap();
        let candidates = generate_candidates(&img, nb_levels, diff_threshold);

        // Save full resolution inverse depth map on disk.
        view::candidates_on_image(&img, &candidates)
            .save(&format!("{}{}.png", OUT_DIR, id_img))
            .unwrap();
    }
}

fn generate_candidates(img: &DMatrix<u8>, nb_levels: usize, diff_threshold: u16) -> DMatrix<bool> {
    // Compute pyramid of matrices.
    let multires_img = multires::mean_pyramid(nb_levels, img.clone());

    // Compute pyramid of gradients (without first level).
    let mut multires_gradients_squared_norm = multires::gradients_squared_norm(&multires_img);
    multires_gradients_squared_norm.insert(0, grad_squared_norm(&multires_img[0]));

    // Choose candidates based on gradients norms.
    // The first candidate matrix in the vector is the lowest resolution.
    let mut multires_candidates =
        candidates::select(diff_threshold, &multires_gradients_squared_norm);
    multires_candidates.pop().unwrap()
}

// Helper ######################################################################

fn grad_squared_norm(im: &DMatrix<u8>) -> DMatrix<u16> {
    let (nb_rows, nb_cols) = im.shape();
    let top = im.slice((0, 1), (nb_rows - 2, nb_cols - 2));
    let bottom = im.slice((2, 1), (nb_rows - 2, nb_cols - 2));
    let left = im.slice((1, 0), (nb_rows - 2, nb_cols - 2));
    let right = im.slice((1, 2), (nb_rows - 2, nb_cols - 2));
    let mut grad_squared_norm_mat = DMatrix::zeros(nb_rows, nb_cols);
    let mut grad_inner = grad_squared_norm_mat.slice_mut((1, 1), (nb_rows - 2, nb_cols - 2));
    for j in 0..nb_cols - 2 {
        for i in 0..nb_rows - 2 {
            let gx = right[(i, j)] as i32 - left[(i, j)] as i32;
            let gy = bottom[(i, j)] as i32 - top[(i, j)] as i32;
            grad_inner[(i, j)] = ((gx * gx + gy * gy) / 4) as u16;
        }
    }
    grad_squared_norm_mat
}
