extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use cv::camera::{Camera, Extrinsics, Intrinsics};
use cv::candidates;
use cv::helper;
use cv::interop;
use cv::inverse_depth;
use cv::multires;

use inverse_depth::InverseDepth;
use na::{DMatrix, Point2};

fn open_icl_data(id: u32) -> Result<(DMatrix<u8>, DMatrix<u16>), image::ImageError> {
    let img_mat =
        interop::matrix_from_image(image::open(&format!("icl-rgb/{}.png", id))?.to_luma());
    let (w, h, buffer) = helper::read_png_16bits(&format!("icl-depth/{}.png", id))?;
    let depth_map = DMatrix::from_row_slice(h, w, buffer.as_slice());
    Ok((img_mat, depth_map))
}

// #[allow(dead_code)]
fn main() {
    // Load the rgb and depth images of frame 1.
    let (rgb_1, depth_1) = open_icl_data(1).unwrap();

    // Compute candidates points.
    let multires_rgb_1 = multires::mean_pyramid(6, rgb_1);
    let candidates = candidates::select(&multires::gradients(&multires_rgb_1))
        .pop()
        .unwrap();

    // Create an inverse depth map with values only at point candidates.
    // This is to emulate result of back projection of known points into a new keyframe.
    let half_res_depth = multires::halve(&depth_1, |a, b, c, d| {
        ((a as u32 + b as u32 + c as u32 + d as u32) / 4) as u16
    }).unwrap();
    let idepth_candidates_1 = helper::zip_mask_map(
        &half_res_depth,
        &candidates,
        InverseDepth::Unknown,
        inverse_depth::from_depth,
    );

    // Create a multires inverse depth map pyramid.
    let fuse_statistical =
        |a, b, c, d| inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar);
    let multires_idepth_statistical = multires::pyramid_with_max_n_levels(
        5,
        idepth_candidates_1,
        |mat| mat,
        |mat| multires::halve(&mat, fuse_statistical),
    );

    // Load camera extrinsics ground truth and camera intrinsics.
    let intrinsics = Intrinsics {
        principal_point: (319.5, 239.5),
        focal_length: 1.0,
        scaling: (481.20, -480.00),
        skew: 0.0,
    };
    let extrinsics = Extrinsics::read_from_tum_file("data/trajectory-gt.txt").unwrap();
    println!("nb images: {}", extrinsics.len());
    let camera_1 = Camera::new(intrinsics.clone(), extrinsics[1].clone());
    let camera_600 = Camera::new(intrinsics.clone(), extrinsics[600].clone());

    // On lower res image, re-project candidates on new image.
    // let (rgb_600, _) = open_icl_data(600).unwrap();
    // let multires_rgb_600 = multires::mean_pyramid(6, rgb_600);
    let lower_res_idepth = multires_idepth_statistical.last().unwrap();
    // let lower_res_rgb_1 = multires_rgb_1.last().unwrap();
    // let lower_res_rgb_600 = multires_rgb_600.last().unwrap();
    // let mut reprojection_error = 0.0;
    let mut total_weight = 0.0;
    let (nrows, ncols) = lower_res_idepth.shape();
    lower_res_idepth
        .iter()
        .enumerate()
        .for_each(|(index, idepth_enum)| {
            if let InverseDepth::WithVariance(idepth, variance) = idepth_enum {
                let (col, row) = helper::div_rem(index, nrows);
                let current_weight = 1.0 / variance;
                let reprojected = camera_600.project(
                    camera_1.back_project(Point2::new(col as f32, row as f32), 1.0 / idepth),
                );
                let new_pos = reprojected.as_slice();
                let x = new_pos[0] / new_pos[2];
                let y = new_pos[1] / new_pos[2];
                if helper::in_image_bounds((x, y), (nrows, ncols)) {
                    total_weight += current_weight;
                }
            }
        });
    println!("total weight: {}", total_weight);
}
