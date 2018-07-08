extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra;

use cv::camera::{Camera, Extrinsics, Intrinsics};
use cv::candidates;
use cv::helper;
use cv::interop;
use cv::inverse_depth::{self, InverseDepth};
use cv::multires;
use cv::optimization;

use nalgebra::DMatrix;

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
    let idepth_candidates = helper::zip_mask_map(
        &half_res_depth,
        &candidates,
        InverseDepth::Unknown,
        inverse_depth::from_depth,
    );

    // Load camera extrinsics ground truth and camera intrinsics.
    let intrinsics = Intrinsics {
        principal_point: (319.5, 239.5),
        focal_length: 1.0,
        scaling: (481.20, -480.00),
        skew: 0.0,
    };
    let extrinsics = Extrinsics::read_from_tum_file("data/trajectory-gt.txt").unwrap();
    let camera_1_multires = Camera::new(intrinsics.clone(), extrinsics[0].clone()).multi_res(6);
    let camera_600_multires = Camera::new(intrinsics.clone(), extrinsics[599].clone()).multi_res(6);

    let (rgb_600, _) = open_icl_data(600).unwrap();
    let multires_rgb_600 = multires::mean_pyramid(6, rgb_600);

    let eval_strat = |strat: fn(_) -> _| {
        eval_strategy_reprojection(
            idepth_candidates.clone(),
            &multires_rgb_1,
            &multires_rgb_600,
            &camera_1_multires,
            &camera_600_multires,
            strat,
        )
    };
    let dso_eval = eval_strat(inverse_depth::strategy_dso_mean);
    let stat_eval = eval_strat(inverse_depth::strategy_statistically_similar);
    let _random_eval = eval_strat(inverse_depth::strategy_random);
    println!("Reprojection error (statistical): {:?}", stat_eval);
    println!("Reprojection error (DSO): {:?}", dso_eval);
    println!("Reprojection error (random): {:?}", _random_eval);
}

fn eval_strategy_reprojection<F>(
    idepth_candidates: DMatrix<InverseDepth>,
    multires_rgb_1: &Vec<DMatrix<u8>>,
    multires_rgb_2: &Vec<DMatrix<u8>>,
    multires_camera_1: &Vec<Camera>,
    multires_camera_2: &Vec<Camera>,
    strategy: F,
) -> Vec<f32>
where
    F: Fn(Vec<(f32, f32)>) -> InverseDepth,
{
    // Create a multires inverse depth map pyramid.
    let fuse = |a, b, c, d| inverse_depth::fuse(a, b, c, d, &strategy);
    let multires_idepth = multires::pyramid_with_max_n_levels(
        5,
        idepth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, fuse),
    );

    // Re-project candidates on new image at each level.
    (1..6)
        .map(|n| {
            optimization::reprojection_error(
                &multires_idepth[n - 1],
                &multires_camera_1[n],
                &multires_camera_2[n],
                &multires_rgb_1[n],
                &multires_rgb_2[n],
            )
        })
        .collect()
}

// fn inverse_depth_visual(inverse_mat: &DMatrix<InverseDepth>) -> DMatrix<u8> {
//     inverse_mat.map(|idepth| inverse_depth::visual_enum(&idepth))
// }
