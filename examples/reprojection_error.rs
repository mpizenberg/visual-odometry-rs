extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra;

use cv::camera::{self, Camera};
use cv::candidates;
use cv::helper;
use cv::icl_nuim;
use cv::inverse_depth::{self, InverseDepth};
use cv::multires;
use cv::optimization;

use nalgebra::DMatrix;

// #[allow(dead_code)]
fn main() {
    let all_extrinsics = camera::extrinsics::read_from_tum_file("data/trajectory-gt.txt").unwrap();
    let (multires_camera_1, multires_rgb_1, depth_1) =
        icl_nuim::prepare_data(1, &all_extrinsics).unwrap();
    let (multires_camera_2, multires_rgb_2, _) =
        icl_nuim::prepare_data(100, &all_extrinsics).unwrap();

    let candidates = candidates::select(&multires::gradients(&multires_rgb_1))
        .pop()
        .unwrap();

    let eval_strat = |strat: fn(_) -> _| {
        eval_strategy_reprojection(
            &multires_rgb_1,
            &multires_rgb_2,
            &multires_camera_1,
            &multires_camera_2,
            &depth_1,
            &candidates,
            strat,
        )
    };
    // let _random_eval = eval_strat(inverse_depth::strategy_random);
    let dso_eval = eval_strat(inverse_depth::strategy_dso_mean);
    let stat_eval = eval_strat(inverse_depth::strategy_statistically_similar);
    // println!("Reprojection error (random): {:?}", _random_eval);
    println!("Reprojection error (DSO): {:?}", dso_eval);
    println!("Reprojection error (statistical): {:?}", stat_eval);
}

fn eval_strategy_reprojection<F>(
    multires_rgb_1: &Vec<DMatrix<u8>>,
    multires_rgb_2: &Vec<DMatrix<u8>>,
    multires_camera_1: &Vec<Camera>,
    multires_camera_2: &Vec<Camera>,
    depth_map: &DMatrix<u16>,
    candidates: &DMatrix<bool>,
    strategy: F,
) -> Vec<f32>
where
    F: Fn(Vec<(f32, f32)>) -> InverseDepth,
{
    // Create an inverse depth map with values only at point candidates.
    // This is to emulate result of back projection of known points into a new keyframe.
    let fuse = |a, b, c, d| inverse_depth::fuse(a, b, c, d, &strategy);
    let half_res_depth = multires::halve(&depth_map.map(inverse_depth::from_depth), fuse).unwrap();
    let idepth_candidates = helper::zip_mask_map(
        &half_res_depth,
        &candidates,
        InverseDepth::Unknown,
        |idepth| idepth,
    );

    // Create a multires inverse depth map pyramid.
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
