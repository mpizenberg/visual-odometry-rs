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

const INTRINSICS: Intrinsics = Intrinsics {
    principal_point: (319.5, 239.5),
    focal_length: 1.0,
    scaling: (481.20, -480.00),
    skew: 0.0,
};

// #[allow(dead_code)]
fn main() {
    let all_extrinsics = Extrinsics::read_from_tum_file("data/trajectory-gt.txt").unwrap();
    let (multires_camera_1, multires_rgb_1, depth_1) =
        prepare_icl_data(1, &all_extrinsics).unwrap();
    let (multires_camera_2, multires_rgb_2, _) = prepare_icl_data(600, &all_extrinsics).unwrap();

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

fn prepare_icl_data(
    id: usize,
    extrinsics: &Vec<Extrinsics>,
) -> Result<(Vec<Camera>, Vec<DMatrix<u8>>, DMatrix<u16>), image::ImageError> {
    // Load the rgb and depth image.
    let (rgb, depth) = open_icl_data(id)?;
    Ok((
        Camera::new(INTRINSICS.clone(), extrinsics[id - 1].clone()).multi_res(6),
        multires::mean_pyramid(6, rgb),
        depth,
    ))
}

fn open_icl_data(id: usize) -> Result<(DMatrix<u8>, DMatrix<u16>), image::ImageError> {
    let img_mat =
        interop::matrix_from_image(image::open(&format!("icl-rgb/{}.png", id))?.to_luma());
    let (w, h, buffer) = helper::read_png_16bits(&format!("icl-depth/{}.png", id))?;
    let depth_map = DMatrix::from_row_slice(h, w, buffer.as_slice());
    Ok((img_mat, depth_map))
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
            if n == 0 {
                display_few(&multires_idepth[n - 1]);
            }
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

fn display_few(mat: &DMatrix<InverseDepth>) -> () {
    println!(
        "{:?}",
        mat.iter()
            .filter_map(|d| inverse_depth::with_variance(&d))
            .take(40)
            .collect::<Vec<_>>()
    );
}

// fn inverse_depth_visual(inverse_mat: &DMatrix<InverseDepth>) -> DMatrix<u8> {
//     inverse_mat.map(|idepth| inverse_depth::visual_enum(&idepth))
// }
