extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use std::fs;

use cv::candidates;
use cv::helper;
use cv::icl_nuim;
use cv::inverse_depth;
use cv::multires;
use cv::view;

use inverse_depth::InverseDepth;

const OUT_DIR: &str = "out/example/depth_map_candidates/";

// #[allow(dead_code)]
fn main() {
    // Load image as grayscale (u8) and depth map (u16).
    let (img, depth_map) = icl_nuim::open_imgs(1).unwrap();

    // Compute pyramid of matrices.
    let multires_img = multires::mean_pyramid(6, img);

    // Compute pyramid of gradients (without first level).
    let multires_gradients_squared_norm = multires::gradients_squared_norm(&multires_img);

    // Choose candidates based on gradients norms.
    // The fisrst candidate matrix in the vector is the lowest resolution.
    let diff_threshold = 7;
    let multires_candidates = candidates::select(diff_threshold, &multires_gradients_squared_norm);

    // Construct an half resolution inverse depth map that will be the base for
    // the inverse depth map on candidates.
    let from_depth = |depth| inverse_depth::from_depth(icl_nuim::DEPTH_SCALE, depth);
    let half_res_idepth = multires::halve(&depth_map, |a, b, c, d| {
        let a = from_depth(a);
        let b = from_depth(b);
        let c = from_depth(c);
        let d = from_depth(d);
        inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar)
    })
    .unwrap();

    // Create an inverse depth map with known values only at point candidates.
    // This is to emulate the fact that the depth info is supposed to be sparse.
    let inverse_depth_candidates = helper::zip_mask_map(
        &half_res_idepth,
        &multires_candidates.last().unwrap(),
        InverseDepth::Unknown,
        |idepth| idepth,
    );

    // Create a multires inverse depth map pyramid
    // with one less level than the multires image (starts at half res).
    let fuse =
        |a, b, c, d| inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar);
    let multires_inverse_depth = multires::limited_sequence(
        5,
        inverse_depth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, fuse),
    );

    // ----------------------------------------------------- (visualizations)

    // Create output directory.
    fs::create_dir_all(OUT_DIR).unwrap();

    // Save full resolution inverse depth map on disk.
    view::idepth_image(&depth_map.map(from_depth))
        .save([OUT_DIR, "full_depth_color.png"].concat())
        .unwrap();

    // Save half resolution inverse depth map on disk.
    view::idepth_image(&half_res_idepth)
        .save([OUT_DIR, "half_res_depth_color.png"].concat())
        .unwrap();

    // Save inverse depth pyramid on disk.
    multires_inverse_depth
        .iter()
        .enumerate()
        .for_each(|(i, bitmap)| {
            let out_name = [OUT_DIR, "idepth_", i.to_string().as_str(), ".png"].concat();
            view::idepth_image(bitmap).save(out_name).unwrap();
        });
}
