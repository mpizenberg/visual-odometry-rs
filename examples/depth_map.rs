extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use cv::candidates;
use cv::helper;
use cv::icl_nuim;
use cv::inverse_depth;
use cv::multires;
use cv::view;

use inverse_depth::InverseDepth;

// #[allow(dead_code)]
fn main() {
    // Load image as grayscale (u8) and depth map (u16).
    let (img, depth_map) = icl_nuim::open_imgs(1).unwrap();

    // Save on disk for visualizations.
    let from_depth = |depth| inverse_depth::from_depth(icl_nuim::DEPTH_SCALE, depth);
    view::color_idepth(&depth_map.map(from_depth))
        .save("out/full_depth_color.png")
        .unwrap();

    // Compute pyramid of matrices.
    let multires_img = multires::mean_pyramid(6, img);

    // Compute pyramid of gradients (without first level).
    let multires_gradient_norm = multires::gradients(&multires_img);

    // Canditates.
    let multires_candidates = candidates::select(7, &multires_gradient_norm);

    // Get an half resolution depth map that will be the base for
    // the inverse depth map on candidates.
    let half_res_depth = multires::halve(&depth_map, |a, b, c, d| {
        let a = from_depth(a);
        let b = from_depth(b);
        let c = from_depth(c);
        let d = from_depth(d);
        inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar)
    })
    .unwrap();

    // Save on disk for visualizations.
    view::color_idepth(&half_res_depth)
        .save("out/half_res_depth_color.png")
        .unwrap();

    // Create an inverse depth map with values only at point candidates.
    // This is to emulate result of back projection of known points into a new keyframe.
    let inverse_depth_candidates = helper::zip_mask_map(
        &half_res_depth,
        &multires_candidates.last().unwrap(),
        InverseDepth::Unknown,
        |idepth| idepth,
    );

    // Create a multires inverse depth map pyramid
    // with same number of levels than the multires image.
    let fuse =
        |a, b, c, d| inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar);
    let multires_inverse_depth = multires::pyramid_with_max_n_levels(
        5,
        inverse_depth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, fuse),
    );

    // Save inverse depth pyramid on disk.
    multires_inverse_depth
        .iter()
        .enumerate()
        .for_each(|(i, bitmap)| {
            let out_name = &["out/idepth_", i.to_string().as_str(), ".png"].concat();
            view::color_idepth(bitmap).save(out_name).unwrap();
        });
}
