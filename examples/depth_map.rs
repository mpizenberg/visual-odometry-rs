extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use cv::candidates;
use cv::helper;
use cv::interop;
use cv::inverse_depth;
use cv::multires;
use cv::view;

use inverse_depth::InverseDepth;
use na::DMatrix;

// #[allow(dead_code)]
fn main() {
    // Load a color image and transform into grayscale.
    let img = image::open("icl-rgb/0.png")
        .expect("Cannot open image")
        .to_luma();

    // Create an equivalent matrix.
    let img_matrix = interop::matrix_from_image(img);

    // Compute pyramid of matrices.
    let multires_img = multires::mean_pyramid(6, img_matrix);

    // Compute pyramid of gradients (without first level).
    let multires_gradient_norm = multires::gradients(&multires_img);

    // canditates
    let multires_candidates = candidates::select(&multires_gradient_norm);

    // Read 16 bits PNG image.
    let (width, height, buffer_u16) = helper::read_png_16bits("icl-depth/0.png").unwrap();

    // Transform depth map image into a matrix.
    let depth_mat: DMatrix<u16> = DMatrix::from_row_slice(height, width, buffer_u16.as_slice());

    // Create an inverse depth map with values only at point candidates.
    // This is to emulate result of back projection of known points into a new keyframe.
    let half_res_depth = multires::halve(&depth_mat, |a, b, c, d| {
        let a = inverse_depth::from_depth(a);
        let b = inverse_depth::from_depth(b);
        let c = inverse_depth::from_depth(c);
        let d = inverse_depth::from_depth(d);
        inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar)
    }).unwrap();
    view::color_idepth(&half_res_depth)
        .save("out/half_res_depth_color.png")
        .unwrap();
    let inverse_depth_candidates = helper::zip_mask_map(
        &half_res_depth,
        &multires_candidates.last().unwrap(),
        InverseDepth::Unknown,
        |idepth| idepth,
    );
    view::color_idepth(&inverse_depth_candidates)
        .save("out/idepth_candidates_color.png")
        .unwrap();

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
