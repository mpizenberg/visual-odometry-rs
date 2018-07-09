extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra as na;

use cv::candidates;
use cv::colormap;
use cv::helper;
use cv::interop;
use cv::inverse_depth;
use cv::multires;

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
    save_color_depth(&half_res_depth, "out/half_res_depth_color.png");
    let inverse_depth_candidates = helper::zip_mask_map(
        &half_res_depth,
        &multires_candidates.last().unwrap(),
        InverseDepth::Unknown,
        |idepth| idepth,
    );
    save_color_depth(&inverse_depth_candidates, "out/idepth_candidates_color.png");

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
        .map(inverse_depth_visual)
        .map(|mat| interop::image_from_matrix(&mat))
        .enumerate()
        .for_each(|(i, bitmap)| {
            let out_name = &["out/idepth_", i.to_string().as_str(), ".png"].concat();
            bitmap.save(out_name).unwrap();
        });
}

// Inverse Depth stuff ###############################################

fn inverse_depth_visual(inverse_mat: &DMatrix<InverseDepth>) -> DMatrix<u8> {
    inverse_mat.map(|idepth| inverse_depth::visual_enum(&idepth))
}

fn min_max(idepth_map: &DMatrix<InverseDepth>) -> Option<(f32, f32)> {
    let mut min_temp = 10000.0_f32;
    let mut max_temp = 0.0_f32;
    idepth_map.iter().for_each(|idepth| {
        if let Some((d,_)) = inverse_depth::with_variance(idepth) {
            min_temp = min_temp.min(d);
            max_temp = max_temp.max(d);
        }
    });
    Some((min_temp, max_temp))
}

fn save_color_depth(idepth_map: &DMatrix<InverseDepth>, path: &str) -> () {
    let viridis = &colormap::viridis()[0..256];
    let (d_min, d_max) = min_max(idepth_map).unwrap();
    interop::rgb_from_matrix(&idepth_map.map(
            |idepth| inverse_depth::to_color(viridis, d_min, d_max, &idepth))
        ).save(path).unwrap();
}
