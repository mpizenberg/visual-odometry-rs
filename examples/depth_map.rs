extern crate image;
extern crate nalgebra as na;

mod candidates;
mod helper;
mod interop;
mod inverse_depth;
mod multires;

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

    // Create a half resolution depth map to fit resolution of candidates map.
    let half_res_depth = multires::halve(&depth_mat, |a, b, c, d| {
        ((a as u32 + b as u32 + c as u32 + d as u32) / 4) as u16
    }).unwrap();

    // Transform depth map into an InverseDepth matrix.
    let inverse_depth_mat = half_res_depth.map(inverse_depth::from_depth);

    // Only keep InverseDepth values corresponding to point candidates.
    // This is to emulate result of back projection of known points into a new keyframe.
    let higher_res_candidate = multires_candidates.last().unwrap();
    let inverse_depth_candidates =
        higher_res_candidate.zip_map(&inverse_depth_mat, |is_candidate, idepth| {
            if is_candidate {
                idepth
            } else {
                InverseDepth::Unknown
            }
        });

    // Create a multires inverse depth map pyramid
    // with same number of levels than the multires image.
    let multires_inverse_depth = multires::pyramid_with_max_n_levels(
        5,
        inverse_depth_candidates,
        |mat| mat,
        |mat| multires::halve(&mat, inverse_depth::fuse),
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
