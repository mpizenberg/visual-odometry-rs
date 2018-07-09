extern crate nalgebra;

use camera::Camera;
use helper;
use inverse_depth::InverseDepth;
use nalgebra::{DMatrix, Point2};

pub fn reprojection_error(
    idepth: &DMatrix<InverseDepth>,
    camera_1: &Camera,
    camera_2: &Camera,
    rgb_1: &DMatrix<u8>,
    rgb_2: &DMatrix<u8>,
) -> f32 {
    let mut reprojection_error_accum = 0.0;
    let mut total_count = 0;
    let (nrows, ncols) = idepth.shape();
    let mut projected = DMatrix::repeat(nrows, ncols, InverseDepth::Unknown);
    idepth.iter().enumerate().for_each(|(index, idepth_enum)| {
        if let InverseDepth::WithVariance(idepth, _variance) = idepth_enum {
            let (col, row) = helper::div_rem(index, nrows);
            let reprojected = camera_2
                .project(camera_1.back_project(Point2::new(col as f32, row as f32), 1.0 / idepth));
            let new_pos = reprojected.as_slice();
            let x = new_pos[0] / new_pos[2];
            let y = new_pos[1] / new_pos[2];
            if helper::in_image_bounds((x, y), (nrows, ncols)) {
                // let current_weight = 1.0 / variance;
                total_count += 1;
                let u = x.floor() as usize;
                let v = y.floor() as usize;
                let a = x - u as f32;
                let b = y - v as f32;
                // to be optimized
                let img_xy = (1.0 - a) * (1.0 - b) * rgb_2[(v, u)] as f32
                    + (1.0 - a) * b * rgb_2[(v + 1, u)] as f32
                    + a * (1.0 - b) * rgb_2[(v, u + 1)] as f32
                    + a * b * rgb_2[(v + 1, u + 1)] as f32;
                // to be optimized
                let img_orig = rgb_1[(row, col)] as f32;
                reprojection_error_accum += (img_xy - img_orig).abs();
                unsafe {
                    *(projected.get_unchecked_mut(y.round() as usize, x.round() as usize)) =
                        idepth_enum.clone();
                }
            }
        }
    });
    // println!("total weight: {}", total_weight);
    // interop::image_from_matrix(&inverse_depth_visual(&projected))
    //     .save("out/idepth_projected.png")
    //     .unwrap();
    reprojection_error_accum / total_count as f32
}
