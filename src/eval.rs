use camera::Camera;
use helper;
use inverse_depth::InverseDepth;
use nalgebra::{DMatrix, Point2, Scalar};
use num_traits::{self, NumCast};
use std::f32;
use std::ops::{Add, Mul};

pub type Float = f32;

pub fn reprojection_error(
    idepth: &DMatrix<InverseDepth>,
    camera_1: &Camera,
    camera_2: &Camera,
    rgb_1: &DMatrix<u8>,
    rgb_2: &DMatrix<u8>,
) -> Float {
    let mut reprojection_error_accum = 0.0;
    let mut total_count = 0;
    let (nrows, ncols) = idepth.shape();
    idepth.iter().enumerate().for_each(|(index, idepth_enum)| {
        if let InverseDepth::WithVariance(idepth, _variance) = idepth_enum {
            let (col, row) = helper::div_rem(index, nrows);
            let reprojected = camera_2.project(
                camera_1.back_project(Point2::new(col as Float, row as Float), 1.0 / idepth),
            );
            let x = reprojected.x / reprojected.z;
            let y = reprojected.y / reprojected.z;
            if helper::in_image_bounds((x, y), (nrows, ncols)) {
                total_count += 1;
                let (indices, coefs) = linear_interpolator(x, y);
                let img_xy = interpolate_with(indices, coefs, rgb_2);
                let img_orig = rgb_1[(row, col)] as f32;
                reprojection_error_accum += (img_xy - img_orig).abs();
            }
        }
    });

    if total_count == 0 {
        0.0
    } else {
        reprojection_error_accum / total_count as Float
    }
}

pub fn interpolate_with<T, F>(
    indices: (usize, usize),
    coefs: (F, F, F, F),
    matrix: &DMatrix<T>,
) -> F
where
    T: Scalar + NumCast,
    F: NumCast + Add<F, Output = F> + Mul<F, Output = F>,
{
    let (u, v) = indices;
    let (a, b, c, d) = coefs;
    let v_u: F = num_traits::cast(matrix[(v, u)]).unwrap();
    let v1_u: F = num_traits::cast(matrix[(v + 1, u)]).unwrap();
    let v_u1: F = num_traits::cast(matrix[(v, u + 1)]).unwrap();
    let v1_u1: F = num_traits::cast(matrix[(v + 1, u + 1)]).unwrap();
    a * v_u + b * v1_u + c * v_u1 + d * v1_u1
}

pub fn linear_interpolator(x: Float, y: Float) -> ((usize, usize), (Float, Float, Float, Float)) {
    let u = x.floor() as usize;
    let v = y.floor() as usize;
    let a = x - u as Float;
    let b = y - v as Float;
    let _a = 1.0 - a;
    let _b = 1.0 - b;
    ((u, v), (_a * _b, _a * b, a * _b, a * b))
}
