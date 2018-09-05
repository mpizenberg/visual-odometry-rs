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
    rgb_1: &DMatrix<Float>,
    rgb_2: &DMatrix<Float>,
) -> Float {
    let mut reprojection_error_accum = 0.0;
    let mut total_count = 0;
    let (nrows, ncols) = idepth.shape();
    let mut projected = DMatrix::repeat(nrows, ncols, InverseDepth::Unknown);
    idepth.iter().enumerate().for_each(|(index, idepth_enum)| {
        if let InverseDepth::WithVariance(idepth, _variance) = idepth_enum {
            let (col, row) = helper::div_rem(index, nrows);
            let reprojected = camera_2.project(
                camera_1.back_project(Point2::new(col as Float, row as Float), 1.0 / idepth),
            );
            let new_pos = reprojected.as_slice();
            let x = new_pos[0] / new_pos[2];
            let y = new_pos[1] / new_pos[2];
            if helper::in_image_bounds((x, y), (nrows, ncols)) {
                // let current_weight = 1.0 / variance;
                total_count += 1;
                let u = x.floor() as usize;
                let v = y.floor() as usize;
                let a = x - u as Float;
                let b = y - v as Float;
                // to be optimized
                let img_xy = (1.0 - a) * (1.0 - b) * rgb_2[(v, u)]
                    + (1.0 - a) * b * rgb_2[(v + 1, u)]
                    + a * (1.0 - b) * rgb_2[(v, u + 1)]
                    + a * b * rgb_2[(v + 1, u + 1)];
                // to be optimized
                let img_orig = rgb_1[(row, col)];
                reprojection_error_accum += (img_xy - img_orig).abs();
                unsafe {
                    // Copying idepth_enum is wrong for the new inverse depth
                    // but simplest for enum visualization.
                    *(projected.get_unchecked_mut(y.round() as usize, x.round() as usize)) =
                        idepth_enum.clone();
                }
            }
        }
    });

    reprojection_error_accum / total_count as Float
}

pub fn iterative<Observation, Model, Derivatives, Residual, EvalFn, StepFn, CriterionFn>(
    eval: EvalFn,
    step: StepFn,
    stop_criterion: CriterionFn,
    observation: &Observation,
    initial_model: Model,
) -> (Model, usize)
where
    EvalFn: Fn(&Observation, &Model) -> (Derivatives, Residual),
    StepFn: Fn(&Derivatives, &Residual, &Model) -> Model,
    CriterionFn: Fn(usize, Float, &Residual) -> (Float, Continue),
{
    // Manual first iteration enable avoiding to have Clone for model.
    // Otherwise, the compiler doesn't know if previous_model has
    // been initialized in the backward branch.
    let mut energy = f32::INFINITY;
    let (mut derivatives, mut residual) = eval(observation, &initial_model);
    match stop_criterion(0, energy, &residual) {
        (new_energy, Continue::Forward) => {
            energy = new_energy;
        }
        _ => return (initial_model, 0),
    }
    let mut nb_iter = 1;
    let mut model = step(&derivatives, &residual, &initial_model);
    let mut previous_model = initial_model;
    let (new_derivatives, new_residual) = eval(observation, &model);
    derivatives = new_derivatives;
    residual = new_residual;

    // After first iteration, loop until stop criterion.
    loop {
        let (new_energy, continuation) = stop_criterion(nb_iter, energy, &residual);
        match continuation {
            Continue::Stop => break,
            Continue::Backward => {
                model = previous_model;
                break;
            }
            Continue::Forward => {
                nb_iter = nb_iter + 1;
                energy = new_energy;
                previous_model = model;
                model = step(&derivatives, &residual, &previous_model);
                let (new_derivatives, new_residual) = eval(observation, &model);
                derivatives = new_derivatives;
                residual = new_residual;
            }
        }
    }
    (model, nb_iter)
}

pub enum Continue {
    Stop,
    Forward,
    Backward,
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

pub fn linear_interpolator(
    coordinates: Point2<Float>,
) -> ((usize, usize), (Float, Float, Float, Float)) {
    let x = coordinates[0];
    let y = coordinates[1];
    let u = x.floor() as usize;
    let v = y.floor() as usize;
    let a = x - u as Float;
    let b = y - v as Float;
    let _a = 1.0 - a;
    let _b = 1.0 - b;
    ((u, v), (_a * _b, _a * b, a * _b, a * b))
}
