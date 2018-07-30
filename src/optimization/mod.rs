use camera::Camera;
use helper;
use inverse_depth::InverseDepth;
use nalgebra::{DMatrix, Point2};
use std::f32;

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
                    // Copying idepth_enum is wrong for the new inverse depth
                    // but simplest for enum visualization.
                    *(projected.get_unchecked_mut(y.round() as usize, x.round() as usize)) =
                        idepth_enum.clone();
                }
            }
        }
    });

    reprojection_error_accum / total_count as f32
}

// pub fn gauss_newton<Observation, Model, Jacobian, Residual, Energy>(
//     eval: fn(&Observation, &Model) -> (Jacobian, Residual),
//     step: fn(&Jacobian, &Residual, &Model) -> Model,
//     stop_criterion: fn(usize, Energy, &Residual) -> (Energy, bool),
//     observation: &Observation,
//     initial_model: Model,
// ) -> (Model, usize) {
//     (initial_model, 0)
// }

pub fn gauss_newton_fn<Observation, Model, Jacobian, Residual, EvalFn, StepFn, CriterionFn>(
    eval: EvalFn,
    step: StepFn,
    stop_criterion: CriterionFn,
    observation: &Observation,
    initial_model: Model,
) -> (Model, usize)
where
    Model: Clone,
    EvalFn: Fn(&Observation, &Model) -> (Jacobian, Residual),
    StepFn: Fn(&Jacobian, &Residual, &Model) -> Model,
    CriterionFn: Fn(usize, f32, &Residual) -> (f32, Continue),
{
    let mut nb_iter = 0;
    let mut energy = f32::INFINITY;
    let mut model = initial_model;
    let mut previous_model = model.clone();
    // For algorithm simplicity, we accept to compute the jacobian
    // one time more than needed, since eval returns both jacobian and residual
    // and we need residual to evaluate the stop criterion.
    let (mut jacobian, mut residual) = eval(observation, &model);
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
                model = step(&jacobian, &residual, &previous_model);
                let (new_jacobian, new_residual) = eval(observation, &model);
                jacobian = new_jacobian;
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
