extern crate computer_vision_rs as cv;
extern crate csv;
extern crate image;
extern crate nalgebra;

#[macro_use]
extern crate itertools;

use cv::camera::{self, Camera, Intrinsics};
use cv::candidates;
use cv::helper;
use cv::icl_nuim;
use cv::interop;
use cv::inverse_depth::{self, InverseDepth};
use cv::multires_float;
use cv::optimization::{self, Continue};
use cv::se3;

use nalgebra::base::dimension::Dynamic;
use nalgebra::{
    DMatrix, DVector, Isometry3, Matrix6, Matrix6x2, MatrixMN, Point2, Vector2, Vector6,
};
use std::f32::EPSILON;
use std::iter;

fn main() {
    let all_extrinsics = icl_nuim::read_extrinsics("data/trajectory-gt.txt").unwrap();
    let (multires_camera_1, multires_img_1, depth_1) =
        icl_nuim::prepare_data(1, &all_extrinsics).unwrap();
    let (multires_camera_2, multires_img_2, _) =
        icl_nuim::prepare_data(2, &all_extrinsics).unwrap();
    let multires_gradients_1_norm = multires_float::gradients(&multires_img_1);
    let multires_gradients_1_xy = multires_float::gradients_xy(&multires_img_1);
    let candidates = candidates::select(7.0 / 255.0, &multires_gradients_1_norm)
        .pop()
        .unwrap();
    let fuse =
        |a, b, c, d| inverse_depth::fuse(a, b, c, d, inverse_depth::strategy_statistically_similar);
    let from_depth = |depth| inverse_depth::from_depth(icl_nuim::DEPTH_SCALE, depth);
    let half_res_depth = multires_float::halve(&depth_1.map(from_depth), fuse).unwrap();
    let idepth_candidates = helper::zip_mask_map(
        &half_res_depth,
        &candidates,
        InverseDepth::Unknown,
        |idepth| idepth,
    );
    let multires_idepth_1 = multires_float::pyramid_with_max_n_levels(
        5,
        idepth_candidates,
        |mat| mat,
        |mat| multires_float::halve(&mat, fuse),
    );
    let _ = track(
        &multires_camera_1,
        &multires_camera_2,
        &multires_idepth_1,
        &multires_img_1,
        &multires_img_2,
        &multires_gradients_1_xy,
    );
}

struct Model {
    level: usize,
    iteration: usize,
    motion: Isometry3<Float>,
}

fn track(
    multires_camera_1: &Vec<Camera>,
    multires_camera_2: &Vec<Camera>,
    multires_idepth_1: &Vec<DMatrix<InverseDepth>>,
    multires_img_1: &Vec<DMatrix<Float>>,
    multires_img_2: &Vec<DMatrix<Float>>,
    multires_gradients_1: &Vec<(DMatrix<Float>, DMatrix<Float>)>,
) -> Isometry3<Float> {
    // * For each pyramid level (starting with lowest resolution):
    //     * (1) Compute the smallest residual energy threshold, such that more than 40% residuals of points ar under this threshold.
    //     * (2) Compute and save all residuals (the energy functional)
    //     * While the energy decreases fast enough and we are under the limit of iterations:
    //         * (3) Compute Jacobians for each point
    //         * (4) Accumulate Jacobians products in accumulated Hessian
    //         * (5) Solve the step computation using Levenberg-Marquardt dampening instead of the Gauss-Newton
    //         * (6) Update (recompute) the residuals

    let motion = Isometry3::identity();
    let mut model = Model {
        level: 0,
        iteration: 0,
        motion,
    };
    for level in (1..6).rev() {
        model.level = level;
        model.iteration = 0;
        println!("--- level {}", level);
        let cam_1 = &multires_camera_1[level];
        let cam_2 = &multires_camera_2[level];
        let intrinsics = &cam_1.intrinsics;
        let img_1 = &multires_img_1[level];
        // let img_1_name = &["out/lvl_", level.to_string().as_str(), "_1.png"].concat();
        // interop::image_from_matrix(&img_1).save(img_1_name).unwrap();
        let img_2 = &multires_img_2[level];
        // let img_2_name = &["out/lvl_", level.to_string().as_str(), "_2.png"].concat();
        // interop::image_from_matrix(&img_2).save(img_2_name).unwrap();
        let idepth_map = &multires_idepth_1[level - 1];
        let (gx_1, gy_1) = &multires_gradients_1[level - 1];
        let template = &(idepth_map, img_1, gx_1, gy_1);
        let observation = &(intrinsics, img_2, template);

        let (new_model, _) = optimization::iterative(
            &eval,
            &step_hessian_cholesky,
            &stop_criterion,
            observation,
            model,
        );
        model = new_model;
        // Some ground truth logging.
        let gt_energy = optimization::reprojection_error(idepth_map, cam_1, cam_2, img_1, img_2);
        println!("GT energy: {}", gt_energy);
    }

    // Some printing of results.
    let cam_1 = &multires_camera_1[1];
    let cam_2 = &multires_camera_2[1];
    let gt_motion = cam_2.extrinsics.inverse() * cam_1.extrinsics;
    println!("------------------------------- results");
    println!(
        "Computed translation: {:?}",
        model.motion.translation.vector
    );
    println!("GT       translation: {:?}", gt_motion.translation.vector);
    println!("Computed rotation: {:?}", model.motion.rotation);
    println!("GT       rotation: {:?}", gt_motion.rotation);

    // Return motion.
    model.motion
}

fn stop_criterion(nb_iter: usize, energy: f32, residuals: &Vec<Residual>) -> (f32, Continue) {
    let new_energy: f32 = residuals.iter().map(|x| x.abs()).sum::<f32>() / residuals.len() as f32;
    println!("iter {}, energy: {}", nb_iter, new_energy);
    let continuation = if new_energy > energy {
        Continue::Backward
    } else if nb_iter >= 10 {
        Continue::Stop
    } else {
        Continue::Forward
    };
    (new_energy, continuation)
}

fn step_hessian_cholesky(
    jacobians: &Vec<Jacobian>,
    residuals: &Vec<Residual>,
    model: &Model,
) -> Model {
    let mut hessian: Matrix6<Float> = Matrix6::zeros();
    let mut rhs: Vector6<Float> = Vector6::zeros();
    for (jac, &res) in jacobians.iter().zip(residuals.iter()) {
        hessian = hessian + jac * jac.transpose();
        rhs = rhs + res * jac;
    }
    let twist_step = hessian.cholesky().unwrap().solve(&rhs);
    let new_motion = model.motion * se3::exp(se3::from_vector(twist_step)).inverse();
    Model {
        level: model.level,
        iteration: 1 + model.iteration,
        motion: new_motion,
    }
}

type Observation<'a> = (
    &'a Intrinsics,     // camera intrinsics
    &'a DMatrix<Float>, // image to track
    &'a Template<'a>,   // reference template (image, gradients, ...)
);

fn eval(observation: &Observation, model: &Model) -> (Vec<Jacobian>, Vec<Residual>) {
    let (intrinsics, img_2, template) = observation;
    let mut jacobians_kept = Vec::new();
    let mut residuals_kept = Vec::new();
    // We should extract precomputation, but let's leave it here for now ...
    let (all_coordinates, all_jacobians, all_intensities, all_idepth) =
        precompute(intrinsics, template);
    for (coordinates, jac, color_ref, _z) in
        izip!(all_coordinates, all_jacobians, all_intensities, all_idepth)
    {
        let (x, y) = reproject(coordinates, _z, intrinsics, &model.motion);
        if helper::in_image_bounds((x, y), img_2.shape()) {
            let (indices, coefs) = optimization::linear_interpolator(Point2::new(x, y));
            let color = optimization::interpolate_with(indices, coefs, img_2);
            let residual = color - color_ref;
            if residual.abs() < 100000.0 {
                jacobians_kept.push(jac);
                residuals_kept.push(residual);
            }
        }
    }
    // Return jacobians and residuals kept.
    (jacobians_kept, residuals_kept)
}

fn reproject(
    coordinates: (Float, Float),
    idepth: Float,
    intrinsics: &Intrinsics,
    motion: &Isometry3<Float>,
) -> (Float, Float) {
    let (x, y) = coordinates;
    let point = Point2::new(x, y);
    let homogeneous = intrinsics.project(motion * intrinsics.back_project(point, 1.0 / idepth));
    let new_idepth = 1.0 / homogeneous[2];
    let new_point = Point2::new(homogeneous[0] * new_idepth, homogeneous[1] * new_idepth);
    (new_point[0], new_point[1])
}

type Float = f32;
struct Gradient(Float, Float);
type Jacobian = Vector6<Float>;
type Residual = Float;

type Template<'a> = (
    &'a DMatrix<InverseDepth>, // idepth map
    &'a DMatrix<Float>,        // img
    &'a DMatrix<Float>,        // img gradient x
    &'a DMatrix<Float>,        // img gradient y
);

fn precompute(
    intrinsics: &Intrinsics,
    template: &Template,
) -> (Vec<(Float, Float)>, Vec<Jacobian>, Vec<Float>, Vec<Float>) {
    let (idepth_map, img, img_gx, img_gy) = template;
    // img, idepth_map, img_gx and img_gy should have the same shape.
    let (nrows, ncols) = idepth_map.shape();
    let mut all_coordinates = Vec::new();
    let mut all_jacobians = Vec::new();
    let mut all_intensities = Vec::new();
    let mut all_idepth = Vec::new();
    for (index, (intensity, idepth, gx, gy)) in
        izip!(img.iter(), idepth_map.iter(), img_gx.iter(), img_gy.iter()).enumerate()
    {
        if let InverseDepth::WithVariance(_z, _) = idepth {
            let (col, row) = helper::div_rem(index, nrows);
            let gradient = Gradient(*gx, *gy);
            let jac = compute_jacobian(col as Float, row as Float, *_z, intrinsics, gradient);
            all_coordinates.push((col as Float, row as Float));
            all_jacobians.push(jac);
            all_intensities.push(*intensity);
            all_idepth.push(*_z);
        }
    }
    // return jacobians and intensities.
    (all_coordinates, all_jacobians, all_intensities, all_idepth)
}

fn compute_jacobian(x: Float, y: Float, idepth: Float, k: &Intrinsics, g: Gradient) -> Jacobian {
    // Named variables
    let _z = idepth;
    let z = 1.0 / _z;
    let (cx, cy) = k.principal_point;
    let (sx, sy) = k.scaling;
    let (fx, fy) = (k.focal_length * sx, k.focal_length * sy);
    let Gradient(gx, gy) = g;
    let skew = k.skew;

    // Intermediate computations
    let u = (x - cx) / fx;
    let v = (y - cy) / fy;
    let t = skew * v / fx - u;

    // Partial derivatives
    #[cfg_attr(rustfmt, rustfmt_skip)]
    let partial_derivatives = Matrix6x2::new(
        fx       * _z,          0.0,                 // nu 1
        skew     * _z,          fy * _z,             // nu 2
        (cx - x) * _z,          (cy - y) * _z,       // nu 3
        v * (x + cx) - skew,    v * (y + cy) - fy,   // omega 1
        t * (x + cx) + fx,      t * (y + cy),        // omega 2
        -t * skew - fx * v,    -t * fy,              // omega 3
    );

    // Return jacobian
    let gradient = Vector2::new(gx, gy);
    partial_derivatives * gradient
}
