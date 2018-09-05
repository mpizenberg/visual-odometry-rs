extern crate computer_vision_rs as cv;
extern crate image;
extern crate nalgebra;

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
use nalgebra::{DMatrix, DVector, Isometry3, Matrix6, MatrixMN, Point2, Vector6};
use std::f32::EPSILON;

fn main() {
    let all_extrinsics = icl_nuim::read_extrinsics("data/trajectory-gt.txt").unwrap();
    let (multires_camera_1, multires_img_1, depth_1) =
        icl_nuim::prepare_data(1, &all_extrinsics).unwrap();
    let (multires_camera_2, multires_img_2, _) =
        icl_nuim::prepare_data(20, &all_extrinsics).unwrap();
    let multires_gradients_1_norm = multires_float::gradients(&multires_img_1);
    let multires_gradients_2_xy = multires_float::gradients_xy(&multires_img_2);
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
        &multires_gradients_2_xy,
    );
}

fn track(
    multires_camera_1: &Vec<Camera>,
    multires_camera_2: &Vec<Camera>,
    multires_idepth_1: &Vec<DMatrix<InverseDepth>>,
    multires_img_1: &Vec<DMatrix<Float>>,
    multires_img_2: &Vec<DMatrix<Float>>,
    multires_gradients_2: &Vec<(DMatrix<Float>, DMatrix<Float>)>,
) -> Isometry3<Float> {
    // * For each pyramid level (starting with lowest resolution):
    //     * (1) Compute the smallest residual energy threshold, such that more than 40% residuals of points ar under this threshold.
    //     * (2) Compute and save all residuals (the energy functional)
    //     * While the energy decreases fast enough and we are under the limit of iterations:
    //         * (3) Compute Jacobians for each point
    //         * (4) Accumulate Jacobians products in accumulated Hessian
    //         * (5) Solve the step computation using Levenberg-Marquardt dampening instead of the Gauss-Newton
    //         * (6) Update (recompute) the residuals

    let mut motion = Isometry3::identity();
    for level in (1..6).rev() {
        println!("--- level {}", level);
        let cam_1 = &multires_camera_1[level];
        let cam_2 = &multires_camera_2[level];
        let intrinsics = &cam_1.intrinsics;
        let img_1 = &multires_img_1[level];
        let img_1_name = &["out/lvl_", level.to_string().as_str(), "_1.png"].concat();
        // interop::image_from_matrix(&img_1).save(img_1_name).unwrap();
        let img_2 = &multires_img_2[level];
        let img_2_name = &["out/lvl_", level.to_string().as_str(), "_2.png"].concat();
        // interop::image_from_matrix(&img_2).save(img_2_name).unwrap();
        let idepth_map = &multires_idepth_1[level - 1];
        let (gx_2, gy_2) = &multires_gradients_2[level - 1];

        let (new_motion, _) = optimization::iterative(
            &eval,
            &_step_levenberg_marquardt,
            &stop_criterion,
            &(intrinsics, idepth_map, img_1, img_2, gx_2, gy_2),
            motion,
        );
        motion = new_motion;
        // Some ground truth logging.
        let gt_energy = optimization::reprojection_error(idepth_map, cam_1, cam_2, img_1, img_2);
        println!("GT energy: {}", gt_energy);
    }

    // Some printing of results.
    let cam_1 = &multires_camera_1[1];
    let cam_2 = &multires_camera_2[1];
    let gt_motion = cam_2.extrinsics.inverse() * cam_1.extrinsics;
    println!("------------------------------- results");
    println!("Computed translation: {:?}", motion.translation.vector);
    println!("GT       translation: {:?}", gt_motion.translation.vector);
    println!("Computed rotation: {:?}", motion.rotation);
    println!("GT       rotation: {:?}", gt_motion.rotation);

    // Return motion.
    motion
}

fn stop_criterion(nb_iter: usize, energy: f32, residuals: &Vec<Residual>) -> (f32, Continue) {
    let new_energy: f32 = residuals.iter().map(|x| x.abs()).sum::<f32>() / residuals.len() as f32;
    println!("iter {}, energy: {}", nb_iter, new_energy);
    let continuation = if new_energy > energy {
        Continue::Backward
    } else if nb_iter >= 50 {
        Continue::Stop
    } else {
        Continue::Forward
    };
    (new_energy, continuation)
}

fn _step_jacobian_svd(
    jacobian: &Vec<Jacobian>,
    residuals: &Vec<Residual>,
    motion: &Isometry3<f32>,
) -> Isometry3<f32> {
    let full_jacobian = MatrixMN::<_, _, Dynamic>::from_columns(jacobian.as_slice());
    let full_residual = DVector::from_column_slice(residuals.len(), residuals.as_slice());
    let twist_step = full_jacobian
        .transpose()
        .svd(true, true)
        .solve(&(-full_residual), EPSILON);
    let new_motion = se3::exp(se3::from_vector(twist_step)) * motion;
    // println!("Computed translation: {:?}", new_motion.translation.vector);
    new_motion
}

fn _step_hessian_cholesky(
    jacobian: &Vec<Jacobian>,
    residuals: &Vec<Residual>,
    model: &Vector6<f32>,
) -> Vector6<f32> {
    let mut hessian: Matrix6<Float> = Matrix6::zeros();
    let mut rhs: Vector6<Float> = Vector6::zeros();
    for (jac, &res) in jacobian.iter().zip(residuals.iter()) {
        hessian = hessian + jac * jac.transpose();
        rhs = rhs + res * jac;
    }
    let twist_step = hessian.cholesky().unwrap().solve(&rhs);
    model - 0.1 * twist_step
}

fn _step_levenberg_marquardt(
    jacobian: &Vec<Jacobian>,
    residuals: &Vec<Residual>,
    motion: &Isometry3<f32>,
) -> Isometry3<f32> {
    let mut hessian: Matrix6<Float> = Matrix6::zeros();
    let mut rhs: Vector6<Float> = Vector6::zeros();
    let mean_coef = 1.0 / jacobian.len() as f32;
    // println!("nb points: {}", jacobian.len());

    // ----- debug
    let full_jacobian = MatrixMN::<_, _, Dynamic>::from_columns(jacobian.as_slice());
    let full_jacobian_64 = full_jacobian.map(|x| x as f64);
    // println!("Size Jac: {:?}", full_jacobian.shape());
    let singular_values = full_jacobian.clone().singular_values();
    let singular_values_64 = full_jacobian_64.clone().singular_values();
    // println!("Singular values Jac: {:?}", singular_values.data.data());
    // println!(
    //     "Singular values Jac (64): {:?}",
    //     singular_values_64.data.data()
    // );
    let jac_jac_t = full_jacobian.clone() * full_jacobian.transpose();
    let jac_jac_t_64 = full_jacobian_64.clone() * full_jacobian_64.transpose();
    // println!("Jac * JacT: {}", jac_jac_t / 100000000.0);
    let eigen_values = jac_jac_t.eigenvalues().unwrap();
    let eigen_values_64 = jac_jac_t_64.eigenvalues().unwrap();
    // println!(
    //     "Eigenvalues Jac * JacT: {:?}",
    //     (eigen_values / 10000000000.0).data
    // );
    // println!("Eigenvalues Jac * JacT (64): {:?}", (eigen_values_64).data);
    // ----- end debug

    for (jac, &res) in jacobian.iter().zip(residuals.iter()) {
        // println!("jacobian: {:?}", jac.data);
        hessian = hessian + jac * jac.transpose();
        rhs = rhs + res * jac;
    }

    // Scale changement do not change cholesky decomposition in theory
    // but may avoid overflowing max f32 values (3.402823 * 10^38)
    hessian = mean_coef * hessian;
    rhs = mean_coef * rhs;
    // println!("hessian: {}", &hessian);
    // println!("rhs: {}", &rhs);

    // SVD to check condition number
    let singular_values = hessian.singular_values();
    // println!("Singular values hessian: {:?}", singular_values.data);

    let levenberg_marquardt_coef = 1.2;
    hessian.m11 = levenberg_marquardt_coef * hessian.m11;
    hessian.m22 = levenberg_marquardt_coef * hessian.m22;
    hessian.m33 = levenberg_marquardt_coef * hessian.m33;
    hessian.m44 = levenberg_marquardt_coef * hessian.m44;
    hessian.m55 = levenberg_marquardt_coef * hessian.m55;
    hessian.m66 = levenberg_marquardt_coef * hessian.m66;
    let twist_step = 0.2 * hessian.cholesky().unwrap().solve(&(-rhs));
    // println!("twist_step: {}", &twist_step);
    let new_motion = se3::exp(se3::from_vector(twist_step)) * motion;
    // println!("Computed translation: {:?}", new_motion.translation.vector);
    new_motion
}

type Observation<'a> = (
    &'a Intrinsics,
    &'a DMatrix<InverseDepth>,
    &'a DMatrix<Float>,
    &'a DMatrix<Float>,
    &'a DMatrix<Float>,
    &'a DMatrix<Float>,
);

fn eval(observation: &Observation, motion: &Isometry3<f32>) -> (Vec<Jacobian>, Vec<Residual>) {
    let (intrinsics, idepth_map, img_1, img_2, gx_2, gy_2) = observation;
    let mut all_jacobians = Vec::new();
    let mut all_residuals = Vec::new();
    let (nrows, ncols) = idepth_map.shape();
    let (f, (sx, sy)) = (intrinsics.focal_length, intrinsics.scaling);
    let focale = Focale(f * sx, f * sy);
    let eval_at = |point, idepth, color_ref| {
        jacobian_at(point, idepth, img_2, gx_2, gy_2, &focale, color_ref)
    };
    for (index, idepth_enum) in idepth_map.iter().enumerate() {
        if let InverseDepth::WithVariance(idepth, _variance) = idepth_enum {
            // nb_candidates += 1;
            let (col, row) = helper::div_rem(index, nrows);
            let point = Point2::new(col as f32, row as f32);
            let (reprojected, new_idepth) = reproject(point, *idepth, intrinsics, motion);
            if helper::in_image_bounds((reprojected[0], reprojected[1]), (nrows, ncols)) {
                // nb_in_frame += 1;
                let color_ref = img_1[(row, col)];
                let (jacobian, residual) = eval_at(reprojected, new_idepth, color_ref);
                all_jacobians.push(jacobian);
                all_residuals.push(residual);
            }
        }
    }
    (all_jacobians, all_residuals)
}

// Step (1) is done by multiplying by 2 the threshold and recomputing the residuals every time (max 5 times) which seems very ineficient. But maybe it hapens rarely if the default threshold already allows more than 40% of points to have lower residuals.
//
// Step (2) is done with the function `CoarseTracker::calcRes()` (https://github.com/JakobEngel/dso/blob/master/src/FullSystem/CoarseTracker.cpp#L358). I'll summarize it as follows:

fn reproject(
    point: Point2<Float>,
    idepth: Float,
    intrinsics: &Intrinsics,
    motion: &Isometry3<Float>,
) -> (Point2<Float>, Float) {
    let homogeneous = intrinsics.project(motion * intrinsics.back_project(point, 1.0 / idepth));
    let new_idepth = 1.0 / homogeneous[2];
    let new_point = Point2::new(homogeneous[0] * new_idepth, homogeneous[1] * new_idepth);
    (new_point, new_idepth)
}

type Float = f32;
struct Focale(Float, Float);
struct Gradient(Float, Float);
type Jacobian = Vector6<Float>;
type Residual = Float;

fn jacobian_at(
    point: Point2<Float>,
    idepth: Float,
    img: &DMatrix<Float>,
    gradient_x: &DMatrix<Float>,
    gradient_y: &DMatrix<Float>,
    focale: &Focale,
    img_origin_pixel: Float,
) -> (Jacobian, Residual) {
    let (indices, coefs) = optimization::linear_interpolator(point);
    let g_x = optimization::interpolate_with(indices, coefs, gradient_x);
    let g_y = optimization::interpolate_with(indices, coefs, gradient_y);
    let gradient = Gradient(g_x, g_y);
    let img_xy = optimization::interpolate_with(indices, coefs, img);
    let residual = img_xy - img_origin_pixel as f32;
    (compute_jacobian(point, idepth, focale, gradient), residual)
}

fn compute_jacobian(point: Point2<Float>, idepth: Float, f: &Focale, g: Gradient) -> Jacobian {
    // * For each point:
    //     * (3.1) Scale saved gradients with the focale: dx = fx * dx and dy = fy * dy
    //     * (3.2) Compute 8 Jacobians (6 geometric, 2 photometric)
    //         J0 = idepth * dx
    //         J1 = idepth * dy
    //         J2 = - ( idepth * ( u * dx + v * dy ) )
    //         J3 = - ( dx * u * v + dy * ( 1 + v^2 ) )
    //         J4 = u * v * dy + dx * (1 + u^2 )
    //         J5 = u * dy - v * dx
    //         J6 = a * ( b - refColor )
    //         J7 = -1
    //     * (4.1) Accumulate the products of Jacobians in an Hessian matrix (H_ij += (Ji * Jj)).
    let (u, v) = (point[0], point[1]);
    let uv = u * v;
    let Focale(fx, fy) = f;
    let Gradient(dx, dy) = g;
    let fdx = fx * dx;
    let fdy = fy * dy;
    Vector6::new(
        idepth * fdx,
        idepth * fdy,
        -(idepth * (u * fdx + v * fdy)),
        -(uv * fdx + (1.0 + v * v) * fdy),
        uv * fdy + (1.0 + u * u) * fdx,
        u * fdy - v * fdx,
    )

    // Mathematically, we could compute the jacobian like below,
    // as a product of the gradient and partial derivates,
    // but it's just less efficient to do so:
    //
    // let gradient_transposed = Vector2::new(fx * dx, fy * dy);
    // let partial_derivates = Matrix6x2::from_columns(&[
    //     Vector6::new(idepth, 0.0, -idepth * u, -uv, 1.0 + u * u, -v),
    //     Vector6::new(0.0, idepth, -idepth * v, -(1.0 + v * v), uv, u),
    // ]);
    // let jacobian = partial_derivates * gradient_transposed;
}
