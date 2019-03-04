//! Levenberg-Marquardt implementation of the Optimizer trait
//! for the inverse compositional tracking algorithm.

use nalgebra::{DMatrix, UnitQuaternion};
use std::f32;

use crate::core::camera::Intrinsics;
use crate::math::optimizer::{self as optim, Optimizer, State};
use crate::math::se3;
use crate::misc::type_aliases::{Float, Iso3, Mat6, Point2, Vec6};

/// Empty struct, implementor of `Optimizer`.
pub struct LMOptimizer;

/// State of the Levenberg-Marquardt optimizer.
pub struct LMState {
    /// Levenberg-Marquardt hessian diagonal coefficient.
    pub lm_coef: Float,
    /// Data used during the optimizer iterations.
    pub data: LMData,
}

/// Data needed during the optimizer iterations.
pub struct LMData {
    /// The hessian matrix of the system.
    pub hessian: Mat6,
    /// The gradient of the system.
    pub gradient: Vec6,
    /// Energy associated with the current model.
    pub energy: Float,
    /// Estimated pose at the current state of iterations.
    pub model: Iso3,
}

/// Either a fully constructed `LMData`
/// or an error containing the energy of the new iteration.
///
/// The error is returned when the new computed energy
/// is higher than the previous iteration energy.
pub type LMPartialState = Result<LMData, Float>;

/// (inside_indices, residuals, energy).
pub type PreEval = (Vec<usize>, Vec<Float>, Float);

/// Precomputed data available for the optimizer iterations:
pub struct Obs<'a> {
    /// Intrinsic parameters of the camera.
    pub intrinsics: &'a Intrinsics,
    /// Reference ("keyframe") image.
    pub template: &'a DMatrix<u8>,
    /// Current image to track.
    pub image: &'a DMatrix<u8>,
    /// Coordinates of the points used for the tracking.
    pub coordinates: &'a Vec<(usize, usize)>,
    /// Inverse depth of the points used for the tracking.
    pub _z_candidates: &'a Vec<Float>,
    /// Jacobians precomputed for the points used for the tracking.
    pub jacobians: &'a Vec<Vec6>,
    /// Hessian matrices precomputed for the points used for the tracking.
    pub hessians: &'a Vec<Mat6>,
}

impl State<Iso3, Float> for LMState {
    fn model(&self) -> &Iso3 {
        &self.data.model
    }
    fn energy(&self) -> Float {
        self.data.energy
    }
}

/// impl<'a> Optimizer<Obs<'a>, LMState, Vec6, Iso3, PreEval, LMPartialState, Float> for LMOptimizer.
impl<'a> Optimizer<Obs<'a>, LMState, Vec6, Iso3, PreEval, LMPartialState, Float> for LMOptimizer {
    /// Initial energy is f32::INFINITY.
    fn initial_energy() -> Float {
        f32::INFINITY
    }

    /// Compute the step using Levenberg-Marquardt.
    /// May return an error at the Cholesky decomposition of the hessian.
    fn compute_step(state: &LMState) -> Option<Vec6> {
        let mut hessian = state.data.hessian.clone();
        hessian.m11 = (1.0 + state.lm_coef) * hessian.m11;
        hessian.m22 = (1.0 + state.lm_coef) * hessian.m22;
        hessian.m33 = (1.0 + state.lm_coef) * hessian.m33;
        hessian.m44 = (1.0 + state.lm_coef) * hessian.m44;
        hessian.m55 = (1.0 + state.lm_coef) * hessian.m55;
        hessian.m66 = (1.0 + state.lm_coef) * hessian.m66;
        hessian.cholesky().map(|ch| ch.solve(&state.data.gradient))
    }

    /// Apply the step of an inverse compositional approach to compute the next pose estimation.
    fn apply_step(delta: Vec6, model: &Iso3) -> Iso3 {
        let delta_warp = se3::exp(delta);
        let mut not_normalized = model * delta_warp.inverse();
        not_normalized.rotation = re_normalize(not_normalized.rotation);
        not_normalized
    }

    /// Compute residuals and energy of the new model.
    fn pre_eval(obs: &Obs, model: &Iso3) -> PreEval {
        let mut inside_indices = Vec::new();
        let mut residuals = Vec::new();
        let mut energy = 0.0;
        for (idx, &(x, y)) in obs.coordinates.iter().enumerate() {
            let _z = obs._z_candidates[idx];
            // check if warp(x,y) is inside the image
            let (u, v) = warp(model, x as Float, y as Float, _z, &obs.intrinsics);
            if let Some(im) = interpolate(u, v, &obs.image) {
                // precompute residuals and energy
                let tmp = obs.template[(y, x)];
                let r = im - tmp as Float;
                energy = energy + r * r;
                residuals.push(r);
                inside_indices.push(idx); // keep only inside points
            }
        }
        energy = energy / residuals.len() as Float;
        (inside_indices, residuals, energy)
    }

    /// Evaluate the new hessian and gradient if the energy is lower than previously.
    fn eval(obs: &Obs, energy: Float, pre_eval: PreEval, model: Iso3) -> LMPartialState {
        let (inside_indices, residuals, new_energy) = pre_eval;
        if new_energy > energy {
            Err(new_energy)
        } else {
            let mut gradient = Vec6::zeros();
            let mut hessian = Mat6::zeros();
            for (i, idx) in inside_indices.iter().enumerate() {
                let jac = obs.jacobians[*idx];
                let hes = obs.hessians[*idx];
                let r = residuals[i];
                gradient = gradient + jac * r;
                hessian = hessian + hes;
            }
            Ok(LMData {
                hessian,
                gradient,
                energy: new_energy,
                model,
            })
        }
    }

    /// Stop after too many iterations,
    /// or if the energy variation is too low.
    ///
    /// Also updates the Levenberg-Marquardt coefficient
    /// depending on if the energy increased or decreased.
    fn stop_criterion(
        nb_iter: usize,
        s0: LMState,
        s1: LMPartialState,
    ) -> (LMState, optim::Continue) {
        let too_many_iterations = nb_iter > 20;
        match (s1, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (s0, optim::Continue::Stop),
            (Ok(data), true) => {
                // eprintln!("Energy: {}", data.energy);
                // eprintln!("Gradient norm: {}", data.gradient.norm());
                let kept_state = LMState {
                    lm_coef: s0.lm_coef, // does not matter actually
                    data: data,
                };
                (kept_state, optim::Continue::Stop)
            }
            // Can continue to iterate:
            (Err(_energy), false) => {
                // eprintln!("\t back from: {}", energy);
                let mut kept_state = s0;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
                (kept_state, optim::Continue::Forward)
            }
            (Ok(data), false) => {
                let d_energy = s0.data.energy - data.energy;
                let _gradient_norm = data.gradient.norm();
                // 1.0 is totally empiric here
                let continuation = if d_energy > 1.0 {
                    optim::Continue::Forward
                } else {
                    optim::Continue::Stop
                };
                // eprintln!("Energy: {}", data.energy);
                // eprintln!("Gradient norm: {}", gradient_norm);
                let kept_state = LMState {
                    lm_coef: 0.1 * s0.lm_coef,
                    data: data,
                };
                (kept_state, continuation)
            }
        }
    } // fn stop_criterion
} // impl Optimizer<...> for LMOptimizer

// Helper ######################################################################

/// First order Taylor approximation for unit quaternion re-normalization.
fn re_normalize(uq: UnitQuaternion<Float>) -> UnitQuaternion<Float> {
    let q = uq.into_inner();
    let sq_norm = q.norm_squared();
    UnitQuaternion::new_unchecked(0.5 * (3.0 - sq_norm) * q)
}

/// Warp a point from an image to another by a given rigid body motion.
fn warp(model: &Iso3, x: Float, y: Float, _z: Float, intrinsics: &Intrinsics) -> (Float, Float) {
    // TODO: maybe move into the camera module?
    let x1 = intrinsics.back_project(Point2::new(x, y), 1.0 / _z);
    let x2 = model * x1;
    let uvz2 = intrinsics.project(x2);
    (uvz2.x / uvz2.z, uvz2.y / uvz2.z)
}

/// Simple linear interpolation of a pixel with floating point coordinates.
/// Return `None` if the point is outside of the image boundaries.
fn interpolate(x: Float, y: Float, image: &DMatrix<u8>) -> Option<Float> {
    let (height, width) = image.shape();
    let u = x.floor();
    let v = y.floor();
    if u >= 0.0 && u < (width - 2) as Float && v >= 0.0 && v < (height - 2) as Float {
        let u_0 = u as usize;
        let v_0 = v as usize;
        let u_1 = u_0 + 1;
        let v_1 = v_0 + 1;
        let vu_00 = image[(v_0, u_0)] as Float;
        let vu_10 = image[(v_1, u_0)] as Float;
        let vu_01 = image[(v_0, u_1)] as Float;
        let vu_11 = image[(v_1, u_1)] as Float;
        let a = x - u;
        let b = y - v;
        Some(
            (1.0 - b) * (1.0 - a) * vu_00
                + b * (1.0 - a) * vu_10
                + (1.0 - b) * a * vu_01
                + b * a * vu_11,
        )
    } else {
        None
    }
}
