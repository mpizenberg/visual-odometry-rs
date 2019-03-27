// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Levenberg-Marquardt implementation of the `optimizer::State` trait
//! for the inverse compositional tracking algorithm.

use nalgebra::{DMatrix, UnitQuaternion, U6};

use crate::core::camera::Intrinsics;
use crate::math::optimizer::{self, Continue};
use crate::math::se3;
use crate::misc::type_aliases::{DMat, DVec, Float, Iso3, Point2, Vec6};

/// State of the Levenberg-Marquardt optimizer.
pub struct LMOptimizerState {
    /// Levenberg-Marquardt hessian diagonal coefficient.
    pub lm_coef: Float,
    /// Data resulting of a successful model evaluation.
    pub eval_data: EvalData,
}

/// Either a successfully constructed `EvalData`
/// or an error containing the energy of a given model.
///
/// The error is returned when the new computed energy
/// is higher than the previous iteration energy.
pub type EvalState = Result<EvalData, Float>;

/// Motion and photometric parameters.
pub type Model = (Iso3, Float, Float);

/// Data resulting of a successful model evaluation.
pub struct EvalData {
    /// The hessian matrix of the system.
    pub hessian: DMat,
    /// The gradient of the system.
    pub gradient: DVec,
    /// Energy associated with the current model.
    pub energy: Float,
    /// Estimated motion and exposure at the current state of iterations.
    pub model: Model,
}

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
    /// Exposure parameters (a, b) of the reference image.
    pub template_exposure: &'a (Float, Float),
}

/// `(energy, inside_indices, residuals)`.
type Precomputed = (Float, Vec<usize>, Vec<Float>);

impl LMOptimizerState {
    /// Precompute the energy of a model.
    /// Also return the residuals vector and the indices of candidate points used.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::used_underscore_binding)]
    fn eval_energy(obs: &Obs, model: &Model) -> Precomputed {
        let mut inside_indices = Vec::new();
        let mut residuals = Vec::new();
        let mut energy_sum = 0.0;
        let (motion, a, b) = model;
        let (a_tmp, b_tmp) = obs.template_exposure;
        for (idx, &(x, y)) in obs.coordinates.iter().enumerate() {
            let _z = obs._z_candidates[idx];
            // check if warp(x,y) is inside the image
            let (u, v) = warp(motion, x as Float, y as Float, _z, obs.intrinsics);
            if let Some(im) = interpolate(u, v, obs.image) {
                // precompute residuals and energy
                let tmp = obs.template[(y, x)];
                let r = im - b - (a - a_tmp).exp() * (Float::from(tmp) - b_tmp);
                energy_sum += r * r;
                residuals.push(r);
                inside_indices.push(idx); // keep only inside points
            }
        }
        let energy = energy_sum / residuals.len() as Float;
        (energy, inside_indices, residuals)
    }

    /// Fully evaluate a model.
    fn compute_eval_data(obs: &Obs, model: Model, pre: Precomputed) -> EvalData {
        let (energy, inside_indices, residuals) = pre;
        let mut gradient = DVec::zeros(8);
        let mut hessian = DMat::zeros(8, 8);
        let (a_tmp, b_tmp) = obs.template_exposure;
        let (_, a, _) = &model;
        // Build gradient with inside points.
        for (i, idx) in inside_indices.into_iter().enumerate() {
            // Compute photometric terms of jac.
            let mut jac = DVec::zeros(8);
            let (x, y) = obs.coordinates[idx];
            let tmp_color = obs.template[(y, x)];
            let alpha_ratio = (a - a_tmp).exp();
            jac[7] = -1.0;
            jac[6] = alpha_ratio * (b_tmp - Float::from(tmp_color));

            // Fill geometric terms of jac.
            // *jac.fixed_rows_mut::<U6>(0) = *obs.jacobians[idx];
            *jac.fixed_rows_mut::<U6>(0) = *(alpha_ratio * obs.jacobians[idx]);

            // TODO: replace with accumulator
            let r = residuals[i];
            gradient += &jac * r;
            hessian += &jac * &jac.transpose();
        }
        EvalData {
            hessian,
            gradient,
            energy,
            model,
        }
    }
}

/// `impl<'a> optimizer::State<Obs<'a>, EvalState, Model, String> for LMOptimizerState`.
impl<'a> optimizer::State<Obs<'a>, EvalState, Model, String> for LMOptimizerState {
    /// Initialize the optimizer state.
    fn init(obs: &Obs, model: Model) -> Self {
        Self {
            lm_coef: 0.1,
            eval_data: Self::compute_eval_data(obs, model, Self::eval_energy(obs, &model)),
        }
    }

    /// Compute the step using Levenberg-Marquardt.
    /// Apply the step in an inverse compositional approach to compute the next motion estimation.
    /// May return an error at the Cholesky decomposition of the hessian.
    fn step(&self) -> Result<Model, String> {
        let mut hessian = self.eval_data.hessian.clone();
        hessian[(0, 0)] *= 1.0 + self.lm_coef;
        hessian[(1, 1)] *= 1.0 + self.lm_coef;
        hessian[(2, 2)] *= 1.0 + self.lm_coef;
        hessian[(3, 3)] *= 1.0 + self.lm_coef;
        hessian[(4, 4)] *= 1.0 + self.lm_coef;
        hessian[(5, 5)] *= 1.0 + self.lm_coef;
        hessian[(6, 6)] *= 1.0 + self.lm_coef;
        hessian[(7, 7)] *= 1.0 + self.lm_coef;
        let cholesky = hessian
            .cholesky()
            .ok_or("Error at Cholesky decomposition of hessian")?;
        let delta = cholesky.solve(&self.eval_data.gradient);
        let delta_warp = se3::exp(delta.fixed_rows::<U6>(0).into_owned());
        let (motion, photo_a, photo_b) = self.eval_data.model;
        let new_motion = renormalize(motion * delta_warp.inverse());
        let new_photo_a = photo_a - delta[6];
        let new_photo_b = photo_b - delta[7];
        Ok((new_motion, new_photo_a, new_photo_b))
    }

    /// Compute residuals and energy of the new model.
    /// Then, evaluate the new hessian and gradient if the energy has decreased.
    fn eval(&self, obs: &Obs, model: Model) -> EvalState {
        let pre = Self::eval_energy(obs, &model);
        let energy = pre.0;
        let old_energy = self.eval_data.energy;
        if energy > old_energy {
            Err(energy)
        } else {
            Ok(Self::compute_eval_data(obs, model, pre))
        }
    }

    /// Stop after too many iterations,
    /// or if the energy variation is too low.
    ///
    /// Also update the Levenberg-Marquardt coefficient
    /// depending on if the energy increased or decreased.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue) {
        let too_many_iterations = nb_iter > 20;
        match (eval_state, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (self, Continue::Stop),
            (Ok(eval_data), true) => {
                // eprintln!("Energy: {}", eval_data.energy);
                let kept_state = Self {
                    lm_coef: self.lm_coef, // does not matter actually
                    eval_data,
                };
                (kept_state, Continue::Stop)
            }
            // Can continue to iterate:
            (Err(_energy), false) => {
                // eprintln!("\t back from: {}", energy);
                let mut kept_state = self;
                kept_state.lm_coef *= 10.0;
                (kept_state, Continue::Forward)
            }
            (Ok(eval_data), false) => {
                let d_energy = self.eval_data.energy - eval_data.energy;
                // 1.0 is totally empiric here
                let continuation = if d_energy > 1.0 {
                    Continue::Forward
                } else {
                    Continue::Stop
                };
                // eprintln!("Energy: {}", eval_data.energy);
                let kept_state = Self {
                    lm_coef: 0.1 * self.lm_coef,
                    eval_data,
                };
                (kept_state, continuation)
            }
        }
    } // fn stop_criterion
} // impl optimizer::State<...> for LMOptimizerState

// Helper ######################################################################

/// First order Taylor approximation for renormalization of rotation part of motion.
fn renormalize(motion: Iso3) -> Iso3 {
    let mut motion = motion;
    motion.rotation = renormalize_unit_quaternion(motion.rotation);
    motion
}

/// First order Taylor approximation for unit quaternion re-normalization.
fn renormalize_unit_quaternion(uq: UnitQuaternion<Float>) -> UnitQuaternion<Float> {
    let q = uq.into_inner();
    let sq_norm = q.norm_squared();
    UnitQuaternion::new_unchecked(0.5 * (3.0 - sq_norm) * q)
}

/// Warp a point from an image to another by a given rigid body motion.
#[allow(clippy::used_underscore_binding)]
fn warp(pose: &Iso3, x: Float, y: Float, _z: Float, intrinsics: &Intrinsics) -> (Float, Float) {
    // TODO: maybe move into the camera module?
    let x1 = intrinsics.back_project(Point2::new(x, y), 1.0 / _z);
    let x2 = pose * x1;
    let uvz2 = intrinsics.project(x2);
    (uvz2.x / uvz2.z, uvz2.y / uvz2.z)
}

/// Simple linear interpolation of a pixel with floating point coordinates.
/// Return `None` if the point is outside of the image boundaries.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_precision_loss)]
fn interpolate(x: Float, y: Float, image: &DMatrix<u8>) -> Option<Float> {
    let (height, width) = image.shape();
    let u = x.floor();
    let v = y.floor();
    if u >= 0.0 && u < (width - 2) as Float && v >= 0.0 && v < (height - 2) as Float {
        let u_0 = u as usize;
        let v_0 = v as usize;
        let u_1 = u_0 + 1;
        let v_1 = v_0 + 1;
        let vu_00 = Float::from(image[(v_0, u_0)]);
        let vu_10 = Float::from(image[(v_1, u_0)]);
        let vu_01 = Float::from(image[(v_0, u_1)]);
        let vu_11 = Float::from(image[(v_1, u_1)]);
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
