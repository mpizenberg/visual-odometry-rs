//! Levenberg-Marquardt implementation of the Optimizer trait
//! for the inverse compositional tracking algorithm.

use nalgebra::{DMatrix, UnitQuaternion};

use crate::core::camera::Intrinsics;
use crate::math::optimizer::{Continue, OptimizerState};
use crate::math::se3;
use crate::misc::type_aliases::{Float, Iso3, Mat6, Point2, Vec6};

/// State of the Levenberg-Marquardt optimizer.
pub struct LMOptimizerState {
    /// Levenberg-Marquardt hessian diagonal coefficient.
    pub lm_coef: Float,
    /// Data used during the optimizer iterations.
    pub eval_data: EvalData,
}

/// Either a fully constructed `LMData`
/// or an error containing the energy of the new iteration.
///
/// The error is returned when the new computed energy
/// is higher than the previous iteration energy.
pub type EvalState = Result<EvalData, Float>;

/// Data needed during the optimizer iterations.
pub struct EvalData {
    /// The hessian matrix of the system.
    pub hessian: Mat6,
    /// The gradient of the system.
    pub gradient: Vec6,
    /// Energy associated with the current model.
    pub energy: Float,
    /// Estimated pose at the current state of iterations.
    pub model: Iso3,
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
    /// Hessian matrices precomputed for the points used for the tracking.
    pub hessians: &'a Vec<Mat6>,
}

/// (energy, inside_indices, residuals)
type Precomputed = (Float, Vec<usize>, Vec<Float>);

impl LMOptimizerState {
    fn eval_energy(obs: &Obs, model: &Iso3) -> Precomputed {
        let mut inside_indices = Vec::new();
        let mut residuals = Vec::new();
        let mut energy_sum = 0.0;
        for (idx, &(x, y)) in obs.coordinates.iter().enumerate() {
            let _z = obs._z_candidates[idx];
            // check if warp(x,y) is inside the image
            let (u, v) = warp(model, x as Float, y as Float, _z, &obs.intrinsics);
            if let Some(im) = interpolate(u, v, &obs.image) {
                // precompute residuals and energy
                let tmp = obs.template[(y, x)];
                let r = im - tmp as Float;
                energy_sum = energy_sum + r * r;
                residuals.push(r);
                inside_indices.push(idx); // keep only inside points
            }
        }
        let energy = energy_sum / residuals.len() as Float;
        (energy, inside_indices, residuals)
    }

    fn compute_eval_data(obs: &Obs, model: Iso3, pre: Precomputed) -> EvalData {
        let (energy, inside_indices, residuals) = pre;
        let mut gradient = Vec6::zeros();
        let mut hessian = Mat6::zeros();
        for (i, idx) in inside_indices.iter().enumerate() {
            let jac = obs.jacobians[*idx];
            let hes = obs.hessians[*idx];
            let r = residuals[i];
            gradient = gradient + jac * r;
            hessian = hessian + hes;
        }
        EvalData {
            hessian,
            gradient,
            energy,
            model,
        }
    }
}

/// impl<'a> Optimizer<Obs<'a>, LMState, Iso3, Float, EvalState, String> for LMOptimizer.
impl<'a> OptimizerState<Obs<'a>, EvalState, Iso3, String> for LMOptimizerState {
    /// Initialize the optimizer state.
    fn init(obs: &Obs, model: Iso3) -> Self {
        Self {
            lm_coef: 0.1,
            eval_data: Self::compute_eval_data(obs, model, Self::eval_energy(obs, &model)),
        }
    }

    /// Compute the step using Levenberg-Marquardt.
    /// Apply the step of an inverse compositional approach to compute the next pose estimation.
    /// May return an error at the Cholesky decomposition of the hessian.
    fn step(&self) -> Result<Iso3, String> {
        let mut hessian = self.eval_data.hessian.clone();
        hessian.m11 = (1.0 + self.lm_coef) * hessian.m11;
        hessian.m22 = (1.0 + self.lm_coef) * hessian.m22;
        hessian.m33 = (1.0 + self.lm_coef) * hessian.m33;
        hessian.m44 = (1.0 + self.lm_coef) * hessian.m44;
        hessian.m55 = (1.0 + self.lm_coef) * hessian.m55;
        hessian.m66 = (1.0 + self.lm_coef) * hessian.m66;
        let delta = hessian
            .cholesky()
            .map(|ch| ch.solve(&self.eval_data.gradient))
            .ok_or("Error at Cholesky decomposition of hessian")?;
        let delta_warp = se3::exp(delta);
        let mut motion = self.eval_data.model * delta_warp.inverse();
        motion.rotation = re_normalize(motion.rotation);
        Ok(motion)
    }

    /// Compute residuals and energy of the new model.
    /// Then, evaluate the new hessian and gradient if the energy is lower than previously.
    fn eval(&self, obs: &Obs, model: Iso3) -> EvalState {
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
    /// Also updates the Levenberg-Marquardt coefficient
    /// depending on if the energy increased or decreased.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue) {
        let too_many_iterations = nb_iter > 20;
        match (eval_state, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (self, Continue::Stop),
            (Ok(eval_data), true) => {
                // eprintln!("Energy: {}", eval_data.energy);
                let kept_state = LMOptimizerState {
                    lm_coef: self.lm_coef, // does not matter actually
                    eval_data,
                };
                (kept_state, Continue::Stop)
            }
            // Can continue to iterate:
            (Err(_energy), false) => {
                // eprintln!("\t back from: {}", energy);
                let mut kept_state = self;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
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
                let kept_state = LMOptimizerState {
                    lm_coef: 0.1 * self.lm_coef,
                    eval_data,
                };
                (kept_state, continuation)
            }
        }
    } // fn stop_criterion
} // impl OptimizerState<...> for LMOptimizerState

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
