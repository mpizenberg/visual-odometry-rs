// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate nalgebra as na;
extern crate visual_odometry_rs as vors;

use std::process::exit;
use vors::math::optimizer::{Continue, OptimizerState};

/// In this example, we implement the OptimizerState trait to estimate a rosenbrock function.
/// rosenbrock: (a, b, x, y) -> (a-x)^2 + b*(y-x^2)^2
///
/// The two residuals are: r1 = (a-x)^2  and  r2 = b*(y-x^2)^2
/// So the jacobian matrix is:
///    | -2.0 * (a - x)            , 0.0                   |
///    | -4.0 * b * x * (y - x * x), 2.0 * b * (y - x * x) |
fn main() {
    if let Err(err) = run() {
        eprintln!("{}", err);
        exit(1);
    };
}

fn run() -> Result<(), String> {
    let initial_model = (-2.0, -2.0);
    let (final_state, nb_iteration) = LMOptimizerState::iterative_solve(&(), initial_model)?;
    println!("After {} iterations:", nb_iteration);
    println!("Computed: {:?}", final_state.eval_data.model);
    println!("Solution: (1.0, 1.0)");
    Ok(())
}

type Vec2 = na::Vector2<f32>;
type Mat2 = na::Matrix2<f32>;

const A: f32 = 1.0;
const B: f32 = 100.0;

/// Residuals
fn rosenbrock_res(x: f32, y: f32) -> Vec2 {
    Vec2::new((A - x).powi(2), B * (y - x * x).powi(2))
}

/// Jacobian
fn rosenbrock_jac(x: f32, y: f32) -> Mat2 {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Mat2::new(
        -2.0 * (A - x)            , 0.0                  , // for 1st residual
        -4.0 * B * x * (y - x * x), 2.0 * B * (y - x * x), // for 2nd residual
    )
}

/// Levenberg-Marquardt (LM) state of the optimizer.
struct LMOptimizerState {
    lm_coef: f32,
    eval_data: EvalData,
}

/// State useful to evaluate the model.
struct EvalData {
    model: (f32, f32),
    energy: f32,
    gradient: Vec2,
    hessian: Mat2,
}

/// Same role as EvalData except we might not need to fully compute the EvalData.
/// In such cases (evaluation stopped after noticing an increasing energy),
/// the partial state only contains an error with the corresponding energy.
type EvalState = Result<EvalData, f32>;

impl LMOptimizerState {
    /// Compute evaluation data needed for the next iteration step.
    fn compute_eval_data(model: (f32, f32), residuals: Vec2) -> EvalData {
        let (x, y) = model;
        let energy = residuals.norm_squared();
        let jacobian = rosenbrock_jac(x, y);
        let jacobian_t = jacobian.transpose();
        let gradient = jacobian_t * residuals;
        let hessian = jacobian_t * jacobian;
        EvalData {
            model,
            energy,
            gradient,
            hessian,
        }
    }
}

impl OptimizerState<(), EvalState, (f32, f32), String> for LMOptimizerState {
    /// Initialize the optimizer state.
    /// Levenberg-Marquardt coefficient start at 0.1.
    fn init(_obs: &(), model: (f32, f32)) -> Self {
        Self {
            lm_coef: 0.1,
            eval_data: Self::compute_eval_data(model, rosenbrock_res(model.0, model.1)),
        }
    }

    /// Compute the Levenberg-Marquardt step.
    fn step(&self) -> Result<(f32, f32), String> {
        let mut hessian = self.eval_data.hessian.clone();
        hessian.m11 = (1.0 + self.lm_coef) * hessian.m11;
        hessian.m22 = (1.0 + self.lm_coef) * hessian.m22;
        let cholesky = hessian.cholesky().ok_or("Issue in Cholesky")?;
        let delta = cholesky.solve(&self.eval_data.gradient);
        let (x, y) = self.eval_data.model;
        Ok((x - delta.x, y - delta.y))
    }

    /// Evaluate the new model.
    fn eval(&self, _obs: &(), model: (f32, f32)) -> EvalState {
        let (x, y) = model;
        let residuals = rosenbrock_res(x, y);
        let energy = residuals.norm_squared();
        let old_energy = self.eval_data.energy;
        if energy > old_energy {
            Err(energy)
        } else {
            Ok(Self::compute_eval_data(model, residuals))
        }
    }

    /// Decide if iterations should continue.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue) {
        let too_many_iterations = nb_iter >= 100;
        match (eval_state, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (self, Continue::Stop),
            (Ok(eval_data), true) => {
                println!(
                    "model = {:?}, energy = {}",
                    eval_data.model, eval_data.energy
                );
                let mut kept_state = self;
                kept_state.eval_data = eval_data;
                (kept_state, Continue::Stop)
            }
            // Can continue to iterate:
            (Err(model), false) => {
                let mut kept_state = self;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
                println!("\t back from {:?}, lm_coef = {}", model, kept_state.lm_coef);
                (kept_state, Continue::Forward)
            }
            (Ok(eval_data), false) => {
                println!(
                    "model = {:?}, energy = {}",
                    eval_data.model, eval_data.energy
                );
                let delta_energy = self.eval_data.energy - eval_data.energy;
                let mut kept_state = self;
                kept_state.lm_coef = 0.1 * kept_state.lm_coef;
                kept_state.eval_data = eval_data;
                let continuation = if delta_energy > 1e-10 {
                    Continue::Forward
                } else {
                    Continue::Stop
                };
                (kept_state, continuation)
            }
        }
    }
}
