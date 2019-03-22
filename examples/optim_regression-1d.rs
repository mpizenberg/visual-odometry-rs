// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate nalgebra as na;
extern crate rand;
extern crate visual_odometry_rs as vors;

use na::DVector;
use rand::{distributions::Uniform, rngs::StdRng, SeedableRng};
use std::{f32, process::exit};
use vors::math::optimizer::{Continue, OptimizerState};

/// In this example, we implement the `OptimizerState` trait to find the correct parameter `a`
/// for modelling a noisy curve of the form: y = exp( -a * x ).
/// Let f be the function: (a, x) -> exp( -a * x ).
///
/// For this, we are using the Levenberg-Marquardt optimization schema.
/// Note that any iterative optimization algorithm fitting the Optimizer trait
/// could have been used instead. This just serves as an example.
///
/// We thus try to minimize the expression: Sum_i ( |residual_i|^2 )
/// where each residual is: y_i - f(a, x_i) for a data point ( x_i, y_i ).
///
/// The jacobian of the vector of residuals is obtained by deriving the expression with regard to a.
/// Let f_1 be the derivative of f with regard a.
/// f_1: (a, x) -> -x * f(a, x)
/// For each residual, we thus obtain:
/// jacobian_i = -x_i * f(a, x_i)
///
/// The jacobian is 1-dimensional, to the hessian is a scalar:
/// hessian = transpose(jacobian) * jacobian
///
/// And the gradient is also a scalar:
/// gradient = transpose(jacobian) * residuals
///
/// So finally, if we note hessian_lm the slightly modified hessian (Levenberg-Marquardt coef),
/// the iteration step is:
/// step = - gradient / hessian

fn main() {
    if let Err(err) = run() {
        eprintln!("{}", err);
        exit(1);
    };
}

fn run() -> Result<(), String> {
    // Ground truth value for the parameter a.
    let a_ground_truth = 1.5;

    // Let's generate some noisy observations.
    let nb_data: usize = 100;
    let x_domain = linspace(-5.0, 3.0, nb_data);
    let seed = [0; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut distribution = Uniform::from(-1.0..1.0);
    let noise = DVec::from_distribution(nb_data, &mut distribution, &mut rng);
    let y_data_noisy = f(a_ground_truth, &x_domain) + 0.1 * noise;
    let noisy_observations = Obs {
        x: x_domain,
        y: y_data_noisy,
    };

    // Now we run the Levenberg-Marquardt optimizer
    // as defined below in the Optimizer trait implementation.
    let (final_state, nb_iteration) = LMOptimizerState::iterative_solve(&noisy_observations, 0.0)?;
    println!("After {} iterations:", nb_iteration);
    println!("Ground truth: a = {}", a_ground_truth);
    println!("Computed:     a = {}", final_state.eval_data.model);
    Ok(())
}

/// Simpler type alias for a vector of floats.
type DVec = DVector<f32>;

/// Helper function to generate regularly spaced values.
fn linspace(start: f32, end: f32, nb: usize) -> DVec {
    DVec::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - 1 - i) as f32 * start) / (nb as f32 - 1.0)
    })
}

/// The function modelling our observations: y = exp(-a x)
fn f(a: f32, x_domain: &DVec) -> DVec {
    x_domain.map(|x| (-a * x).exp())
}

/// Type of the data observed: x and y coordinates of data points.
struct Obs {
    x: DVec,
    y: DVec,
}

/// Levenberg-Marquardt (LM) state of the optimizer.
struct LMOptimizerState {
    lm_coef: f32,
    eval_data: EvalData,
}

/// State useful to evaluate the model.
struct EvalData {
    model: f32,
    energy: f32,
    gradient: f32,
    hessian: f32,
}

/// Same role as EvalData except we might not need to fully compute the EvalData.
/// In such cases (evaluation stopped after noticing an increasing energy),
/// the partial state only contains an error with the corresponding energy.
type EvalState = Result<EvalData, f32>;

impl LMOptimizerState {
    /// Evaluate energy associated with a model.
    fn eval_energy(obs: &Obs, model: f32) -> (f32, DVec, DVec) {
        let f_model = f(model, &obs.x);
        let residuals = &f_model - &obs.y;
        let new_energy: f32 = residuals.iter().map(|r| r * r).sum();
        (new_energy / residuals.len() as f32, residuals, f_model)
    }

    /// Compute evaluation data needed for the next iteration step.
    fn compute_eval_data(obs: &Obs, model: f32, pre: (f32, DVec, DVec)) -> EvalData {
        let (energy, residuals, f_model) = pre;
        let jacobian = -1.0 * f_model.component_mul(&obs.x);
        let gradient = jacobian.dot(&residuals);
        let hessian = jacobian.dot(&jacobian);
        EvalData {
            model,
            energy,
            gradient,
            hessian,
        }
    }
}

impl OptimizerState<Obs, EvalState, f32, String> for LMOptimizerState {
    /// Initialize the optimizer state.
    /// Levenberg-Marquardt coefficient start at 0.1.
    fn init(obs: &Obs, model: f32) -> Self {
        Self {
            lm_coef: 0.1,
            eval_data: Self::compute_eval_data(obs, model, Self::eval_energy(obs, model)),
        }
    }

    /// Compute the Levenberg-Marquardt step.
    fn step(&self) -> Result<f32, String> {
        let hessian = (1.0 + self.lm_coef) * self.eval_data.hessian;
        Ok(self.eval_data.model - self.eval_data.gradient / hessian)
    }

    /// Evaluate the new model.
    fn eval(&self, obs: &Obs, model: f32) -> EvalState {
        let pre = Self::eval_energy(obs, model);
        let energy = pre.0;
        let old_energy = self.eval_data.energy;
        if energy > old_energy {
            Err(energy)
        } else {
            Ok(Self::compute_eval_data(obs, model, pre))
        }
    }

    /// Decide if iterations should continue.
    fn stop_criterion(self, nb_iter: usize, eval_state: EvalState) -> (Self, Continue) {
        let too_many_iterations = nb_iter >= 20;
        match (eval_state, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (self, Continue::Stop),
            (Ok(eval_data), true) => {
                println!("a = {}, energy = {}", eval_data.model, eval_data.energy);
                let mut kept_state = self;
                kept_state.eval_data = eval_data;
                (kept_state, Continue::Stop)
            }
            // Can continue to iterate:
            (Err(model), false) => {
                let mut kept_state = self;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
                println!("\t back from {}, lm_coef = {}", model, kept_state.lm_coef);
                (kept_state, Continue::Forward)
            }
            (Ok(eval_data), false) => {
                println!("a = {}, energy = {}", eval_data.model, eval_data.energy);
                let delta_energy = self.eval_data.energy - eval_data.energy;
                let mut kept_state = self;
                kept_state.lm_coef = 0.1 * kept_state.lm_coef;
                kept_state.eval_data = eval_data;
                let continuation = if delta_energy > 0.01 {
                    Continue::Forward
                } else {
                    Continue::Stop
                };
                (kept_state, continuation)
            }
        }
    }
}
