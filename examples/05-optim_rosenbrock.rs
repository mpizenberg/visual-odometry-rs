extern crate computer_vision_rs as cv;
extern crate nalgebra as na;

use cv::optimization_bis::{Continue, Optimizer, State};
use na::{Matrix2, Vector2};
use std::f32;

fn main() {
    let _ = run();
}

type Vec2 = Vector2<f32>;
type Mat2 = Matrix2<f32>;

const A: f32 = 1.0;
const B: f32 = 100.0;

fn rosenbrock_res(x: f32, y: f32) -> Vec2 {
    Vector2::new((A - x).powi(2), B * (y - x * x).powi(2))
}

fn rosenbrock_jac(x: f32, y: f32) -> Mat2 {
    Matrix2::new(
        -2.0 * (A - x),
        0.0, // for 1st residual
        -4.0 * B * x * (y - x * x),
        2.0 * B * (y - x * x), // for 2nd residual
    )
}

fn run() -> Result<(), (f32, f32)> {
    let x_0 = -2.0;
    let y_0 = -2.0;

    // Run iterative optimization with Levenberg-Marquardt.
    let qn_state = LMOptimizer::init(&(), (x_0, y_0))?;
    let state = LMState {
        lm_coef: 0.1,
        data: qn_state,
    };
    LMOptimizer::iterative(&(), state);
    Ok(())
}

// Levenberg-Marquardt #########################################################

struct LMState {
    lm_coef: f32,
    data: QNState,
}

struct QNState {
    model: (f32, f32),
    energy: f32,
    gradient: Vec2,
    hessian: Mat2,
}

impl State<(f32, f32), f32> for LMState {
    fn model(&self) -> &(f32, f32) {
        &self.data.model
    }
    fn energy(&self) -> f32 {
        self.data.energy
    }
}

type LMPartialState = Result<QNState, (f32, f32)>;
type PreEval = Vec2;

struct LMOptimizer;

impl Optimizer<(), LMState, Vec2, (f32, f32), PreEval, LMPartialState, f32> for LMOptimizer {
    fn initial_energy() -> f32 {
        f32::INFINITY
    }

    fn compute_step(state: &LMState) -> Option<Vec2> {
        let mut hessian = state.data.hessian.clone();
        hessian.m11 = (1.0 + state.lm_coef) * hessian.m11;
        hessian.m22 = (1.0 + state.lm_coef) * hessian.m22;
        hessian.cholesky().map(|ch| ch.solve(&state.data.gradient))
    }

    fn apply_step(delta: Vec2, model: &(f32, f32)) -> (f32, f32) {
        let (x, y) = *model;
        (x - delta.x, y - delta.y)
    }

    fn pre_eval(_obs: &(), model: &(f32, f32)) -> PreEval {
        let (x, y) = *model;
        rosenbrock_res(x, y)
    }

    fn eval(_obs: &(), energy: f32, residuals: PreEval, model: (f32, f32)) -> LMPartialState {
        let new_energy = residuals.norm_squared();
        if new_energy > energy {
            Err(model)
        } else {
            let (x, y) = model;
            let jacobian = rosenbrock_jac(x, y);
            let jacobian_t = jacobian.transpose();
            let gradient = jacobian_t * residuals;
            let hessian = jacobian_t * jacobian;
            Ok(QNState {
                model,
                energy: new_energy,
                gradient,
                hessian,
            })
        }
    }

    fn stop_criterion(nb_iter: usize, s0: LMState, s1: LMPartialState) -> (LMState, Continue) {
        let too_many_iterations = nb_iter > 50;
        match (s1, too_many_iterations) {
            // Max number of iterations reached:
            (Err(_), true) => (s0, Continue::Stop),
            (Ok(qn_state), true) => {
                println!("(x, y) = ({}, {})", qn_state.model.0, qn_state.model.1);
                let kept_state = LMState {
                    lm_coef: s0.lm_coef, // does not matter actually
                    data: qn_state,
                };
                (kept_state, Continue::Stop)
            }
            // Can continue to iterate:
            (Err(model), false) => {
                println!("\t back from ({}, {})", model.0, model.1);
                let mut kept_state = s0;
                kept_state.lm_coef = 10.0 * kept_state.lm_coef;
                (kept_state, Continue::Forward)
            }
            (Ok(qn_state), false) => {
                println!("(x, y) = ({}, {})", qn_state.model.0, qn_state.model.1);
                let kept_state = LMState {
                    lm_coef: 0.1 * s0.lm_coef,
                    data: qn_state,
                };
                (kept_state, Continue::Forward)
            }
        }
    }
}
