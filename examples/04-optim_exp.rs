extern crate computer_vision_rs as cv;
extern crate nalgebra as na;
extern crate rand;

use cv::optimization::{Continue, QuasiNewtonOptimizer, State};
use na::{DVector, Matrix1};
use rand::distributions::Uniform;
use rand::{SeedableRng, StdRng};

type Vect = DVector<f32>;
type Scalar = Matrix1<f32>;
type OptimState<P> = State<P, f32, f32, Vect, Vect, Scalar>;
type GNState = OptimState<()>;
type LMState = OptimState<f32>;
struct Obs {
    x: Vect,
    y: Vect,
}

fn main() {
    let a_model = 1.5;
    let initial_model = 0.0;
    let nb_data: usize = 100;

    // Noisy data to avoid perfect conditions.
    let domain = linspace(-2.0, 3.0, nb_data);
    let seed = [0; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut distribution = Uniform::from(-1.0..1.0);
    let noise: DVector<f32> = DVector::from_distribution(nb_data, &mut distribution, &mut rng);
    let data_noisy = f(a_model, &domain) + 0.05 * noise;
    let observations = Obs {
        x: domain,
        y: data_noisy,
    };

    // Run iterative optimization with Gauss Newton.
    let state = GaussNewtonOptimizer::eval(&observations, initial_model, ());
    GaussNewtonOptimizer::iterative(&observations, state);

    // Run iterative optimization with Gauss Newton.
    let state = LMOptimizer::eval(&observations, initial_model, 0.1);
    LMOptimizer::iterative(&observations, state);
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - 1 - i) as f32 * start) / (nb as f32 - 1.0)
    })
}

fn f(a: f32, domain: &Vect) -> Vect {
    domain.map(|x| (-a * x).exp())
}

struct GaussNewtonOptimizer;

impl QuasiNewtonOptimizer<Obs, (), f32, f32, Vect, Vect, Scalar, Scalar> for GaussNewtonOptimizer {
    fn eval(observations: &Obs, model: f32, params: ()) -> GNState {
        eval_f(observations, model, params)
    }

    fn step(state: &GNState) -> Scalar {
        step_gauss_newton(state)
    }

    fn apply(delta: Scalar, model: &f32) -> f32 {
        model - delta.x
    }

    fn stop_criterion(nb_iter: usize, s0: GNState, s1: GNState) -> (GNState, Continue) {
        stop_criterion_gauss_newton(nb_iter, s0, s1)
    }
}

struct LMOptimizer;

impl QuasiNewtonOptimizer<Obs, f32, f32, f32, Vect, Vect, Scalar, Scalar> for LMOptimizer {
    fn eval(observations: &Obs, model: f32, params: f32) -> LMState {
        eval_f(observations, model, params)
    }

    fn step(state: &LMState) -> Scalar {
        step_levenberg_marquardt(state)
    }

    fn apply(delta: Scalar, model: &f32) -> f32 {
        model - delta.x
    }

    fn stop_criterion(nb_iter: usize, s0: LMState, s1: LMState) -> (LMState, Continue) {
        stop_criterion_levenberg_marquardt(nb_iter, s0, s1)
    }
}

// Levenberg Marquardt specific ######################################

fn step_levenberg_marquardt(state: &LMState) -> Scalar {
    let mut hessian: Scalar = state.jacobian.transpose() * &state.jacobian;
    hessian.x = (1.0 + state.params) * hessian.x;
    hessian.cholesky().unwrap().solve(&state.gradient)
}

fn stop_criterion_levenberg_marquardt(
    nb_iter: usize,
    old_state: LMState,
    new_state: LMState,
) -> (LMState, Continue) {
    println!("(old_a, new_a) = ({},{})", old_state.model, new_state.model);
    let higher_energy = new_state.energy > old_state.energy;
    let too_many_iters = nb_iter > 20;
    match (higher_energy, too_many_iters) {
        // Cases with too many iterations:
        (false, true) => (new_state, Continue::Stop),
        (true, true) => (old_state, Continue::Stop),
        // Cases when we continue to iterate:
        (false, false) => {
            let mut kept_state = new_state;
            kept_state.params = 0.1 * kept_state.params;
            (kept_state, Continue::Forward)
        }
        (true, false) => {
            let mut kept_state = old_state;
            kept_state.params = 10.0 * kept_state.params;
            (kept_state, Continue::Forward)
        }
    }
}

// Gauss-Newton specific #############################################

fn step_gauss_newton(state: &GNState) -> Scalar {
    state.jacobian.clone().pseudo_inverse(0.0) * &state.residuals
}

fn stop_criterion_gauss_newton(
    nb_iter: usize,
    old_state: GNState,
    new_state: GNState,
) -> (GNState, Continue) {
    let continuation = if nb_iter < 5 {
        Continue::Forward
    } else {
        Continue::Stop
    };
    println!("a = {}", new_state.model);
    (new_state, continuation)
}

// Shared functions ##################################################

fn eval_f<P>(observations: &Obs, model: f32, params: P) -> OptimState<P> {
    let f_model = f(model, &observations.x);
    let residuals = &f_model - &observations.y;
    let energy = residuals.iter().map(|r| r * r).sum();
    // Is it possible to avoid computing the jacobian and gradient
    // (in case of higher energy) without complexifying the optimization API?
    let jacobian = -1.0 * f_model.component_mul(&observations.x);
    let gradient = &jacobian.transpose() * &residuals;
    State {
        params,
        energy,
        model,
        residuals,
        jacobian,
        gradient,
    }
}
