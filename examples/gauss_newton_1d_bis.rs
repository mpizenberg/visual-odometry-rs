extern crate computer_vision_rs as cv;
extern crate nalgebra as na;
extern crate rand;

use cv::optimization_bis::{Continue, Optimizer, State};
use na::DVector;
use rand::{distributions::Uniform, SeedableRng, StdRng};
use std::f32;

type Vect = DVector<f32>;
struct Obs {
    x: Vect,
    y: Vect,
}

fn main() {
    let _ = run();
}

fn run() -> Result<(), f32> {
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
    let state = GNOptimizer::init(&observations, initial_model)?;
    GNOptimizer::iterative(&observations, state);
    Ok(())
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - 1 - i) as f32 * start) / (nb as f32 - 1.0)
    })
}

fn f(a: f32, domain: &Vect) -> Vect {
    domain.map(|x| (-a * x).exp())
}

// Gauss-Newton ################################################################

struct GNState {
    model: f32,
    residuals: Vect,
    energy: f32,
    jacobian: Vect,
}

impl State<f32, f32> for GNState {
    fn model(&self) -> &f32 {
        &self.model
    }
    fn energy(&self) -> f32 {
        self.energy
    }
}

type GNPartialState = Result<GNState, f32>;
type PreEval = (Vect, Vect, f32);

struct GNOptimizer;

impl Optimizer<Obs, GNState, f32, f32, PreEval, GNPartialState, f32> for GNOptimizer {
    fn initial_energy() -> f32 {
        f32::INFINITY
    }

    fn compute_step(state: &GNState) -> f32 {
        (state.jacobian.clone().pseudo_inverse(0.0) * &state.residuals).x
    }

    fn apply_step(delta: f32, model: &f32) -> f32 {
        model - delta
    }

    fn pre_eval(obs: &Obs, model: &f32) -> PreEval {
        let f_model = f(*model, &obs.x);
        let residuals = &f_model - &obs.y;
        let new_energy = residuals.iter().map(|r| r * r).sum();
        (f_model, residuals, new_energy)
    }

    fn eval(obs: &Obs, energy: f32, pre_eval: PreEval, model: f32) -> GNPartialState {
        let (f_model, residuals, new_energy) = pre_eval;
        if new_energy > energy {
            Err(new_energy)
        } else {
            let jacobian = -1.0 * f_model.component_mul(&obs.x);
            Ok(GNState {
                model,
                residuals,
                energy: new_energy,
                jacobian,
            })
        }
    }

    fn stop_criterion(nb_iter: usize, s0: GNState, s1: GNPartialState) -> (GNState, Continue) {
        let too_many_iterations = nb_iter > 5;
        match (s1, too_many_iterations) {
            (Err(_energy), _) => (s0, Continue::Stop),
            (Ok(state), true) => (state, Continue::Stop),
            (Ok(state), false) => {
                println!("a = {}", state.model);
                (state, Continue::Forward)
            }
        }
    }
}
