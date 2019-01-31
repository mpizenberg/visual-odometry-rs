extern crate computer_vision_rs as cv;
extern crate nalgebra as na;
extern crate rand;

use cv::optimization::{Continue, QuasiNewtonOptimizer, State};
use na::{DVector, Matrix1};
use rand::distributions::Uniform;
use rand::{SeedableRng, StdRng};

type Vect = DVector<f32>;
type Scalar = Matrix1<f32>;
type OptimState = State<(), f32, f32, Vect, Vect, Scalar>;
struct Obs {
    x: Vect,
    y: Vect,
}

fn f(a: f32, domain: &Vect) -> Vect {
    domain.map(|x| (-a * x).exp())
}

struct MyOptimizer;

impl QuasiNewtonOptimizer<Obs, (), f32, f32, Vect, Vect, Scalar, Scalar> for MyOptimizer {
    fn eval(observations: &Obs, model: f32) -> OptimState {
        let f_model = f(model, &observations.x);
        let jacobian = -1.0 * observations.x.component_mul(&f_model);
        let residuals = f_model - &observations.y;
        let gradient = &jacobian.transpose() * &residuals;
        let energy = residuals.iter().map(|r| r * r).sum();
        let params = ();
        State {
            params,
            energy,
            model,
            residuals,
            jacobian,
            gradient,
        }
    }

    fn step(state: &OptimState) -> Scalar {
        state.jacobian.clone().pseudo_inverse(0.0) * &state.residuals
    }

    fn apply(delta: Scalar, model: &f32) -> f32 {
        model - delta.x
    }

    fn stop_criterion(
        nb_iter: usize,
        old_state: OptimState,
        new_state: OptimState,
    ) -> (OptimState, Continue) {
        let continuation = if nb_iter < 5 {
            Continue::Forward
        } else {
            Continue::Stop
        };
        println!("a = {}", new_state.model);
        (new_state, continuation)
    }
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

    // Run iterative optimization.
    let state = MyOptimizer::eval(&observations, initial_model);
    MyOptimizer::iterative(&observations, state);
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - 1 - i) as f32 * start) / (nb as f32 - 1.0)
    })
}
