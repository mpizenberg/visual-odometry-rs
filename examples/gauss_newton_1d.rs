//! Getting started with nalgebra

extern crate computer_vision_rs as cv;
extern crate nalgebra as na;
extern crate rand;

use cv::optimization::{self, Continue};
use na::DVector;
use rand::distributions::Uniform;
use rand::{SeedableRng, StdRng};

fn main() {
    // Create vector [0, 0.01, ..., 3]
    let nb: usize = 100;
    let domain = linspace(0.0, 3.0, nb);

    // Exponential model
    let f = |a: f32| domain.map(|x| (-a * x).exp());

    // Noisy data
    let seed = [0; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut distribution = Uniform::from(-1.0..1.0);
    let noise: DVector<f32> = DVector::from_distribution(nb, &mut distribution, &mut rng);
    let data_noise = f(1.5) + 0.05 * noise;

    // Using the optimization module.
    let eval = |observation: &DVector<f32>, model: &f32| {
        let f_model = f(*model);
        let jacobian = -1.0 * domain.component_mul(&f_model);
        let residual = f_model - observation;
        (jacobian, residual)
    };
    let step_gauss_newton = |jacobian: &DVector<f32>, residual: &DVector<f32>, model: &f32| {
        println!("a_n: {}", model);
        let gradient = 2.0 * jacobian.component_mul(residual).iter().sum::<f32>();
        let hessian = 2.0 * jacobian.norm_squared();
        model - gradient / hessian
    };
    let stop_criterion = |nb_iter, _, residual: &DVector<f32>| {
        let new_energy = residual.norm_squared();
        // println!("E_n: {}", new_energy);
        let continuation = if nb_iter < 5 {
            Continue::Forward
        } else {
            Continue::Stop
        };
        (new_energy, continuation)
    };

    let (model, _) =
        optimization::iterative(eval, step_gauss_newton, stop_criterion, &data_noise, 0.0);
    println!("a: {}", model);
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - i) as f32 * start) / (nb as f32 - 1.0)
    })
}
