//! Getting started with nalgebra

extern crate computer_vision_rs as cv;
extern crate nalgebra as na;
extern crate rand;

use cv::optimization::{self, Continue};
use na::DVector;
use rand::distributions::Uniform;
use rand::{SeedableRng, StdRng};

fn main() {
    // Create vector [0, 0.01, ..., 3].
    let nb: usize = 100;
    // [-2, 3] -> converges
    // [-3, 3] -> converges slowly
    // [-4, 3] -> does not converge
    let domain = linspace(-3.0, 3.0, nb);

    // Exponential model (unknown a): f(a,x) = exp(-ax).
    let f = |a: f32| domain.map(|x| (-a * x).exp());
    let a_model = 1.5;
    let initial_model = 0.0;

    // Noisy data to avoid perfect conditions.
    let seed = [0; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut distribution = Uniform::from(-1.0..1.0);
    let noise: DVector<f32> = DVector::from_distribution(nb, &mut distribution, &mut rng);
    let data_noisy = f(a_model) + 0.05 * noise;

    // The eval function computes the residual and derivatives
    // from a current model and observations.
    let eval = |observation: &DVector<f32>, model: &f32| {
        let f_model = f(*model);
        let jacobian = -1.0 * domain.component_mul(&f_model);
        let residuals = f_model - observation;
        (jacobian, residuals)
    };

    // The Gauss-Newton step consists in computing the new model
    // with the formula x_n+1 = x_n - (J^T J)^(-1) J^T r_n
    let step_gauss_newton = |jacobian: &DVector<f32>, residual: &DVector<f32>, model: &f32| {
        println!("a_n: {}", model);
        model - (jacobian.clone().pseudo_inverse(0.0) * residual).x
    };

    // Stop iterations after max 5 iterations.
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

    // Run the iterative Gauss-Newton optimization.
    let (model, _) = optimization::iterative(
        eval,
        step_gauss_newton,
        stop_criterion,
        &data_noisy,
        initial_model,
    );
    println!("a: {}", model);
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - 1 - i) as f32 * start) / (nb as f32 - 1.0)
    })
}
