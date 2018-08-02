//! Gauss Newton optimization for 2D

extern crate computer_vision_rs as cv;
extern crate nalgebra as na;
extern crate rand;

use cv::optimization::{self, Continue};
use na::{DMatrix, DVector, MatrixMN, Vector2};
use rand::distributions::Uniform;
use rand::thread_rng;
use rand::{SeedableRng, StdRng};
use std::f32::EPSILON;

fn main() {
    // Create a mesh grid
    let nb: usize = 33;
    let (x_grid, y_grid) = meshgrid(-8.0, 8.0, nb);
    let x_grid = DVector::<f32>::from_column_slice(nb * nb, x_grid.as_slice());
    let y_grid = DVector::<f32>::from_column_slice(nb * nb, y_grid.as_slice());

    // Functions of the model
    let radius = |_: f32, b: f32| {
        x_grid.zip_map(&y_grid, |x, y| {
            ((x - b).powi(2) + y.powi(2)).sqrt() + EPSILON
        })
    };
    let f = |a, b| radius(a, b).map(|x| a * sinc(x));

    // Noisy data
    let seed = [1; 32];
    let mut _rng_seed: StdRng = SeedableRng::from_seed(seed);
    let mut _rng = thread_rng();
    let mut distribution = Uniform::new(-1.0, 1.0);
    let noise: DVector<f32> =
        DVector::from_distribution(nb * nb, &mut distribution, &mut _rng_seed);
    let data_noise = f(10.0, 3.0) + 0.1 * noise;

    // Using the optimization module.
    let eval = |observation: &DVector<f32>, model: &Vector2<f32>| {
        let a = model[0];
        let b = model[1];
        let radius_grid = radius(a, b);
        let f_model = radius_grid.map(|r| a * sinc(r));

        let residual = f_model - observation;
        let d_res_a = radius_grid.map(&sinc);
        let d_res_b = radius_grid
            .map(|r| a * (cos(r) - sinc(r)) / (r * r))
            .component_mul(&x_grid.map(|x| b - x));
        // let jacobian: MatrixMN<_, _, na::U2> = MatrixMN::from_columns(&[d_res_a, d_res_b]);
        let jacobian = MatrixMN::from_columns(&[d_res_a, d_res_b]);

        (jacobian, residual)
    };

    let step = |jacobian: &MatrixMN<f32, na::Dynamic, na::U2>,
                residual: &DVector<f32>,
                model: &Vector2<f32>| {
        println!("(a_n, b_n): ({}, {})", model[0], model[1]);
        model - jacobian.clone().svd(true, true).solve(residual, EPSILON)
    };

    let stop_criterion = |nb_iter, _, residual: &DVector<f32>| {
        let new_energy = residual.norm_squared();
        let continuation = if nb_iter < 6 {
            Continue::Forward
        } else {
            Continue::Stop
        };
        (new_energy, continuation)
    };

    let _ = optimization::gauss_newton(
        eval,
        step,
        stop_criterion,
        &data_noise,
        Vector2::new(5.0, 1.0),
    );
}

// Helper functions

fn meshgrid(start: f32, end: f32, size: usize) -> (DMatrix<f32>, DMatrix<f32>) {
    let x_grid = DMatrix::from_fn(size, size, |_, j| {
        (j as f32 * end + (size - j) as f32 * start) / (size as f32 - 1.0)
    });
    let y_grid = DMatrix::from_fn(size, size, |i, _| {
        (i as f32 * end + (size - i) as f32 * start) / (size as f32 - 1.0)
    });
    (x_grid, y_grid)
}

fn sinc(x: f32) -> f32 {
    x.sin() / x
}

fn cos(x: f32) -> f32 {
    x.cos()
}
