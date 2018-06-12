//! Gauss Newton optimization for 2D

extern crate nalgebra as na;
extern crate rand;

use na::{DMatrix, DVector, Vector2};
use rand::distributions::Uniform;
use rand::thread_rng;
use rand::{SeedableRng, StdRng};
use std::f32::EPSILON;

fn main() {
    // Create a mesh grid
    let nb: usize = 33;
    let (x_grid, y_grid) = meshgrid(-8.0, 8.0, nb);
    let x_grid = x_grid.resize(nb * nb, 1, 0.0).column(0).into_owned();
    let y_grid = y_grid.resize(nb * nb, 1, 0.0).column(0).into_owned();

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
    let noise: DVector<f32> = DVector::from_distribution(nb * nb, &mut distribution, &mut _rng);
    let data_noise = f(10.0, 3.0) + 0.1 * noise;

    // Energy
    let res = |a, b| f(a, b) - &data_noise;
    let energy = |a, b| res(a, b).norm_squared();

    // Partial derivatives
    let d_res_a = |a, b| radius(a, b).map(&sinc);
    let d_res_b = |a, b| {
        radius(a, b)
            .map(|r| a * (cos(r) - sinc(r)) / (r * r))
            .component_mul(&x_grid.map(|x| b - x))
    };
    let jacobian = |a, b| DMatrix::from_columns(&[d_res_a(a, b), d_res_b(a, b)]);

    // Gradient and Hessian
    // let gradient = |a, b| jacobian(a, b).tr_mul(&res(a, b));
    // let hessian = |a, b| {
    //     let jac = jacobian(a, b);
    //     jac.tr_mul(&jac)
    // };

    // Iteration step
    let step = |a, b| jacobian(a, b).svd(true, true).solve(&res(a, b), EPSILON);
    // let step = |a, b| hessian(a, b).try_inverse().unwrap() * gradient(a, b);

    // Initialization
    let mut x_n = Vector2::new(5.0, 1.0);
    let (mut a_n, mut b_n) = (x_n[0], x_n[1]);
    let mut e_n = energy(a_n, b_n);
    println!("a_n: {}, b_n: {}, E_n: {}", a_n, b_n, e_n);

    // Iterate
    for _ in 0..8 {
        x_n = x_n - step(a_n, b_n);
        a_n = x_n[0];
        b_n = x_n[1];
        e_n = energy(a_n, b_n);
        println!("a_n: {}, b_n: {}, E_n: {}", a_n, b_n, e_n);
    }
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
