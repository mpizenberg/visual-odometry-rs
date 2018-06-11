//! Getting started with nalgebra

extern crate nalgebra as na;
extern crate rand;

use na::DVector;
use rand::distributions::Uniform;
use rand::thread_rng;

fn main() {
    // Create vector [0, 0.01, ..., 3]
    let nb: usize = 4;
    let domain = linspace(0.0, 3.0, nb);

    // Exponential model
    let f = |a: f32| domain.map(|x| (-a * x).exp());

    // Noisy data
    let mut distribution = Uniform::from(0.0..1.0);
    let mut rng = thread_rng();
    let noise: DVector<f32> = DVector::from_distribution(nb, &mut distribution, &mut rng);
    let data_noise = f(1.3) + 0.05 * noise;

    // Energy
    let res = |a| f(a) - &data_noise;
    let energy = |a| res(a).norm_squared();

    // Gauss Newton
    let d_res = |a| -1.0 * domain.component_mul(&f(a));
    let gradient: &Fn(f32) -> f32 = &|a| 2.0 * d_res(a).component_mul(&res(a)).iter().sum::<f32>();
    let hessian = |a| 2.0 * d_res(a).norm_squared();

    // Initialization
    let mut x_n = 0.0;
    let mut e_n = energy(x_n);
    println!("x_n: {}, E_n: {}", x_n, e_n);

    // Iterate
    for _ in 0..5 {
        x_n = x_n - gradient(x_n) / hessian(x_n);
        e_n = energy(x_n);
        println!("x_n: {}, E_n: {}", x_n, e_n);
    }
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - i) as f32 * start) / (nb as f32 - 1.0)
    })
}
