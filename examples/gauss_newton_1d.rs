//! Getting started with nalgebra

extern crate nalgebra as na;
extern crate rand;

use na::DVector;
use rand::distributions::Uniform;
use rand::thread_rng;

fn main() {
    // Create vector [0, 0.01, ..., 3]
    let nb: usize = 4;
    let d = linspace(0.0, 3.0, nb);
    // println!("d = {}", d);
    let a = 1.3;
    let y: DVector<f32> = d.iter().map(|x| (-a * x).exp()).collect();
    let mut distribution = Uniform::from(0.0..1.0);
    let mut rng = thread_rng();
    // let noise: DVector<f32> = DVector::from_distribution(nb, &mut distribution, &mut rng);
    // let y_noise = y + 0.05 * noise;
}

fn linspace(start: f32, end: f32, nb: usize) -> DVector<f32> {
    DVector::from_fn(nb, |i, _| {
        (i as f32 * end + (nb - i) as f32 * (start as f32)) / (nb as f32 - 1.0)
    })
}
