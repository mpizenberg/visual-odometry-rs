// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion, Fun};
use visual_odometry_rs::math::accumulator;
use visual_odometry_rs::misc::type_aliases::{Mat6, Vec6};

// Functions.

fn normal_sum(nb_iter: u32, vec: &Vec6) -> Mat6 {
    let mut mat = Mat6::zeros();
    for _ in 0..nb_iter {
        mat += vec * vec.transpose();
    }
    mat
}

fn accum_sum(nb_iter: u32, vec: &Vec6) -> Mat6 {
    let mut accum = accumulator::SymMat6::new();
    for _ in 0..nb_iter {
        accum.add_vec(vec);
    }
    accum.flush();
    accum.to_mat()
}

// Benches.

fn bench_normal_sum(b: &mut Bencher, nb_iter: &u32) {
    let vec = Vec6::repeat(1.0);
    b.iter(|| black_box(normal_sum(*nb_iter, &vec)));
}

fn bench_accum_sum(b: &mut Bencher, nb_iter: &u32) {
    let vec = Vec6::repeat(1.0);
    b.iter(|| black_box(accum_sum(*nb_iter, &vec)));
}

fn criterion_benchmark(c: &mut Criterion) {
    let funs = vec![
        Fun::new("Normal", bench_normal_sum),
        Fun::new("Accum", bench_accum_sum),
    ];
    c.bench_functions("Accumulator", funs, 1000);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
