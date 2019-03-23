// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use visual_odometry_rs::core::multires;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("mean_pyramid 5 480x640", |b| {
        let mat: DMatrix<u8> = DMatrix::repeat(480, 640, 1);
        b.iter(|| multires::mean_pyramid(5, mat.clone()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
