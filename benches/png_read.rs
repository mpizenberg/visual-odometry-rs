// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate computer_vision_rs as cv;
extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion, Fun, ParameterizedBenchmark};

fn criterion_benchmark(c: &mut Criterion) {
    let basis = Fun::new("Basis", |b, img| {
        b.iter(|| cv::helper::read_png_16bits(img))
    });
    let second = Fun::new("No-0", |b, img| {
        b.iter(|| cv::helper::read_png_16bits_bis(img))
    });
    c.bench_functions("Read png", vec![basis, second], "icl-depth/0.png");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
