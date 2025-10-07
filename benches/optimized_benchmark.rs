use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use eskf::{Builder, ESKF};
use nalgebra::Vector3;

fn benchmark_predict(
    c: &mut Criterion,
    predict: fn(&mut ESKF, Vector3<f32>, Vector3<f32>, f32),
    id: &'static str,
) {
    let mut filter = Builder::new()
        .acceleration_variance(0.01)
        .rotation_variance(0.001)
        .acceleration_bias(0.0001)
        .rotation_bias(0.0001)
        .initial_covariance(1e-6)
        .build();

    let acc = Vector3::new(0.1, 0.2, -9.81);
    let gyr = Vector3::new(0.01, -0.02, 0.005);
    let dt = 0.01;

    c.bench_function(id, |b| {
        b.iter(|| predict(&mut filter, black_box(acc), black_box(gyr), black_box(dt)))
    });
}

fn predict_original(c: &mut Criterion) {
    benchmark_predict(c, ESKF::predict, "predict_original")
}

fn predict_optimized(c: &mut Criterion) {
    benchmark_predict(c, ESKF::predict_optimized, "predict_optimized")
}

criterion_group!(
    benches,
    predict_original,
    predict_optimized,
);

criterion_main!(benches);
