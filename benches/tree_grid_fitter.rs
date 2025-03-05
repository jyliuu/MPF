use criterion::{criterion_group, criterion_main, Criterion};
use mpf::grid::{fitter, TreeGridParams};
use mpf::test_data::setup_data_csv;
use rand::thread_rng;

fn bench_tree_grid_fitter(c: &mut Criterion) {
    let (x, y) = setup_data_csv();

    let mut group = c.benchmark_group("TreeGridFitter");

    group.bench_function("OptimizedFitter", |b| {
        b.iter(|| {
            let mut rng = thread_rng();
            fitter::fit(x.view(), y.view(), &TreeGridParams::default(), &mut rng);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_tree_grid_fitter);
criterion_main!(benches);
